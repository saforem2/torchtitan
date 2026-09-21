# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.parallel import DistributedDataParallel

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims


class NativeDDP(DistributedDataParallel):
    """DDP wrapper that preserves access to the underlying model interface."""

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


_AGPT_DTYPE_POLICIES = {
    "uniform_bfloat16": {
        "tok_embedding_weight": "bfloat16",
        "tok_embedding_output": "bfloat16",
        "block0_input": "bfloat16",
        "attention_norm_weight": "bfloat16",
        "attention_norm_output": "bfloat16",
        "final_norm_output": "bfloat16",
        "lm_head_weight": "bfloat16",
        "lm_head_input": "bfloat16",
        "lm_head_output": "bfloat16",
    },
    # XPU autocast leaves embedding/RMSNorm/residual tensors and stored weights
    # in FP32 while autocasting eligible linear algebra, including the LM head,
    # to BF16. This is deliberately not described as BF16 parameter parity with
    # FSDP: it is the correctness contract for ordinary DDP's safe fallback.
    "autocast": {
        "tok_embedding_weight": "float32",
        "tok_embedding_output": "float32",
        "block0_input": "float32",
        "attention_norm_weight": "float32",
        "attention_norm_output": "float32",
        "final_norm_output": "float32",
        "lm_head_weight": "float32",
        "lm_head_input": "float32",
        "lm_head_output": "bfloat16",
    },
}


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None


def scale_native_ddp_loss(loss: torch.Tensor, dp_degree: int) -> torch.Tensor:
    """Preserve the local loss value while compensating DDP gradient averaging."""
    if dp_degree <= 1:
        raise ValueError(f"native DDP requires dp_degree > 1, got {dp_degree}")
    return _ScaleGradient.apply(loss, float(dp_degree))


def wrap_native_ddp_loss(loss_fn: object, dp_degree: int) -> Callable[..., object]:
    """Scale native-DDP gradients without hiding the validated loss object."""
    if not isinstance(loss_fn, CrossEntropyLoss):
        raise ValueError("native DDP currently requires standard CrossEntropyLoss")

    def wrapped(*args, **kwargs):
        result = loss_fn(*args, **kwargs)
        if isinstance(result, tuple):
            return (scale_native_ddp_loss(result[0], dp_degree), *result[1:])
        return scale_native_ddp_loss(result, dp_degree)

    return wrapped


def native_ddp_autocast_context(
    base_context: Callable[[], AbstractContextManager[None]],
    device_type: str,
) -> Callable[[], AbstractContextManager[None]]:
    """Compose the trainer SPMD context with native-DDP BF16 autocast."""

    @contextmanager
    def context():
        with base_context(), torch.autocast(
            device_type=device_type, dtype=torch.bfloat16
        ):
            yield

    return context


def validate_native_ddp(
    *,
    model_name: str,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    loss_fn: object,
    gradient_accumulation_steps: int,
    fault_tolerance_enabled: bool,
    create_seed_checkpoint: bool,
    optimizer_has_param_groups: bool,
) -> None:
    if not parallelism.enable_data_parallel_native_ddp:
        return
    if model_name != "ezpz.agpt":
        raise ValueError("native DDP is currently supported only for ezpz.agpt")
    if not parallel_dims.dp_replicate_enabled or parallel_dims.dp_shard != 1:
        raise ValueError(
            "native DDP requires data_parallel_replicate_degree > 1 and "
            "data_parallel_shard_degree = 1"
        )
    if parallel_dims.tp != 1 or parallel_dims.cp != 1:
        raise ValueError("native DDP currently requires TP=CP=1")
    if parallel_dims.ep != 1:
        raise ValueError("native DDP does not support expert parallelism")
    if parallel_dims.pp != 1:
        raise ValueError("native DDP does not support pipeline parallelism")
    if getattr(parallelism, "native_ddp_compute_policy", None) != "autocast":
        raise ValueError("native DDP only supports the autocast compute policy")
    if training.enable_cpu_offload:
        raise ValueError("native DDP does not support CPU offload")
    if training.dtype != "float32" or training.mixed_precision_reduce != "float32":
        raise ValueError("native DDP requires FP32 master parameters and reductions")
    if training.mixed_precision_param != "bfloat16":
        raise ValueError("native DDP currently requires BF16 autocast")
    if not isinstance(loss_fn, CrossEntropyLoss):
        raise ValueError("native DDP currently requires standard CrossEntropyLoss")
    if gradient_accumulation_steps != 1:
        raise ValueError("native DDP currently requires one gradient accumulation step")
    if fault_tolerance_enabled:
        raise ValueError("native DDP does not support fault tolerance")
    if create_seed_checkpoint:
        raise ValueError("native DDP cannot create a seed checkpoint")
    if optimizer_has_param_groups:
        raise ValueError("native DDP does not yet support optimizer param_groups")
    if parallelism.enable_data_parallel_replicate_module:
        raise ValueError("native DDP and ReplicateModule are mutually exclusive")
    # These three fields exist on the upstream Aurora MoE branch's core
    # ParallelismConfig but NOT on ours -- EzpzParallelismConfig deliberately
    # ports only the five native-DDP fields our code reads, and core
    # torchtitan has never had them. Reading them directly raised
    # AttributeError on a slotted dataclass, so this validator could not run
    # at all. getattr with the upstream default keeps the rejection
    # meaningful if the fields are ever added, without inventing config.
    if getattr(parallelism, "enable_fsdp_async_all_reduce", False):
        raise ValueError("native DDP does not use the FSDP async all-reduce path")
    if getattr(parallelism, "pipeline_parallel_fsdp_overlap", False) or (
        getattr(parallelism, "pipeline_parallel_fsdp_overlap_policy", "bulk") != "bulk"
    ):
        raise ValueError("native DDP does not use pipeline FSDP overlap")
    if parallelism.native_ddp_bucket_cap_mb <= 0:
        raise ValueError("native_ddp_bucket_cap_mb must be positive")


def wrap_native_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    bucket_cap_mb: float,
    compute_policy: str = "autocast",
    bucketize_first_iteration: bool = False,
) -> NativeDDP:
    if compute_policy != "autocast":
        raise ValueError("native DDP only supports the autocast compute policy")
    device = next(model.parameters()).device
    device_ids = None if device.type == "cpu" else [device.index]
    mixed_precision = None
    process_group = dp_mesh.get_group()
    initial_bucket_kwargs = {}
    if bucketize_first_iteration:
        signature = inspect.signature(DistributedDataParallel)
        if "bucket_cap_mb_list" not in signature.parameters:
            raise RuntimeError(
                "native DDP first-iteration bucketization requires "
                "DistributedDataParallel.bucket_cap_mb_list"
            )
        if torch._dynamo.utils.get_optimize_ddp_mode() == "python_reducer":
            raise RuntimeError(
                "native DDP first-iteration bucketization is incompatible "
                "with the Python reducer"
            )
        initial_bucket_kwargs["bucket_cap_mb_list"] = [bucket_cap_mb]

    def validate_initial_buckets(wrapped: NativeDDP) -> None:
        if not bucketize_first_iteration:
            return
        expected = int(bucket_cap_mb * 1024 * 1024)
        if wrapped.bucket_bytes_cap_list != [expected]:
            raise RuntimeError(
                "native DDP initial bucket limit mismatch: "
                f"{wrapped.bucket_bytes_cap_list} != {[expected]}"
            )
        limits = wrapped._bucket_config.compute_bucket_size_limits(
            static_graph=False,
            find_unused_parameters=False,
        )
        if limits != ([expected], [expected]):
            raise RuntimeError(
                "native DDP initial/rebuild bucket semantics mismatch: "
                f"{limits} != {([expected], [expected])}"
            )

    wrapped = NativeDDP(
        model,
        device_ids=device_ids,
        process_group=process_group,
        broadcast_buffers=False,
        bucket_cap_mb=bucket_cap_mb,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        mixed_precision=mixed_precision,
        **initial_bucket_kwargs,
    )
    validate_initial_buckets(wrapped)
    return wrapped


def _first_tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if (tensor := _first_tensor(item)) is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            if (tensor := _first_tensor(item)) is not None:
                return tensor
    return None


def install_agpt_dtype_probe(model: nn.Module) -> None:
    """Record first-forward parameter and activation dtypes for an AGPT model."""
    root = model.module if isinstance(model, NativeDDP) else model
    try:
        first_block = next(iter(root.layers.values()))
        modules = {
            "tok_embedding": root.tok_embeddings,
            "attention_norm": first_block.attention_norm,
            "final_norm": root.norm,
            "lm_head": root.lm_head,
        }
    except (AttributeError, StopIteration) as exc:
        raise TypeError("AGPT dtype probe requires a non-pipeline decoder") from exc
    if any(module is None for module in modules.values()):
        raise TypeError(
            "AGPT dtype probe requires embedding, norm, and LM head modules"
        )

    state: dict[str, str] = {}

    def record(key: str, value: object) -> None:
        if key not in state:
            tensor = _first_tensor(value)
            if tensor is None:
                raise RuntimeError(f"AGPT dtype probe found no tensor for {key}")
            state[key] = str(tensor.dtype).removeprefix("torch.")

    def pre_hook(key: str | None = None, parameter_key: str | None = None):
        def hook(module, args):
            # FSDP installs its pre-forward unshard/cast hook before this
            # diagnostic is registered.  Therefore a parameter read here is
            # the dtype consumed by compute.  Reading it from a post-forward
            # hook is too late: FSDP's earlier post hook may already have
            # resharded/restored the FP32 master.  Ordinary DDP has no such
            # cast hook, so this continues to observe its FP32 storage.
            if parameter_key is not None:
                parameter = next(module.parameters(recurse=False), None)
                if parameter is None:
                    raise RuntimeError(
                        f"AGPT dtype probe found no parameter for {parameter_key}"
                    )
                record(parameter_key, parameter)
            if key is not None:
                record(key, args)

        return hook

    def post_hook(key: str):
        def hook(_module, _args, output):
            record(key, output)

        return hook

    modules["tok_embedding"].register_forward_pre_hook(
        pre_hook(parameter_key="tok_embedding_weight")
    )
    modules["tok_embedding"].register_forward_hook(post_hook("tok_embedding_output"))
    first_block.register_forward_pre_hook(pre_hook("block0_input"))
    modules["attention_norm"].register_forward_pre_hook(
        pre_hook(parameter_key="attention_norm_weight")
    )
    modules["attention_norm"].register_forward_hook(post_hook("attention_norm_output"))
    modules["final_norm"].register_forward_hook(post_hook("final_norm_output"))
    modules["lm_head"].register_forward_pre_hook(
        pre_hook("lm_head_input", "lm_head_weight")
    )
    modules["lm_head"].register_forward_hook(post_hook("lm_head_output"))
    model._agpt_dtype_probe = state


def get_agpt_dtype_probe_data(
    model: nn.Module, expected_policy: str = "uniform_bfloat16"
) -> dict[str, str]:
    try:
        expected = _AGPT_DTYPE_POLICIES[expected_policy]
    except KeyError as exc:
        raise ValueError(
            f"unsupported AGPT dtype probe policy: {expected_policy}"
        ) from exc
    required = expected.keys()
    state = getattr(model, "_agpt_dtype_probe", None)
    if not isinstance(state, dict) or state.keys() != required:
        missing = set(required) - (state.keys() if isinstance(state, dict) else set())
        raise RuntimeError(f"AGPT dtype probe incomplete; missing={sorted(missing)}")
    if invalid := {
        key: (value, expected[key])
        for key, value in state.items()
        if value != expected[key]
    }:
        raise RuntimeError(f"AGPT dtype parity failure: {invalid}")
    return dict(sorted(state.items()))


def get_native_ddp_logging_data(model: nn.Module) -> dict[str, object]:
    """Return reducer timing and rebuilt-bucket evidence for an active DDP model."""
    if not isinstance(model, NativeDDP):
        raise TypeError(f"expected NativeDDP, got {type(model).__name__}")
    data = model._get_ddp_logging_data()
    keys = (
        "rank",
        "world_size",
        "backend_name",
        "bucket_cap_bytes",
        "bucket_sizes",
        "has_rebuilt_buckets",
        "rebuilt_bucket_sizes",
        "rebuilt_per_bucket_param_indices",
        "prev_iteration_grad_ready_order_indices",
        "num_buckets_reduced",
        "total_parameter_size_bytes",
        "iteration",
        "avg_forward_compute_time",
        "avg_backward_compute_time",
        "avg_backward_comm_time",
        "avg_backward_compute_comm_overlap_time",
    )
    result = {key: data[key] for key in keys if key in data}
    result["rank"] = dist.get_rank()
    result["world_size"] = dist.get_world_size()
    return result
