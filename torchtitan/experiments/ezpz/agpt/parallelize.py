# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply PT-D parallelisms + AC + compile + FSDP to the agpt model.

This is the agpt mirror of `torchtitan.models.llama3.parallelize`. It uses
the new config-based DTensor sharding API: TP is applied via
`model.parallelize(parallel_dims)`, which reads `sharding_config` declarations
that were filled in by `AgptModel.Config.update_from_config`.

Differences vs upstream `parallelize_llama`:

- `disable_fsdp_gradient_division` additionally enables
  `set_force_sum_reduction_for_comms(True)` for non-NCCL backends (CCL on
  XPU). Upstream's version only sets the divide factor.
- After `apply_compile`, resets `torch._dynamo.config.capture_scalar_outputs`
  to False. apply_compile sets it True for MoE; that breaks the
  separately-compiled CrossEntropyLoss for dense models.
- Names the FSDP grouping `[norm, lm_head]` together with
  `reshard_after_forward=False` (upstream uses
  `reshard_after_forward=reshard_after_forward_policy == "always"`).
"""

import ezpz
import ezpz.distributed
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
import os

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import (
    _maybe_regional_inductor_backend,
    apply_compile,
)
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.tools.logging import logger


# [ezpz] max-autotune (and other torch.compile modes) on XPU.
# The shared torchtitan apply_compile (distributed/compile.py) calls
# transformer_block.compile(backend=, fullgraph=True) with NO mode=, and its
# CompileConfig has no `mode` field. torch.compile mode="max-autotune" IS
# functional on Sunspot XPU (Triton GEMM autotune runs + selects triton_mm_*
# kernels), so this ezpz-scoped override threads a mode through WITHOUT editing
# core. Opt in via env: AGPT_COMPILE_MODE=max-autotune (or reduce-overhead, etc).
# Unset / "default" -> None -> byte-identical to the core apply_compile path.
# Reuses the core _maybe_regional_inductor_backend so FlexAttention handling is
# unchanged (its own inductor_configs still set max_autotune=False for the
# backward, which OOMs on XPU -- that is per-kernel and independent of this).
def _apply_compile_with_mode(model, compile_config) -> None:
    mode = os.environ.get("AGPT_COMPILE_MODE", "").strip() or None
    if mode in (None, "default"):
        apply_compile(model, compile_config)
        return
    # Mirror apply_compile's dynamo flags + backend resolution, adding mode=.
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    backend = _maybe_regional_inductor_backend(model, compile_config.backend)
    for _layer_id, transformer_block in model.layers.named_children():
        transformer_block.compile(backend=backend, mode=mode, fullgraph=True)
    logger.info(
        "Compiling each TransformerBlock with torch.compile "
        "(backend=%s, mode=%s) [ezpz AGPT_COMPILE_MODE]",
        compile_config.backend,
        mode,
    )



def parallelize_llama(
    model: Llama3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    skip_dp: bool = False,
):
    """Apply TP, AC, compile, and FSDP to an agpt model.

    The passed-in model preferably should be on meta device. Otherwise
    the model must fit on GPU or CPU memory.
    """
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # CP: wrap inner attention forward BEFORE parallelize() so CP logic
    # runs inside the local_map boundary on local tensors.
    if parallel_dims.cp_enabled:
        apply_cp_to_forward(
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    # TP via the config-based sharding API. The model's sharding_config
    # declarations were filled in by update_from_config (see model.py).
    # Upstream #3159 changed Module.parallelize to take ParallelDims (not a
    # bare tp_mesh) so each Module can resolve its own SPMD submesh.
    if parallel_dims.tp_enabled:
        model.parallelize(parallel_dims)
        maybe_enable_async_tp(
            parallelism, compile_config, parallel_dims.get_mesh("tp")
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    # 57th sync: PR #3674 refactored AC into a Configurable policy
    # hierarchy. ac_config is now an ActivationCheckpointing.Config
    # subclass (FullAC.Config / SelectiveAC.Config / MemoryBudgetAC.Config)
    # or None. The old `mode="none"` sentinel is replaced by `None`.
    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if model_compile_enabled:
        _apply_compile_with_mode(model, compile_config)
        # apply_compile unconditionally sets capture_scalar_outputs=True
        # (needed for MoE dynamic shapes). For dense models this breaks
        # the separately-compiled loss_fn when loss_parallel + ignore_index
        # produce unbacked symbols in cross_entropy.
        torch._dynamo.config.capture_scalar_outputs = False

    # [ezpz] Skip FSDP for the vLLM generator: FSDP forward hooks are
    # incompatible with torch.inference_mode() used by vLLM. The generator's
    # vllm_wrapper passes skip_dp=True. Mirrors qwen3/gpt_oss parallelize.
    if skip_dp:
        return model

    names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    dp_mesh = parallel_dims.get_mesh(names)

    # [ezpz] Ablation arm B ("norms-only fp32 master"), opt-in via
    # EZPZ_FP32_NORMS=1. Only meaningful with training.dtype=bfloat16 (bf16
    # master everywhere): promote the affine norm weights to an fp32 master
    # while the bulk stays bf16. FSDP2 rejects mixed dtypes inside one
    # fully_shard group, so the promoted norms must also be sharded
    # separately -- see agpt/fp32_norms.py. Settles whether the shipped
    # full-fp32 master is over-broad.
    fp32_norm_modules: list[nn.Module] = []
    if os.environ.get("EZPZ_FP32_NORMS", "0").strip() not in ("", "0", "false"):
        from torchtitan.experiments.ezpz.agpt.fp32_norms import promote_norms_to_fp32

        if training.dtype != "bfloat16":
            logger.warning(
                "EZPZ_FP32_NORMS=1 with training.dtype=%s: the master copy is "
                "already float32, so promoting norms is a no-op.",
                training.dtype,
            )
        fp32_norm_modules = promote_norms_to_fp32(model)

    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        separate_fsdp_modules=fp32_norm_modules,
    )

    if parallel_dims.dp_replicate_enabled:
        logger.info("Applied HSDP to the model")
    else:
        logger.info("Applied FSDP to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    return model


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """Disable FSDP's automatic gradient division and (on XPU/CCL) force
    sum reduction for cross-rank gradient comms.

    On NCCL the default reduce-mean works correctly. On CCL (XPU) we need
    SUM and divide ourselves to avoid losing precision.
    """
    force_sum_reduction = False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        backend = ezpz.distributed.get_torch_backend() or str(
            torch.distributed.get_backend()
        )
        if backend and "nccl" not in str(backend).lower():
            force_sum_reduction = True

    fsdp_modules_updated = 0
    for module in model.modules():
        # Be resilient to FSDPModule class location changes across PyTorch
        # releases by going through the public method.
        set_divide_factor = getattr(module, "set_gradient_divide_factor", None)
        if callable(set_divide_factor):
            set_divide_factor(1.0)
            fsdp_modules_updated += 1
            if force_sum_reduction:
                set_force_sum = getattr(
                    module, "set_force_sum_reduction_for_comms", None
                )
                if callable(set_force_sum):
                    set_force_sum(True)

    logger.info(
        "Configured FSDP gradient division for %d modules (force_sum_reduction=%s)",
        fsdp_modules_updated,
        force_sum_reduction,
    )


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    separate_fsdp_modules: list[nn.Module] | None = None,
):
    """FSDP2 with the same per-block grouping as upstream llama3.

    Note: matches upstream's `[norm, lm_head]` joint grouping with
    `reshard_after_forward=reshard_after_forward_policy == "always"`
    (last layers don't reshard after forward by default — FSDP would
    prefetch them immediately).

    `separate_fsdp_modules` (ezpz, ablation arm B) each get their OWN
    `fully_shard` group, applied before the enclosing block/model groups so
    the inner wrap wins. This exists because FSDP2 asserts a uniform
    `orig_dtype` per group: the norms-only-fp32 arm has fp32 norm weights
    inside otherwise-bf16 blocks, which is only expressible by regrouping.
    Empty (the default) leaves the production grouping untouched.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    # [ezpz] Wrap the dtype-divergent modules first so each owns its group;
    # the enclosing block/model wraps below then only see the remaining
    # (uniformly bf16) parameters.
    for module in separate_fsdp_modules or ():
        fully_shard(
            module,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    if separate_fsdp_modules:
        logger.info(
            "Applied %d separate FSDP groups (dtype-divergent modules)",
            len(separate_fsdp_modules),
        )

    # [ezpz] When embeddings are tied (enable_weight_tying), tok_embeddings
    # and lm_head share one weight tensor -- FSDP2 requires shared/tied
    # parameters to live in the SAME fully_shard group, so group tok_embeddings
    # + norm + lm_head together here instead of the two separate calls the
    # untied path below uses. Mirrors
    # torchtitan.distributed.fsdp.apply_fsdp_to_decoder:151-163. The untied
    # branch (default, currently working) is unchanged.
    tied = getattr(model, "enable_weight_tying", False)

    # [ezpz] Modules already given their own group above must not appear in a
    # second fully_shard call ("can only be applied to a module once").
    # Nesting inside an enclosing block/model wrap is fine -- that is how the
    # per-layer norms keep their own group -- but a DIRECT re-wrap is not.
    _separate = {id(m) for m in (separate_fsdp_modules or ())}

    def _not_separate(modules):
        return [m for m in modules if m is not None and id(m) not in _separate]

    if tied:
        modules = _not_separate([model.tok_embeddings, model.norm, model.lm_head])
        fully_shard(
            modules,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )
    elif model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    for transformer_block in model.layers.values():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    if not tied and model.norm is not None and model.lm_head is not None:
        tail_modules = _not_separate([model.norm, model.lm_head])
        if tail_modules:
            fully_shard(
                tail_modules,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward_policy == "always",
            )

    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)
