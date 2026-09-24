# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import time
from contextlib import AbstractContextManager
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, cast

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data.loader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import apply_overrides
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.ezpz import signal_stop
from torchtitan.experiments.ezpz.ckpt_key_compat import (
    maybe_install_flat_attention_compat,
)
from torchtitan.experiments.ezpz.ckpt_owner_claim import check_and_claim
from torchtitan.experiments.ezpz.config import EzpzParallelismConfig
from torchtitan.experiments.ezpz.logging import logger
from torchtitan.experiments.ezpz.lr_finder import LRFinderConfig
from torchtitan.experiments.ezpz.native_ddp import (
    install_agpt_dtype_probe,
    native_ddp_autocast_context,
    validate_native_ddp,
    wrap_native_ddp,
    wrap_native_ddp_loss,
)
from torchtitan.experiments.ezpz.xpu_graph import (
    maybe_wrap_with_xpu_graph,
    xpu_graph_teardown,
)
from torchtitan.experiments.torchft.config.job_config import FaultTolerance
from torchtitan.experiments.torchft.manager import maybe_semi_sync_training
from torchtitan.experiments.torchft.trainer import (
    FaultTolerantTrainer as TorchFTTrainer,
    FaultTolerantTrainingEngine as TorchFTTrainingEngine,
)
from torchtitan.models.common.aux_loss import collect_aux_loss_metrics
from torchtitan.training_engine import TrainingEngine


_CLIP_FOREACH_LOGGED = False


def _clip_foreach() -> bool:
    """Whether to use the fused multi-tensor path in ``clip_grad_norm_``.

    Defaults to True, matching upstream. ``EZPZ_CLIP_NO_FOREACH=1`` selects the
    unfused loop, which exists to test whether the fused path is what exhausts
    level_zero resources under HSDP on XPU (30B dies in
    ``clip_grad.py:106 torch.stack([norm.to(first_device) ...])`` with
    UR_RESULT_ERROR_OUT_OF_RESOURCES at 71.68% memory -- resource exhaustion,
    not an OOM). Logged once so a run's own output proves which path it took,
    rather than the caller assuming the env var reached the ranks.
    """
    global _CLIP_FOREACH_LOGGED
    foreach = os.environ.get("EZPZ_CLIP_NO_FOREACH", "0") != "1"
    if not _CLIP_FOREACH_LOGGED:
        _CLIP_FOREACH_LOGGED = True
        logger.info("EZPZ_CLIP_NO_FOREACH active: clip_grad foreach=%s", foreach)
    return foreach


def _set_pg_timeouts_xpu_aware(
    timeout: timedelta,
    parallel_dims: ParallelDims,
) -> None:
    """Apply ``timeout`` to every PG in the mesh, with explicit XPU support.

    Upstream ``dist_utils.set_pg_timeouts`` delegates to
    ``torch.distributed.distributed_c10d._set_pg_timeout``, whose device
    dispatch only knows about cpu (gloo) and cuda (nccl / gloo /
    torchcomms). On XPU the loop adds no backends, emits the warning
    ``"Set timeout is now only supported for either nccl or gloo."``,
    and never calls ``_set_default_timeout``. Result: ``train_timeout_seconds``
    silently no-ops and a hung collective burns the full PBS walltime
    instead of aborting (see
    ``docs/upstream-issues/train_timeout_xpu_silent_noop.md``).

    Workaround: set the timeout on every mesh PG ourselves (with the
    safety barrier the upstream helper uses), then additionally call
    ``ProcessGroupXCCL.set_timeout`` on any xccl-backed groups.

    61st sync: we no longer delegate to ``dist_utils.set_pg_timeouts``.
    The spmd_types series switched it from
    ``distributed_c10d._set_pg_timeout`` to ``torch.distributed.set_timeout``,
    which does NOT exist in our pinned torch 2.13 (AttributeError ->
    every collective unguarded). So we inline the pre-sync behavior
    against ``_set_pg_timeout`` (present in torch 2.13's
    ``distributed_c10d``), keeping this shim independent of upstream's
    timeout-API churn. Remove once PyTorch's timeout dispatch learns
    about XPU and our torch exposes the matching API.
    """
    from torch.distributed import distributed_c10d as c10d

    device_module = dist_utils.device_module
    # Flush in-flight work under the old timeout before lowering it
    # (mirrors upstream set_pg_timeouts' safety barrier). The gloo (CPU)
    # backend does not take device_ids, and any integer index there is
    # resolved against the default accelerator (e.g. MPS on a Mac), which has
    # no c10d::barrier -- so only pass device_ids for real accelerators.
    if dist_utils.device_type == "cpu":
        torch.distributed.barrier()
    else:
        torch.distributed.barrier(device_ids=[device_module.current_device()])
    device_module.synchronize()

    timeout_groups: list[torch.distributed.ProcessGroup | None] = [
        mesh.get_group()
        for mesh in parallel_dims.get_all_one_dimensional_meshes().values()
    ] + [None]
    for group in timeout_groups:
        c10d._set_pg_timeout(timeout, group)

    xpu_device = torch.device("xpu")
    if not (torch.distributed.is_xccl_available() and torch.xpu.is_available()):
        return

    from torch._C._distributed_c10d import ProcessGroupXCCL

    groups: list[torch.distributed.ProcessGroup | None] = [
        mesh.get_group()
        for mesh in parallel_dims.get_all_one_dimensional_meshes().values()
    ] + [None]
    patched = 0
    for group in groups:
        if group is None:
            group = torch.distributed.distributed_c10d._get_default_group()
        if xpu_device not in group._device_types:
            continue
        backend = group._get_backend(xpu_device)
        if isinstance(backend, ProcessGroupXCCL):
            backend.set_timeout(timeout)
            patched += 1
    if patched:
        logger.info(
            f"Applied train timeout {timeout} to {patched} xccl ProcessGroup(s) "
            "(upstream _set_pg_timeout has no xpu branch)."
        )


class _DictTrainingMicrobatch(TrainingMicrobatch):
    """Adapter for ezpz dataloaders that still yield batch dictionaries."""

    def __init__(self, batch: dict[str, Any]) -> None:
        self._batch = batch
        labels = batch.get("labels")
        if not isinstance(labels, torch.Tensor):
            raise ValueError("ezpz dict microbatches must include tensor labels")
        self.labels = labels
        valid_tokens = batch.get("num_valid_tokens")
        if valid_tokens is None:
            valid_tokens = int((labels != IGNORE_INDEX).sum().item())
        elif isinstance(valid_tokens, torch.Tensor):
            valid_tokens = int(valid_tokens.item())
        else:
            valid_tokens = int(valid_tokens)
        self.num_valid_tokens = valid_tokens

    def as_input_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self._batch.items() if k != "num_valid_tokens"}


class EzpzTrainingEngine(TorchFTTrainingEngine):
    """TorchFT engine with ezpz runtime hooks and execution state."""

    _diag_prev_weight_norms: dict[str, float]
    _history_bridge: Any | None
    _last_grad_norm: float | None

    def _model_name(self) -> str:
        """Return the stable registry identity formerly carried by ModelSpec."""
        module = self.model_cls.__module__.split(".")
        if "ezpz" in module:
            index = module.index("ezpz")
            if len(module) > index + 1:
                return ".".join(module[index : index + 2])
        return self.model_cls.__qualname__

    @property
    def diloco_fragment_fn(self):
        """Resolve DiLoCo fragmentation from the model's lifecycle owner."""
        return getattr(self.model_cls, "_fragment", None)

    def _initialize_distributed_runtime(self) -> None:
        from torchtitan.experiments.ezpz.gloo_new_group_workaround import (
            maybe_install_gloo_new_group_workaround,
        )
        from torchtitan.experiments.ezpz.xccl_split_group_workaround import (
            maybe_install_xccl_split_group_workaround,
        )

        maybe_install_xccl_split_group_workaround()
        maybe_install_gloo_new_group_workaround()
        super()._initialize_distributed_runtime()

    def _initialize_model(
        self,
        *,
        compile_config,
        hf_assets_path,
        create_seed_checkpoint: bool = False,
    ) -> None:
        super()._initialize_model(
            compile_config=compile_config,
            hf_assets_path=hf_assets_path,
            create_seed_checkpoint=create_seed_checkpoint,
        )
        config = self.config
        if getattr(config.parallelism, "enable_data_parallel_native_ddp", False):
            validate_native_ddp(
                model_name=self._model_name(),
                parallel_dims=self.parallel_dims,
                training=config.training,
                parallelism=config.parallelism,
                loss_fn=self.loss_fn,
                gradient_accumulation_steps=getattr(
                    self, "configured_gradient_accumulation_steps", 1
                ),
                fault_tolerance_enabled=config.fault_tolerance.enable,
                create_seed_checkpoint=create_seed_checkpoint,
                optimizer_has_param_groups=(
                    len(config.optimizer.param_groups) != 1
                    or config.optimizer.param_groups[0].pattern != r".*"
                ),
            )
            self.loss_fn = cast(
                Any, wrap_native_ddp_loss(self.loss_fn, self.parallel_dims.dp_replicate)
            )
            assert len(self.model_parts) == 1
            self.model_parts[0] = wrap_native_ddp(
                self.model_parts[0],
                self.parallel_dims.get_mesh("dp_replicate"),
                config.parallelism.native_ddp_bucket_cap_mb,
                config.parallelism.native_ddp_compute_policy,
                config.parallelism.native_ddp_bucketize_first_iteration,
            )

        if os.getenv("TORCHTITAN_AGPT_DTYPE_PROBE") == "1":
            if self.parallel_dims.pp_enabled or len(self.model_parts) != 1:
                raise ValueError("AGPT dtype probe requires a non-pipeline model")
            install_agpt_dtype_probe(self.model_parts[0])

    def _initialize_forward_backward(self) -> None:
        super()._initialize_forward_backward()
        if not self.parallel_dims.pp_enabled:
            self.forward_backward_body_fn = maybe_wrap_with_xpu_graph(
                self._non_pp_forward_backward_body
            )

    def _get_train_context(self) -> AbstractContextManager[None]:
        context = super()._get_train_context
        if getattr(self.config.parallelism, "enable_data_parallel_native_ddp", False):
            return native_ddp_autocast_context(context, self.device.type)()
        return context()

    def initialize(
        self,
        *,
        compile_config,
        hf_assets_path,
        dataloader: BaseDataLoader | None = None,
        create_seed_checkpoint: bool = False,
    ) -> None:
        super().initialize(
            compile_config=compile_config,
            hf_assets_path=hf_assets_path,
            dataloader=dataloader,
            create_seed_checkpoint=create_seed_checkpoint,
        )
        self._diag_prev_weight_norms = {}
        self._history_bridge = None
        self._last_grad_norm = None
        config = self.config
        if getattr(config, "diagnostics", False) or getattr(
            config, "history_bridge", False
        ):
            from torchtitan.experiments.ezpz.diagnostics import attention as _attn
            from torchtitan.experiments.ezpz.diagnostics.history_bridge import (
                HistoryBridge,
            )

            _attn.configure(
                enabled=getattr(config, "diagnostics_attention", False),
                every=getattr(config, "diagnostics_interval", 50),
            )
            self._history_bridge = HistoryBridge(
                enabled=getattr(config, "history_bridge", False),
                outdir=os.path.join(config.dump_folder, "history"),
                world_size=int(os.environ.get("WORLD_SIZE", "1")),
            )

    def optimizer_step(self) -> torch.Tensor:
        current_step = self.num_completed_steps + 1
        if getattr(self.config, "diagnostics_attention", False):
            from torchtitan.experiments.ezpz.diagnostics import attention as _attn

            _attn.set_step(current_step)

        grad_norm = dist_utils.clip_grad_norm_(
            [p for model in self.model_parts for p in model.parameters()],
            self.config.training.max_norm,
            foreach=_clip_foreach(),
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )
        if not self.parallel_dims.pp_enabled or self.pp_has_last_stage:
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
            if loss_mesh is not None:
                torch.distributed.all_reduce(
                    self.loss_is_finite,
                    op=torch.distributed.ReduceOp.MIN,
                    group=loss_mesh.get_group(),
                )
        pp_mesh = self.parallel_dims.get_optional_mesh("pp")
        if pp_mesh is not None:
            torch.distributed.all_reduce(
                self.loss_is_finite,
                op=torch.distributed.ReduceOp.MIN,
                group=pp_mesh.get_group(),
            )
        if hasattr(self, "checkpointer"):
            self.checkpointer.maybe_wait_for_staging()

        step_is_finite = self.loss_is_finite.logical_and(
            torch.isfinite(grad_norm).all()
        )
        if bool(step_is_finite.item()):
            self.optimizers.step()
        else:
            self._capture_nonfinite_gradients(current_step, grad_norm)
            self.optimizers.zero_grad()
            logger.error(
                "non-finite loss or grad_norm (%s) at step %s: SKIPPING the "
                "optimizer step to avoid writing NaN into the weights.",
                grad_norm,
                current_step,
            )
        self.lr_schedulers.step()
        self.num_completed_steps = current_step
        self._num_optimizer_steps_since_cuda_graph_init += 1
        self._last_grad_norm = float(grad_norm.item())
        return grad_norm

    def _capture_nonfinite_gradients(self, step: int, grad_norm: torch.Tensor) -> None:
        try:
            from torchtitan.experiments.ezpz import diagnostics as _diag_nan

            stats = _diag_nan.collect_param_stats(self.model_parts, per_layer=True)
            bad = [
                k
                for k, v in stats.items()
                if isinstance(v, float) and not math.isfinite(v)
            ]
            named = [
                f"{stats[k]} (gradnorm={stats.get(k[:-6], '?')})"
                for k in sorted(stats)
                if k.endswith("_layer")
                and isinstance(stats.get(k[:-6]), float)
                and not math.isfinite(stats[k[:-6]])
            ]
            logger.error(
                "NON-FINITE GRADIENT CAPTURE step %s: %s",
                step,
                ", ".join(
                    f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in sorted(stats.items())
                ),
            )
            if named:
                logger.error(
                    "NON-FINITE GRADIENT CAPTURE step %s: THE TENSORS THAT "
                    "WENT NON-FINITE: %s",
                    step,
                    "; ".join(named),
                )
            elif bad:
                logger.error(
                    "NON-FINITE GRADIENT CAPTURE step %s: non-finite "
                    "AGGREGATES only (%s).",
                    step,
                    ", ".join(sorted(bad)),
                )
        except Exception as exc:
            logger.error(
                "NON-FINITE GRADIENT CAPTURE step %s FAILED: %s: %s",
                step,
                type(exc).__name__,
                exc,
            )

    def collect_extra_metrics(
        self,
        *,
        step: int,
        loss: torch.Tensor,
        grad_norm: torch.Tensor,
    ) -> dict[str, Any]:
        extra_metrics: dict[str, Any] = collect_aux_loss_metrics(self.parallel_dims)
        try:
            from torchtitan.experiments.ezpz import zloss as _zloss

            extra_metrics.update(_zloss.drain())
        except Exception:
            pass

        if getattr(self.config, "diagnostics", False) and (
            step % max(1, getattr(self.config, "diagnostics_interval", 50)) == 0
        ):
            try:
                from torchtitan.experiments.ezpz import diagnostics as _diag
                from torchtitan.experiments.ezpz.diagnostics import attention as _attn

                extra_metrics.update(
                    _diag.clipping_metrics(grad_norm, self.config.training.max_norm)
                )
                extra_metrics.update(
                    _diag.collect_param_stats(
                        self.model_parts,
                        per_layer=getattr(self.config, "diagnostics_per_layer", False),
                    )
                )
                ratios, self._diag_prev_weight_norms = _diag.collect_update_ratios(
                    self.model_parts, self._diag_prev_weight_norms
                )
                extra_metrics.update(ratios)
                extra_metrics.update(_diag.collect_optimizer_stats(self.optimizers))
                extra_metrics.update(_attn.drain())
            except Exception as exc:
                logger.warning("diagnostics failed at step %s: %s", step, exc)

        if self._history_bridge is not None and self._history_bridge.enabled:
            extra_metrics.update(
                self._history_bridge.update(
                    {
                        "loss": float(loss.detach().item()),
                        "grad_norm": float(grad_norm.item()),
                    },
                    step=step,
                )
            )
        if os.environ.get("EZPZ_DIAG_ECHO") == "1" and extra_metrics:
            logger.info(
                "DIAG_ECHO "
                + ", ".join(f"{k}={v}" for k, v in sorted(extra_metrics.items()))
            )
        return extra_metrics

    def close(self) -> None:
        super().close()
        xpu_graph_teardown()


class FaultTolerantTrainer(TorchFTTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(TorchFTTrainer.Config):
        fault_tolerance: FaultTolerance = field(default_factory=FaultTolerance)
        lr_finder: LRFinderConfig = field(default_factory=LRFinderConfig)
        parallelism: EzpzParallelismConfig = field(
            default_factory=EzpzParallelismConfig
        )
        checkpoint: CheckpointManager.Config | None = None
        batch_ramp_steps: int = 0
        batch_ramp_start_gas: int = 1
        walltime_deadline_epoch: int = 0
        walltime_seconds: int = 0
        walltime_checkpoint_margin_seconds: int = 600
        save_on_signal: bool = True
        grad_norm_abort: float = 0.0
        grad_norm_abort_window: int = 50
        nan_abort_consecutive: int = 0
        diagnostics: bool = False
        diagnostics_interval: int = 50
        diagnostics_per_layer: bool = False
        diagnostics_attention: bool = False
        history_bridge: bool = False
        wandb_watch: bool = False

        def __post_init__(self):
            # ``@dataclass(slots=True)`` returns a replacement class object;
            # call the parent explicitly so Python 3.12 does not use the stale
            # class captured by zero-argument ``super()``.
            TorchFTTrainer.Config.__post_init__(self)
            # ``checkpointer`` is canonical. Legacy ``--checkpoint.*`` CLI
            # options are translated before parsing, so copying the alias back
            # here would silently erase canonical CLI overrides and specialized
            # checkpointer subclasses.
            self.checkpoint = self.checkpointer

    engine_cls: type[TrainingEngine] = EzpzTrainingEngine

    @record
    def __init__(self, config: Config):
        self.config = config
        model_config = config.model
        model_config.update_from_config(config=config)
        if config.override.imports:
            apply_overrides(config.override, config)
        config.__post_init__()

        self.engine = self.engine_cls(
            config,
            model_config=model_config,
            max_num_documents=config.dataloader.max_num_documents,
            output_dir=config.dump_folder,
            fault_tolerance=config.fault_tolerance,
        )
        engine = self.engine
        parallel_dims = engine.parallel_dims
        config.maybe_log()

        if parallel_dims.dp_enabled:
            dp_mesh = parallel_dims.get_mesh("dp")
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0
        dp_degree, dp_rank = engine.ft_manager.get_dp_info(dp_degree, dp_rank)

        self.tokenizer = (
            config.tokenizer.build(tokenizer_path=config.hf_assets_path)
            if config.tokenizer is not None
            else None
        )
        self.num_pp_microbatches = (
            config.parallelism.num_pp_microbatches if parallel_dims.pp_enabled else 1
        )
        num_tokens_per_microbatch = (
            config.training.num_tokens_per_microbatch_per_dp_rank
        )
        num_tokens_per_dp_rank = num_tokens_per_microbatch * self.num_pp_microbatches
        num_tokens_per_train_step = config.training.num_tokens_per_train_step
        if num_tokens_per_train_step < 0:
            num_tokens_per_train_step = num_tokens_per_dp_rank * dp_degree
        if num_tokens_per_train_step % (num_tokens_per_dp_rank * dp_degree) != 0:
            raise ValueError(
                "training.num_tokens_per_train_step "
                f"({num_tokens_per_train_step}) must be divisible by the number "
                "of tokens processed globally in one gradient accumulation "
                f"iteration ({num_tokens_per_dp_rank * dp_degree})."
            )
        self.gradient_accumulation_steps = num_tokens_per_train_step // (
            num_tokens_per_dp_rank * dp_degree
        )
        engine.configured_gradient_accumulation_steps = self.gradient_accumulation_steps

        self.dataloader = self._build_dataloader(
            dp_degree=dp_degree,
            dp_rank=dp_rank,
            tokenizer=self.tokenizer,
            max_context_length=config.training.max_context_length,
            num_tokens_per_microbatch=num_tokens_per_microbatch,
            training_steps=config.training.steps * self.num_pp_microbatches,
            parallel_dims=parallel_dims,
            num_tokens_per_train_step=num_tokens_per_train_step,
        )

        from torchtitan.experiments.ezpz.agpt import set_ezpz_max_context_length

        set_ezpz_max_context_length(config.training.max_context_length)

        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            device_memory_monitor=engine.device_memory_monitor,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            ft_enable=config.fault_tolerance.enable,
            ft_replica_id=config.fault_tolerance.replica_id,
            config_dict=config.to_dict(),
            has_quantization=engine.has_quantization,
        )
        color = self.metrics_processor.color

        engine.initialize(
            compile_config=config.compile,
            dataloader=self.dataloader,
            hf_assets_path=config.hf_assets_path,
            create_seed_checkpoint=config.create_seed_checkpoint,
        )

        if parallel_dims.pp_enabled:
            from torchtitan.observability.metrics import ensure_pp_loss_visible

            ensure_pp_loss_visible(
                parallel_dims=parallel_dims,
                pp_schedule=config.parallelism.pipeline_parallel_schedule,
                color=color,
            )
        self.metrics_processor.num_flops_per_token = engine.num_flops_per_token
        self.metrics_processor.optimizers = engine.optimizers
        self.metrics_processor.model_parts = engine.model_parts

        logger.info(
            "Peak FLOPS used for computing MFU: "
            f"{self.metrics_processor.gpu_peak_flops:.3e}"
        )
        logger.info(
            f"{engine.device.type.upper()} memory usage for model: "
            f"{engine.model_device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({engine.model_device_mem_stats.max_reserved_pct:.2f}%)"
        )

        validator_config = config.validator
        validator_enabled = validator_config is not None and getattr(
            validator_config, "enable", True
        )
        if validator_enabled:
            assert validator_config is not None
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    engine.pp_schedule,
                    engine.pp_has_first_stage,
                    engine.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )
            self.validator = validator_config.build(
                parallelism=config.parallelism,
                dp_world_size=dp_degree,
                dp_rank=dp_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=engine.loss_fn,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.max_context_length,
                num_tokens_per_microbatch=num_tokens_per_microbatch,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        self.batch_ramp_steps = config.batch_ramp_steps
        self.batch_ramp_start_gas = config.batch_ramp_start_gas
        if self.batch_ramp_steps < 0:
            raise ValueError(
                f"batch_ramp_steps must be >= 0, got {self.batch_ramp_steps}"
            )
        if self.batch_ramp_steps > 0 and not (
            1 <= self.batch_ramp_start_gas <= self.gradient_accumulation_steps
        ):
            raise ValueError(
                "batch_ramp_start_gas must be in "
                f"[1, {self.gradient_accumulation_steps}], got "
                f"{self.batch_ramp_start_gas}"
            )

        logger.info(
            "Trainer is initialized with "
            f"tokens/microbatch/dp-rank {num_tokens_per_microbatch}, "
            f"tokens/train-step {num_tokens_per_train_step}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {config.training.max_context_length}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.lr_scheduler.warmup_steps})"
        )

    def _build_dataloader(
        self,
        *,
        dp_degree: int,
        dp_rank: int,
        tokenizer: Any,
        max_context_length: int,
        num_tokens_per_microbatch: int,
        training_steps: int,
        parallel_dims: ParallelDims,
        num_tokens_per_train_step: int | None = None,
    ) -> BaseDataLoader:
        local_batch_size = num_tokens_per_microbatch // max_context_length
        if local_batch_size < 1:
            raise ValueError(
                "training.num_tokens_per_microbatch_per_dp_rank "
                f"({num_tokens_per_microbatch}) must be at least one full "
                f"sequence of max_context_length ({max_context_length})."
            )
        return self.config.dataloader.build(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            seq_len=max_context_length,
            local_batch_size=local_batch_size,
            max_context_length=max_context_length,
            num_tokens_per_batch=num_tokens_per_microbatch,
            num_tokens_per_microbatch=num_tokens_per_microbatch,
            training_steps=training_steps,
            global_batch_size=(
                (num_tokens_per_train_step or num_tokens_per_microbatch)
                // max_context_length
            ),
            parallel_dims=parallel_dims,
        )

    def microbatch_generator(
        self, data_iterable: Iterable[TrainingMicrobatch | dict[str, Any]]
    ) -> Iterator[TrainingMicrobatch]:
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = time.perf_counter()
            try:
                microbatch = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            ntokens_microbatch = (
                self.config.training.num_tokens_per_microbatch_per_dp_rank
            )
            self.metrics_processor.ntokens_since_last_log += ntokens_microbatch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )
            if isinstance(microbatch, dict):
                yield _DictTrainingMicrobatch(microbatch)
            else:
                yield microbatch

    def state_dict(self) -> dict[str, Any]:
        return self.engine.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.engine.load_state_dict(state_dict)

    def _effective_gas(self) -> int:
        if self.batch_ramp_steps <= 0:
            return self.gradient_accumulation_steps
        step = self.engine.num_completed_steps
        if step >= self.batch_ramp_steps:
            return self.gradient_accumulation_steps
        start = self.batch_ramp_start_gas
        full = self.gradient_accumulation_steps
        frac = step / self.batch_ramp_steps
        gas = round(start + (full - start) * frac)
        return max(start, min(full, gas))

    def train_step(self, data_iterator: Iterator[TrainingMicrobatch]) -> float | None:
        original_gas = self.gradient_accumulation_steps
        self.gradient_accumulation_steps = self._effective_gas()
        try:
            return self._train_step(data_iterator)
        finally:
            self.gradient_accumulation_steps = original_gas

    def _train_step(self, data_iterator: Iterator[TrainingMicrobatch]) -> float | None:
        engine = self.engine
        current_step = engine.num_completed_steps + 1
        should_log = self.metrics_processor.should_log(current_step)
        parallel_dims = engine.parallel_dims

        microbatch_groups: list[list[TrainingMicrobatch]] = []
        local_valid_tokens = 0
        for _ in range(self.gradient_accumulation_steps):
            microbatch_group = []
            for _ in range(self.num_pp_microbatches):
                microbatch = next(data_iterator)
                local_valid_tokens += microbatch.num_valid_tokens
                microbatch_group.append(microbatch)
            microbatch_groups.append(microbatch_group)

        local_valid_tokens_tensor = torch.tensor(
            local_valid_tokens,
            dtype=torch.int64,
            device=engine.device,
        )
        if parallel_dims.dp_enabled:
            dp_mesh = parallel_dims.get_mesh("dp")
            global_valid_tokens = dist_utils.dist_sum_tensor(
                local_valid_tokens_tensor, dp_mesh
            )
        else:
            global_valid_tokens = local_valid_tokens_tensor

        global_valid_tokens = engine.prepare_step(
            global_valid_tokens,
            num_accumulation_steps=self.gradient_accumulation_steps,
        )

        accumulated_loss: torch.Tensor | None = None
        for fwd_bwd_index, microbatch_group in enumerate(microbatch_groups):
            detached_loss = engine.forward_backward_microbatch(
                microbatch_group=microbatch_group,
                global_valid_tokens=global_valid_tokens,
                accumulation_index=fwd_bwd_index,
            )
            if accumulated_loss is None:
                accumulated_loss = detached_loss.clone()
            else:
                accumulated_loss.add_(detached_loss)

        lr_metrics = engine.lr_schedulers.get_metrics() if should_log else {}
        grad_norm = engine.optimizer_step()
        if accumulated_loss is None:
            return None

        if not should_log:
            return float(accumulated_loss.detach().item())

        if parallel_dims.dp_cp_enabled:
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            ft_pg = engine.ft_manager.loss_sync_pg
            local_avg_loss = (
                accumulated_loss * global_valid_tokens / local_valid_tokens_tensor
            )
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_sum(accumulated_loss, loss_mesh, ft_pg),
                dist_utils.dist_max(local_avg_loss, loss_mesh, ft_pg),
                dist_utils.dist_sum(
                    torch.tensor(
                        engine.ntokens_seen,
                        dtype=torch.int64,
                        device=engine.device,
                    ),
                    loss_mesh,
                    ft_pg,
                ),
            )
            if ft_pg is not None:
                global_avg_loss /= ft_pg.size()
        else:
            global_avg_loss = global_max_loss = float(accumulated_loss.item())
            global_ntokens_seen = engine.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            **lr_metrics,
            **engine.collect_extra_metrics(
                step=engine.num_completed_steps,
                loss=accumulated_loss,
                grad_norm=grad_norm,
            ),
        }
        if "lr" not in extra_metrics and hasattr(engine.lr_schedulers, "schedulers"):
            extra_metrics["lr"] = engine.lr_schedulers.schedulers[0].get_last_lr()[0]
        self.metrics_processor.log(
            engine.num_completed_steps,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )

        if isinstance(global_avg_loss, torch.Tensor):
            return float(global_avg_loss.item())
        return float(global_avg_loss)

    @record
    def train(self) -> None:
        import collections
        import statistics as _stats

        config = self.config
        engine = self.engine
        checkpointer_config = config.checkpointer
        if checkpointer_config is not None:
            maybe_install_flat_attention_compat(
                engine.checkpointer,
                checkpointer_config.folder,
                checkpointer_config.load_step,
                dump_folder=config.dump_folder,
                initial_load_path=getattr(checkpointer_config, "initial_load_path", "")
                or "",
            )
            if not getattr(checkpointer_config, "load_only", False):
                check_and_claim(
                    checkpointer_config.folder,
                    dump_folder=config.dump_folder,
                    world_size=int(os.environ.get("WORLD_SIZE", -1)),
                    is_rank_zero=(
                        not torch.distributed.is_initialized()
                        or torch.distributed.get_rank() == 0
                    ),
                )

        engine.load_checkpoint()
        loaded_step = engine.num_completed_steps
        logger.info(f"Training starts at step {engine.num_completed_steps + 1}")

        leaf_folder = (
            ""
            if not engine.ft_manager.enabled
            else f"replica_{engine.ft_manager.replica_id}"
        )
        profiler = config.profiler.build(
            global_step=engine.num_completed_steps,
            base_folder=config.dump_folder,
            leaf_folder=leaf_folder,
        )
        profiler.__enter__()
        engine.profiler = profiler
        try:
            with maybe_semi_sync_training(
                config.fault_tolerance,
                ft_manager=engine.ft_manager,
                model=engine.model_parts[0],
                n_layers=(
                    len(engine.model_config.layers)
                    if hasattr(engine.model_config, "layers")
                    else 0
                ),
                optimizer=engine.optimizers,
                fragment_fn=engine.diloco_fragment_fn,
            ):
                wall_deadline = self._resolve_walltime_deadline()
                if config.save_on_signal:
                    if signal_stop.install():
                        logger.info(
                            "signal-ckpt: SIGTERM/SIGINT will force a final "
                            "checkpoint and stop cleanly after the current step"
                        )
                    else:
                        logger.warning(
                            "signal-ckpt: could not install signal handlers; "
                            "a SIGTERM may lose work since the last checkpoint"
                        )

                nan_abort_n = config.nan_abort_consecutive
                consecutive_nonfinite = 0
                gn_hist: collections.deque = collections.deque(
                    maxlen=max(2, config.grad_norm_abort_window)
                )
                numerics_capture = self._maybe_start_numerics_capture()
                data_iterator = self.microbatch_generator(self.dataloader)
                try:
                    while self.should_continue_training():
                        try:
                            loss_val = self.train_step(data_iterator)
                        except DataloaderExhaustedError:
                            logger.warning("Ran out of data; last step was canceled.")
                            break

                        if self._should_abort_for_grad_norm(
                            gn_hist,
                            median_fn=_stats.median,
                        ):
                            break
                        if nan_abort_n > 0:
                            if loss_val is None or not math.isfinite(loss_val):
                                consecutive_nonfinite += 1
                                logger.warning(
                                    "non-finite loss at step %s (%s/%s consecutive)",
                                    engine.num_completed_steps,
                                    consecutive_nonfinite,
                                    nan_abort_n,
                                )
                                if consecutive_nonfinite >= nan_abort_n:
                                    logger.error(
                                        "aborting: %s consecutive non-finite losses "
                                        "(nan_abort_consecutive=%s); run has "
                                        "diverged, stopping to reclaim walltime",
                                        consecutive_nonfinite,
                                        nan_abort_n,
                                    )
                                    break
                            else:
                                consecutive_nonfinite = 0

                        saved_this_step = engine.save_checkpoint(
                            last_step=(
                                engine.num_completed_steps == config.training.steps
                            )
                        )
                        if self._maybe_stop_for_walltime(
                            wall_deadline, saved_this_step
                        ):
                            break
                        if self._maybe_stop_for_signal(saved_this_step):
                            break

                        if (
                            self.config.validator is not None
                            and getattr(self.config.validator, "enable", True)
                            and self.validator.should_validate(
                                engine.num_completed_steps
                            )
                        ):
                            self.validator.validate(
                                engine.model_parts, engine.num_completed_steps
                            )

                        engine.step_profiler()
                        if numerics_capture is not None:
                            numerics_capture.step()

                        if engine.num_completed_steps - loaded_step == 1:
                            _set_pg_timeouts_xpu_aware(
                                timeout=timedelta(
                                    seconds=config.comm.train_timeout_seconds
                                ),
                                parallel_dims=engine.parallel_dims,
                            )
                finally:
                    if numerics_capture is not None:
                        numerics_capture.__exit__(None, None, None)
        finally:
            engine.close_profiler()

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)
        logger.info("Training completed")

    def _resolve_walltime_deadline(self) -> float | None:
        config = self.config
        wall_margin = config.walltime_checkpoint_margin_seconds
        if config.walltime_deadline_epoch > 0:
            wall_deadline = float(config.walltime_deadline_epoch)
            logger.info(
                "walltime-ckpt: absolute deadline %.0f (~%.0f min from now), margin %ss",
                wall_deadline,
                (wall_deadline - time.time()) / 60,
                wall_margin,
            )
            return wall_deadline
        if config.walltime_seconds > 0:
            wall_deadline = time.time() + config.walltime_seconds
            logger.info(
                "walltime-ckpt: relative budget %ss from loop entry, margin %ss",
                config.walltime_seconds,
                wall_margin,
            )
            return wall_deadline
        return None

    def _maybe_start_numerics_capture(self):
        if os.environ.get("EZPZ_DUMP_NUMERICS", "0") != "1":
            return None
        from agent_tooling.numerics_debugging.activation_tracer import (
            ActivationCaptureProfiler,
        )

        ops_env = os.environ.get("EZPZ_NUMERICS_OPS", "mm,bmm,softmax")
        op_filter = {op for op in ops_env.split(",") if op} or None
        capture = ActivationCaptureProfiler(
            enabled=True,
            model=self.engine.model_parts[0],
            dump_dir=os.path.join(self.config.dump_folder, "numerics"),
            capture_step=int(os.environ.get("EZPZ_NUMERICS_STEP", "6")),
            op_filter=op_filter,
            min_numel=int(os.environ.get("EZPZ_NUMERICS_MIN_NUMEL", "1000")),
        )
        capture.__enter__()
        return capture

    def _should_abort_for_grad_norm(self, gn_hist, *, median_fn) -> bool:
        config = self.config
        grad_norm = self.engine._last_grad_norm
        if (
            config.grad_norm_abort <= 0
            or grad_norm is None
            or self.engine.num_completed_steps <= config.lr_scheduler.warmup_steps
        ):
            return False
        if not math.isfinite(grad_norm):
            return False
        if len(gn_hist) == gn_hist.maxlen:
            med = median_fn(gn_hist)
            if med > 0 and grad_norm > config.grad_norm_abort * med:
                logger.error(
                    "grad_norm runaway at step %d: %.4g is %.1fx the trailing "
                    "median (%.4g) over %d steps; aborting before it burns the "
                    "window (threshold %.1fx)",
                    self.engine.num_completed_steps,
                    grad_norm,
                    grad_norm / med,
                    med,
                    gn_hist.maxlen,
                    config.grad_norm_abort,
                )
                return True
        gn_hist.append(grad_norm)
        return False

    def _maybe_stop_for_walltime(
        self, wall_deadline: float | None, saved_this_step: bool
    ) -> bool:
        if wall_deadline is None:
            return False
        remaining = wall_deadline - time.time()
        margin = self.config.walltime_checkpoint_margin_seconds
        if remaining > margin:
            return False
        logger.info(
            "walltime deadline near at step %s (%.0fs left <= margin %ss); "
            "forcing final checkpoint and stopping",
            self.engine.num_completed_steps,
            remaining,
            margin,
        )
        if not saved_this_step:
            self.engine.save_checkpoint(last_step=True)
        self.engine.checkpointer.maybe_wait_for_saving()
        return True

    def _maybe_stop_for_signal(self, saved_this_step: bool) -> bool:
        if not signal_stop.stop_requested():
            return False
        logger.warning(
            "%s received at step %s; forcing final checkpoint and stopping",
            signal_stop.stop_signal_name(),
            self.engine.num_completed_steps,
        )
        if not saved_this_step:
            self.engine.save_checkpoint(last_step=True)
        self.engine.checkpointer.maybe_wait_for_saving()
        logger.info(
            "signal-ckpt: checkpoint for step %s is on disk; exiting cleanly",
            self.engine.num_completed_steps,
        )
        return True
