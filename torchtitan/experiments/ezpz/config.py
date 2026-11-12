# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz-local config extensions.

The upstream Aurora MoE branch declares its native-DDP knobs directly on
core ``ParallelismConfig`` (torchtitan/config/configs.py). We do not: the
project rule is that experiments must not modify core to accommodate
themselves, so the fields live here instead and core gets zero lines.

This works because the native-DDP call sites were already inverted out of
core -- ``wrap_native_ddp`` and friends are invoked from
``experiments/ezpz/trainer.py``, not ``torchtitan/trainer.py``. Upstream
needs core to know these fields because upstream's hook lives in core.
Ours does not, so nothing in core ever reads them.

Subclassing works even though ``ParallelismConfig`` sets ``slots=True``:
a slotted dataclass can still be extended with new fields, the subclass
passes ``isinstance(..., ParallelismConfig)`` so every core annotation
accepts it unchanged, and tyro generates the ``--parallelism.*`` CLI
flags for the added fields.

Field definitions and documentation are taken from the upstream branch so
the two stay comparable; ``disable_degree_one_fsdp`` and
``enable_fsdp_async_all_reduce`` are deliberately omitted because nothing
in our port reads them.
"""

from dataclasses import dataclass
from typing import Literal

from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.observability.metrics import (
    BaseLogger,
    MetricsProcessor,
    TensorBoardLogger,
    WandBLogger,
)

__all__ = [
    "EzpzMetricsConfig",
    "EzpzParallelismConfig",
    "EzpzTrainingConfig",
    "apply_legacy_metrics_fields",
    "set_metrics_logger",
]


@dataclass(kw_only=True)
class EzpzTrainingConfig(TrainingConfig):
    """``TrainingConfig`` plus the ezpz CLI fields upstream moved elsewhere.

    Upstream moved the model dtypes to ``precision`` and FSDP CPU offload to
    ``parallelism`` (#5776), and dropped ``max_context_length`` in favour of the
    model config's own field (#5779). ezpz launchers, PBS seats and JSON configs
    still pass ``--training.max-context-length`` / ``--training.dtype`` /
    ``--training.mixed-precision-*``, and ezpz cannot take
    ``--model.max-context-length``: the model config is opaque to the CLI
    (``config_manager.py``). These fields keep that surface.

    They are one-way inputs, ``None`` unless set. ``FaultTolerantTrainer.Config``
    copies each set field to its upstream home in ``__post_init__``; no code
    reads them back. Code reads ``model.max_context_length``, ``precision.*`` and
    ``parallelism.fsdp_enable_cpu_offloading``.
    """

    max_context_length: int | None = None
    """Legacy alias for ``model.max_context_length`` (also resizes RoPE caches)."""

    dtype: Literal["bfloat16", "float32"] | None = None
    """Legacy alias for ``precision.dtype``."""

    mixed_precision_param: Literal["bfloat16", "float32"] | None = None
    """Legacy alias for ``precision.param_dtype``."""

    mixed_precision_reduce: Literal["bfloat16", "float32"] | None = None
    """Legacy alias for ``precision.reduce_dtype``."""

    enable_cpu_offload: bool | None = None
    """Legacy alias for ``parallelism.fsdp_enable_cpu_offloading``."""


@dataclass(kw_only=True, slots=True)
class EzpzMetricsConfig(MetricsProcessor.Config):
    """``MetricsProcessor.Config`` plus the legacy exporter switches.

    Upstream replaced ``enable_wandb`` / ``enable_tensorboard`` with
    ``loggers: list[BaseLogger.Config]`` (#5775). ezpz launchers and PBS seats
    still pass ``--metrics.enable-wandb`` / ``--metrics.no-enable-wandb`` /
    ``--metrics.enable-tensorboard`` and JSON launch configs still set
    ``metrics.enable_wandb``; these fields keep that surface.

    They are one-way inputs, ``None`` unless set, like ``EzpzTrainingConfig``:
    ``FaultTolerantTrainer.Config.__post_init__`` applies each set field to
    ``loggers`` with :func:`apply_legacy_metrics_fields` and resets it to
    ``None``. ``True`` adds the exporter unless one of that type is already
    listed; ``False`` removes every exporter of that type (what
    ``--metrics.no-enable-wandb`` did to a registry that enabled W&B). Code
    reads ``loggers`` only. ``build()`` still constructs ``MetricsProcessor``:
    the owner is inherited from ``MetricsProcessor.Config``.
    """

    enable_wandb: bool | None = None
    """Legacy switch for a ``WandBLogger.Config`` entry in ``loggers``."""

    enable_tensorboard: bool | None = None
    """Legacy switch for a ``TensorBoardLogger.Config`` entry in ``loggers``."""


# Legacy field -> exporter config type. Pre-sync core built W&B before
# TensorBoard; this order keeps that when both switches are set.
_LEGACY_METRICS_LOGGERS: dict[str, type[BaseLogger.Config]] = {
    "enable_wandb": WandBLogger.Config,
    "enable_tensorboard": TensorBoardLogger.Config,
}


def set_metrics_logger(
    metrics: MetricsProcessor.Config,
    logger_config_cls: type[BaseLogger.Config],
    enabled: bool,
) -> None:
    """Enable or disable one exporter type in ``metrics.loggers``.

    Enabling appends ``logger_config_cls()`` unless an exporter of that type
    (or a subclass) is already listed, so a registry entry with settings is
    kept as is. Disabling removes every entry of that type.
    """
    present = any(isinstance(c, logger_config_cls) for c in metrics.loggers)
    if enabled and not present:
        metrics.loggers = [*metrics.loggers, logger_config_cls()]
    elif not enabled and present:
        metrics.loggers = [
            c for c in metrics.loggers if not isinstance(c, logger_config_cls)
        ]


def apply_legacy_metrics_fields(metrics: MetricsProcessor.Config) -> None:
    """Copy set legacy exporter switches into ``loggers``, then reset them."""
    for name, logger_config_cls in _LEGACY_METRICS_LOGGERS.items():
        value = getattr(metrics, name, None)
        if value is None:
            continue
        set_metrics_logger(metrics, logger_config_cls, bool(value))
        setattr(metrics, name, None)


@dataclass(kw_only=True)
class EzpzParallelismConfig(ParallelismConfig):
    """``ParallelismConfig`` plus the ezpz native-DDP knobs."""

    enable_data_parallel_replicate_module: bool = False
    """
    Use PyTorch's experimental FSDP2 ReplicateModule implementation. For AGPT
    this selects pure replicated data parallelism. For MoE it selects
    node-local EP plus parameter-specific replicated DP: routed experts reduce
    across corresponding EP ranks on other nodes and shared parameters reduce
    across the full batch mesh. The default keeps the fully_shard path.
    """

    enable_data_parallel_native_ddp: bool = False
    """
    Use native ``DistributedDataParallel`` for pure replicated AGPT data
    parallelism. This experimental path keeps FP32 master parameters and
    gradients. Its forward compute policy is selected separately below. It is
    disabled by default and supports no model parallelism. PP, TP, CP, and EP
    remain unsupported; data-parallel gradient accumulation is supported.
    """

    native_ddp_compute_policy: Literal["autocast"] = "autocast"
    """
    Native DDP forward/backward compute policy. Only BF16 autocast over FP32
    parameter storage is supported.
    """

    native_ddp_communication_dtype: Literal["float32", "bfloat16"] = "float32"
    """
    Native DDP collective payload precision. The BF16 compression hook
    quantizes gradient buckets for communication and copies the result back
    into FP32 gradients; parameter storage and local gradient accumulation
    remain FP32. BF16 communication can change numerical results.
    """

    native_ddp_bucket_cap_mb: float = 25.0
    """Gradient bucket capacity for the native DDP backend, in MiB."""

    native_ddp_zero1: bool = False
    """
    Shard the optimizer across the native-DDP replicas, ZeRO-1 style. Model
    parameters are flattened into one FP32 buffer before DDP wraps them; each
    rank steps a contiguous ``1/world`` slice and one all-gather per update
    redistributes the result. Elementwise optimizers only.
    """

    native_ddp_zero1_allgather_dtype: Literal["float32", "bfloat16"] = "float32"
    """
    ZeRO-1 parameter all-gather payload. ``bfloat16`` keeps a separate FP32
    master shard and leaves BF16-rounded values in the replicated model
    parameters, as Megatron-DeepSpeed's BF16 optimizer does. ``float32``
    keeps the model parameters themselves as the FP32 masters.
    """

    native_ddp_zero1_param_dtype: Literal["float32", "bfloat16"] = "float32"
    """
    Storage dtype of the replicated ZeRO-1 model parameters. ``bfloat16``
    (requires ``native_ddp_zero1_allgather_dtype=bfloat16``) stores the
    model in BF16, as Megatron-DeepSpeed does: autocast no longer casts the
    weights on every microbatch, gradients accumulate in BF16 across
    microbatches, and the FP32 master exists only in each rank's optimizer
    shard. Changes numerics; ``float32`` keeps FP32 storage and FP32
    gradient accumulation.
    """

    native_ddp_bucketize_first_iteration: bool = False
    """
    Apply ``native_ddp_bucket_cap_mb`` when DDP constructs its initial
    reduction buckets. By default PyTorch deliberately uses one whole-model
    bucket on the first backward and rebuilds by gradient readiness afterward.
    This option uses the pinned DDP ``bucket_cap_mb_list`` API to avoid that
    initial full-model collective while retaining the normal later rebuild.
    """
