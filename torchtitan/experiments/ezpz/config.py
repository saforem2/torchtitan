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

from torchtitan.config.configs import ParallelismConfig

__all__ = ["EzpzParallelismConfig"]


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
    disabled by default and supports either no model parallelism or a guarded
    one-stage-per-rank Schedule1F1B pipeline. TP, CP, EP, and outer gradient
    accumulation remain unsupported.
    """

    native_ddp_compute_policy: Literal[
        "autocast",
        "ddp_mixed_precision",
        "ddp_mixed_precision_xpu_overlap",
    ] = "autocast"
    """
    Native DDP forward/backward compute policy. ``autocast`` retains FP32
    parameter storage during forward and is not dtype-equivalent to FSDP for
    embeddings and residuals. ``ddp_mixed_precision`` uses DDP's experimental
    BF16 parameter-copy path with FP32 gradient reduction and optimizer state.
    ``ddp_mixed_precision_xpu_overlap`` keeps those dtype semantics but replaces
    a blocking private PyTorch communication hook with a pinned-version XPU
    stream implementation. It is experimental and rejected off the validated
    Aurora software/environment combination.
    """

    native_ddp_bucket_cap_mb: float = 25.0
    """Gradient bucket capacity for the native DDP backend, in MiB."""

    native_ddp_bucketize_first_iteration: bool = False
    """
    Apply ``native_ddp_bucket_cap_mb`` when DDP constructs its initial
    reduction buckets. By default PyTorch deliberately uses one whole-model
    bucket on the first backward and rebuilds by gradient readiness afterward.
    This option uses the pinned DDP ``bucket_cap_mb_list`` API to avoid that
    initial full-model collective while retaining the normal later rebuild.
    """
