# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host-layout helper for DP_REPLICATE semantics (XPU trial).

Single source of truth for how ``dp_replicate`` / ``dp_shard`` / ``num_generators``
from Guoqiong's ``_qwen3_0_6b_host_layout`` map onto ezpz's
``ParallelismConfig.data_parallel_{replicate,shard}_degree``.  Parameterized
over the FFN weight's inner dim (``ffn_dim1``) so the same logic serves both
Qwen3-0.6B (3072, ``hidden_dim``) and AGPT-2B (8192, ``compute_ffn_hidden_dim(3072, multiple_of=1024)``)
— see analysis doc ``/tmp/guoqiong-recipe-analysis.md``.

Do not copy ``torchtitan/rl/multinode_launcher.py`` into this trial; ezpz's RL
path uses ``train_upstream.EzpzPerHostProvisioner`` for node splitting. Only
the layout arithmetic is portable.  See also ``/tmp/guoqiong-port-analysis.md``,
sec 7, for the PBS/MPI overlap and the oneCCL env the caller must export
(e.g. ``CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD=0`` for ``_gtp``).

Example (Qwen, 8 hosts x ppn4): trainer=16, rep=4 shard=4.  Override with
DP_REPLICATE=1 -> rep=1 shard=16 (one copy across 4 nodes).  The caller
(``config_registry.py`` or a shell sweep) applies the returned tuple to
``trainer.parallelism`` and ``num_generators`` and, for ``_gtp``, to
``generator.parallelism.tp``.
"""

from __future__ import annotations


def ezpz_host_layout(
    num_hosts: int,
    gpus_per_host: int,
    *,
    ffn_dim1: int = 3072,
) -> tuple[int, int, int]:
    """Return ``(dp_replicate, dp_shard, num_generators)`` for ``num_hosts`` hosts.

    Policy: trainer claims the first ``num_hosts // 2`` nodes (or half of one
    node's GPUs for ``num_hosts == 1``), generators the rest.  One model copy
    per trainer node keeps FSDP's per-layer all-gather intra-node
    (``dp_shard <= gpus_per_host``); when that shard cannot evenly partition the
    FFN stacked weight (``ffn_dim1 % dp_shard == 0`` unless ``dp_shard <= 2``),
    the smallest ``dp_replicate`` that can is used.

    Guoqiong semantics: ``dp_replicate=1`` shards one copy across all trainer
    nodes (the "model too large for one node" layout); ``dp_replicate=2`` is
    two copies each across half.  The caller sets those explicitly; this helper
    returns the default (node-local HSDP) layout.
    """
    if num_hosts == 1:
        trainer = gpus_per_host // 2
        generators = gpus_per_host - gpus_per_host // 2
        replicas = 1
    else:
        trainer = num_hosts // 2 * gpus_per_host
        generators = (num_hosts - num_hosts // 2) * gpus_per_host
        replicas = num_hosts // 2
    for rep in (replicas, *range(1, trainer + 1)):
        if trainer % rep != 0:
            continue
        shard = trainer // rep
        if shard <= gpus_per_host and (shard <= 2 or ffn_dim1 % shard == 0):
            return rep, shard, generators
    raise ValueError(f"no trainer mesh for {num_hosts} hosts x {gpus_per_host} GPUs")


def resolve_dp_degrees(
    num_hosts: int,
    gpus_per_host: int,
    *,
    dp_replicate: int | None = None,
    ffn_dim1: int = 3072,
) -> tuple[int, int, int]:
    """Resolve concrete ``(dp_replicate, dp_shard, num_generators)``.

    ``dp_replicate`` overrides the default node-local choice; the shard is
    scaled inversely (``rep * shard == trainer_gpus`` invariant).  Matches
    Guoqiong's ``_rl_grpo_qwen3_0_6b_hosts(dp_replicate=...)`` arithmetic.
    Used by shell sweeps that compute ``--parallelism.data-parallel-*`` CLI
    flags without importing recipe Python (also by ``config_registry.py``).
    """
    rep, shard, generators = ezpz_host_layout(
        num_hosts, gpus_per_host, ffn_dim1=ffn_dim1
    )
    if dp_replicate is not None:
        trainer_gpus = rep * shard
        if trainer_gpus % dp_replicate != 0:
            raise ValueError(
                f"dp_replicate={dp_replicate} does not divide trainer_gpus={trainer_gpus} "
                f"(from layout {rep}x{shard}); would truncate world-size"
            )
        shard = trainer_gpus // dp_replicate
        rep = dp_replicate
        if shard > 2 and ffn_dim1 % shard != 0:
            raise ValueError(
                f"resolved dp_shard={shard} does not divide ffn_dim1={ffn_dim1}; "
                f"no valid colwise sharding for w13:[2,{ffn_dim1},D]"
            )
    return rep, shard, generators


def generator_count_for_tp(
    num_hosts: int,
    gpus_per_host: int,
    *,
    generator_tp: int = 1,
    ffn_dim1: int = 3072,
) -> int:
    """Generators after folding in ``generator_tp`` (``_gtp`` semantics).

    ``num_generators -> generators // generator_tp`` per
    ``config.num_generators = generators // generator_tp`` in
    ``_rl_grpo_qwen3_0_6b_hosts``.  Emitted in the recipe only when
    ``generator_gpus % tp == 0`` (where ``generator_gpus`` is intra-host
    generator GPUs for 1 host else ``gpus_per_host``).
    """
    _, _, generators = ezpz_host_layout(num_hosts, gpus_per_host, ffn_dim1=ffn_dim1)
    if generators % generator_tp != 0:
        raise ValueError(
            f"generators={generators} not divisible by generator_tp={generator_tp}"
        )
    return generators // generator_tp


# Usage (shell sweep / PBS launcher):
#   python -c "from torchtitan.experiments.ezpz.rl.xpu_trial.layouts import resolve_dp_degrees; print(resolve_dp_degrees(8, 4, dp_replicate=1))"  # noqa: B950
#   # -> (1, 16, 16)
#   python -c "from torchtitan.experiments.ezpz.rl.xpu_trial.layouts import generator_count_for_tp; print(generator_count_for_tp(2, 4, generator_tp=2))"  # noqa: B950
#   # -> 2  (4 generator GPUs -> 2 generators x TP2)
# Invariants to check at launch:
#   dp_replicate * dp_shard == trainer_gpus
#   dp_shard <= gpus_per_host
#   ffn_dim1 % dp_shard == 0 or dp_shard <= 2
# Known hang: 8n dp=2x8 full-arm and 16n dp=2x16 both arms hang in FSDP2 HSDP
# backward all-gather on XPU — use _rep1 or default node-local instead.
