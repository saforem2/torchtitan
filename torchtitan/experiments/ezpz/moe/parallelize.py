# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply PT-D parallelisms + AC + compile + FSDP to the ezpz/moe model.

This is the moe mirror of the ``BaseModel.parallelize`` lifecycle that
``torchtitan.models.deepseek_v3`` uses. Sharding (TP/EP) is declared on the
model config (``moe/sharding.py``) and applied by ``model._parallelize``.

Differences vs the upstream lifecycle:

- FSDP is upstream's ``apply_fsdp_to_decoder`` (dense and routed-expert
  meshes). The previous local copy existed to avoid ``ShardPlacementResult``
  on Aurora's PyTorch and to add a Shard(0) fallback; both are covered by
  current upstream, and the ``efsdp`` mesh axis it indexed no longer exists
  (``edp_shard``).
- ``disable_fsdp_gradient_division`` additionally enables
  ``set_force_sum_reduction_for_comms(True)`` for non-NCCL backends (CCL on
  XPU). Upstream's version only sets the divide factor.
- Per-block ``block.compile`` is gone with ``CompileConfig``; compilation is
  selected through ``local_compile_regions`` (see ``ezpz/compile.py``).
"""

import contextlib

import ezpz
import ezpz.distributed
import torch
import torch.distributed
import torch.nn as nn
from ezpz.models import summarize_model

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import fsdp as core_fsdp, ParallelismContext
from torchtitan.distributed.activation_remat import ActivationRematConfig
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.experiments.ezpz.fsdp_compat import (
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)
from torchtitan.experiments.ezpz.logging import logger
from torchtitan.experiments.ezpz.moe import moeModel


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


@contextlib.contextmanager
def _evenly_divisible_expert_row_sharding(fsdp_shard_size: int):
    """Fall back to expert-dim sharding when expert rows do not divide evenly.

    When the FSDP degree exceeds ``num_experts``, upstream shards routed-expert
    weights along their matrix-row dimension instead of the expert dimension
    (``linear_param_shard_placements(include_unstacked_grouped=True)``). FSDP2
    rejects uneven row sharding, so a hidden size that is not a multiple of the
    shard group (e.g. ``moe_debugmodel`` F=256 over 24 ranks) fails with
    ``FSDP does not support uneven sharding``. ezpz's previous local
    ``apply_fsdp`` fell back to ``Shard(0)`` (the expert dim, which FSDP2 pads)
    in that case; drop non-divisible entries so the default ``Shard(0)`` applies.
    """
    original = core_fsdp.linear_param_shard_placements

    def divisible_only(module, *, include_unstacked_grouped=False):
        placements = original(
            module, include_unstacked_grouped=include_unstacked_grouped
        )
        if not include_unstacked_grouped:
            return placements
        return {
            param: placement
            for param, placement in placements.items()
            if param.shape[placement.dim] % fsdp_shard_size == 0
        }

    core_fsdp.linear_param_shard_placements = divisible_only
    try:
        yield
    finally:
        core_fsdp.linear_param_shard_placements = original


def parallelize_moe(
    model: moeModel,
    *,
    parallelism_context: ParallelismContext,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    local_compile_regions: list[str],
    activation_remat: ActivationRematConfig | None,
    dump_folder: str,
):
    """Apply local compile + CP/TP/EP sharding + remat + FSDP to the moe model.

    The passed-in model preferably should be on meta device. Otherwise
    the model must fit on GPU or CPU memory. Same lifecycle as
    ``BaseModel.parallelize``; FSDP is upstream's ``apply_fsdp_to_decoder``
    (dense + routed-expert meshes, ``edp_shard`` prefetching), followed by
    ezpz's XPU gradient-division fix.
    """
    del dump_folder
    # Bind the @local_compile regions before any module runs (process-wide).
    apply_local_compile(local_compile_regions)
    parallel_dims = parallelism_context
    assert (
        training.max_context_length % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.max_context_length} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # TP/EP via the config-based sharding API: the model's ``sharding_config``
    # declarations (moe/sharding.py) cover dense (attention, dense FFN) and
    # MoE (router, shared/routed experts) submodules. BaseModel.parallelize()
    # owns this lifecycle and calls this custom implementation from
    # moeModel.parallelize(); invoke only the internal sharding pass here,
    # since calling model.parallelize() would recurse.
    model._parallelize(parallel_dims)

    # Upstream moved activation checkpointing to torch_remat (#4893, #5101).
    if activation_remat is not None:
        activation_remat.apply(model)

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    if parallel_dims.ep > 1:
        assert edp_mesh is not None
        fsdp_shard_size = edp_mesh["edp_shard"].size()
    else:
        fsdp_shard_size = dp_mesh["dp_shard"].size()
        if "cp" in (dp_mesh.mesh_dim_names or ()):
            fsdp_shard_size *= dp_mesh["cp"].size()
    with _evenly_divisible_expert_row_sharding(fsdp_shard_size):
        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            dp_mesh_dims=dp_mesh_dims,
            edp_mesh_dims=edp_mesh_dims,
            symm_mem_scope=parallelism.fsdp_symm_mem_scope,
        )
    # On XPU/CCL the sum reduction must be forced explicitly.
    disable_fsdp_gradient_division(model)

    logger.info(f"\n+{summarize_model(model)}")

    return model
