# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply PT-D parallelisms + AC + compile + FSDP to the ezpz/moe model.

This is the moe mirror of `torchtitan.models.deepseek_v3.parallelize`.
Post upstream PR #3386 (37th sync), MoE TP/EP is no longer applied by a
separate `apply_moe_ep_tp` pass — it is now folded into the config-based
DTensor sharding API (`ShardingConfig` declarations populated by
`moeModel.Config.update_from_config` and applied by
`model.parallelize(parallel_dims)`).

Differences vs upstream `parallelize_deepseekv3`:

- `disable_fsdp_gradient_division` enables
  `set_force_sum_reduction_for_comms(True)` for non-NCCL backends
  (CCL on XPU). Upstream's version only sets the divide factor.
- `apply_compile`: upstream uses fullgraph=True via `apply_compile_sparse`,
  which fails on XPU (MoE routing's dynamic shapes). We compile each
  block with `block.compile(backend=...)` (no fullgraph).
- `apply_fsdp` is inlined locally to avoid importing
  `ShardPlacementResult`, which doesn't exist in Aurora's PyTorch. Also
  adds a Shard(0) fallback when expert hidden dim isn't divisible by the
  FSDP world size.
"""

from typing import Any

import ezpz
import ezpz.distributed
import torch
import torch.distributed
import torch.nn as nn
from ezpz.models import summarize_model
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Shard

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torch.distributed.fsdp import DataParallelMeshDims
from torchtitan.experiments.ezpz.fsdp_compat import (
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
# 78th sync (upstream #4045): maybe_enable_async_tp was REMOVED -- async TP is
# now enabled inside apply_compile from parallel_dims. Importing it is an
# ImportError. This file compiles per-block directly (see the compile block
# below) rather than calling apply_compile, so there is nothing to thread
# parallel_dims into here; async TP simply is not available on this path.
# That is not a regression: the call below only ran under tp_enabled, and the
# MoE path has never been validated with async TP on XPU.
from torchtitan.experiments.ezpz.moe import moeModel
from torchtitan.tools.logging import logger


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


def parallelize_moe(
    model: moeModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Apply CP + TP + EP + AC + compile + FSDP to the moe model.

    The passed-in model preferably should be on meta device. Otherwise
    the model must fit on GPU or CPU memory.
    """
    assert (
        training.max_context_length % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.max_context_length} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # CP: wrap inner attention forward BEFORE parallelize() so CP logic
    # runs inside the local_map boundary on local tensors.
    # CP removed upstream (#4218): apply_cp_to_forward is gone, and the
    # replacement validate_cp_backend() rejects cp>1 on anything but
    # spmd_types. We pin partial_dtensor and every ezpz config sets
    # context_parallel_degree=1, so this branch was already dead.

    # TP/EP via the config-based sharding API. The model's
    # ``sharding_config`` declarations were filled in by
    # ``update_from_config`` (see model.py + sharding.py), covering both
    # dense (attention, dense FFN) and MoE (router, shared/routed experts)
    # submodules. ``GroupedExperts.parallelize`` additionally wires the
    # EP/TP meshes onto the token dispatcher.
    # 79th sync (#4085): under full_dtensor/spmd_types this must run
    # UNCONDITIONALLY -- it is what makes the params DTensors on the SPMD mesh,
    # which resolve_fsdp_mesh's DataParallelMeshDims then requires. Gating on
    # tp/ep (right for the legacy backend) leaves plain tensors when both are
    # off. Core: llama3/parallelize.py:42-53.
    # Upstream #4217 removed validate_config outright; deepseek_v3 (our base)
    # now just calls model.parallelize. "full_dtensor" stays in the tuple only
    # until the sync lands, since it is still a legal value on this tree.
    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        model.parallelize(parallel_dims)
    elif parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

    # 78th sync (#4045): the maybe_enable_async_tp call that lived here is
    # gone -- see the import-site note above.

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    # 57th sync: PR #3674 refactored AC into a Configurable policy
    # hierarchy. ac_config is now an ActivationCheckpointing.Config
    # subclass or None. The MoE save-set override lives in
    # ``ezpz/moe/activation_checkpoint.py`` as a SelectiveAC subclass
    # (MoeSelectiveAC) which drops _c10d_functional.all_to_all_single
    # from the save list -- see that file for the rationale.
    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if model_compile_enabled:
        # Upstream apply_compile_sparse uses fullgraph=True which fails on
        # XPU after 00b7f569 removed maybe_enable_amp — MoE routing's
        # dynamic shapes cause recompilation that fullgraph=True forbids.
        # Apply compile per-block without fullgraph instead.
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        for layer_id, block in model.layers.named_children():
            block.compile(backend=compile_config.backend)
            model.layers.register_module(layer_id, block)

    # 79th sync: upstream #4085 made spmd_types the DEFAULT backend, and under
    # spmd_types/full_dtensor there is no flattened "fsdp" mesh axis -- asking
    # for it raises ValueError: Invalid mesh dim: 'fsdp'. Mirror core's
    # backend branch (llama3/parallelize.py:71) instead of hardcoding a name.
    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        # BOTH meshes must come from the new resolvers. Resolving only the
        # dense one and leaving edp on the legacy "efsdp" name makes both
        # resolve to the same dp_shard mesh, and FSDP then refuses:
        #   RuntimeError: Cannot concatenate overlapping meshes:
        #   [DeviceMesh((dp_shard=24)...), DeviceMesh((dp_shard=24)...)]
        # Core pairs them in one branch (gpt_oss/parallelize.py:108-110).
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    else:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        dp_mesh_dims = None
        edp_mesh = None
        edp_mesh_dims = None
        if parallel_dims.ep_enabled:
            edp_mesh_names = (
                ["dp_replicate", "efsdp"]
                if parallel_dims.dp_replicate_enabled
                else ["efsdp"]
            )
            edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp(
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
    )

    if parallel_dims.dp_replicate_enabled:
        logger.info("Applied HSDP to the model")
    else:
        logger.info("Applied FSDP to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    logger.info(f"\n+{summarize_model(model)}")

    return model


# ---------------------------------------------------------------------------
# Inlined from torchtitan.models.llama4.parallelize to avoid importing
# ShardPlacementResult, which doesn't exist in Aurora's PyTorch framework
# release. Also adds a Shard(0) fallback when the expert hidden dim isn't
# divisible by the FSDP world size — upstream's version assumes divisibility
# and crashes otherwise.
# ---------------------------------------------------------------------------


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    dp_mesh_dims: DataParallelMeshDims | None = None,
    edp_mesh_dims: DataParallelMeshDims | None = None,
):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if dp_mesh_dims is not None:
        # Multi-axis storage mesh under full_dtensor/spmd_types: fully_shard
        # must be told which axes are data-parallel (core does the same, see
        # distributed/fsdp.py:106). None under the legacy backend, where the
        # flattened "fsdp" axis already encodes it.
        fsdp_config["dp_mesh_dims"] = dp_mesh_dims
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    if model.norm is not None and model.lm_head is not None:
        fully_shard(
            [model.norm, model.lm_head],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    for layer_id, transformer_block in model.layers.items():
        if transformer_block.moe_enabled:
            assert hasattr(transformer_block, "moe")
            expert_params = set(
                transformer_block.moe.routed_experts.inner_experts.parameters()
            )
            num_experts = (
                transformer_block.moe.routed_experts.inner_experts.num_experts
            )

            if ep_degree > 1:
                assert edp_mesh is not None
                efsdp_ep_size = edp_mesh["efsdp"].size() * ep_degree
            else:
                efsdp_ep_size = fsdp_config["mesh"].size()

            # Shard(1) shards the hidden dim instead of expert dim when
            # there are more FSDP ranks than experts. But this requires
            # the hidden dim to be evenly divisible by the world size.
            # Fall back to Shard(0) if not (avoids uneven sharding error).
            if efsdp_ep_size > num_experts:
                expert_w = next(
                    iter(transformer_block.moe.routed_experts.inner_experts.parameters())
                )
                if expert_w.shape[1] % efsdp_ep_size == 0:
                    expert_shard_placement = Shard(1)
                else:
                    expert_shard_placement = Shard(0)
            else:
                expert_shard_placement = Shard(0)

            if ep_degree == 1 and expert_shard_placement == Shard(0):
                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )
            elif ep_degree == 1:
                def _experts_shard_placement_fn(
                    param: nn.Parameter,
                    _expert_params: set = expert_params,
                ) -> Shard | None:
                    if param in _expert_params:
                        return Shard(1)
                    return None

                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    shard_placement_fn=_experts_shard_placement_fn,
                )
            else:
                # ep_degree > 1: per-param mesh with ShardPlacementResult.
                # Imported lazily to avoid hard dependency on a private
                # PyTorch API path that may not exist in older releases.
                from torch.distributed.fsdp._fully_shard._fsdp_common import (
                    FSDPMeshInfo,
                    ShardPlacementResult,
                )
                from torch.distributed.fsdp._fully_shard._fsdp_init import (
                    _get_mesh_info,
                )

                assert edp_mesh is not None

                # 79th sync: build the mesh infos via FSDP2's own builder
                # rather than FSDPMeshInfo(mesh=..., shard_mesh_dim=0).
                # Under full_dtensor/spmd_types the meshes handed in are FULL
                # SPMD meshes; _get_mesh_info EXTRACTS AND FLATTENS the DP
                # submesh out of them using mesh_dims. Constructing
                # FSDPMeshInfo directly leaves both at full width, so the
                # dense and sparse infos both present as dp_shard and
                # fully_shard rejects them:
                #   RuntimeError: Cannot concatenate overlapping meshes:
                #   [DeviceMesh((dp_shard=24)...), DeviceMesh((dp_shard=24)...)]
                # Core: distributed/fsdp.py:274-279.
                edp_mesh_info = _get_mesh_info(edp_mesh, edp_mesh_dims)
                dp_mesh_info = _get_mesh_info(dp_mesh, dp_mesh_dims)
                # _get_mesh_info is typed to the DataParallelMeshInfo base;
                # with a shard dim it always yields FSDPMeshInfo/HSDPMeshInfo.
                assert isinstance(edp_mesh_info, FSDPMeshInfo)
                assert isinstance(dp_mesh_info, FSDPMeshInfo)

                def _shard_placement_fn(
                    param: nn.Parameter,
                    _expert_params: set = expert_params,
                    _expert_placement: Shard = expert_shard_placement,
                    _edp_mesh_info: FSDPMeshInfo = edp_mesh_info,
                    _dp_mesh_info: FSDPMeshInfo = dp_mesh_info,
                ) -> ShardPlacementResult:
                    if param in _expert_params:
                        return ShardPlacementResult(
                            placement=_expert_placement, mesh_info=_edp_mesh_info
                        )
                    return ShardPlacementResult(
                        placement=Shard(0), mesh_info=_dp_mesh_info
                    )

                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    shard_placement_fn=_shard_placement_fn,
                )
        else:
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)
