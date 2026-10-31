# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""RoutedExperts variant with portable and Aurora expert kernels.

Upstream now owns expert weights as ``GroupedLinear`` children directly under
``RoutedExperts``. This subclass preserves that layout and only replaces the
local expert-compute portion when a non-default backend is selected.
"""

from dataclasses import dataclass

import torch_remat as remat
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh, spmd_sparse_mesh
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import dense_sequence_parallel_placement
from torchtitan.models.common.moe import RoutedExperts

from .experts import (
    _env_flag_enabled,
    _run_experts_aurora_full_sonic,
    _run_experts_aurora_sycl,
    _run_experts_bmm,
    _run_experts_bmm_nodrop,
    _run_experts_for_loop,
    ExpertComputeBackend,
)


class EzpzRoutedExperts(RoutedExperts):
    """RoutedExperts that can hand the routing decision to inner_experts.

    Upstream renamed the 5th arg from ``num_local_tokens_per_expert_E`` to
    ``routing_map_TE`` (one-hot map, #5323). Ezpz keeps the old count-based
    contract, so we provide a matching ShardingConfig here.
    """

    _EZPZ_ROUTED_EXPERTS_SHARDING = ShardingConfig(
        in_src_shardings={
            "x_TD": dense_sequence_parallel_placement(),
            "topk_scores_TK": dense_sequence_parallel_placement(),
            "topk_expert_ids_TK": dense_sequence_parallel_placement(),
            "num_local_tokens_per_expert_E": dense_sequence_parallel_placement(),
        },
        out_src_shardings=dense_sequence_parallel_placement(),
        local_spmd=True,
    )

    # Without its own nested Config the subclass inherits RoutedExperts.Config,
    # whose build() constructs the PARENT -- the override would never run and
    # the sonic backend would silently get the two-argument contract. Follows
    # the fork's other specialized module configs (kw_only + slots).
    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        compute_backend: ExpertComputeBackend = "grouped_mm"
        capacity_factor: float = 1.25

    def __init__(self, config: Config):
        super().__init__(config)
        self.compute_backend = config.compute_backend
        self.capacity_factor = config.capacity_factor
        if self._wants_routing() and self.output_postprocess is not None:
            raise ValueError(
                "aurora_full_sonic does not support output_postprocess: Sonic "
                "owns dispatch, expert compute, and combine, so TorchTitan "
                "cannot apply a route-wise module before combine"
            )

    def _wants_routing(self) -> bool:
        return self.compute_backend == "aurora_full_sonic"

    def _resolve_ep_mesh(self):
        """Return the live sparse EP mesh, falling back for older lifecycles."""
        sparse_mesh = spmd_sparse_mesh()
        return (
            sparse_mesh["ep"]
            if sparse_mesh is not None
            else getattr(self.token_dispatcher, "ep_mesh", None)
        )

    def forward(
        self,
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
        *,
        padding_mask_T=None,
    ):
        if self.compute_backend == "grouped_mm":
            return super().forward(
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
                padding_mask_T=padding_mask_T,
            )
        del padding_mask_T  # padding rows are routed like real tokens

        # sycl_sonic owns dispatch AND combine: it consumes unrouted tokens
        # plus the routing decision and returns combined output, so we must
        # NOT run the token dispatcher around it. Doing both would route
        # twice.
        # Current upstream owns sparse-mesh lifetime in the SPMD runtime
        # context. Prefer that authoritative mesh; retain the dispatcher field
        # as compatibility for older TorchTitan lifecycles.
        if self._wants_routing():
            return self._run_backend(
                x_TD,
                num_local_tokens_per_expert_E,
                topk_scores=topk_scores_TK,
                topk_indices=topk_expert_ids_TK,
                ep_mesh=self._resolve_ep_mesh(),
            )

        routed_input_RD, counts_e, metadata = self.token_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        with maybe_set_sparse_mesh():
            # The portable expert kernels below read the dispatched tokens and
            # counts with bare ops (not torch_remat regions).
            remat.recompute_needs_tensor(routed_input_RD, counts_e)
            routed_output_RD = self._run_backend(routed_input_RD, counts_e)
            if self.output_postprocess is not None:
                routed_output_RD = self.output_postprocess(routed_output_RD)
        remat.recompute_needs_tensor(routed_output_RD)
        return self.token_dispatcher.combine(routed_output_RD, metadata, x_TD)

    def _weights(self):
        w13 = self.w13.weight
        w2 = self.w2.weight
        if isinstance(w13, DTensor):
            w13 = w13.to_local()
            w2 = w2.to_local()
        return w13[:, 0], w2, w13[:, 1]

    def _run_backend(
        self, x, counts, *, topk_scores=None, topk_indices=None, ep_mesh=None
    ):
        w1, w2, w3 = self._weights()
        if self.compute_backend == "for_loop":
            if _env_flag_enabled("TT_MOE_CHECKPOINT_EXPERTS"):
                return checkpoint(
                    _run_experts_for_loop,
                    w1,
                    w2,
                    w3,
                    x,
                    counts,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            return _run_experts_for_loop(w1, w2, w3, x, counts)
        if self.compute_backend == "bmm":
            return _run_experts_bmm(w1, w2, w3, x, counts, self.capacity_factor)
        if self.compute_backend == "bmm_nodrop":
            return _run_experts_bmm_nodrop(w1, w2, w3, x, counts)
        if self.compute_backend == "aurora_sycl":
            return _run_experts_aurora_sycl(w1, w2, w3, x, counts)
        if self.compute_backend == "aurora_full_sonic":
            return _run_experts_aurora_full_sonic(
                w1,
                w2,
                w3,
                x,
                counts,
                topk_scores=topk_scores,
                topk_indices=topk_indices,
                ep_mesh=ep_mesh,
            )
        raise ValueError(f"Unknown expert compute backend: {self.compute_backend!r}")
