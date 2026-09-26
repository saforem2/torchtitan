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

from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh, spmd_sparse_mesh
from torchtitan.models.common.moe import RoutedExperts

from .experts import (
    _run_experts_aurora_full_sonic,
    _run_experts_aurora_sycl,
    _run_experts_bmm,
    _run_experts_bmm_nodrop,
    _run_experts_for_loop,
    ExpertComputeBackend,
    _env_flag_enabled,
)

class EzpzRoutedExperts(RoutedExperts):
    """RoutedExperts that can hand the routing decision to inner_experts."""

    # Without its own nested Config the subclass inherits RoutedExperts.Config,
    # whose build() constructs the PARENT -- the override would never run and
    # the sonic backend would silently get the two-argument contract. Follows
    # the EzpzGroupedExperts pattern (kw_only + slots).
    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        compute_backend: ExpertComputeBackend = "grouped_mm"
        capacity_factor: float = 1.25

    def __init__(self, config: Config):
        super().__init__(config)
        self.compute_backend = config.compute_backend
        self.capacity_factor = config.capacity_factor

    def _wants_routing(self) -> bool:
        return (
            self.compute_backend == "aurora_full_sonic"
        )

    def _parallelize(self, parallel_dims) -> None:
        """Parallelize children, then install dispatcher mesh ownership.

        Before upstream moved to recursive ``Module._parallelize``,
        ``RoutedExperts.parallelize`` performed this handoff explicitly. The
        custom ezpz dispatcher is not itself a ``Module``, so recursion cannot
        discover or wire it. Restore the lifecycle at the owning routed-experts
        boundary and preserve TP coordinates for sequence-parallel dispatch.
        """
        super()._parallelize(parallel_dims)
        self.token_dispatcher.wire_meshes(
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
        )

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
    ):
        if self.compute_backend == "grouped_mm":
            return super().forward(
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )

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
            routed_output_RD = self._run_backend(routed_input_RD, counts_e)
        return self.token_dispatcher.combine(routed_output_RD, metadata, x_TD)

    def _weights(self):
        w13 = self.w13.weight
        w2 = self.w2.weight
        if isinstance(w13, DTensor):
            w13 = w13.to_local()
            w2 = w2.to_local()
        return w13[:, 0], w2, w13[:, 1]

    def _run_backend(self, x, counts, *, topk_scores=None, topk_indices=None, ep_mesh=None):
        w1, w2, w3 = self._weights()
        if self.compute_backend == "for_loop":
            if _env_flag_enabled("TT_MOE_CHECKPOINT_EXPERTS"):
                return checkpoint(_run_experts_for_loop, w1, w2, w3, x, counts,
                                  use_reentrant=False, preserve_rng_state=False)
            return _run_experts_for_loop(w1, w2, w3, x, counts)
        if self.compute_backend == "bmm":
            return _run_experts_bmm(w1, w2, w3, x, counts, self.capacity_factor)
        if self.compute_backend == "bmm_nodrop":
            return _run_experts_bmm_nodrop(w1, w2, w3, x, counts)
        if self.compute_backend == "aurora_sycl":
            return _run_experts_aurora_sycl(w1, w2, w3, x, counts)
        if self.compute_backend == "aurora_full_sonic":
            return _run_experts_aurora_full_sonic(
                w1, w2, w3, x, counts, topk_scores=topk_scores,
                topk_indices=topk_indices, ep_mesh=ep_mesh)
        raise ValueError(f"Unknown expert compute backend: {self.compute_backend!r}")
