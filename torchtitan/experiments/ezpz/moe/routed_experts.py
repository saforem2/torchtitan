# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""RoutedExperts variant that forwards the routing decision to the experts.

Core's ``RoutedExperts.forward`` receives ``topk_scores_TK`` and
``topk_expert_ids_TK`` (models/common/moe.py:143-144) and then drops them,
calling ``inner_experts(routed_input_RD, num_global_tokens_per_local_expert_e)``
at line 163. Five of the six ezpz expert backends want exactly that two-argument
contract. ``aurora_full_sonic`` does not: it performs its own expert-parallel
all-to-all and needs the router's decision plus an EP mesh.

Rather than restructure the dispatcher, this subclass forwards those tensors to
``inner_experts`` when -- and only when -- the inner module advertises that it
wants them. Everything else takes the identical path as before.
"""

from dataclasses import dataclass

from torchtitan.models.common.moe import RoutedExperts


class EzpzRoutedExperts(RoutedExperts):
    """RoutedExperts that can hand the routing decision to inner_experts."""

    # Without its own nested Config the subclass inherits RoutedExperts.Config,
    # whose build() constructs the PARENT -- the override would never run and
    # the sonic backend would silently get the two-argument contract. Follows
    # the EzpzGroupedExperts pattern (kw_only + slots).
    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        pass

    def _wants_routing(self) -> bool:
        return getattr(self.inner_experts, "compute_backend", None) == "aurora_full_sonic"

    def forward(
        self,
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
    ):
        if not self._wants_routing():
            # Byte-for-byte the upstream path. The other five backends, and
            # every dense config, must not move.
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
        ep_mesh = getattr(self.token_dispatcher, "ep_mesh", None)
        return self.inner_experts(
            x_TD,
            num_local_tokens_per_expert_E,
            topk_scores=topk_scores_TK,
            topk_indices=topk_expert_ids_TK,
            ep_mesh=ep_mesh,
        )
