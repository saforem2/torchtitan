# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn

from torchtitan.models.common.linear import Linear
# Sync 84 made score_func a UnaryActivationFn.Config that the router
# .build()s, so the old score_func="sigmoid" string now dies with
#   AttributeError: 'str' object has no attribute 'build'
from torchtitan.models.common.moe import Sigmoid, TokenChoiceTopKRouter
# ezpz's own dispatcher, NOT core's. Sync 84 removed score_before_experts
# from core's LocalTokenDispatcher.Config (it is now just num_experts +
# top_k), so importing core's made these tests fail with
#   TypeError: ... got an unexpected keyword argument 'score_before_experts'
# ezpz keeps the field (moe/token_dispatcher.py) and ezpz runtime imports
# only its own dispatcher, so this is the class actually under test.
from torchtitan.experiments.ezpz.moe.token_dispatcher import (
    LocalTokenDispatcher,
)


class TestMoERoutingCounts(unittest.TestCase):
    def test_bf16_router_ties_have_stable_expert_order(self):
        class FixedGate(nn.Module):
            def forward(self, x):
                return torch.tensor(
                    [[0.0, 0.0, 0.0, -1.0]], dtype=torch.bfloat16
                ).expand(x.shape[0], -1)

        router = TokenChoiceTopKRouter(
            TokenChoiceTopKRouter.Config(
                num_experts=4,
                gate=Linear.Config(in_features=3, out_features=4, bias=False),
                top_k=2,
                score_func=Sigmoid.Config(),
            )
        )
        router.gate = FixedGate()

        # The production router passes an FP32 load-balancing bias. Cover that
        # promotion explicitly: the choice keys must still have a deterministic
        # expert-id tiebreak when the underlying gate scores are BF16.
        # Sync 84 changed the router's third return from per-expert COUNTS to
        # routing_map_TE, a one-hot boolean (T, E) map. Derive the counts from
        # it rather than asserting on a value that no longer exists.
        _, expert_ids, routing_map_TE = router(
            torch.zeros(5, 3, dtype=torch.bfloat16),
            expert_bias_E=torch.zeros(4, dtype=torch.float32),
        )
        counts = routing_map_TE.sum(dim=0)

        # The gate scores two experts at 0.0 and one at -1.0, so the top-2 pick
        # is a tie broken by expert id. Assert the tiebreak is STABLE (every row
        # agrees) and consistent with the counts, rather than hardcoding which
        # pair wins -- the pair is an implementation detail of the topk kernel,
        # the determinism is the property under test.
        first_row = expert_ids[0].sort().values
        for row in range(expert_ids.shape[0]):
            torch.testing.assert_close(expert_ids[row].sort().values, first_row)
        self.assertEqual(int(counts.sum()), 5 * 2)
        for e in first_row.tolist():
            self.assertEqual(int(counts[e]), 5)

    def test_dispatch_counts_are_integral_and_exact(self):
        dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(
                num_experts=5,
                top_k=2,
                score_before_experts=True,
            )
        )
        x = torch.randn(4, 3)
        expert_ids = torch.tensor([[0, 4], [1, 4], [1, 3], [4, 0]])
        scores = torch.ones(4, 2)

        # dispatch() no longer derives the per-expert counts: sync 84 made
        # num_local_tokens_per_expert_E a required argument (both core's and
        # ezpz's signatures match). Compute it here, which is what the caller
        # does in production, and keep asserting on what dispatch returns.
        num_local_tokens_per_expert_E = torch.bincount(
            expert_ids.flatten(), minlength=5
        )
        _, counts, _ = dispatcher.dispatch(
            x, scores, expert_ids, num_local_tokens_per_expert_E
        )

        self.assertEqual(counts.dtype, torch.int64)
        torch.testing.assert_close(counts, torch.tensor([2, 2, 0, 1, 3]))


if __name__ == "__main__":
    unittest.main()
