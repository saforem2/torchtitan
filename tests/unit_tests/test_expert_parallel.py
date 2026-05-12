# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import (
    _run_experts_for_loop,
    GroupedExperts,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.token_dispatcher import (
    _MOE_FASTPATH_COUNTERS,
    _normal_equal_a2a_padding_enabled,
    _record_moe_fastpath,
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
    TorchAOTokenDispatcher,
)
from torchtitan.ops.scatter_add import (
    deterministic_scatter_add,
    deterministic_scatter_add_,
)


class ScaleSharedExperts(nn.Module):
    def forward(self, x):
        return x * 0.25


class IdentitySharedExperts(nn.Module):
    def forward(self, x):
        return x


class TestMoEFastPathCounters(unittest.TestCase):
    def test_counters_are_environment_gated(self):
        previous = os.environ.pop("TT_MOE_DEBUG_FASTPATHS", None)
        _MOE_FASTPATH_COUNTERS.clear()
        try:
            _record_moe_fastpath("disabled_counter")
            self.assertNotIn("disabled_counter", _MOE_FASTPATH_COUNTERS)

            os.environ["TT_MOE_DEBUG_FASTPATHS"] = "1"
            _record_moe_fastpath("enabled_counter")
            self.assertEqual(_MOE_FASTPATH_COUNTERS["enabled_counter"], 1)
        finally:
            _MOE_FASTPATH_COUNTERS.clear()
            if previous is None:
                os.environ.pop("TT_MOE_DEBUG_FASTPATHS", None)
            else:
                os.environ["TT_MOE_DEBUG_FASTPATHS"] = previous

    def test_normal_equal_a2a_padding_is_environment_gated(self):
        previous = os.environ.pop("TT_MOE_NORMAL_EQUAL_A2A_PADDING", None)
        try:
            self.assertFalse(_normal_equal_a2a_padding_enabled())
            os.environ["TT_MOE_NORMAL_EQUAL_A2A_PADDING"] = "1"
            self.assertTrue(_normal_equal_a2a_padding_enabled())
        finally:
            if previous is None:
                os.environ.pop("TT_MOE_NORMAL_EQUAL_A2A_PADDING", None)
            else:
                os.environ["TT_MOE_NORMAL_EQUAL_A2A_PADDING"] = previous


class TestDeterministicScatterAdd(unittest.TestCase):
    def test_in_place_matches_out_of_place_and_mutates_out(self):
        out = torch.zeros(3, 2)
        index = torch.tensor([[0, 0], [2, 2], [0, 0], [1, 1]])
        src = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        )

        expected = deterministic_scatter_add(out.clone(), index, src)
        original_storage = out.untyped_storage().data_ptr()
        actual = deterministic_scatter_add_(out, index, src)

        self.assertEqual(actual.untyped_storage().data_ptr(), original_storage)
        self.assertEqual(out.untyped_storage().data_ptr(), original_storage)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(out, expected)


class TestLocalTokenDispatcherCorrectness(unittest.TestCase):
    def _run_case(self, score_before_experts: bool, use_shared_experts: bool):
        torch.manual_seed(1234)

        num_experts = 4
        top_k = 3
        num_tokens = 6
        dim = 5

        dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(
                num_experts=num_experts,
                top_k=top_k,
                score_before_experts=score_before_experts,
            )
        )

        x = torch.randn(num_tokens, dim)
        selected_experts_indices_ref = torch.tensor(
            [
                [2, 0, 3],
                [1, 2, 0],
                [3, 3, 1],
                [0, 2, 1],
                [1, 0, 2],
                [2, 3, 0],
            ],
            dtype=torch.long,
        )
        selected_experts_indices = selected_experts_indices_ref.to(x.dtype)
        top_scores = torch.tensor(
            [
                [0.55, 0.30, 0.15],
                [0.20, 0.50, 0.30],
                [0.60, 0.25, 0.15],
                [0.10, 0.70, 0.20],
                [0.40, 0.35, 0.25],
                [0.50, 0.20, 0.30],
            ],
            dtype=x.dtype,
        )
        expert_scales = torch.tensor([1.0, -0.5, 2.0, 0.75], dtype=x.dtype)
        num_tokens_per_expert_precomputed = torch.bincount(
            selected_experts_indices_ref.view(-1), minlength=num_experts
        ).to(x.dtype)

        routed_input, num_tokens_per_expert, metadata = dispatcher.dispatch(
            x,
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert_precomputed,
        )
        torch.testing.assert_close(
            num_tokens_per_expert, num_tokens_per_expert_precomputed
        )

        expert_outputs = []
        offset = 0
        for expert_idx, count in enumerate(num_tokens_per_expert.to(torch.int64)):
            count_int = count.item()
            expert_input = routed_input[offset : offset + count_int]
            expert_outputs.append(expert_input * expert_scales[expert_idx])
            offset += count_int
        routed_output = torch.cat(expert_outputs, dim=0)

        shared_experts = ScaleSharedExperts() if use_shared_experts else None
        actual = dispatcher.combine(routed_output, metadata, x, shared_experts)

        expected = (
            ScaleSharedExperts()(x) if use_shared_experts else torch.zeros_like(x)
        )
        for token_idx in range(num_tokens):
            for choice_idx in range(top_k):
                expert_idx = selected_experts_indices_ref[token_idx, choice_idx].item()
                expected[token_idx] += (
                    x[token_idx]
                    * expert_scales[expert_idx]
                    * top_scores[token_idx, choice_idx]
                )

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_score_before_experts_matches_direct_reference(self):
        self._run_case(score_before_experts=True, use_shared_experts=False)

    def test_score_after_experts_matches_direct_reference_with_shared_experts(self):
        self._run_case(score_before_experts=False, use_shared_experts=True)

    def test_score_after_experts_bfloat16_is_close_to_fp32_scoring_reference(self):
        num_experts = 4
        top_k = 3
        num_tokens = 5
        dim = 7
        dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(
                num_experts=num_experts,
                top_k=top_k,
                score_before_experts=False,
            )
        )

        x = torch.randn(num_tokens, dim, dtype=torch.bfloat16)
        selected_experts_indices_ref = (
            torch.arange(num_tokens * top_k).reshape(num_tokens, top_k) % num_experts
        )
        selected_experts_indices = selected_experts_indices_ref.to(x.dtype)
        top_scores = torch.randn(num_tokens, top_k, dtype=torch.float32).softmax(dim=1)
        counts = torch.bincount(
            selected_experts_indices_ref.view(-1), minlength=num_experts
        )

        routed_input, _, metadata = dispatcher.dispatch(
            x,
            top_scores,
            selected_experts_indices,
            counts,
        )
        actual = dispatcher.combine(routed_input, metadata, x)

        expected = torch.zeros(num_tokens, dim, dtype=torch.float32)
        for token_idx in range(num_tokens):
            for choice_idx in range(top_k):
                expected[token_idx] += (
                    x[token_idx].float() * top_scores[token_idx, choice_idx]
                )

        torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-2)

    def test_no_grad_combine_does_not_mutate_input_when_shared_experts_aliases_x(self):
        dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(
                num_experts=2,
                top_k=1,
                score_before_experts=False,
            )
        )
        x = torch.randn(4, 3)
        x_before = x.clone()
        selected_experts_indices = torch.tensor([[0], [1], [0], [1]]).to(x.dtype)
        top_scores = torch.ones(4, 1)

        routed_input, _, metadata = dispatcher.dispatch(
            x,
            top_scores,
            selected_experts_indices,
        )
        with torch.no_grad():
            out = dispatcher.combine(
                routed_input,
                metadata,
                x,
                shared_experts=IdentitySharedExperts(),
            )

        torch.testing.assert_close(x, x_before)
        torch.testing.assert_close(out, x_before * 2)

    def test_no_grad_combine_mutates_non_aliasing_shared_output(self):
        dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(
                num_experts=2,
                top_k=1,
                score_before_experts=False,
            )
        )
        x = torch.randn(4, 3)
        selected_experts_indices = torch.tensor([[0], [1], [0], [1]]).to(x.dtype)
        top_scores = torch.ones(4, 1)

        routed_input, _, metadata = dispatcher.dispatch(
            x,
            top_scores,
            selected_experts_indices,
        )
        shared_out = x * 0.25
        shared_experts = unittest.mock.Mock(return_value=shared_out)
        with torch.no_grad():
            out = dispatcher.combine(
                routed_input,
                metadata,
                x,
                shared_experts=shared_experts,
            )

        self.assertEqual(
            out.untyped_storage().data_ptr(), shared_out.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(out, x * 1.25)


class TestMoEExpertsFastPaths(unittest.TestCase):
    def test_batched_no_grad_experts_matches_loop_reference(self):
        torch.manual_seed(123)
        num_experts = 3
        tokens_per_expert = 4
        dim = 5
        hidden_dim = 7
        x = torch.randn(num_experts * tokens_per_expert, dim)
        w1 = torch.randn(num_experts, hidden_dim, dim)
        w2 = torch.randn(num_experts, dim, hidden_dim)
        w3 = torch.randn(num_experts, hidden_dim, dim)
        counts = [tokens_per_expert] * num_experts

        reference = _run_experts_for_loop(w1, w2, w3, x, counts)
        with torch.no_grad():
            actual = _run_experts_for_loop(w1, w2, w3, x, counts)

        torch.testing.assert_close(actual, reference)

    def test_no_grad_expert_weight_caches_hit_and_invalidate(self):
        previous = os.environ.get("TT_MOE_DEBUG_FASTPATHS")
        os.environ["TT_MOE_DEBUG_FASTPATHS"] = "1"
        _MOE_FASTPATH_COUNTERS.clear()
        try:
            experts = GroupedExperts(
                GroupedExperts.Config(
                    dim=5,
                    hidden_dim=7,
                    num_experts=3,
                    use_grouped_mm=False,
                    token_dispatcher=LocalTokenDispatcher.Config(
                        num_experts=3,
                        top_k=1,
                        score_before_experts=False,
                    ),
                )
            )
            with torch.no_grad():
                experts.w1.copy_(torch.randn_like(experts.w1))
                experts.w2.copy_(torch.randn_like(experts.w2))
                experts.w3.copy_(torch.randn_like(experts.w3))

            x = torch.randn(12, 5)
            counts = [4, 4, 4]
            with torch.no_grad():
                experts._experts_forward(x, counts)
                experts._experts_forward(x, counts)

                self.assertEqual(_MOE_FASTPATH_COUNTERS["cached_w13_miss"], 1)
                self.assertEqual(_MOE_FASTPATH_COUNTERS["cached_w13_hit"], 1)
                self.assertEqual(_MOE_FASTPATH_COUNTERS["cached_w2_t_miss"], 1)
                self.assertEqual(_MOE_FASTPATH_COUNTERS["cached_w2_t_hit"], 1)

                experts.w1.add_(1.0)
                experts._experts_forward(x, counts)

            self.assertEqual(_MOE_FASTPATH_COUNTERS["cached_w13_miss"], 2)
            self.assertEqual(_MOE_FASTPATH_COUNTERS["cached_w2_t_hit"], 2)
        finally:
            _MOE_FASTPATH_COUNTERS.clear()
            if previous is None:
                os.environ.pop("TT_MOE_DEBUG_FASTPATHS", None)
            else:
                os.environ["TT_MOE_DEBUG_FASTPATHS"] = previous


class TestForceLoadBalanceRouting(unittest.TestCase):
    def test_round_robin_counts_match_selected_experts(self):
        router = TokenChoiceTopKRouter(
            TokenChoiceTopKRouter.Config(
                num_experts=4,
                gate=Linear.Config(in_features=3, out_features=4, bias=False),
                top_k=3,
                _debug_force_load_balance=True,
            )
        )
        scores = torch.randn(5, 4)

        (
            selected_experts_indices,
            top_scores,
            num_tokens_per_expert,
        ) = router._debug_force_load_balance_routing(scores)

        expected_indices = torch.arange(15).reshape(5, 3) % 4
        expected_counts = torch.bincount(expected_indices.reshape(-1), minlength=4)
        torch.testing.assert_close(selected_experts_indices, expected_indices)
        torch.testing.assert_close(top_scores, scores.gather(1, expected_indices))
        torch.testing.assert_close(num_tokens_per_expert, expected_counts)


class TestForceLoadBalanceSplits(unittest.TestCase):
    def test_pad_and_compact_equal_splits_round_trip(self):
        x = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        splits = [2, 1, 2]

        padded = AllToAllTokenDispatcher._pad_to_equal_splits(
            x,
            splits,
            equal_split_size=2,
        )
        self.assertEqual(tuple(padded.shape), (6, 2))
        torch.testing.assert_close(padded[0:2], x[0:2])
        torch.testing.assert_close(padded[2], x[2])
        torch.testing.assert_close(padded[3], torch.zeros(2))
        torch.testing.assert_close(padded[4:6], x[3:5])

        compacted = AllToAllTokenDispatcher._compact_equal_splits(
            padded,
            splits,
            equal_split_size=2,
        )
        torch.testing.assert_close(compacted, x)

    def test_direct_equal_split_routed_input_matches_sorted_then_padded(self):
        dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(
                num_experts=4,
                top_k=2,
                score_before_experts=False,
                force_load_balance=True,
            )
        )
        x = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        selected_experts_indices = (
            torch.arange(x.shape[0] * dispatcher.top_k).reshape(x.shape[0], 2)
            % dispatcher.num_experts
        )
        _, token_indices = dispatcher._force_load_balance_sort_indices(
            selected_experts_indices
        )
        sorted_routed = x[token_indices]
        expected = dispatcher._pad_to_equal_splits(
            sorted_routed,
            splits=[6, 4],
            equal_split_size=6,
        )

        actual = dispatcher._force_load_balance_equal_split_routed_input(
            x,
            ep_size=2,
            num_local_experts=2,
            input_splits=[6, 4],
            equal_split_size=6,
        )

        torch.testing.assert_close(actual, expected)

    def test_direct_equal_split_routed_input_rejects_fractional_local_counts(self):
        x = torch.arange(10, dtype=torch.float32).reshape(5, 2)

        with self.assertRaises(AssertionError):
            AllToAllTokenDispatcher._force_load_balance_equal_split_routed_input(
                x,
                ep_size=2,
                num_local_experts=2,
                input_splits=[5, 5],
                equal_split_size=6,
            )

    def test_sort_indices_cache_matches_stable_argsort(self):
        dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(
                num_experts=4,
                top_k=3,
                score_before_experts=False,
                force_load_balance=True,
            )
        )
        selected_experts_indices = (
            torch.arange(15).reshape(5, 3) % dispatcher.num_experts
        )

        assignment_indices, token_indices = dispatcher._force_load_balance_sort_indices(
            selected_experts_indices
        )
        expected_assignment_indices = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        torch.testing.assert_close(assignment_indices, expected_assignment_indices)
        torch.testing.assert_close(
            token_indices, expected_assignment_indices // dispatcher.top_k
        )
        self.assertEqual(len(dispatcher._force_load_balance_sort_cache), 1)

    def test_splits_match_round_robin_layout(self):
        dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(
                num_experts=36,
                top_k=3,
                score_before_experts=False,
                force_load_balance=True,
            )
        )

        (
            counts,
            input_splits,
            output_splits,
            uniform_local_count,
        ) = dispatcher._force_load_balance_splits(
            num_tokens=16_384,
            ep_size=12,
            ep_rank=4,
            dtype=torch.int64,
            device=torch.device("cpu"),
        )

        self.assertEqual(input_splits, [4098, 4098, 4098, 4098] + [4095] * 8)
        self.assertEqual(output_splits, [4095] * 12)
        self.assertEqual(uniform_local_count, 1365)
        torch.testing.assert_close(
            counts,
            torch.tensor([1365, 1365, 1365] * 12),
        )

    def test_splits_detect_non_uniform_local_counts(self):
        dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(
                num_experts=4,
                top_k=3,
                score_before_experts=False,
                force_load_balance=True,
            )
        )

        _, _, _, uniform_local_count = dispatcher._force_load_balance_splits(
            num_tokens=5,
            ep_size=2,
            ep_rank=1,
            dtype=torch.int64,
            device=torch.device("cpu"),
        )

        self.assertIsNone(uniform_local_count)


class TestPermute(unittest.TestCase):
    """Test AllToAllTokenDispatcher._permute which reorders tokens from rank-major to expert-major layout.

    Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
    Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
    """

    def _make_dispatcher(self) -> AllToAllTokenDispatcher:
        """Create a minimal AllToAllTokenDispatcher for testing _permute."""
        cfg = AllToAllTokenDispatcher.Config(
            num_experts=1, top_k=1, score_before_experts=True
        )
        return AllToAllTokenDispatcher(cfg)

    def _permute(self, tokens_per_expert_group, experts_per_rank, num_ranks):
        """Helper that calls _permute and returns (permuted_indices, num_tokens_per_expert)."""
        dispatcher = self._make_dispatcher()
        total = tokens_per_expert_group.sum().item()
        dummy_input = torch.zeros(total, 1)
        _, _, permuted_indices, num_tokens_per_expert = dispatcher._permute(
            dummy_input, tokens_per_expert_group, num_ranks, experts_per_rank
        )
        return permuted_indices, num_tokens_per_expert

    def test_basic_2ranks_2experts(self):
        # 2 ranks, 2 experts per rank
        # tokens_per_expert_group: [r0e0, r0e1, r1e0, r1e1] = [2, 3, 1, 4]
        tokens_per_expert_group = torch.tensor([2, 3, 1, 4])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        # Expert-major layout: e0r0(2), e0r1(1), e1r0(3), e1r1(4)
        # Input positions:
        #   r0e0: [0, 1], r0e1: [2, 3, 4], r1e0: [5], r1e1: [6, 7, 8, 9]
        # Output order:
        #   e0r0: [0, 1], e0r1: [5], e1r0: [2, 3, 4], e1r1: [6, 7, 8, 9]
        expected_indices = torch.tensor([0, 1, 5, 2, 3, 4, 6, 7, 8, 9])
        torch.testing.assert_close(permuted_indices, expected_indices)

        # num_tokens_per_expert: sum across ranks for each expert
        # e0: r0e0 + r1e0 = 2 + 1 = 3, e1: r0e1 + r1e1 = 3 + 4 = 7
        expected_num_tokens = torch.tensor([3, 7])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_single_rank(self):
        # 1 rank, 3 experts: no reordering needed
        tokens_per_expert_group = torch.tensor([4, 2, 5])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=3, num_ranks=1
        )

        expected_indices = torch.arange(11)
        torch.testing.assert_close(permuted_indices, expected_indices)
        torch.testing.assert_close(num_tokens_per_expert, tokens_per_expert_group)

    def test_single_expert(self):
        # 3 ranks, 1 expert per rank: no reordering needed
        tokens_per_expert_group = torch.tensor([3, 5, 2])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=1, num_ranks=3
        )

        expected_indices = torch.arange(10)
        torch.testing.assert_close(permuted_indices, expected_indices)

        # Single expert gets all tokens
        expected_num_tokens = torch.tensor([10])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_equal_count_fast_path_matches_index_permute_and_unpermute(self):
        dispatcher = self._make_dispatcher()
        ep_size = 3
        num_local_experts = 2
        uniform_count = 4
        dim = 5
        counts = torch.full((ep_size * num_local_experts,), uniform_count)
        routed_input = torch.arange(
            ep_size * num_local_experts * uniform_count * dim,
            dtype=torch.float32,
        ).reshape(-1, dim)

        (
            expected_shape,
            expected_routed_input,
            _,
            expected_counts,
        ) = dispatcher._permute(
            routed_input,
            counts,
            ep_size,
            num_local_experts,
        )
        (
            input_shape,
            actual_routed_input,
            actual_counts,
            actual_counts_list,
            rank_major_shape,
        ) = dispatcher._permute_equal_counts(
            routed_input,
            counts,
            ep_size,
            num_local_experts,
            uniform_count,
        )

        self.assertEqual(input_shape, expected_shape)
        self.assertEqual(rank_major_shape, (ep_size, num_local_experts, uniform_count))
        self.assertEqual(
            actual_counts_list, [ep_size * uniform_count] * num_local_experts
        )
        torch.testing.assert_close(actual_routed_input, expected_routed_input)
        torch.testing.assert_close(actual_counts, expected_counts)
        torch.testing.assert_close(
            dispatcher._unpermute(
                actual_routed_input,
                input_shape,
                None,
                rank_major_shape,
            ),
            routed_input,
        )

    def test_torchao_unpermute_accepts_alltoall_combine_signature(self):
        dispatcher = TorchAOTokenDispatcher(
            TorchAOTokenDispatcher.Config(
                num_experts=1,
                top_k=1,
                score_before_experts=True,
                pad_multiple=4,
            )
        )
        routed_output = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        # TorchAO permute_and_pad appends a sentinel row to input_shape. The
        # override strips it after scattering through permuted_indices.
        input_shape = (4, 2)
        permuted_indices = torch.tensor([2, 0, 1])

        actual = dispatcher._unpermute(
            routed_output,
            input_shape,
            permuted_indices,
            None,
        )

        expected = torch.tensor(
            [
                [3.0, 4.0],
                [5.0, 6.0],
                [1.0, 2.0],
            ]
        )
        torch.testing.assert_close(actual, expected)

    def test_zero_tokens_for_some_experts(self):
        # 2 ranks, 2 experts, some with zero tokens
        # [r0e0, r0e1, r1e0, r1e1] = [0, 3, 2, 0]
        tokens_per_expert_group = torch.tensor([0, 3, 2, 0])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        # Expert-major: e0r0(0), e0r1(2), e1r0(3), e1r1(0)
        # Input positions: r0e0: [], r0e1: [0, 1, 2], r1e0: [3, 4], r1e1: []
        # Output order: e0r0: [], e0r1: [3, 4], e1r0: [0, 1, 2], e1r1: []
        expected_indices = torch.tensor([3, 4, 0, 1, 2])
        torch.testing.assert_close(permuted_indices, expected_indices)

        expected_num_tokens = torch.tensor([2, 3])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_all_zero_tokens(self):
        tokens_per_expert_group = torch.tensor([0, 0, 0, 0])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        self.assertEqual(permuted_indices.numel(), 0)
        expected_num_tokens = torch.tensor([0, 0])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_uniform_distribution(self):
        # 3 ranks, 2 experts, uniform token counts
        # [r0e0, r0e1, r1e0, r1e1, r2e0, r2e1] = [2, 2, 2, 2, 2, 2]
        tokens_per_expert_group = torch.tensor([2, 2, 2, 2, 2, 2])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=3
        )

        # Expert-major: e0r0(2), e0r1(2), e0r2(2), e1r0(2), e1r1(2), e1r2(2)
        # Input positions:
        #   r0e0: [0,1], r0e1: [2,3], r1e0: [4,5], r1e1: [6,7], r2e0: [8,9], r2e1: [10,11]
        # Output: e0r0[0,1], e0r1[4,5], e0r2[8,9], e1r0[2,3], e1r1[6,7], e1r2[10,11]
        expected_indices = torch.tensor([0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11])
        torch.testing.assert_close(permuted_indices, expected_indices)

        expected_num_tokens = torch.tensor([6, 6])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_permutation_is_valid(self):
        # The output should be a permutation of [0, total)
        tokens_per_expert_group = torch.tensor([3, 1, 4, 1, 5, 9])
        permuted_indices, _ = self._permute(
            tokens_per_expert_group, experts_per_rank=3, num_ranks=2
        )

        total = tokens_per_expert_group.sum().item()
        self.assertEqual(permuted_indices.numel(), total)
        self.assertEqual(
            set(permuted_indices.tolist()),
            set(range(total)),
        )


class TestSPPaddingRoundTrip(unittest.TestCase):
    """Verify AllToAllTokenDispatcher's SP padding/unpadding recovers the
    original tokens for an uneven ``bs * slen`` (not divisible by ``sp_size``).

    See docs/moe_sp_padding.md. Like TestPermute above, this runs on CPU by
    only exercising the pure-tensor helper ``_split_along_sp`` rather than
    going through dispatch()/combine() (which need CUDA + a real EP mesh).
    """

    def test_round_trip_recovers_original(self):
        original_num_tokens = 7  # not divisible by sp_size=4
        sp_size = 4
        dim = 3

        cfg = AllToAllTokenDispatcher.Config(
            num_experts=4, top_k=1, score_before_experts=True
        )
        dispatcher = AllToAllTokenDispatcher(cfg)
        dispatcher.sp_size = sp_size

        x = torch.randn(original_num_tokens, dim)

        # Pad (matches what dispatch() does on entry).
        pad = (-original_num_tokens) % sp_size
        x_padded = F.pad(x, (0, 0, 0, pad))
        self.assertEqual(x_padded.shape, (8, dim))

        # Split per rank, then reassemble (this is what the EP all-to-all
        # gathers back in real distributed training).
        local_num_tokens = x_padded.shape[0] // sp_size
        per_rank_slices = []
        for rank in range(sp_size):
            dispatcher.sp_rank = rank
            (slice_,) = dispatcher._split_along_sp(x_padded)
            torch.testing.assert_close(
                slice_,
                x_padded[rank * local_num_tokens : (rank + 1) * local_num_tokens],
            )
            per_rank_slices.append(slice_)
        reassembled = torch.cat(per_rank_slices, dim=0)

        # Unpad to recover original prefix bitwise.
        torch.testing.assert_close(reassembled[:original_num_tokens], x)


if __name__ == "__main__":
    unittest.main()
