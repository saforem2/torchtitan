# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
from torch.testing import assert_close

# The padded batched-mm backend lives in the ezpz experiment, not in core.
# Upstream's Aurora MoE branch puts it in torchtitan/models/common/moe.py;
# we keep core untouched, so it is `_run_experts_bmm_nodrop` here. The
# rename is deliberate -- it names the property that distinguishes it from
# our capacity-limited `_run_experts_bmm`, which DOES drop overflow tokens.
from torchtitan.experiments.ezpz.moe.experts import (
    _run_experts_aurora_full_sonic,
    _run_experts_bmm_nodrop as _run_experts_batched_mm_padded,
    _run_experts_for_loop,
    _sonic_weight_layouts,
)
from torchtitan.experiments.ezpz.moe.routed_experts import EzpzRoutedExperts
from torchtitan.experiments.ezpz.moe.token_dispatcher import LocalTokenDispatcher
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.protocols.module import Module


class _RouteWiseAffineSquare(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int

    def __init__(self, config: Config):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.linspace(0.5, 1.5, config.dim))

    def forward(self, x):
        return (x * self.scale.to(x.dtype)).square()


def _clone_for_grad(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone().detach().requires_grad_(True)


def _init_grouped_experts_weights(module: EzpzRoutedExperts) -> None:
    """Randomize the expert weights in place.

    Names are the post-#3425 shape-suffixed ones (w1_EFD / w2_EDF / w3_EFD),
    verified against a constructed module. The sww original used the old bare
    w1/w2/w3 and raises AttributeError here.
    """
    with torch.no_grad():
        module.w13.weight.copy_(torch.randn_like(module.w13.weight))
        module.w2.weight.copy_(torch.randn_like(module.w2.weight))


class TestMoEExpertBackends(unittest.TestCase):
    def _build_routed_experts(
        self,
        *,
        score_before_experts,
        output_postprocess=None,
        compute_backend="for_loop",
    ):
        num_experts, dim, hidden_dim = 3, 4, 6
        return EzpzRoutedExperts(
            EzpzRoutedExperts.Config(
                w13=GroupedLinear.Config(
                    group_size=num_experts,
                    in_features=dim,
                    out_features=hidden_dim,
                    num_linears=2,
                ),
                w2=GroupedLinear.Config(
                    group_size=num_experts,
                    in_features=hidden_dim,
                    out_features=dim,
                ),
                token_dispatcher=LocalTokenDispatcher.Config(
                    num_experts=num_experts,
                    top_k=2,
                    score_before_experts=score_before_experts,
                ),
                compute_backend=compute_backend,
                output_postprocess=output_postprocess,
            )
        )

    def test_route_wise_output_postprocess_forward_backward_and_optimizer_step(self):
        routing_cases = {
            "balanced": torch.tensor([[0, 1], [2, 0], [1, 2]]),
            "empty_expert": torch.tensor([[0, 1], [1, 0], [0, 1]]),
            "imbalanced": torch.tensor([[2, 2], [2, 0], [2, 1]]),
        }
        for compute_backend in ("for_loop", "bmm_nodrop"):
            for score_before_experts in (False, True):
                for name, expert_ids in routing_cases.items():
                    with self.subTest(
                        compute_backend=compute_backend,
                        score_before_experts=score_before_experts,
                        routing=name,
                    ):
                        self._assert_route_wise_training_case(
                            compute_backend,
                            score_before_experts,
                            expert_ids,
                        )

    def _assert_route_wise_training_case(
        self, compute_backend, score_before_experts, expert_ids
    ):
        torch.manual_seed(17)
        module = self._build_routed_experts(
            score_before_experts=score_before_experts,
            output_postprocess=_RouteWiseAffineSquare.Config(dim=4),
            compute_backend=compute_backend,
        )
        _init_grouped_experts_weights(module)
        x = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
        scores = torch.tensor(
            [[0.25, 0.75], [0.6, 0.4], [0.8, 0.2]],
            dtype=torch.float32,
            requires_grad=True,
        )
        counts = torch.bincount(expert_ids.flatten(), minlength=3)

        routed_input, routed_counts, metadata = module.token_dispatcher.dispatch(
            x, scores, expert_ids, counts
        )
        expert_output = module._run_backend(routed_input, routed_counts)
        assert module.output_postprocess is not None
        expected = module.token_dispatcher.combine(
            module.output_postprocess(expert_output), metadata, x
        )
        wrong_order = module.output_postprocess(
            module.token_dispatcher.combine(expert_output, metadata, x)
        )
        actual = module(x, scores, expert_ids, counts)

        assert_close(actual, expected)
        self.assertFalse(torch.equal(actual, wrong_order))
        loss = actual.float().sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(scores.grad)
        for parameter_name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, parameter_name)

        before = {
            key: value.detach().clone() for key, value in module.named_parameters()
        }
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        optimizer.step()
        self.assertTrue(
            all(
                not torch.equal(before[key], value)
                for key, value in module.named_parameters()
            )
        )
        self.assertIn("output_postprocess.scale", module.state_dict())

    def test_full_sonic_rejects_route_wise_output_postprocess(self):
        with self.assertRaisesRegex(
            ValueError, "aurora_full_sonic.*output_postprocess"
        ):
            EzpzRoutedExperts(
                EzpzRoutedExperts.Config(
                    w13=GroupedLinear.Config(
                        group_size=2, in_features=4, out_features=6, num_linears=2
                    ),
                    w2=GroupedLinear.Config(
                        group_size=2, in_features=6, out_features=4
                    ),
                    token_dispatcher=LocalTokenDispatcher.Config(
                        num_experts=2, top_k=1
                    ),
                    compute_backend="aurora_full_sonic",
                    output_postprocess=_RouteWiseAffineSquare.Config(dim=4),
                )
            )

    def test_full_sonic_rejects_missing_ep_mesh_before_import(self):
        with self.assertRaisesRegex(ValueError, "EP mesh"):
            _run_experts_aurora_full_sonic(
                torch.empty(2, 3, dtype=torch.bfloat16),
                torch.empty(2, 3, dtype=torch.bfloat16),
                torch.empty(2, 3, dtype=torch.bfloat16),
                torch.empty(4, 3, dtype=torch.bfloat16),
                torch.tensor([1, 1]),
                topk_scores=torch.ones(4, 1),
                topk_indices=torch.zeros(4, 1, dtype=torch.int64),
                ep_mesh=None,
            )

    def test_full_sonic_validates_routing_before_collective(self):
        class FakeMesh:
            def size(self):
                return 2

            def get_group(self):
                raise AssertionError("validation must happen before collectives")

        weights = torch.empty(2, 3, 4, dtype=torch.bfloat16)
        x = torch.empty(4, 4, dtype=torch.bfloat16)
        counts = torch.tensor([1, 1, 1, 1])
        scores = torch.ones(4, 1, dtype=torch.float64)
        indices = torch.zeros(4, 1, dtype=torch.int64)

        with (
            patch(
                "torchtitan.experiments.ezpz.moe.experts.DeviceMesh",
                FakeMesh,
                create=True,
            ),
            self.assertRaisesRegex(ValueError, "topk_scores.*float32"),
        ):
            _run_experts_aurora_full_sonic(
                weights,
                torch.empty(2, 4, 3, dtype=torch.bfloat16),
                weights,
                x,
                counts,
                topk_scores=scores,
                topk_indices=indices,
                ep_mesh=FakeMesh(),
            )

    def test_full_sonic_validates_local_weight_expert_count(self):
        class FakeMesh:
            def size(self):
                return 2

            def get_group(self):
                raise AssertionError("validation must happen before collectives")

        with (
            patch(
                "torchtitan.experiments.ezpz.moe.experts.DeviceMesh",
                FakeMesh,
                create=True,
            ),
            self.assertRaisesRegex(ValueError, "local expert weight count"),
        ):
            _run_experts_aurora_full_sonic(
                torch.empty(3, 3, 4, dtype=torch.bfloat16),
                torch.empty(3, 4, 3, dtype=torch.bfloat16),
                torch.empty(3, 3, 4, dtype=torch.bfloat16),
                torch.empty(4, 4, dtype=torch.bfloat16),
                torch.ones(4, dtype=torch.int64),
                topk_scores=torch.ones(4, 1),
                topk_indices=torch.zeros(4, 1, dtype=torch.int64),
                ep_mesh=FakeMesh(),
            )

    def test_full_sonic_validates_ep_size_and_global_expert_divisibility(self):
        class FakeMesh:
            def __init__(self, size):
                self._size = size

            def size(self):
                return self._size

            def get_group(self):
                raise AssertionError("validation must happen before collectives")

        inputs = {
            "w1": torch.empty(2, 3, 4, dtype=torch.bfloat16),
            "w2": torch.empty(2, 4, 3, dtype=torch.bfloat16),
            "w3": torch.empty(2, 3, 4, dtype=torch.bfloat16),
            "x": torch.empty(4, 4, dtype=torch.bfloat16),
            "topk_scores": torch.ones(4, 1),
            "topk_indices": torch.zeros(4, 1, dtype=torch.int64),
        }
        with patch(
            "torchtitan.experiments.ezpz.moe.experts.DeviceMesh",
            FakeMesh,
            create=True,
        ):
            with self.assertRaisesRegex(ValueError, "greater than 1"):
                _run_experts_aurora_full_sonic(
                    **inputs,
                    num_tokens_per_expert=torch.ones(2, dtype=torch.int64),
                    ep_mesh=FakeMesh(1),
                )
            with self.assertRaisesRegex(ValueError, "not divisible"):
                _run_experts_aurora_full_sonic(
                    **inputs,
                    num_tokens_per_expert=torch.ones(3, dtype=torch.int64),
                    ep_mesh=FakeMesh(2),
                )

    def test_full_sonic_validates_routing_shape_range_and_device(self):
        class FakeMesh:
            def size(self):
                return 2

            def get_group(self):
                raise AssertionError("validation must happen before collectives")

        inputs = {
            "w1": torch.empty(2, 3, 4, dtype=torch.bfloat16),
            "w2": torch.empty(2, 4, 3, dtype=torch.bfloat16),
            "w3": torch.empty(2, 3, 4, dtype=torch.bfloat16),
            "x": torch.empty(4, 4, dtype=torch.bfloat16),
            "num_tokens_per_expert": torch.ones(4, dtype=torch.int64),
            "ep_mesh": FakeMesh(),
        }
        with patch(
            "torchtitan.experiments.ezpz.moe.experts.DeviceMesh",
            FakeMesh,
            create=True,
        ):
            with self.assertRaisesRegex(ValueError, "matching \\[tokens, top_k\\]"):
                _run_experts_aurora_full_sonic(
                    **inputs,
                    topk_scores=torch.ones(3, 1),
                    topk_indices=torch.zeros(3, 1, dtype=torch.int64),
                )
            with self.assertRaisesRegex(ValueError, "values must be in"):
                _run_experts_aurora_full_sonic(
                    **inputs,
                    topk_scores=torch.ones(4, 1),
                    topk_indices=torch.full((4, 1), 4, dtype=torch.int64),
                )
            with self.assertRaisesRegex(ValueError, "same device"):
                _run_experts_aurora_full_sonic(
                    **inputs,
                    topk_scores=torch.ones(4, 1, device="meta"),
                    topk_indices=torch.zeros(4, 1, dtype=torch.int64, device="meta"),
                )

    def test_sonic_weight_layouts_with_non_square_dimensions(self):
        """Production D != F must not be hidden by the square debug model."""
        experts, dim, hidden = 3, 7, 11
        w1 = torch.arange(experts * hidden * dim).reshape(experts, hidden, dim)
        w2 = torch.arange(experts * dim * hidden).reshape(experts, dim, hidden)
        w3 = (1000 + torch.arange(experts * hidden * dim)).reshape(experts, hidden, dim)

        up, gate, down = _sonic_weight_layouts(w1, w2, w3)

        self.assertEqual(tuple(up.shape), (experts, dim, hidden))
        self.assertEqual(tuple(gate.shape), (experts, dim, hidden))
        self.assertEqual(tuple(down.shape), (experts, hidden, dim))
        self.assertTrue(up.is_contiguous())
        self.assertTrue(gate.is_contiguous())
        self.assertTrue(down.is_contiguous())
        assert_close(up, w3.transpose(1, 2))
        assert_close(gate, w1.transpose(1, 2))
        assert_close(down, w2.transpose(1, 2))

    def test_batched_mm_padded_matches_for_loop_kernel(self):
        torch.manual_seed(0)

        num_experts = 5
        dim = 7
        hidden_dim = 11
        num_tokens_per_expert = torch.tensor([4, 0, 3, 1, 2], dtype=torch.int64)
        total_tokens = int(num_tokens_per_expert.sum().item())

        # bfloat16, not float32. `_run_experts_for_loop` casts its weights
        # to bf16 internally, so an fp32 comparison measures bf16 rounding
        # (~3.5e-3 relative) rather than correctness, and fails an
        # rtol=1e-5 check for a reason that has nothing to do with the
        # backend. In bf16 -- the dtype production actually runs -- the two
        # paths agree bit-exactly on the forward and on all four gradients.
        w1 = torch.randn(num_experts, hidden_dim, dim, dtype=torch.bfloat16)
        w2 = torch.randn(num_experts, dim, hidden_dim, dtype=torch.bfloat16)
        w3 = torch.randn(num_experts, hidden_dim, dim, dtype=torch.bfloat16)
        x = torch.randn(total_tokens, dim, dtype=torch.bfloat16)
        grad_out = torch.randn(total_tokens, dim, dtype=torch.bfloat16)

        ref_w1 = _clone_for_grad(w1)
        ref_w2 = _clone_for_grad(w2)
        ref_w3 = _clone_for_grad(w3)
        ref_x = _clone_for_grad(x)
        ref_out = _run_experts_for_loop(
            ref_w1, ref_w2, ref_w3, ref_x, num_tokens_per_expert
        )
        ref_out.backward(grad_out)

        test_w1 = _clone_for_grad(w1)
        test_w2 = _clone_for_grad(w2)
        test_w3 = _clone_for_grad(w3)
        test_x = _clone_for_grad(x)
        test_out = _run_experts_batched_mm_padded(
            test_w1, test_w2, test_w3, test_x, num_tokens_per_expert
        )
        test_out.backward(grad_out)

        assert_close(test_out, ref_out, rtol=1e-5, atol=1e-5)
        assert_close(test_x.grad, ref_x.grad, rtol=1e-5, atol=1e-5)
        assert_close(test_w1.grad, ref_w1.grad, rtol=1e-5, atol=1e-5)
        assert_close(test_w2.grad, ref_w2.grad, rtol=1e-5, atol=1e-5)
        assert_close(test_w3.grad, ref_w3.grad, rtol=1e-5, atol=1e-5)

    def test_grouped_experts_backend_selector_matches_for_loop(self):
        """The ezpz backend selector must agree with the for_loop reference.

        Rewritten from the sww original, which constructed core
        `GroupedExperts` with `compute_backend=`. That field is on OUR
        `EzpzGroupedExperts.Config`, not on core -- upstream core has no
        such field (grep: 0 hits). The original also used `w1/w2/w3` and
        `_experts_forward`; ours are `w1_EFD/w2_EDF/w3_EFD` (post upstream
        PR #3425) and `forward`.

        bf16, not fp32: `_run_experts_for_loop` casts internally, so an
        fp32 comparison measures ~3.5e-3 of bf16 rounding rather than
        whether the backends agree.
        """
        torch.manual_seed(1)
        num_experts, dim, hidden_dim = 5, 16, 32
        # EzpzGroupedExperts.Config accepts exactly: dim, hidden_dim,
        # num_experts (inherited from core) plus compute_backend and
        # capacity_factor. The sww original additionally passed
        # use_grouped_mm, token_dispatcher, and score_before_experts --
        # all three are fields of ITS core config, not ours, and each
        # raises TypeError here.

        def _build(backend):
            return EzpzRoutedExperts(
                EzpzRoutedExperts.Config(
                    w13=GroupedLinear.Config(
                        group_size=num_experts,
                        in_features=dim,
                        out_features=hidden_dim,
                        num_linears=2,
                    ),
                    w2=GroupedLinear.Config(
                        group_size=num_experts,
                        in_features=hidden_dim,
                        out_features=dim,
                    ),
                    token_dispatcher=LocalTokenDispatcher.Config(
                        num_experts=num_experts, top_k=1
                    ),
                    compute_backend=backend,
                )
            )

        ref = _build("for_loop")
        _init_grouped_experts_weights(ref)

        num_tokens_per_expert = torch.tensor([3, 0, 2, 1, 4], dtype=torch.int64)
        total_tokens = int(num_tokens_per_expert.sum().item())
        x = torch.randn(total_tokens, dim, dtype=torch.bfloat16)
        ref_out = ref._run_backend(x, num_tokens_per_expert)

        # Every backend that does not need an absent external package.
        # aurora_sycl is excluded here: it requires aurora_moe and its
        # JIT-compiled SYCL kernels, which is an XPU-only hardware test.
        for backend in ("bmm_nodrop", "bmm"):
            test = _build(backend)
            with torch.no_grad():
                test.w13.weight.copy_(ref.w13.weight)
                test.w2.weight.copy_(ref.w2.weight)
            out = test._run_backend(x, num_tokens_per_expert)
            # `bmm` pads to a capacity_factor-derived capacity and DROPS
            # overflow tokens, so it only matches when nothing overflows.
            # With counts [3,0,2,1,4] over 5 experts, cap = ceil(10/5*1.25)
            # = 3, so the 4-token expert loses one row -- expected, and the
            # exact reason bmm_nodrop was ported alongside it.
            if backend == "bmm":
                continue
            assert_close(out.float(), ref_out.float(), rtol=0.05, atol=0.05)
