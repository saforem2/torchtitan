# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
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
    EzpzGroupedExperts,
)

from torchtitan.models.common.moe import GroupedExperts


def _clone_for_grad(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone().detach().requires_grad_(True)


def _init_grouped_experts_weights(module: GroupedExperts) -> None:
    """Randomize the expert weights in place.

    Names are the post-#3425 shape-suffixed ones (w1_EFD / w2_EDF / w3_EFD),
    verified against a constructed module. The sww original used the old bare
    w1/w2/w3 and raises AttributeError here.
    """
    with torch.no_grad():
        module.w1_EFD.copy_(torch.randn_like(module.w1_EFD))
        module.w2_EDF.copy_(torch.randn_like(module.w2_EDF))
        module.w3_EFD.copy_(torch.randn_like(module.w3_EFD))


class TestMoEExpertBackends(unittest.TestCase):
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

        inputs = dict(
            w1=torch.empty(2, 3, 4, dtype=torch.bfloat16),
            w2=torch.empty(2, 4, 3, dtype=torch.bfloat16),
            w3=torch.empty(2, 3, 4, dtype=torch.bfloat16),
            x=torch.empty(4, 4, dtype=torch.bfloat16),
            topk_scores=torch.ones(4, 1),
            topk_indices=torch.zeros(4, 1, dtype=torch.int64),
        )
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

        inputs = dict(
            w1=torch.empty(2, 3, 4, dtype=torch.bfloat16),
            w2=torch.empty(2, 4, 3, dtype=torch.bfloat16),
            w3=torch.empty(2, 3, 4, dtype=torch.bfloat16),
            x=torch.empty(4, 4, dtype=torch.bfloat16),
            num_tokens_per_expert=torch.ones(4, dtype=torch.int64),
            ep_mesh=FakeMesh(),
        )
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
            return EzpzGroupedExperts(
                EzpzGroupedExperts.Config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    num_experts=num_experts,
                    compute_backend=backend,
                )
            )

        ref = _build("for_loop")
        _init_grouped_experts_weights(ref)

        num_tokens_per_expert = torch.tensor([3, 0, 2, 1, 4], dtype=torch.int64)
        total_tokens = int(num_tokens_per_expert.sum().item())
        x = torch.randn(total_tokens, dim, dtype=torch.bfloat16)
        ref_out = ref.forward(x, num_tokens_per_expert)

        # Every backend that does not need an absent external package.
        # aurora_sycl is excluded here: it requires aurora_moe and its
        # JIT-compiled SYCL kernels, which is an XPU-only hardware test.
        for backend in ("bmm_nodrop", "bmm"):
            test = _build(backend)
            with torch.no_grad():
                test.w1_EFD.copy_(ref.w1_EFD)
                test.w2_EDF.copy_(ref.w2_EDF)
                test.w3_EFD.copy_(ref.w3_EFD)
            out = test.forward(x, num_tokens_per_expert)
            # `bmm` pads to a capacity_factor-derived capacity and DROPS
            # overflow tokens, so it only matches when nothing overflows.
            # With counts [3,0,2,1,4] over 5 experts, cap = ceil(10/5*1.25)
            # = 3, so the 4-token expert loses one row -- expected, and the
            # exact reason bmm_nodrop was ported alongside it.
            if backend == "bmm":
                continue
            assert_close(out.float(), ref_out.float(), rtol=0.05, atol=0.05)
