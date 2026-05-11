# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing import assert_close

from torchtitan.models.common.moe import (
    GroupedExperts,
    _run_experts_batched_mm_padded,
    _run_experts_for_loop,
)
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher


def _init_weights(
    num_experts: int,
    dim: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w1 = torch.randn(num_experts, hidden_dim, dim, dtype=dtype) * 0.02
    w2 = torch.randn(num_experts, dim, hidden_dim, dtype=dtype) * 0.02
    w3 = torch.randn(num_experts, hidden_dim, dim, dtype=dtype) * 0.02
    return w1, w2, w3


class MoEExpertBackendsTest(unittest.TestCase):
    def test_batched_mm_padded_matches_for_loop_kernel(self) -> None:
        torch.manual_seed(1234)
        num_experts = 6
        dim = 16
        hidden_dim = 24
        counts = torch.tensor([0, 3, 5, 1, 0, 7], dtype=torch.int64)
        total_tokens = int(counts.sum().item())

        w1, w2, w3 = _init_weights(num_experts, dim, hidden_dim)
        x = torch.randn(total_tokens, dim)

        ref_inputs = [t.detach().clone().requires_grad_(True) for t in (w1, w2, w3, x)]
        test_inputs = [
            t.detach().clone().requires_grad_(True) for t in (w1, w2, w3, x)
        ]

        ref_out = _run_experts_for_loop(*ref_inputs[:3], ref_inputs[3], counts)
        test_out = _run_experts_batched_mm_padded(
            *test_inputs[:3], test_inputs[3], counts
        )
        assert_close(test_out, ref_out, rtol=1e-5, atol=1e-5)

        grad = torch.randn_like(ref_out)
        ref_out.backward(grad)
        test_out.backward(grad)

        for ref, test in zip(ref_inputs, test_inputs):
            assert_close(test.grad, ref.grad, rtol=1e-5, atol=1e-5)

    def test_grouped_experts_backend_selector_matches_for_loop(self) -> None:
        torch.manual_seed(5678)
        num_experts = 4
        dim = 12
        hidden_dim = 20
        top_k = 2
        counts = torch.tensor([4, 0, 3, 2], dtype=torch.int64)
        total_tokens = int(counts.sum().item())

        base_cfg = dict(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            token_dispatcher=LocalTokenDispatcher.Config(
                num_experts=num_experts,
                top_k=top_k,
                score_before_experts=True,
            ),
        )
        ref = GroupedExperts.Config(
            **base_cfg,
            use_grouped_mm=False,
            compute_backend="for_loop",
        ).build()
        test = GroupedExperts.Config(
            **base_cfg,
            use_grouped_mm=False,
            compute_backend="batched_mm_padded",
        ).build()

        w1, w2, w3 = _init_weights(num_experts, dim, hidden_dim)
        with torch.no_grad():
            ref.w1.copy_(w1)
            ref.w2.copy_(w2)
            ref.w3.copy_(w3)
            test.w1.copy_(w1)
            test.w2.copy_(w2)
            test.w3.copy_(w3)

        x = torch.randn(total_tokens, dim, requires_grad=True)
        x_test = x.detach().clone().requires_grad_(True)

        ref_out = ref._experts_forward(x, counts)
        test_out = test._experts_forward(x_test, counts)
        assert_close(test_out, ref_out, rtol=1e-5, atol=1e-5)

        grad = torch.randn_like(ref_out)
        ref_out.backward(grad)
        test_out.backward(grad)

        assert_close(x_test.grad, x.grad, rtol=1e-5, atol=1e-5)
        for ref_param, test_param in zip(ref.parameters(), test.parameters()):
            assert_close(test_param.grad, ref_param.grad, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
