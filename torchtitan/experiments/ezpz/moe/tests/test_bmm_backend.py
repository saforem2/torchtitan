# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU correctness gate for the ``"bmm"`` expert compute backend.

Asserts that ``_run_experts_bmm`` matches ``_run_experts_for_loop`` given
the same inputs, when the configured capacity is large enough that no
tokens are dropped. This is the contract the "bmm" backend must satisfy:
it is a batched re-expression of the same per-expert matmuls, not a
different algorithm.
"""

import torch

from torchtitan.experiments.ezpz.moe.experts import (
    _run_experts_bmm,
    _run_experts_for_loop,
)


def _make_inputs(num_tokens_per_expert: list[int], num_experts: int, F: int, D: int):
    torch.manual_seed(0)
    R = sum(num_tokens_per_expert)
    x = torch.randn(R, D, dtype=torch.float32)
    w1 = torch.randn(num_experts, F, D, dtype=torch.float32)
    w2 = torch.randn(num_experts, D, F, dtype=torch.float32)
    w3 = torch.randn(num_experts, F, D, dtype=torch.float32)
    counts = torch.tensor(num_tokens_per_expert, dtype=torch.long)
    return w1, w2, w3, x, counts


def test_bmm_matches_for_loop_no_drop():
    # E=4, F=16, D=8; counts include a zero-token expert (index 1) as an
    # empty-expert edge case. capacity_factor=4.0 gives
    # cap = ceil(10/4*4.0) = 10 >= max(counts)=5, so nothing is dropped.
    num_tokens_per_expert = [3, 0, 5, 2]
    num_experts = 4
    hidden_dim = 16
    dim = 8
    w1, w2, w3, x, counts = _make_inputs(
        num_tokens_per_expert, num_experts, hidden_dim, dim
    )

    out_for_loop = _run_experts_for_loop(w1, w2, w3, x, counts)
    out_bmm = _run_experts_bmm(w1, w2, w3, x, counts, capacity_factor=4.0)

    assert out_bmm.shape == out_for_loop.shape
    torch.testing.assert_close(out_bmm, out_for_loop, atol=2e-2, rtol=2e-2)


def test_bmm_all_experts_nonempty_no_drop():
    # Every expert has at least one token, still well under capacity.
    num_tokens_per_expert = [2, 4, 1, 3]
    num_experts = 4
    hidden_dim = 16
    dim = 8
    w1, w2, w3, x, counts = _make_inputs(
        num_tokens_per_expert, num_experts, hidden_dim, dim
    )

    out_for_loop = _run_experts_for_loop(w1, w2, w3, x, counts)
    out_bmm = _run_experts_bmm(w1, w2, w3, x, counts, capacity_factor=4.0)

    assert out_bmm.shape == out_for_loop.shape
    torch.testing.assert_close(out_bmm, out_for_loop, atol=2e-2, rtol=2e-2)


def test_bmm_drops_tokens_beyond_capacity():
    # capacity_factor small enough that expert 2 (5 tokens) overflows a
    # small capacity; dropped rows should be exactly zero in the bmm
    # output, while for_loop (no capacity limit) keeps them nonzero.
    num_tokens_per_expert = [3, 0, 5, 2]
    num_experts = 4
    hidden_dim = 16
    dim = 8
    w1, w2, w3, x, counts = _make_inputs(
        num_tokens_per_expert, num_experts, hidden_dim, dim
    )

    # R / E * capacity_factor = 10 / 4 * 0.4 = 1.0 -> cap = 1.
    cap_factor = 0.4
    out_bmm = _run_experts_bmm(w1, w2, w3, x, counts, capacity_factor=cap_factor)
    out_for_loop = _run_experts_for_loop(w1, w2, w3, x, counts)

    assert out_bmm.shape == out_for_loop.shape

    # Expert 0 occupies rows [0:3), only row 0 (its first token) survives
    # a capacity of 1; rows 1-2 are dropped (zeroed).
    torch.testing.assert_close(
        out_bmm[0:1], out_for_loop[0:1], atol=2e-2, rtol=2e-2
    )
    assert torch.all(out_bmm[1:3] == 0)

    # Expert 2 occupies rows [3:8), only row 3 (its first token) survives.
    torch.testing.assert_close(
        out_bmm[3:4], out_for_loop[3:4], atol=2e-2, rtol=2e-2
    )
    assert torch.all(out_bmm[4:8] == 0)

    # Expert 3 occupies rows [8:10), only row 8 survives.
    torch.testing.assert_close(
        out_bmm[8:9], out_for_loop[8:9], atol=2e-2, rtol=2e-2
    )
    assert torch.all(out_bmm[9:10] == 0)


def _leaf(t: torch.Tensor) -> torch.Tensor:
    """Detach + clone into an independent leaf tensor with requires_grad set.

    Used so the bmm and for_loop paths backward into separate tensors --
    otherwise calling .backward() on both would accumulate into the same
    .grad and mask a mismatch between the two paths.
    """
    return t.clone().detach().requires_grad_(True)


def test_bmm_backward_matches_for_loop_no_drop():
    # Same shapes/capacity as test_bmm_matches_for_loop_no_drop (nothing is
    # dropped), but with requires_grad=True on x and the weights, checking
    # that x.grad/w1.grad/w2.grad/w3.grad also match for_loop, not just the
    # forward output. The backward path goes through index_copy_/
    # index_select with a shared scratch row for dropped tokens, which is
    # non-obvious enough to need its own coverage beyond the forward gate.
    num_tokens_per_expert = [3, 0, 5, 2]
    num_experts = 4
    hidden_dim = 16
    dim = 8
    w1, w2, w3, x, counts = _make_inputs(
        num_tokens_per_expert, num_experts, hidden_dim, dim
    )

    w1_bmm, w2_bmm, w3_bmm, x_bmm = _leaf(w1), _leaf(w2), _leaf(w3), _leaf(x)
    out_bmm = _run_experts_bmm(
        w1_bmm, w2_bmm, w3_bmm, x_bmm, counts, capacity_factor=4.0
    )
    out_bmm.sum().backward()

    w1_fl, w2_fl, w3_fl, x_fl = _leaf(w1), _leaf(w2), _leaf(w3), _leaf(x)
    out_fl = _run_experts_for_loop(w1_fl, w2_fl, w3_fl, x_fl, counts)
    out_fl.sum().backward()

    torch.testing.assert_close(x_bmm.grad, x_fl.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(w1_bmm.grad, w1_fl.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(w2_bmm.grad, w2_fl.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(w3_bmm.grad, w3_fl.grad, atol=2e-2, rtol=2e-2)


def test_bmm_backward_dropped_tokens_get_zero_grad():
    # capacity_factor forces expert 0's 5 tokens past a capacity of 2:
    # R / E * capacity_factor = 5 / 2 * 0.8 = 2.0 -> cap = 2. Rows 0-1
    # (positions 0-1 within expert 0's run) survive; rows 2-4 (positions
    # 2-4) exceed capacity and are dropped.
    #
    # The bmm backend redirects dropped tokens to a shared scratch row
    # (index E * cap) via index_copy_, which is never read by the bmm
    # matmuls, so its gradient is exactly zero; index_select's backward
    # then hands that same exact zero to every dropped token. This test
    # pins down that non-obvious contract and checks it doesn't produce
    # NaNs anywhere (weights included).
    num_tokens_per_expert = [5, 0]
    num_experts = 2
    hidden_dim = 16
    dim = 8
    w1, w2, w3, x, counts = _make_inputs(
        num_tokens_per_expert, num_experts, hidden_dim, dim
    )

    w1_bmm, w2_bmm, w3_bmm, x_bmm = _leaf(w1), _leaf(w2), _leaf(w3), _leaf(x)
    out_bmm = _run_experts_bmm(
        w1_bmm, w2_bmm, w3_bmm, x_bmm, counts, capacity_factor=0.8
    )
    out_bmm.sum().backward()

    dropped_rows = x_bmm.grad[2:5]
    assert torch.all(dropped_rows == 0)

    assert not torch.isnan(x_bmm.grad).any()
    assert not torch.isnan(w1_bmm.grad).any()
    assert not torch.isnan(w2_bmm.grad).any()
    assert not torch.isnan(w3_bmm.grad).any()
