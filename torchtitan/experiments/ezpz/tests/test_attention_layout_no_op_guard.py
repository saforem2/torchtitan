"""The `[B, L*N, H]` attention reading performs NO attention at all.

Two incompatible fixes were written for the same post-#4121 error,
`token count 1 is not a multiple of max_context_length 8192`:

  A. 5b4a81803 patched the attention wrapper to unflatten as
     [B, L*N, H], leaving blendcorpus emitting [B, L].
  B. 42f4edfaa folded the dataloader to flat [T], matching core.

A is wrong, and wrong in the worst way: reading dim 0 as the batch dim
makes SDPA see a sequence of length 1 with 131072 heads. Causal
attention over one token is softmax of a single score = 1.0, so the
output is exactly `v`. The model becomes a position-wise MLP that still
trains and still descends -- there is no traceback to catch.

The plausible-looking check does NOT catch it: the GQA head ratio
n_q/n_kv is preserved under A (both sides scale by L), so a ratio
assertion confirms the wrong layout. The load-bearing check is
max|out - v| == 0.

CPU-only, no corpus, no GPU.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

# Shape legend (agpt-2b): B batch, L context, D model, H head_dim,
# NQ query heads, NKV kv heads, T = B*L folded tokens.
B, L, D, H, NQ, NKV = 2, 512, 256, 32, 8, 2


def _project(x_2D, wq, wkv):
    """Mimic QKVLinear.forward: dim 0 is read as the token count."""
    n = x_2D.shape[0]
    return (
        (x_2D @ wq).view(n, -1, H),
        (x_2D @ wkv).view(n, -1, H),
        (x_2D @ wkv).view(n, -1, H),
    )


def _sdpa(q_NH, k_NH, v_NH):
    out = F.scaled_dot_product_attention(
        q_NH.transpose(0, 1).unsqueeze(0),
        k_NH.transpose(0, 1).unsqueeze(0),
        v_NH.transpose(0, 1).unsqueeze(0),
        is_causal=True,
        enable_gqa=q_NH.shape[1] != k_NH.shape[1],
    )
    return out.squeeze(0).transpose(0, 1)


@pytest.fixture
def weights():
    torch.manual_seed(0)
    return torch.randn(D, NQ * H) / D**0.5, torch.randn(D, NKV * H) / D**0.5


@pytest.fixture
def x_BLD():
    torch.manual_seed(1)
    return torch.randn(B, L, D)


class TestBadLayoutIsDegenerate:
    """Pin the silent-corruption mode so it cannot be reintroduced.

    Under `[B, L*N, H]` the sequence SDPA sees has length B, not L. Two
    distinct failures follow, and which one you get depends on the batch:

      B == 1 (production blendcorpus): sequence length 1, so causal
        attention is softmax of a single score = 1.0 and the output is
        EXACTLY v. No attention occurs at all.
      B > 1: tokens attend ACROSS BATCH ROWS -- row 1 leaks into row 0,
        which the correct layout forbids absolutely.
    """

    @staticmethod
    def _bad_project(x_BLD, wq, wkv):
        """dim 0 stays the batch dim, so L and N fold together."""
        n = x_BLD.shape[0]
        return (
            (x_BLD @ wq).reshape(n, -1, H),
            (x_BLD @ wkv).reshape(n, -1, H),
            (x_BLD @ wkv).reshape(n, -1, H),
        )

    def test_single_row_makes_attention_the_identity_on_v(self, weights):
        """B == 1 is the production case: max|out - v| is exactly 0."""
        torch.manual_seed(1)
        x_1LD = torch.randn(1, L, D)
        q, k, v = self._bad_project(x_1LD, *weights)
        assert q.shape == (1, L * NQ, H), f"expected folded L*N, got {q.shape}"
        out = _sdpa(q, k, v)
        v_broadcast = v.repeat_interleave(NQ // NKV, dim=1)
        assert torch.equal(out, v_broadcast), (
            "expected a pure no-op at B=1; if this now differs the degenerate "
            "path changed and this guard needs rewriting"
        )

    def test_multi_row_leaks_across_batch_rows(self, x_BLD, weights):
        """B > 1: batch row 0 bleeds into row 1. Rows must be isolated.

        The leak runs FORWARD, not backward: with the batch dim standing in
        for the sequence, row i sits at position i, and `is_causal` lets
        later positions attend to earlier ones. So perturbing row 0 moves
        row 1, while row 0 itself is unreachable from row 1.
        """
        q, k, v = self._bad_project(x_BLD, *weights)
        base = _sdpa(q, k, v)
        v_row0 = v.clone()
        v_row0[0] += 100.0
        leaked = _sdpa(q, k, v_row0)
        assert not torch.allclose(base[1], leaked[1], atol=1e-6), (
            "batch row 1 was unaffected by row 0 -- the cross-row leak this "
            "guards against is gone"
        )

    def test_gqa_ratio_check_does_not_catch_it(self, x_BLD, weights):
        """The obvious sanity check PASSES under the wrong layout."""
        q, k, _ = self._bad_project(x_BLD, *weights)
        assert q.shape[1] // k.shape[1] == NQ // NKV, (
            "head ratio is preserved under the bad layout -- that is the point"
        )

    def test_wo_input_width_is_wrong_by_L(self, x_BLD, weights):
        """GQAttention does out.view(dim0, -1); the bad layout is L x too wide."""
        q, k, v = self._bad_project(x_BLD, *weights)
        out = _sdpa(q, k, v)
        width = out.reshape(out.shape[0], -1).shape[1]
        assert width == NQ * H * L, f"expected {NQ * H * L}, got {width}"
        assert width != NQ * H, "wo expects the model dim, not L times it"


class TestFoldedLayoutAttends:
    """The correct flat [T] path must actually mix tokens."""

    def test_attention_is_not_the_identity(self, x_BLD, weights):
        q, k, v = _project(x_BLD.reshape(B * L, D), *weights)
        assert q.shape == (B * L, NQ, H)
        out = _sdpa(q, k, v)
        v_broadcast = v.repeat_interleave(NQ // NKV, dim=1)
        assert not torch.allclose(out, v_broadcast, atol=1e-5), (
            "attention output equals v -- no attention occurred"
        )

    def test_wo_input_width_is_model_dim(self, x_BLD, weights):
        """out.view(T, -1) must be [T, D]; the bad layout is off by L."""
        q, k, v = _project(x_BLD.reshape(B * L, D), *weights)
        out = _sdpa(q, k, v)
        assert out.reshape(out.shape[0], -1).shape == (B * L, NQ * H)


class TestUnflattenRecoversTheGrid:
    """agpt/__init__.py: dim 0 is TOKENS, B = T / max_context_length."""

    @staticmethod
    def _unflatten_and_attend(q, k, v, seq_len):
        num_tokens, num_heads, head_dim = q.shape
        assert num_tokens % seq_len == 0
        batch = num_tokens // seq_len
        q4 = q.view(batch, seq_len, num_heads, head_dim)
        k4 = k.view(batch, seq_len, -1, head_dim)
        v4 = v.view(batch, seq_len, -1, head_dim)
        out = F.scaled_dot_product_attention(
            q4.transpose(1, 2),
            k4.transpose(1, 2),
            v4.transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2)
        return batch, out

    def test_recovers_the_true_batch(self, x_BLD, weights):
        q, k, v = _project(x_BLD.reshape(B * L, D), *weights)
        batch, _ = self._unflatten_and_attend(q, k, v, L)
        assert batch == B

    def test_causality_holds(self, x_BLD, weights):
        """Perturbing the tail must leave earlier outputs bit-identical."""
        q, k, v = _project(x_BLD.reshape(B * L, D), *weights)
        _, base = self._unflatten_and_attend(q, k, v, L)
        v_tail = v.view(B, L, -1, H).clone()
        v_tail[:, L // 2 :] += 100.0
        _, perturbed = self._unflatten_and_attend(
            q, k, v_tail.reshape(B * L, -1, H), L
        )
        assert torch.equal(base[:, : L // 2], perturbed[:, : L // 2])
        assert not torch.allclose(base[:, L // 2 :], perturbed[:, L // 2 :])

    def test_rows_are_isolated(self, x_BLD, weights):
        """One packed row must not leak into another."""
        q, k, v = _project(x_BLD.reshape(B * L, D), *weights)
        _, base = self._unflatten_and_attend(q, k, v, L)
        v_row1 = v.view(B, L, -1, H).clone()
        v_row1[1] += 100.0
        _, perturbed = self._unflatten_and_attend(
            q, k, v_row1.reshape(B * L, -1, H), L
        )
        assert torch.equal(base[0], perturbed[0])
        assert not torch.allclose(base[1], perturbed[1])
