#!/usr/bin/env python3
"""Regression test for the flat-token port of moe's forked MLA forward.

`moe/model.py` carries a FORKED multi-head latent attention that upstream's
#4121 ("fold batch dim") never touched. Post-`42f4edfaa` the loader folds
`[B, L] -> [T]` unconditionally, so the forward received a 2D `[T, D]` tensor
while every reshape in it still expected `[B, L, D]`:

    bsz, seqlen, _ = x.size()
    ValueError: not enough values to unpack (expected 3, got 2)

The port mirrors `torchtitan/models/deepseek_v3/model.py` -- the upstream MLA
this one forked from, which #4121 DID update. It is used here as the reference
rather than deriving the axes from the traceback, because two of the six
changes are not recoverable that way:

  * `k_pe.unsqueeze(2)` -> `unsqueeze(1)`. k_pe drops from rank 3 to rank 2,
    so the head axis moves. unsqueeze(2) on a 2D tensor does not raise -- it
    appends a trailing axis -- and the error surfaces later as a wrong
    head_dim, at SDPA rather than at the reshape.
  * `k_pe.expand(-1, -1, self.n_heads, -1)` -> `expand(-1, k_nope.size(1), -1)`.
    Under TP each rank holds n_heads/tp_degree heads locally, so expanding to
    the GLOBAL count is correct at tp=1 and wrong at tp>1 -- a bug no
    single-rank smoke can see.

What is asserted here is VALUES, not shapes. Per the lesson recorded in
`test_attn_unflatten.py`: a wrong-but-plausible reshape has the right shape and
the wrong element order, SDPA accepts it, and the only symptom is a quietly
degraded loss curve. So each case builds a tensor whose elements encode their
own (token, head, dim) identity and checks that identity survives.

CPU-only. No corpus, no GPU, no distributed init.

Run from repo root:
    uv run --with pytest python -m pytest \\
        torchtitan/experiments/ezpz/tests/test_moe_mla_flat_layout.py -q
"""

import re
from pathlib import Path

import pytest
import torch

# Shapes small enough to reason about by hand, structured like the real thing:
# GQA (n_heads != n_kv), and qk_head_dim != v_head_dim so the XPU pad_v path
# is exercised.
T = 12  # tokens (post-fold: B*L, e.g. 2 seqs x 6 positions)
N_HEADS = 4
QK_NOPE = 8
QK_ROPE = 3  # deliberately != N_HEADS; see test_unsqueeze_2_*
V_HEAD = 6
QK_HEAD = QK_NOPE + QK_ROPE  # 11


def _src() -> str:
    p = (
        Path(__file__).resolve().parents[1] / "moe" / "model.py"
    )
    return p.read_text()


# ---------------------------------------------------------------------------
# 1. The shipped source really is ported (no stale 3D assumptions left).
# ---------------------------------------------------------------------------


def test_no_bsz_seqlen_unpack_remains():
    """The exact line that raised must be gone, not merely guarded."""
    src = _src()
    assert "bsz, seqlen, _ = x.size()" not in src
    assert not re.search(r"\bbsz\b", src), "bsz still referenced in moe/model.py"
    assert not re.search(r"\bseqlen\b", src), "seqlen still referenced"


def test_forward_uses_num_tokens():
    src = _src()
    assert "num_tokens = x.shape[0]" in src


def test_kpe_unsqueezes_axis_1_not_2():
    """unsqueeze(2) on the now-2D k_pe silently produces the wrong tensor."""
    src = _src()
    assert "k_pe.unsqueeze(1)" in src
    assert "k_pe.unsqueeze(2)" not in src


def test_expand_uses_local_head_count():
    """self.n_heads is the GLOBAL count; under TP the local shard is smaller."""
    src = _src()
    assert "k_pe.expand(-1, k_nope.size(1), -1)" in src
    assert "k_pe.expand(-1, -1, self.n_heads, -1)" not in src


def test_views_are_flat():
    src = _src()
    assert "q.view(num_tokens, -1, self.qk_head_dim)" in src
    assert (
        "kv.view(num_tokens, -1, self.qk_nope_head_dim + self.v_head_dim)" in src
    )
    assert "output.view(num_tokens, -1)" in src


def test_xpu_pad_v_workaround_survived_the_port():
    """moe's XPU fix has no upstream equivalent; the port must not drop it."""
    src = _src()
    assert "pad_v = self.qk_head_dim != self.v_head_dim" in src
    assert "v = F.pad(v, (0, self.qk_head_dim - self.v_head_dim))" in src
    assert "output = output[..., : self.v_head_dim]" in src


# ---------------------------------------------------------------------------
# 2. The reshapes preserve element identity (the load-bearing checks).
# ---------------------------------------------------------------------------


def _identity_tensor(num_tokens: int, heads: int, dim: int) -> torch.Tensor:
    """Every element encodes its own (token, head, dim) coordinate."""
    t = torch.arange(num_tokens, dtype=torch.float64).view(-1, 1, 1) * 10_000
    h = torch.arange(heads, dtype=torch.float64).view(1, -1, 1) * 100
    d = torch.arange(dim, dtype=torch.float64).view(1, 1, -1)
    return t + h + d


def test_q_view_preserves_token_head_order():
    """q: [T, N*QK] -> [T, N, QK] must keep each token's heads contiguous."""
    want = _identity_tensor(T, N_HEADS, QK_HEAD)
    flat = want.reshape(T, N_HEADS * QK_HEAD)          # as wq emits it
    got = flat.view(T, -1, QK_HEAD)                     # the shipped reshape
    assert got.shape == (T, N_HEADS, QK_HEAD)
    assert torch.equal(got, want)


def test_q_view_rejects_the_batch_reading():
    """The [B, L*N, H] misreading is arithmetically consistent but reorders.

    This is the negative control that matters: it is the shape the OLD code
    would have produced. It has a valid shape and the wrong contents.
    """
    want = _identity_tensor(T, N_HEADS, QK_HEAD)
    flat = want.reshape(T, N_HEADS * QK_HEAD)
    # Pretend T folds as B=2, L=6 and unflatten the 3D way.
    wrong = flat.view(2, 6 * N_HEADS, QK_HEAD)
    assert wrong.numel() == want.numel()               # same element count...
    assert wrong.reshape(-1).equal(want.reshape(-1))   # ...same flat order...
    assert wrong.shape != want.shape                   # ...different axes.
    # The point: element count and flat order match, so any check weaker than
    # a shape+value pair on the FINAL axes passes the wrong layout.


def test_kpe_unsqueeze_then_expand_broadcasts_across_heads():
    """k_pe is per-token (head-independent); it must repeat, not stripe."""
    k_pe = torch.arange(T * QK_ROPE, dtype=torch.float64).view(T, QK_ROPE)
    k_pe3 = k_pe.unsqueeze(1)                           # [T, 1, QK_ROPE]
    assert k_pe3.shape == (T, 1, QK_ROPE)
    expanded = k_pe3.expand(-1, N_HEADS, -1)            # [T, N, QK_ROPE]
    assert expanded.shape == (T, N_HEADS, QK_ROPE)
    # every head sees the SAME rope vector for a given token
    for h in range(N_HEADS):
        assert torch.equal(expanded[:, h, :], k_pe)


def test_unsqueeze_2_would_be_wrong_on_a_2d_kpe():
    """unsqueeze(2) does not raise on rank-2 input -- it silently misbuilds."""
    k_pe = torch.arange(T * QK_ROPE, dtype=torch.float64).view(T, QK_ROPE)
    bad = k_pe.unsqueeze(2)                             # [T, QK_ROPE, 1]
    assert bad.shape == (T, QK_ROPE, 1)                 # no error raised
    # Axis 1 is now the ROPE DIM, not a head axis, so expanding heads there
    # is a size mismatch -- when the two happen to differ.
    with pytest.raises(RuntimeError):
        bad.expand(-1, N_HEADS, -1)


def test_wrong_unsqueeze_survives_expand_when_rope_dim_equals_head_count():
    """When rope_dim == n_heads the wrong expand does not raise -- but the
    resulting head_dim is wrong, so it still fails downstream.

    Written after this file's first run. The original constants had
    QK_ROPE == N_HEADS == 4, so `bad.expand(-1, N_HEADS, -1)` found axis 1
    already the right size and succeeded where the test expected a raise.

    MEASURED, not assumed: unsqueeze(2) gives [T, rope_dim, 1], and expanding
    axis 1 leaves axis 2 at 1, so the cat produces head_dim = qk_nope + 1
    instead of qk_nope + rope_dim. That is a shape error SDPA or wo will hit --
    unpleasant to debug, but not the silent value corruption I first claimed
    here. Recorded so the distinction is not re-litigated: the *silent* hazard
    in this port is the q/kv view and the [B, L*N, H] reading, not this one.
    """
    n = 4
    rope_dim = 4                                        # == n, the coincidence
    k_pe = torch.arange(T * rope_dim, dtype=torch.float64).view(T, rope_dim)
    k_nope = torch.zeros(T, n, QK_NOPE, dtype=torch.float64)

    right = k_pe.unsqueeze(1).expand(-1, n, -1)
    assert right.shape == (T, n, rope_dim)
    assert torch.cat([k_nope, right], dim=-1).shape == (T, n, QK_NOPE + rope_dim)

    bad = k_pe.unsqueeze(2)                             # [T, rope_dim, 1]
    assert bad.shape == (T, rope_dim, 1)
    wrong = bad.expand(-1, n, -1)                       # no raise at rope==n
    assert wrong.shape == (T, n, 1)                     # axis 2 stayed 1
    assert torch.cat([k_nope, wrong], dim=-1).shape == (T, n, QK_NOPE + 1)
    assert QK_NOPE + 1 != QK_NOPE + rope_dim            # caught downstream


def test_kv_split_and_concat_round_trip():
    """kv: [T, N*(nope+v)] -> heads, split, recombine with rope."""
    kv_flat = torch.arange(
        T * N_HEADS * (QK_NOPE + V_HEAD), dtype=torch.float64
    ).view(T, N_HEADS * (QK_NOPE + V_HEAD))
    kv = kv_flat.view(T, -1, QK_NOPE + V_HEAD)
    assert kv.shape == (T, N_HEADS, QK_NOPE + V_HEAD)
    k_nope, v = torch.split(kv, [QK_NOPE, V_HEAD], dim=-1)
    assert k_nope.shape == (T, N_HEADS, QK_NOPE)
    assert v.shape == (T, N_HEADS, V_HEAD)

    k_pe = torch.arange(T * QK_ROPE, dtype=torch.float64).view(T, QK_ROPE)
    k = torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, k_nope.size(1), -1)], dim=-1)
    assert k.shape == (T, N_HEADS, QK_NOPE + QK_ROPE)
    # nope half is untouched; rope half is the per-token vector on every head
    assert torch.equal(k[..., :QK_NOPE], k_nope)
    for h in range(N_HEADS):
        assert torch.equal(k[:, h, QK_NOPE:], k_pe)


def test_expand_to_global_head_count_breaks_under_tp():
    """Why k_nope.size(1) and not self.n_heads.

    Simulate tp=2: the local shard holds N_HEADS//2 heads. Expanding k_pe to
    the GLOBAL n_heads mismatches k_nope and raises on the cat -- but only at
    tp>1, which no single-rank smoke reaches.
    """
    local_heads = N_HEADS // 2
    k_nope = torch.zeros(T, local_heads, QK_NOPE, dtype=torch.float64)
    k_pe = torch.zeros(T, 1, QK_ROPE, dtype=torch.float64)

    good = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1)], dim=-1)
    assert good.shape == (T, local_heads, QK_NOPE + QK_ROPE)

    with pytest.raises(RuntimeError):
        torch.cat([k_nope, k_pe.expand(-1, N_HEADS, -1)], dim=-1)


def test_output_view_returns_flat_tokens():
    """SDPA output [T, N, V] -> [T, N*V] for wo, preserving order."""
    out = _identity_tensor(T, N_HEADS, V_HEAD)
    flat = out.contiguous().view(T, -1)
    assert flat.shape == (T, N_HEADS * V_HEAD)
    assert torch.equal(flat.view(T, N_HEADS, V_HEAD), out)


def test_pad_v_round_trip_is_lossless():
    """The XPU pad/unpad must return exactly the original v values."""
    v = _identity_tensor(T, N_HEADS, V_HEAD)
    padded = torch.nn.functional.pad(v, (0, QK_HEAD - V_HEAD))
    assert padded.shape == (T, N_HEADS, QK_HEAD)
    recovered = padded[..., :V_HEAD]
    assert torch.equal(recovered, v)
