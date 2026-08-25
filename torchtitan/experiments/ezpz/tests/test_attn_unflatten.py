#!/usr/bin/env python3
"""Regression test for the folded-attention unflatten in agpt/__init__.py.

#4121 ("fold batch dim") hands attention a 3D tensor. The layout is
``[T, N, H]`` -- the batch dim is folded INTO the token count -- so B is
recovered as ``num_tokens // seq_len``.

MEASURED on 4 A100s (Perlmutter job 57536150), agpt_2b_real, one
sequence per rank::

    q=(8192, 16, 128)  k=(8192, 4, 128)  seq_len=8192

T=8192 tokens, N=16 heads, H=128; k carries n_kv=4 under GQA, which is
why k/v keep the ``-1`` head inference instead of reusing q's count.

An earlier revision of this test asserted ``[B, L*N, H]`` instead --
reading dim 0 as the batch. That was inferred from a single observed
shape, ``(1, 131072, 128)``, which is arithmetically consistent with
BOTH readings; the wrong one was picked and the matching arithmetic
mistaken for proof. It shipped as 5b4a81803 and was reverted after this
measurement. The lesson worth keeping: a test written from an inferred
layout will pass its own negative control, because the control only
shows the test is sensitive to a change -- not that the premise is
right.

Why this extracts the source instead of importing it: importing the
module needs the full distributed stack (``spmd_types`` and friends),
which is not available on a login node or in CI. Extracting keeps the
test honest -- it exercises the SHIPPED lines, so an edit that breaks
the reshape fails here rather than passing against a drifted copy.

What is asserted is element-order preservation (``torch.equal`` against
the pre-fold tensor), not just output shape. A wrong-but-plausible
reshape has the right shape and the wrong values, and SDPA accepts it:
the symptom is a quietly degraded loss curve, not a traceback.

Verified end to end: with the correct unflatten the same config trains
(Perlmutter 57536357, 4/4 steps, loss 12.90 -> 12.01, 48% MFU, rc=0).

Run from repo root:
    python3 torchtitan/experiments/ezpz/tests/test_attn_unflatten.py
"""

from __future__ import annotations

import pathlib
import re
import sys
import textwrap

import torch

_SRC = (
    pathlib.Path(__file__).resolve().parents[1] / "agpt" / "__init__.py"
)


def _committed_unflatten_block() -> str:
    """Return the shipped unflatten lines, dedented for exec()."""
    src = _SRC.read_text()
    m = re.search(
        # Suffix is _TNH since b5f6f712f (the local_map contract matches
        # positional-arg NAMES, so these track upstream's in_dst_shardings
        # keys). Matched loosely so a future rename fails on the STRUCTURE
        # here, not on the letters -- this test broke silently for a day
        # because it pinned the old _BLNH spelling.
        r"num_tokens, num_heads, head_dim = q_\w+\.shape.*?"
        r"v_\w+ = v_\w+\.view\(batch, seq_len, -1, head_dim\)",
        src,
        re.S,
    )
    if not m:
        raise AssertionError(
            f"could not locate the unflatten block in {_SRC}. If it was "
            "intentionally rewritten, update this test to match."
        )
    return textwrap.dedent(" " * 12 + m.group(0))


_BLOCK = _committed_unflatten_block()

# Derive the q/k/v variable names FROM the extracted block instead of hardcoding
# them. The shape suffix is a naming convention that tracks upstream's local_map
# keys (_BLNH -> _TNH in b5f6f712f), and hardcoding it here is what made this
# test start failing silently the day that rename landed.
_SUF = re.search(r"num_tokens, num_heads, head_dim = q_(\w+)\.shape", _BLOCK).group(1)
_Q, _K, _V = f"q_{_SUF}", f"k_{_SUF}", f"v_{_SUF}"


def _roundtrip(B: int, L: int, N: int, H: int, n_kv: int | None = None):
    """Fold a known 4D tensor, run the shipped code, compare exactly."""
    n_kv = n_kv or N
    q_ref = torch.arange(B * L * N * H, dtype=torch.float32).view(B, L, N, H)
    k_ref = torch.arange(B * L * n_kv * H, dtype=torch.float32).view(
        B, L, n_kv, H
    )
    v_ref = k_ref.clone()
    ns = {
        # [T, N, H] with T = B*L -- the batch dim folded into tokens.
        _Q: q_ref.reshape(B * L, N, H),
        _K: k_ref.reshape(B * L, n_kv, H),
        _V: v_ref.reshape(B * L, n_kv, H),
        "seq_len": L,
        "ValueError": ValueError,
    }
    exec(_BLOCK, ns)  # noqa: S102 -- deliberate: run the shipped lines
    assert torch.equal(ns[_Q], q_ref), f"q mismatch B={B} L={L} N={N}"
    assert torch.equal(ns[_K], k_ref), f"k mismatch B={B} L={L} N={N}"
    assert torch.equal(ns[_V], v_ref), f"v mismatch B={B} L={L} N={N}"


def test_smoke_shape_lbs1() -> None:
    """The 2N smoke config: LBS=1, one sequence per microbatch."""
    _roundtrip(1, 8192, 16, 128)


def test_production_shape_lbs2() -> None:
    """The 20B chain's actual config (LBS=2 -> GBS 1024 at 512 ranks)."""
    _roundtrip(2, 8192, 16, 128)


def test_smaller_context() -> None:
    _roundtrip(1, 4096, 8, 128)


def test_gqa_kv_heads_differ_from_q() -> None:
    """Under GQA k/v carry n_kv_heads != num_heads.

    This is why k/v keep the ``-1`` inference instead of reusing q's
    head count -- hardcoding num_heads here would silently mis-shape
    every grouped-query model.
    """
    _roundtrip(2, 8192, 16, 128, n_kv=4)


def test_ragged_batch_raises() -> None:
    """A batch that is not a whole number of sequences must be refused.

    Better a loud failure than a reshape onto the wrong grid.
    """
    ns = {
        # 100 tokens is not a whole number of 8192-token sequences.
        _Q: torch.zeros(100, 16, 128),
        _K: torch.zeros(100, 4, 128),
        _V: torch.zeros(100, 4, 128),
        "seq_len": 8192,
        "ValueError": ValueError,
    }
    try:
        exec(_BLOCK, ns)  # noqa: S102
    except ValueError:
        return
    raise AssertionError("ragged batch was accepted; it must raise")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
