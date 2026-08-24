#!/usr/bin/env python3
"""Regression test for the folded-attention unflatten in agpt/__init__.py.

#4121 ("fold batch dim") makes the LM stack hand attention a 3D tensor.
The first port read it as ``[T, N, H]`` -- batch folded INTO the token
count -- but on the blendcorpus path the batch dim is PRESERVED and it is
L and N that are folded: ``[B, L*N, H]``. Reading dim 0 as tokens yields
B=1, so ``1 % 8192 != 0`` and a well-formed batch was rejected with

    ValueError: token count 1 is not a multiple of max_context_length 8192

Measured, not inferred (probe run 7554521): the decoder receives 2D
tokens ``[B, L]`` -- ``tokens.shape=(1, 8192)``. Both observed attention
shapes agree with ``[B, L*N, H]`` and rule out ``[T, N, H]``:

    LBS=16 -> (1, 131072, 128) == B=1, L*N = 8192*16, H=128
    LBS=1  -> dim 0 == 1        (would be 8192 under the other reading)

Why this test extracts the source instead of importing it: importing the
module needs the full distributed stack (``spmd_types`` and friends),
which is not available on a login node or in CI. Extracting keeps the
test honest -- it exercises the SHIPPED lines, so a future edit that
breaks the reshape fails here rather than passing against a copy that
drifted.

What is actually asserted is element-order preservation
(``torch.equal`` against the pre-fold tensor), not just the output shape.
A wrong-but-plausible reshape has the right shape and the wrong values,
and SDPA accepts it -- the failure mode is a quietly degraded loss curve,
not a traceback.

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
        r"batch, folded_ln, head_dim = q_BLNH\.shape.*?"
        r"v_BLNH = v_BLNH\.view\(batch, seq_len, -1, head_dim\)",
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


def _roundtrip(B: int, L: int, N: int, H: int, n_kv: int | None = None):
    """Fold a known 4D tensor, run the shipped code, compare exactly."""
    n_kv = n_kv or N
    q_ref = torch.arange(B * L * N * H, dtype=torch.float32).view(B, L, N, H)
    k_ref = torch.arange(B * L * n_kv * H, dtype=torch.float32).view(
        B, L, n_kv, H
    )
    v_ref = k_ref.clone()
    ns = {
        "q_BLNH": q_ref.reshape(B, L * N, H),
        "k_BLNH": k_ref.reshape(B, L * n_kv, H),
        "v_BLNH": v_ref.reshape(B, L * n_kv, H),
        "seq_len": L,
        "ValueError": ValueError,
    }
    exec(_BLOCK, ns)  # noqa: S102 -- deliberate: run the shipped lines
    assert torch.equal(ns["q_BLNH"], q_ref), f"q mismatch B={B} L={L} N={N}"
    assert torch.equal(ns["k_BLNH"], k_ref), f"k mismatch B={B} L={L} N={N}"
    assert torch.equal(ns["v_BLNH"], v_ref), f"v mismatch B={B} L={L} N={N}"


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
        "q_BLNH": torch.zeros(1, 100, 128),
        "k_BLNH": torch.zeros(1, 100, 128),
        "v_BLNH": torch.zeros(1, 100, 128),
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
