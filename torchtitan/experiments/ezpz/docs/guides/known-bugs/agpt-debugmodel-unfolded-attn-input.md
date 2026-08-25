# agpt_debugmodel hands attention `[rows, N, H]`, not the folded `[T, N, H]`

> **Canonical record** for this failure mode. Found 2026-08-25 while smoke-testing
> after the grain install. **Production is NOT affected** -- see [Verdict](#verdict).
> The guard that catches this is doing its job; do not "fix" it by relaxing the
> divisibility check.

## Verdict

**`agpt_debugmodel` cannot currently run the smoke.** Every rank dies in the
first forward with:

```
ValueError: token count 2 is not a multiple of max_context_length 8192;
this wrapper assumes the fixed-length rows ConcatThenSplitPacking emits
and cannot reshape a ragged batch
```

**Production (`agpt_2b`, `agpt_20b`) is unaffected, and that is measured twice
over.** The wrapper's premise -- that a folded tensor's dim 0 is a whole number
of `max_context_length` rows -- holds for production and does not hold for the
debugmodel:

| config | `max_context_length` | `num_tokens_per_microbatch_per_dp_rank` | rows | folded `T` should be |
|---|---:|---:|---:|---:|
| `agpt_2b` | 8192 | 8192 | **1** | 8192 |
| `agpt_debugmodel` | 8192 | 16384 | **2** | 16384 |

MEASURED independently on 4 A100s (Perlmutter 57536150, `agpt_2b_real`):
`q=(8192, 16, 128)` at `seq_len=8192`, i.e. `T = 1 x max_context_length`
exactly. A 4-step train on the same config is clean (57536357, loss
12.90 -> 12.01, 48% MFU, rc=0). `tests/test_attn_unflatten.py` passes 5/5,
including `test_ragged_batch_raises`.

## What is actually wrong

The debugmodel should present `T = 2 x 8192 = 16384`. The wrapper saw **`T = 2`**
-- off by exactly `max_context_length`. So the tensor arriving is
`[rows, N, H]`, not `[rows * max_context_length, N, H]`: **the batch dim was
never folded into the token dim for this config**, or was folded by a path that
does not match #4121's layout.

That is a layout mismatch in the debugmodel's data path, not an arithmetic
problem in the wrapper.

## Why NOT to pin `--training.max-context-length=512`

This was tried (commit `a0f17f66b`, reverted). It makes the smoke pass the
check and is WRONG:

```
16384 / 512 = 32     # divisibility satisfied
```

...while the tensor still has 2 rows. The wrapper would then reshape a
`[2, N, H]` tensor against a 32-row grid. On 3D input a bare `transpose(1, 2)`
swaps N with H, **SDPA accepts it**, and the symptom is a quietly degraded loss
curve with no traceback -- exactly the failure the guard exists to prevent.
Relaxing the divisor converts a loud, correct refusal into a silent wrong
answer.

## Fix direction

Establish which is true before changing code:

1. the debugmodel's dataloader does not pack (so `ConcatThenSplitPacking` never
   runs, and dim 0 is rows), or
2. it packs but the fold happens somewhere the wrapper does not see.

If (1), the wrapper needs an explicit non-folded branch rather than a looser
check. If (2), the fold belongs upstream of attention. Either way the guard
stays.

## Related

- `blendcorpus-fold-batch-dim.md` -- the #4121 fold on the production path
- `tests/test_attn_unflatten.py` -- the regression test, and the docstring
  recording the measured production shape
- `fw-rc-compile-sdpa-backward-tp4.md` -- unrelated, different surface
