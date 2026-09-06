# 12474761: the first valid reproduction of 8574385

**Verified at launch, not assumed.** The scheduler build line and the trainer
banner agree, and no clamp fired:

```
banner: total steps 25 (warmup 25)
clamp:  NONE
lr:     5.376344e-09
```

Compare `12474740`, where the banner read `total steps 25 (warmup 4650)` on
the line immediately after the scheduler clamped warmup to 25. Here the two
agree, which is what makes the schedule trustworthy.

## Why this reproduces 8574385 and the earlier attempts did not

Under linear warmup `lr(n) = peak * n / warmup`, so rescaling peak and warmup
by the same factor leaves the trajectory unchanged:

| | peak | warmup | lr@18 |
|---|------|--------|-------|
| `8574385` (original) | 1.000000e-06 | 4650 | 3.8710e-09 |
| `12474761` (this run) | 5.376344e-09 | 25 | 3.8710e-09 |

`5.376344e-09 = 1e-6 * 25/4650`. Equal at steps 1, 5, 10, 15, 18, 20, 25 to
**2.2e-16**.

The two failed attempts, for contrast:

| run | warmup as built | lr@18 | vs original |
|-----|-----------------|-------|-------------|
| `12474733` | 40 (clamped) | 4.500e-07 | 116x |
| `12474740` | 25 (clamped) | 7.200e-07 | 186x |
| `12474761` | 25 (intended) | 3.871e-09 | **1.000x** |

Both earlier runs asked for warmup 4650 and were silently clamped, because the
guard fires whenever `warmup > total_steps` -- so shortening a run tightens the
clamp rather than avoiding it. See
[`known-bugs/warmup-clamp-silently-voids-short-reproductions.md`](../guides/known-bugs/warmup-clamp-silently-voids-short-reproductions.md).

## What it can find that earlier runs could not

It carries commit `3c17d3b55`. Before that, the non-finite capture could
detect an overflow but not name its site: `collect_param_stats` filtered
non-finite layer norms out of its per-layer stats and emitted
`diag/topN_gradnorm` with no companion key for the layer name. The capture
would have printed two global aggregates and nothing else.

Now it prints `THE TENSORS THAT WENT NON-FINITE: <name> (gradnorm=inf)`.

There is a second, independent way to identify the culprit in this run, from
the clipping analysis: after `clip_grad_norm_` sees a non-finite total norm,
`scale = max_norm/inf = 0` zeroes every FINITE gradient while the offending
tensor becomes `nan` (`inf * 0`). Post-clip, exactly one tensor is non-finite
and all others are exactly 0.0. See
[`known-bugs/clipping-hides-and-then-reveals-the-overflow.md`](../guides/known-bugs/clipping-hides-and-then-reveals-the-overflow.md).

## The lead to check first

`12474733` showed one layer holding **44x** the gradient norm of the next
(`top0 = 0.2504`, `top1 = 0.0057`) with skew pinned at ~217 across all five
steps. That log could not name it. This run can: read
`diag/top0_gradnorm_layer` at step 1.

If the dominant block is identifiable at a safe LR, it can be watched
approaching the failure rather than autopsied after it.

## Config

32N on access-verified hosts, 384 ranks, TP=4, GAS 64, GBS 25,165,824 tokens,
SophiaG, `--diagnostics-per-layer --diagnostics-interval=1`, 25 steps, 12h
walltime. Steps cost ~34 min at this GBS, so 25 steps needs ~14h -- the run
will be cut short by walltime around step 20, which still clears the step-18
death point with margin.
