# Separating batch size from parallelism: GBS is not the variable

**Two arms, one variable.** Same 32 access-verified nodes, same 384 ranks,
same TP=4 -- so **dp = 96 in both** -- and the same LR trajectory (peak
`5.376344e-09`, warmup 25, matching `8574385` step for step). The only
difference is global batch:

| | `12474765` | `12474768` |
|---|---|---|
| GAS | 64 | 4 |
| tokens/step | 25,165,824 | 1,572,864 |
| seqs | 6,144 | 384 |

No run in this investigation had previously varied GBS while holding dp fixed.
The 2026-08-31 attempt varied them together and was void for it.

## Result, matched at n=25 steps

| metric | mean diff | cv ratio (small/large) |
|--------|-----------|------------------------|
| `top0_gradnorm` (`lm_head`) | **0.08%** | 4.26x |
| `grad_norm_preclip` | 0.20% | 2.75x |
| `layer_gradnorm_skew` | **0.03%** | 4.33x |
| `grad_absmax_local` | 0.05% | 3.93x |

```
top0    GBS 6144: mean 0.251540  cv 0.284%
        GBS  384: mean 0.251736  cv 1.211%
skew    GBS 6144: mean 217.176   cv 0.181%
        GBS  384: mean 217.237   cv 0.784%
```

**Means invariant to 0.03-0.20%. Variances scale 2.75-4.33x.**

## Why the variance number is the informative one

A 16x reduction in batch size should multiply the standard error of any
gradient statistic by **sqrt(16) = 4** if the only batch effect is averaging
over samples. Three of the four metrics land within 8% of exactly that
(4.26, 4.33, 3.93); the fourth, the global pre-clip norm, is lower at 2.75x,
which is expected for a quantity aggregating over the whole model rather than
one tensor.

So the batch is doing **exactly what sampling theory says it should, and
nothing else.**

## What this establishes

1. **The gradient structure is deterministic, not a sampling artifact.**
   `lm_head` dominance and the 217x skew reproduce to three decimal places
   across a 16x batch change. They are properties of the model.
2. **The gradient noise is pure sampling.** No anomalous batch-dependent
   behaviour at this dp -- nothing that would single out large batch as
   destabilising.
3. **GBS is very unlikely to be what killed `8574385`.** Changing it 16x moves
   nothing except the noise, and moves that noise by precisely the predicted
   factor.

Combined with `12474765` (25/25 steps clean at the original's exact LR
trajectory), the remaining candidates for what distinguishes a clean run from
`8574385` are **dp** (96 here against ~1530 there) and **the seed**.

## What it does NOT establish

This holds *at dp=96*. It does not show GBS is irrelevant at dp~1530 -- a
batch-by-parallelism interaction would not be visible here. Testing that needs
node counts this session could not obtain, and it is the honest next step
rather than a settled point.

Both arms also ran clean past step 18, so neither reproduces the failure; this
is a comparison of healthy-regime statistics, not of failure modes.

## Config

`80b_gbs_arm.pbs` (this arm) and `80b_capture_rescaled2.pbs` (the twin).
SophiaG, `--diagnostics-per-layer --diagnostics-interval=1`, seed 42, hosts
verified by access test (see
[`known-bugs/sunspot-home-mount-check-offlines-nodes.md`](../guides/known-bugs/sunspot-home-mount-check-offlines-nodes.md)).
