# 12474810: a gradient instability developing at a defensible LR

**In progress.** This run was built to test whether `lm_head`'s dp-invariance
survives leaving the near-uniform output regime. It is instead showing
gradient growth, which is more interesting -- and the first such event in this
investigation at an LR nobody can dismiss.

## The sequence

| step | loss | `grad_norm_preclip` | `layer_gradnorm_max` | skew | lr |
|------|------|---------------------|----------------------|------|-----|
| 8 | 12.0666 | 8.670 | 0.240405 | 125.4 | 3.0e-7 |
| 9 | 11.8355 | 8.108 | 0.204738 | 111.2 | 3.5e-7 |
| 10 | 11.6605 | 8.848 **(+9%)** | **0.481659 (+135%)** | 209.4 | 4.0e-7 |
| 11 | 11.5809 | **11.748 (+33%)** | 0.512776 | 220.8 | 4.5e-7 |

## The per-layer signal led the global one by a full step

At step 10, `layer_gradnorm_max` more than doubled while `grad_norm_preclip`
moved **9%**. Anyone watching the standard metric would have seen a wiggle.
The global norm only caught up at step 11, jumping 33% after ten steps pinned
near 8.1-8.8.

Over the run so far: **preclip +43%, layer_max +95%.**

This is exactly what `diagnostics/__init__.py` was written to catch -- its own
comment says one exploding block raises max/mean long before the global L2
notices -- and it is the first time that decoupling has been *observed* here
rather than argued for.

## Why the LR matters

The LR at the onset was **4.0e-7**, ramping under a 20-step warmup toward a
peak of 1e-6. The 80B guide documents a usable AdamW ceiling of **~7.4e-7**,
so this is growth *below* the documented ceiling, during warmup, at an LR that
took job `12473149` cleanly from loss 12.95 to 8.098.

That is what distinguishes it from most events in this investigation's
history: it cannot be dismissed as an over-driven learning rate.

## Current state

- **Zero non-finite events.**
- Loss still falling but **decelerating**: per-step drops 0.23, 0.17, 0.08.
- `clip_headroom` tightening: 0.115 -> 0.085.
- Clipping fires every step, as always, so the optimizer still sees norm 1.0.

## Step 12: the global norm doubles, and the excursion is concentrated

| step | loss | preclip | max | mean | skew |
|------|------|---------|-----|------|------|
| 9 | 11.8355 | 8.108 | 0.204738 | 0.00184192 | 111.2 |
| 10 | 11.6605 | 8.848 | 0.481659 | 0.00230051 | 209.4 |
| 11 | 11.5809 | 11.748 | 0.512776 | 0.00232208 | 220.8 |
| 12 | 11.4945 | **17.246** | 0.433992 | 0.00219622 | 197.6 |

Since step 9: **preclip +113%, max +112%, mean +19%.**

Preclip has more than doubled -- from a baseline pinned near 8.1 for ten steps
to **17.25** -- and it tracks the per-layer max almost exactly. The typical
layer has barely moved. So the global L2 is being driven by a handful of large
contributors rather than by broad growth: **the excursion is concentrated, not
diffuse.**

(An earlier reading here said the growth had "spread beyond the single block",
because max dipped at step 12 while preclip rose. Over the excursion as a
whole the two are matched to within 1%. The dip was one step of scatter.)

## Which tensors carry it

`lm_head.weight` holds rank 0 at every step of the excursion. The runners-up
change character:

| steps | top1 / top2 |
|-------|-------------|
| before | `attention.wo`, mid-stack layers |
| 10-11 | **`layers.0.feed_forward.w2` and `w3`** -- the FIRST block's FFN |
| 12 | `attention.wo` at layers **83 and 77** -- the LAST blocks |

The disturbance appears at the output head, recruits the input-side FFN, then
shows at the far end of the stack. This is the first per-tensor view of an
excursion in progress in this investigation -- previous events were visible
only as a grad_norm spike after the fact.

## State at step 12

- **Zero non-finite events.**
- Loss still descending: 11.5809 -> 11.4945.
- `clip_headroom` 0.085 -> **0.058** and tightening.
- LR 5.0e-7, still ramping toward 1e-6 under a 20-step warmup, still below the
  documented ~7.4e-7 ceiling.

## Step 13: a SECOND phase -- the growth broadens, and neither max nor mean sees it

| step | preclip | max | mean | max/preclip | mean/preclip |
|------|---------|-----|------|-------------|--------------|
| 9 | 8.108 | 0.204738 | 0.00184192 | 0.0253 | 0.000227 |
| 10 | 8.848 | 0.481659 | 0.00230051 | **0.0544** | 0.000260 |
| 11 | 11.748 | 0.512776 | 0.00232208 | 0.0436 | 0.000198 |
| 12 | 17.246 | 0.433992 | 0.00219622 | 0.0252 | 0.000127 |
| 13 | **22.996** | 0.376449 | 0.00208607 | **0.0164** | 0.000091 |

Since step 9: **preclip +184%, max +84%, mean +13%.**

**The two phases are different, and this is the correction to the previous
section.** Steps 10-11 were concentrated: max and preclip rose together and
`max/preclip` jumped to 0.0544. From step 12 the global norm keeps climbing
while the largest tensor *falls* -- `max/preclip` collapses 0.0544 -> 0.0164,
a 3.3x drop, and `mean/preclip` falls too (0.000260 -> 0.000091).

So the L2 is now being driven by tensors that are neither the maximum nor
typical: **a broad middle band lifting together, which neither summary
statistic reports.** Watching max alone would show recovery; watching mean
alone would show almost nothing; only the global norm sees it.

That is the mirror image of the step-10 lesson. There, the per-layer max led
the global norm by a step. Here, the global norm is leading and the per-layer
max is actively misleading. **Neither metric dominates -- they catch different
phases, and the pair is what carries the information.**

## State at step 13

- **Zero non-finite events.**
- Loss still descending, though slowly: 11.4945 -> 11.4454.
- `clip_headroom` 0.058 -> **0.043**, roughly halving every two steps.
- preclip 8.1 -> 23.0 over four steps, ~3x baseline.
- LR ~5.5e-7, still ramping toward 1e-6, still under the documented ~7.4e-7.

At this rate the clip headroom reaches zero within a few steps. What happens
there is the interesting part: clipping has absorbed every step so far, and
the optimizer has seen norm 1.0 throughout.

## Step 15: ESCALATION, loss reversal, and an inverted excursion

| step | loss | preclip | max | mean | max/preclip |
|------|------|---------|-----|------|-------------|
| 10 | 11.6605 | 8.85 | 0.481659 | 0.00230051 | **0.0544** |
| 13 | 11.4454 | 23.00 | 0.376449 | 0.00208607 | 0.0164 |
| 14 | 11.3909 | 25.95 | 0.351594 | 0.00206609 | 0.0136 |
| 15 | **11.5990** | **48.05** | 0.140967 | 0.00169613 | **0.0029** |

- **Loss reversed** for the first time in the run: 11.3909 -> 11.5990.
- **preclip +85% in one step**, to ~6x the 8.1 baseline.
- **`clip_headroom` 0.039 -> 0.021**, near saturation.
- **Concentration collapsed 19x**: `max/preclip` 0.0544 -> 0.0029.

### The excursion inverted

It began as one tensor dominating (step 10: max doubles, global norm barely
moves) and has become **the entire field lifting while the formerly dominant
tensor shrinks** -- `lm_head` is down to 0.141 from a peak of 0.513, even
though it still holds rank 0. The mean fell too. So the L2 growth is coming
from the bulk of the distribution, not from its extremes.

That is a different failure shape from "one block explodes", and it is the
shape no previous 80B run could have revealed: earlier events are recorded
only as a `grad_norm` spike after the fact.

### A prediction I got wrong one step earlier

At step 14 I read the increments (+2.9, +5.5, +5.8, **+3.0**) as deceleration
and said recovery was a live possibility. Step 15 was **+22.1**. The
"deceleration" was a single lower increment inside a rising sequence -- the
same read-a-trend-off-three-points error that has recurred all session, and
the reason the resolution watch was set to fire on thresholds rather than on
my reading of the slope.

### Still no non-finite gradient

Zero, at step 15, with clipping absorbing every step and the optimizer seeing
norm 1.0 throughout. Headroom at 0.021 means clipping is close to its limit,
and what happens at saturation is the open question.

## What to watch

If this reaches a non-finite gradient, the capture instrumentation fires on a
**real** failure at a defensible LR and names the tensor -- the measurement
this investigation has been trying to obtain since the beginning. The watch on
it alerts on non-finite, loss reversal, or clip saturation.

If it recovers instead, that is also worth having: it would be a documented
near-miss with full per-layer telemetry, showing what the approach to
instability looks like from the inside.

## Caveat

One run. The 80B's failures are documented as intermittent (steps 9, 17, 18,
37 across different jobs), so a single trajectory is one draw. What is not a
draw is the *decoupling* -- the per-layer max leading the global norm by a
step is a property of the metrics, and it will hold whether or not this
particular run fails.
