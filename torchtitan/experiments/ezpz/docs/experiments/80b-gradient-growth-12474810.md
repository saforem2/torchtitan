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

## THE ANSWER: the divergence tracks the LR ramp into the documented ceiling

Adding the LR column makes the whole excursion legible:

| step | **lr** | loss | preclip | max | max/preclip |
|------|--------|------|---------|-----|-------------|
| 9 | 3.5e-7 | 11.8355 | 8.11 | 0.204738 | 0.0253 |
| 10 | 4.0e-7 | 11.6605 | 8.85 | 0.481659 | 0.0544 |
| 12 | 5.0e-7 | 11.4945 | 17.25 | 0.433992 | 0.0252 |
| 14 | 6.0e-7 | **11.3909** (min) | 25.95 | 0.351594 | 0.0136 |
| 15 | **6.5e-7** | 11.5990 | 48.05 | 0.140967 | 0.0029 |
| 16 | **7.0e-7** | 12.0158 | 73.16 | 0.073690 | **0.0010** |

**Loss bottoms at step 14 (lr 6.0e-7) and rises +0.62 by step 16.** Gradients
reach **9x** baseline. This is a warmup ramp walking straight into the
**~7.4e-7 usable ceiling** that `agpt_80b.md` documents -- and it fails
essentially on arrival.

### Why this is worth more than the mechanism test it replaced

The ceiling was previously established from runs *started* above it
(`8530891` NaN'd at step 2 at lr=1e-6). This is the first observation of the
80B **walking into the ceiling from below under a warmup ramp**, with
per-tensor telemetry the whole way. It is an independent confirmation from a
direction nobody had tested, and it says the ceiling is a property of the LR
rather than of how the run reached it.

### The shape of the approach

Concentration **collapses** as the LR rises: `max/preclip` 0.0544 -> 0.0010, a
54x fall. `lm_head` peaks at 0.513 (step 11) and ends at 0.074 -- a 7x
*decrease* while the global norm grows 8x. So approaching the ceiling does not
look like one tensor exploding; it looks like **the entire gradient field
lifting off together**, with the previously dominant tensor becoming
relatively insignificant.

That is the opposite of what a "one exploding block" model predicts, and it is
why watching `layer_gradnorm_max` alone would have shown *improvement*
throughout the divergence.

### Still no non-finite gradient

Zero at step 16, with `clip_headroom` at 0.0137. Clipping has absorbed every
step and the optimizer has seen norm 1.0 throughout -- which is presumably why
the loss degrades gradually rather than NaN-ing outright.

## Correction: `clip_headroom` is `1/preclip`, and there is no saturation event

Verified empirically at every step:

```
step 12  headroom 0.05798   1/preclip 0.05798
step 13  headroom 0.04349   1/preclip 0.04349
step 16  headroom 0.01367   1/preclip 0.01367
```

`clip_headroom` **is** the clip scale factor. It cannot reach zero -- it
asymptotes as the gradient grows. So the repeated framing above ("headroom
tightening", "near saturation", "what happens when clipping can't absorb it")
is wrong. **Nothing runs out.** A headroom of 0.014 means the gradient is
being scaled down 73x, and clipping will keep scaling by whatever factor is
required, indefinitely.

### What this changes about the failure mode

This is a better account of why the loss degrades *gradually* instead of
NaN-ing. The optimizer always receives a gradient of norm exactly 1.0 -- that
is what clipping guarantees. What degrades is the **direction**: as one part
of the field grows 9x relative to the rest, the unit-norm vector handed to the
optimizer points more and more at whatever is blowing up, and less at the loss
gradient.

So the model is not being destroyed by large updates. It is being steered by
increasingly bad ones of constant size. That predicts exactly what is
observed: no non-finite values, no sudden collapse, loss rising smoothly by
+0.62 over two steps while gradients grow 9x.

It also means the run may never produce a non-finite gradient at all -- and
that the capture instrumentation, which only fires on non-finite, may be
watching for the wrong event in this regime.

## Two things the full-precision data shows, and one I got backwards

### The weights barely move at all

`weight_norm_global` varies by **1.08e-06 relative** across the entire run
(4967.650433 .. 4967.655816). `update_ratio_mean` runs ~1e-6 to 2.7e-6 --
**three orders of magnitude below** the ~1e-3 this repo's own diagnostics call
healthy.

So the divergence is not the weights being blown apart. Through step 17 the
parameters have hardly changed; what has changed is the *gradient field* being
computed from them. Combined with the clipping analysis above -- the optimizer
receives norm exactly 1.0 every step -- the picture is a model whose weights
are nearly static while the direction it is being pushed degrades.

### Parameter activation tracks the divergence

| step | lr | frozen (of 759) | active | preclip | loss |
|------|----|-----------------|--------|---------|------|
| 1 | 1.0e-6* | 333 | 56.1% | 8.22 | 12.9443 |
| 7 | 2.5e-7 | 154 | 79.7% | 8.35 | 12.3033 |
| 11 | 4.5e-7 | 128 | **83.1%** | 11.75 | 11.5809 |
| 14 | 6.0e-7 | 83 | 89.1% | 25.95 | 11.3909 |
| 16 | 7.0e-7 | 54 | **92.9%** | 73.16 | 12.0158 |

(*step 1 shows the pre-warmup value before the schedule takes over.)

Activation climbs from 56% to 93% as the LR ramps, and the gradient growth
begins around step 11 where activation crosses ~83%. Whether that is cause or
co-symptom is not separable here -- both track the LR.

### The mistake

I first read this column as "92% of the model was frozen early on, and the LR
woke it up". Backwards: the field is `n_params_frozen`, so 333 at step 1 is
**56% active**, not 8%. The model was mostly training from the start, and the
trend is the opposite of what I described -- fewer parameters frozen over
time, not more waking up from near-total dormancy.

The corrected version is a cleaner correlation and needs no dramatic framing.

## Step 18: 68% of all training progress given back, with no NaN

| step | lr | loss | preclip |
|------|-----|------|---------|
| 1 | -- | 12.9443 | 8.22 |
| **14** | 6.0e-7 | **11.3909** (min) | 25.95 |
| 15 | 6.5e-7 | 11.5990 | 48.05 |
| 16 | 7.0e-7 | 12.0158 | 73.16 |
| 17 | 7.5e-7 | 12.2274 | 61.09 |
| 18 | **8.0e-7** | **12.4424** | 66.62 |

```
progress made:  1.5534   (12.9443 -> 11.3909)
progress lost:  1.0515   (11.3909 -> 12.4424)
                = 68% given back in four steps
```

The LR is now **8.0e-7**, past the documented ~7.4e-7 ceiling, and still
ramping to 1e-6 at step 20.

Preclip has stopped escalating (73 -> 61 -> 67) and settled into a high,
unstable plateau while the loss climbs steadily. So the failure mode is not
runaway gradient growth -- it is **sustained misdirection**: gradients pinned
at ~8x baseline, clipped to norm 1.0, pointing somewhere that undoes training.

## What this run establishes

**The ~7.4e-7 ceiling is real, and crossing it destroys training without ever
producing a NaN.** Eighteen steps, zero non-finite gradients, weights that
moved by 1e-6 relative -- and 68% of the run's progress erased.

That reframes what "the 80B failure" means. The documented failures are NaN
events, and the capture instrumentation built for them fires only on
non-finite values. This run shows a **failure mode that instrumentation cannot
see at all**: no NaN, no inf, no skipped step, nothing for the capture to
catch, and the model quietly walking backwards.

If production runs have been crossing the ceiling during warmup, this is what
it looks like from the metrics -- and only `grad_norm_preclip` and the loss
show it. `layer_gradnorm_max` *fell* 7x through the whole event.

## Step 20: it RECOVERS -- at a higher LR than the divergence

| step | lr | loss | preclip |
|------|-----|------|---------|
| 14 | 6.0e-7 | 11.3909 (min) | 25.95 |
| 16 | 7.0e-7 | 12.0158 | 73.16 |
| 18 | 8.0e-7 | **12.4424** (worst) | 66.62 |
| 19 | 8.5e-7 | 12.3768 | 68.06 |
| 20 | **9.0e-7** | **11.7424** | **38.87** |

Step 20 is the largest single-step loss drop of the run (**-0.63**) with
preclip falling 43% at the same time. Both metrics improving together, two
steps running.

**And the LR is higher than it was during the divergence.** The worst of the
damage happened at 6.5e-7 to 8.0e-7; the recovery is happening at 9.0e-7 --
well past the ~7.4e-7 ceiling.

### This complicates the ceiling reading recorded above

The earlier section here concluded that "the ~7.4e-7 ceiling is real and
crossing it destroys training". That is now too strong. The model crossed the
ceiling at step 17 and is training *better* at step 20, at a higher LR still.

A reading that fits all the data: **the damage tracked the RATE of LR
increase, not the level.** Every step of the divergence added 5e-8 to the LR;
the model was chasing a moving target. What changed at step 20 is not the
level -- it is that the loss surface had time to catch up. That is a
hypothesis, and the clean test is a fixed-LR run at 9e-7, which this run
cannot provide.

The honest status of the ceiling claim: **`8530891` NaN-ing at step 2 at
lr=1e-6 remains the evidence for it. This run does not confirm it and does not
refute it** -- it shows a transient degradation during a ramp through that
region, followed by recovery above it.

I recorded the stronger claim two steps earlier. It was the natural reading of
four monotone points, and the fifth point broke it -- the same pattern as the
four functional forms and the step-14 deceleration call.

### What survives unchanged

- 68% of progress was given back, transiently, with **zero non-finite
  gradients** -- a failure mode the NaN-triggered capture cannot see.
- `layer_gradnorm_max` fell 7x through the entire event while
  `grad_norm_preclip` rose 9x: the per-layer max is actively misleading here.
- Weights moved by 1e-6 relative throughout. Nothing was destroyed; the
  direction degraded and then improved.

## THE COMPLETE EXCURSION: a reversible 7-step transient

| step | lr | loss | preclip | layer_max |
|------|-----|------|---------|-----------|
| 14 | 6.0e-7 | **11.3909** min | 25.95 | 0.351594 |
| 15 | 6.5e-7 | 11.5990 | 48.05 | 0.140967 |
| 16 | 7.0e-7 | 12.0158 | **73.16** peak | **0.073690** min |
| 17 | 7.5e-7 | 12.2274 | 61.09 | 0.135083 |
| 18 | 8.0e-7 | **12.4424** worst | 66.62 | 0.179820 |
| 19 | 8.5e-7 | 12.3768 | 68.06 | 0.165771 |
| 20 | 9.0e-7 | 11.7424 | 38.87 | 0.228469 |
| 21 | 9.5e-7 | **11.4395** | 22.79 | 0.299819 |

**Onset step 15, recovered by step 21. Seven steps. Zero non-finite
gradients.** Loss returned to within 0.05 of its pre-excursion minimum while
the LR rose 58% above where the trouble began.

### `layer_gradnorm_max` traces a perfect V, anti-correlated with the global norm

`layer_max` bottoms at **0.0737** exactly where preclip peaks at **73.16**,
and recovers in lockstep as preclip falls. The two are anti-correlated across
the *entire* event, not merely at onset.

So the concentration did not collapse into the bulk and stay there. It swung
out and came back. Whatever redistributes gradient away from `lm_head` during
the excursion also reverses.

This is worth stating plainly because it inverts the intuitive model twice
over: a metric named "max" *falling* is the signature of trouble here, and
that metric recovering is the signature of health.

### What this run is

**The first complete, reversible, fully-instrumented gradient excursion in
this investigation.** Every prior event is recorded as a NaN or a grad_norm
spike, seen after the fact. This one has per-tensor telemetry through onset,
peak, and recovery.

What it demonstrates:

1. The 80B can lose **68% of its training progress** and recover it, with no
   non-finite value at any point.
2. The NaN-triggered capture would not have fired once. An entire class of
   degradation is invisible to it.
3. `grad_norm_preclip` and loss are the only two metrics that showed the event
   correctly. `layer_gradnorm_max` was anti-correlated -- watching it alone
   would have reported improvement at the worst moment and trouble during
   recovery.

### What it does not settle

Whether this transient is related to the documented NaN failures at all. It
may be the same mechanism caught in a survivable regime, or an unrelated
warmup artifact. The run continues to step 150 and the LR is about to stop
ramping, which will separate "the ramp caused it" from "that LR region caused
it".

## A SECOND excursion is starting -- at CONSTANT LR

| step | lr | loss | preclip | layer_max |
|------|-----|------|---------|-----------|
| 22 | 1.0e-6 | 11.3480 | 16.18 | 0.305022 |
| 26 | 1.0e-6 | 10.5492 | 12.60 | 0.360582 |
| 27 | 1.0e-6 | 10.3590 | 14.96 | 0.414344 |
| 28 | 1.0e-6 | 10.3580 | **31.83** | **0.298656** |

Preclip doubled in one step while `layer_max` fell -- **the same
anti-correlated signature as the first excursion**, where preclip peaked at
73.16 exactly as layer_max bottomed at 0.0737.

### This tests the "rate not level" hypothesis, and it fails it

After the first excursion I proposed that the damage tracked the **rate** of
LR increase (5e-8 per step) rather than the level, because recovery happened
at a higher LR than the divergence.

**The LR reached its 1e-6 peak at step 22 and has been flat for seven steps.**
This second excursion is developing with no LR change at all. So the rate
cannot be the driver -- or at least, it is not necessary for one.

That leaves the level (1e-6 is above the documented ~7.4e-7 ceiling) or
something intrinsic to this phase of training. The first excursion cannot
distinguish those; this one is at least consistent with the level mattering.

### Status

Loss has stalled: 10.3590 -> 10.3580, essentially flat after seven steps of
steady descent. Preclip at 31.83 has not reached the 40 threshold set for a
"new excursion" alert, and the first excursion peaked at 73.2 -- so this may
be smaller, or may be one step into the same shape.

Worth watching but not yet worth a conclusion. The first excursion took four
steps to reach its peak.

## CONFIRMED: this is a REPEATING instability, not a warmup transient

Two excursions, near-identical in shape, onsets **13 steps apart**:

| | #1 (LR ramping) | #2 (LR **flat** at 1e-6) |
|---|---|---|
| onset | step 15 | step 28 |
| preclip at onset+1 | 48.05 | 57.97 |
| `layer_max` at onset+1 | 0.140967 | 0.123221 |
| loss | rising | rising |

The second excursion's step 29 reproduces the first's step 15 to within ~20%
on preclip and ~13% on `layer_max`, with the same anti-correlation.

### What this settles

**It is not a warmup artifact.** The LR has been flat at 1e-6 since step 22;
the second onset came six steps later. Whatever drives this recurs on its own.

**It is not the LR ramp rate.** That hypothesis, offered after the first
excursion, is dead -- there is no ramp during the second.

**It is periodic, or at least repeating.** 13 steps between onsets. One
interval is not a period, but it is now a quantity worth measuring, and the
run has ~120 steps left to measure it in.

### Why this matters beyond this run

The documented 80B failures are single NaN events at steps 9, 17, 18, 37 in
different jobs. If those are the tail of a *recurring* instability rather than
isolated accidents, then:

- the step number of a NaN is a draw from a repeating process, not a property
  of the configuration -- which would explain why the failure step has never
  been reproducible
- a run that "trains cleanly" may simply have sampled the quiet phase
- **the relevant measurement is the excursion RATE, not whether a given run
  failed** -- and this run is the first to make that measurable

That last point reframes every clean-run result in this investigation,
including tonight's five dp arms, all of which ran 40 steps or fewer at LRs
too small to leave the near-uniform regime.

Still **zero non-finite gradients** across both excursions.

## CORRECTION: not periodic -- an ATTENUATING sequence

A threshold-independent count (local peaks in `grad_norm_preclip` above 25,
rather than crossings of an arbitrary 40) gives four events, not two:

| event | step | peak preclip |
|-------|------|--------------|
| 1 | 16 | **73.16** |
| 2 | 19 | 68.06 |
| 3 | 29 | 57.97 |
| 4 | 41 | **36.18** |

```
peaks:  73 -> 68 -> 58 -> 36     monotonically attenuating
gaps:        3    10    12       lengthening, not constant
```

**This is not a periodic instability.** The "13 steps apart" and then "14
steps apart" figures I recorded both came from counting threshold crossings,
which merged the step-16 and step-19 peaks into a single event and produced a
spurious constant interval.

What the data actually shows is a **decaying sequence with lengthening
intervals** -- a transient settling out, not a recurring cycle. The fourth
event peaked at half the first and resolved without any loss damage: loss went
8.8716 -> 9.0161 -> **8.7764**, a new run minimum.

### What this does to the earlier reframing

The commit `eaec6c9fc` argued that if the documented NaN failures are draws
from a *repeating* process, that would explain why the failure step was never
reproducible. **That argument is weakened.** A decaying transient is much less
likely to produce a NaN at step 37 of one run and step 9 of another; a
stationary repeating process would.

What survives: this run had four gradient events of decreasing severity, all
survivable, none producing a non-finite value. Whether the documented NaN
failures are the same phenomenon caught at a worse phase, or something else
entirely, remains unresolved -- and the attenuation makes the "same
phenomenon" reading harder to sustain, not easier.

### The methodological point

Both the "period" and its disappearance came from the same data. The
difference was the event definition: a fixed threshold merges nearby peaks and
invents regularity. Counting local maxima instead is threshold-free and gave
the real structure immediately.

## FINAL: 64 steps, loss 12.94 -> 8.07, zero non-finite, four attenuating events

`Exit_status = 0` at 9h30m -- stopped by the inner `timeout 34200`, not by
instability. 64 of a planned 150 steps.

```
loss     12.9443 -> 8.0747 (min)     no non-finite gradients in 64 steps
events   step 16 (73.2), 19 (68.1), 29 (58.0), 41 (36.2)
peaks    73 -> 68 -> 58 -> 36        attenuating
gaps          3    10    12          lengthening
last 23 steps: max preclip 14.6      quiet, no further events
```

The attenuation reading holds to the end. Four events, decreasing in
magnitude, then nothing for the final third of the run.

**Efficiency note:** this reached loss 8.0747 in 63 steps at dp=24 (8 nodes).
Job `12473149` reached 8.098 in 120 steps at dp=192 (64 nodes). Comparable
loss, half the steps, an eighth of the hardware -- though the two differ in
precision settings (`12473149` ran fp32 activations) so this is not a clean
throughput comparison.

## The mechanism test, final answer: refuted by a factor of three

Quiet steps only (`preclip < 25`, all four events excluded):

| | loss | skew |
|---|------|------|
| early (steps 1-8) | 12.6193 | **131.0** |
| late (final 5) | 8.1240 | **361.0** |
| change | **-4.50** | **+175.5%** |

**Predicted: skew falls materially as the model leaves the near-uniform
regime. Observed: it nearly TRIPLED.**

The model ended 4.3 loss units below `ln(256128) = 12.45` -- decisively out of
the uniform regime -- and `lm_head`'s dominance grew by a factor of 2.75.

This is the strongest form of the refutation. Earlier measurements gave +12.5%
(contaminated by excursion steps) and +32.3% (quiet steps, loss 9.77). At loss
8.12 it is +175%. **The further the model gets from a uniform output
distribution, the more `lm_head` dominates** -- the exact opposite of the
proposed mechanism, and a monotone relationship rather than a marginal one.

Whatever explains `lm_head`'s dp-invariance has to explain this too: its
dominance is not a warmup artifact, it is a property that strengthens as the
model learns.

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
