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

## RESULT: the dominant layer is `lm_head.weight`

Step 1 of the valid reproduction, first time this has ever been named:

```
top0  lm_head.weight                                   0.2520
top1  layers.65..attention.wo.weight                   0.005693     <- 44x smaller
top2  layers.36..attention.wo.weight                   0.005658
top3  layers.57..attention.wo.weight                   0.005624
top4  layers.5..attention.wo.weight                    0.005574

layer_gradnorm_skew = 217.40      clip_fired = 1.0
grad_norm_preclip   = 8.066       lr = 2.1505376e-10
```

`lm_head.weight` -- the output projection -- carries **44x** the gradient norm
of the next-largest tensor and ~217x the mean.

### Why this matters

**It is not in the transformer stack.** Every standing hypothesis about the
80B instability targets attention scores inside the blocks: softcap, QK-norm,
depth effects. The gradient mass is concentrated somewhere none of those
interventions reach. If the failure originates here, the two routes on the
standing list were never going to bound it.

**Ranks 1-4 are depth-independent.** They are all `attention.wo` from layers
65, 36, 57 and 5 -- scattered through the stack -- and their norms span 2%
(0.005693 to 0.005574). That is a flat population, which is evidence against
the depth-scaling story that three unreplicated points once suggested (see
[`known-bugs/80b-nan-what-we-know.md`](../guides/known-bugs/80b-nan-what-we-know.md)).

**`lm_head` is where bf16 range pressure would concentrate.** It is the widest
matmul in the model -- hidden x vocab -- and the only place a full-vocabulary
logit tensor exists. A 256k-vocab head is the natural candidate for the
largest intermediate magnitudes in the graph.

**Naming ambiguity to watch.** Production 2B/20B checkpoints spell this tensor
`output.weight` while current code wants `lm_head.weight` (see
`project_output_weight_lm_head_rename`). Same tensor. Anyone grepping older
logs or checkpoints for the culprit needs both spellings.

### The whole ranking is frozen, not just the top

Five steps of `12474761`, every rank identical every time:

| rank | tensor | gradnorm | spread over 5 steps |
|------|--------|----------|---------------------|
| top0 | `lm_head.weight` | 0.2520 | 0.5% |
| top1 | `layers.65..attention.wo.weight` | 0.005693 | -- |
| top2 | `layers.36..attention.wo.weight` | 0.005658 | -- |
| top3 | `layers.57..attention.wo.weight` | 0.005624 | -- |
| top4 | `layers.5..attention.wo.weight` | 0.005574 | -- |

**No rotation at any rank across any step**, even though ranks 1-4 sit within
**2%** of one another. Values that close would swap order constantly under
sampling noise. They do not, so the ordering is a structural property of the
model, not a measurement artifact.

This corrects a looser reading earlier in this document: calling ranks 1-4 a
"flat, depth-independent population" suggests they are interchangeable. They
are not -- each holds a fixed position. What is depth-independent is the
*pattern*: layers 65, 36, 57 and 5 are scattered through an 84-layer stack
with no monotonic trend, so whatever fixes their relative magnitudes is not
depth.

Two things this rules in and out:

- **Not sampling noise.** A 2% spread that never reorders across 5
  independent steps is deterministic structure.
- **Not depth ordering.** 65 > 36 > 57 > 5 is not monotonic in either
  direction.

Worth checking on the successor, which runs ~40 steps: whether the ranking
survives as the LR climbs toward the failure point, and whether the tensor
that eventually goes non-finite is `lm_head` (the dominant one) or one of the
fixed runners-up. **A mismatch would be the more informative outcome** -- it
would mean the overflow starts somewhere that was never carrying much
gradient, which no magnitude-based hypothesis predicts.

### What this does NOT establish

This is the gradient distribution at a **healthy** step, lr 2.15e-10, no
overflow. It says where the gradient mass sits, not that this tensor is what
goes non-finite first. Those coincide only if the failure is a magnitude
effect in the dominant tensor, which is a hypothesis, not a result.

The run continues toward step 18. If a capture fires, it names the tensor
directly and the two can be compared. **A mismatch would be the more
interesting outcome** -- it would mean the overflow starts somewhere that was
never carrying much gradient.

## OUTCOME: step 18 passed clean

`12474765`, the run whose LR trajectory matches `8574385` at every step:

```
step 14  loss 12.94641  grad_norm 8.0754      <- original logged its first inf here
step 15  loss 12.94461  grad_norm 8.0347
step 16  loss 12.94192  grad_norm 8.0412
step 17  loss 12.93991  grad_norm 8.0512
step 18  loss 12.93713  grad_norm 8.0249      <- original NaN'd here
```

**18 steps, ZERO non-finite events.** Effective LR at step 18 read
`3.87096768e-09` against the intended `3.871e-09` -- the trajectory
reproduction was exact to 9 significant figures. `lm_head.weight` held top0
throughout at 0.252 +/- 0.4%.

### What this licenses, per the criteria fixed at step 15

This is the **weak** outcome, and the pre-registered reading applies
unchanged: it is consistent with *"dp is the variable and 96 is safe"* and
equally consistent with *"the failure is stochastic and this seed missed"*.

**It does NOT show the LR trajectory is safe.** dp here is 96 against
`8574385`'s ~1530 -- a 16x gap that could not be closed at 32 nodes while
holding GBS fixed. The confound stated before the outcome remains
unresolved by this run.

What it does establish, narrowly:

- The 80B completes 18 steps at production GBS on this stack with SophiaG at
  the exact LR trajectory that killed `8574385`, when dp is 96.
- So the failure is **not** a deterministic function of (LR trajectory, GBS,
  optimizer, step count) alone. At least one further variable matters --
  dp and seed being the obvious candidates.

That is a real narrowing. Before this run, the LR trajectory could not be
excluded as sufficient on its own; now it can, at this dp.

### Next

`12474768` is queued (`afterany`), same 32 nodes, same dp=96, same LR
trajectory, GBS dropped 16x to 384 seqs. If both arms behave alike, GBS is not
the variable at this dp -- which would leave dp itself, or the seed, as what
distinguishes this clean run from `8574385`.

## The one axis this reproduction does NOT match: dp

Written at step 15, before the outcome is known, so it cannot be shaped by it.

| | `8574385` | `12474761` / `12474765` |
|---|---|---|
| nodes | 510 | 32 |
| **dp** | **~1530** | **96** |
| GAS | 4 | 64 |
| GBS | 6120 seqs | 6144 seqs (0.4% high) |
| peak LR / warmup | 1e-6 / 4650 | 5.376344e-09 / 25 |
| effective LR at step n | identical to 2.2e-16 | identical |

Holding GBS fixed on 32 nodes requires trading dp for GAS -- there is no other
way to reach 25M tokens/step at that node count. So dp is **16x lower** here.

**The canonical doc lists the failure as dp-sensitive, in the ESTABLISHED
column.** If dp is the operative variable rather than batch size, this
configuration was never capable of reproducing the failure, and a clean run
proves nothing about the mechanism. I should have stated that when building
the run rather than after step 14 passed.

### What each outcome would and would not license

- **NaN at or near step 18.** Strong: the trajectory matched, the failure
  followed, and the capture names the tensor. dp being 16x lower would then
  argue the failure is NOT primarily dp-driven.
- **Clean through step 18.** Weak on its own. It is consistent with "dp is
  the variable and 96 is safe", and equally consistent with "the failure is
  stochastic and this seed missed". It does NOT show the LR trajectory is
  safe, because the confound is unresolved.
- **Clean through ~40 steps.** Somewhat stronger -- it would make a
  stochastic miss less likely -- but still cannot separate dp from batch.

### The experiment that would separate them

Two arms at the SAME dp, differing only in GBS, both at this LR trajectory.
That isolates batch from parallelism. At 32N the reachable dp is 96, so the
arms would be GAS 64 (GBS 6144) against GAS 4 (GBS 384). Cheap: the second
arm is 16x less compute per step.

Until that runs, "GBS not dp" and "dp not GBS" remain unseparated -- which is
exactly the state the 2026-08-31 investigation left them in, for the same
reason (see the void arms in
[`known-bugs/80b-nan-what-we-know.md`](../guides/known-bugs/80b-nan-what-we-know.md)).

## Config

32N on access-verified hosts, 384 ranks, TP=4, GAS 64, GBS 25,165,824 tokens,
SophiaG, `--diagnostics-per-layer --diagnostics-interval=1`, 25 steps, 12h
walltime. Steps cost ~34 min at this GBS, so 25 steps needs ~14h -- the run
will be cut short by walltime around step 20, which still clears the step-18
death point with margin.
