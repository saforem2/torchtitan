# The 80B NaN

> [!CAUTION]
> **EVERY EXPERIMENT IN THIS DOCUMENT RAN AT THE WRONG LEARNING RATE.**
> Jobs `12474403` (depth bisect) and `12474423` (GAS sweep) used
> `agpt_80b`'s config default of **8e-4**. The 80B training guide
> ([`guides/training/agpt_80b.md`](../training/agpt_80b.md)) documents the
> AdamW NaN onset at production batch as **1.36e-6**, with a usable ceiling of
> **~7.4e-7** -- so these runs were roughly **1000x past the last stable
> point**. The guide even records job `8530891` NaN-ing at step 2 at `1e-6`,
> which is already 800x below what I used.
>
> **So the step-2 blow-up characterized throughout this document is expected
> behaviour at that LR, not a discovery.** Invalidated: the depth bisect's
> 0/1/8 event counts, the "GBS not dp" conclusion, the frozen-at-43
> characterization, and the accumulation analysis. All of it measured an 80B
> driven far past its stable LR.
>
> The guide also already contains the batch relationship, in the direction
> opposite to what I concluded: *"it NaN'd at GBS=192 but runs all 15 finder
> steps finite at GBS=6144, where the larger batch smooths the Hessian
> estimate."*
>
> **What survives, because it does not depend on my runs:**
> - The three refutations of the documented root cause (below). They come from
>   that diagnosis's OWN evidence and from arithmetic -- same exponent range,
>   fp32-residual failing its own test, and a "masking" claim that describes an
>   impossible operation.
> - The magnitude measurements. `qk_q_absmax_local` = 61.2 across every arm and
>   36 orders below the bf16 ceiling. Activation scale does not depend on the
>   LR being correct, and the same value appears in the clean-regime run at
>   dp=12 (`12472477`, 62.25).
>
> **I should have read the training guide before launching an 80B job.** The
> LR ceiling is in it, stated plainly, with the failing job id.
: an accelerating rate of transient non-finite gradients

> 2026-08-31. Job `12474403`, the depth bisect. **This supersedes the
> "bf16 forward-activation overflow" diagnosis**, which is refuted on three
> independent grounds recorded below.

## The result

Three arms, identical in everything but depth: same dp_shard=192, TP=4,
world_size=768, seed 42, data, and per-layer geometry (dim 9216, ffn 25600,
vocab 256128). 60 steps each, uncompiled, per-layer diagnostics at interval 1.

| arm | layers | params | grad events | loss NaNs | final state |
|---|---:|---:|---:|---:|---|
| L48 | 48 | 48.2B | **0** | 0 | loss 6.54, healthy |
| L72 | 72 | 70.0B | **1** (step 55) | 0 | loss 6.80, healthy |
| L84 | 84 | 80.8B | **8** | **2** (40, 59) | loss 6.66, **healthy** |

> [!CAUTION]
> **NO ARM DIED, AND I REPORTED TWICE THAT ONE DID.** L84's loss went
> non-finite at step 40 and at step 59, and it recovered IMMEDIATELY both
> times -- step 41 reads loss 6.98 / grad_norm 7.42, step 60 reads loss 6.66 /
> grad_norm 3.73. I called step 40 terminal without reading step 41, then
> built a "two back-to-back events are fatal" mechanism on top of it. Both
> claims are withdrawn. **The production failure was NOT reproduced.**

W&B: [L48 `gvkphx8l`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/gvkphx8l)
· [L72 `lwwffftb`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/lwwffftb)
· [L84 `7m0bcmvr`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/7m0bcmvr)

**0 -> 1 -> 8, monotone in depth, with the deepest arm reproducing the
production failure in a controlled experiment.** This is the configuration in
which `8671243` NaN'd at step 19, `8673658` at ~37 and bf16 `12473142` at step
30, so the bisect ran inside the failure regime rather than beside it.

## The mechanism, and why the failure looked sudden

L84's events land at steps **2, 39, 40, 45, 49, 55, 58, 59**. The inter-event
gaps are:

```
37 -> 1 -> 5 -> 4 -> 6 -> 3 -> 1
```

A long quiet stretch, then events crowding together until the loss itself goes
non-finite. **That gap collapse is the runup everyone was looking for.** It
was always there, in a quantity nobody logged.

**WITHDRAWN: "the terminal step is the gap collapse".** I claimed the
back-to-back pair at 39, 40 was fatal. It was not -- step 41 recovered to loss
6.98 and the run finished healthy at step 60. The gap sequence is real; the
fatality attached to it was not, and the mechanism built on it is withdrawn.

What survives is weaker and still worth having: the model **absorbs** these
events. Every one of L84's 8 grad events and both of its loss NaNs was
followed by a normal step. Over 60 steps at this batch size, a rising event
rate does not compound into termination.

Each individual event is instantaneous: the gradient is non-finite for one
step, the trainer's guard skips the optimizer step
(`trainer.py:1063`), and training resumes. The aggregate pre-clip `grad_norm`
between events is unremarkable. So the documented signature -- "grad_norm
dead-flat ~6.0, then a sudden step to inf, no runup" -- is **an artifact of
watching the wrong quantity**. The degradation lives in the event RATE.

**How a run eventually dies is NOT established by this experiment.** Every
event here was absorbed. The bridge from "transient events at a
depth-dependent rate" to "production runs terminate" is inference, not
observation: 60 steps at this batch size was not enough, and the production
failures occurred at production batch -- a variable this bisect deliberately
held fixed.

## Why the previous explanations failed

**Not overflow.** MEASURED in the failing configuration:
`diag/qk_q_absmax_local` = **61.2 at all three depths**, and 62.25 at dp=12 in
the clean regime (`12472477`). Three configurations spanning 16x in dp and
1.75x in depth, and the activation peak does not move. The bf16 ceiling is
3.39e38 -- **36 orders of magnitude away**. `grad_absmax_local` peaks at 0.03.

Independently: bf16 max is 3.3895e38 against fp32's 3.4028e38, the SAME 8-bit
exponent. fp32 cannot fix an overflow because it has no meaningful extra
range; it buys mantissa (8 bits -> 24). And the "bf16 masks true grad_norms of
21K-79K down to ~5-7" claim describes an operation that does not exist: bf16
represents those values to 0.29%, and an actual overflow yields inf, which
propagates through a norm. No path maps a true 21309 to a reported 4.97.

**Not the residual stream.** fp32-ing the residual add -- the direct test of
the stated mechanism -- failed twice, but it did DELAY failure from step 19 to
step 37. Under the rate model that is exactly right: fp32 lowers the event
rate without removing the cause.

**Why fp32 activations work.** More mantissa bits make individual events
rarer. Job `12473149` ran 120/120 clean at production dp where bf16
(`12473142`) died at step 30. It is a rate reduction, not a range fix.

**Why depth matters.** Not accumulation toward a threshold -- more layers mean
more opportunities per step for a single tensor to go non-finite. The
per-layer data supports this: `layer_gradnorm_max` is 0.99 at BOTH 48 and 72
layers while the global norm scales 3.3x, so the global grows only because
more layers sum into it.

## What is still unknown

**Which tensor goes non-finite first.** The trainer zeroed the gradients
(`zero_grad`, line 1064) about 90 lines before the diagnostics ran, so the one
step that mattered was the only step with no data -- L72's step-55 event is
absent from W&B entirely while 51-54 and 56-57 logged normally. Fixed in
commit `f01548f6a`: `collect_param_stats(per_layer=True)` now runs inside the
non-finite branch before zeroing, and separately logs the metrics that are
themselves non-finite, which name the affected tensors. The next 80B run at
dp=192 will record it.

**Whether the rate model explains the dp dependence.** Depth is established;
dp is not tested by this bisect. If dp raises the rate the same way, the two
triggers unify.

## The 0/1/8 counts are SINGLE RUNS with no error bar

Each arm ran once. An event count of 8 versus 1 versus 0 could reflect depth,
or it could reflect ordinary run-to-run variance in a stochastic process --
nothing here bounds that. The monotone ordering across three points is
suggestive, not a measurement of a rate.

### REPLICATE RESULT: the event count is NOISY, and 0/1/8 was over-read

G1 finished. **2 events (steps 53, 59), 60 steps, zero loss NaNs** -- against
L84's **8** in the identical configuration, differing only in data order.

| run | config | events |
|---|---|---:|
| bisect L84 | 84L, dp=192, GAS=1 | 8 |
| GAS G1 | 84L, dp=192, GAS=1 | **2** |

**A 4x spread from run-to-run variance alone.** So the bisect's 0 / 1 / 8
across 48 / 72 / 84 layers cannot carry a monotone depth trend: if one
configuration spans 2-8, then L72's 1 and L84's 8 are not clearly
distinguishable from each other.

**What survives:** L48 produced ZERO events in 60 steps where both 84L runs
produced some (8 and 2). That is suggestive of a depth effect at the extremes
and is still a single run per depth.

**What does not:** any claim about the SHAPE of the depth relationship, the
"rate scales with depth" phrasing, or reading L84's early step-2 event as a
depth signature -- G1 had no early event at all, so that one was variance.

I recorded this limit before the data arrived rather than after, which is the
only reason this is a sharpening and not a retraction of something published
as settled. The lesson is the same one that produced the two withdrawn claims
above: a suggestive pattern over unreplicated points is a hypothesis, and I
stated it as a result.

**A replicate arrives for free.** Job `12474423`'s G1 arm is agpt_80b at
GAS=1, dp=192, 60 steps -- the same configuration as the bisect's L84 arm,
differing only in data order. If it produces ~8 events, the depth trend is
robust. If it produces 2 or 20, the single-run counts cannot carry the weight
the 0/1/8 table puts on them, and the depth claim needs restating with much
weaker language.

Noted before that result lands so it cannot look like a retrofit either way.
Related: G1's step-1 grad_norm is 7.96 against L84's 6.19 at the same step and
seed -- data order alone moves it, which is a reminder not to read small
between-run differences as signal.

## GBS, NOT dp: the non-recovering state reproduced at fixed dp (job 12474423)

The GAS sweep pins dp=192 and varies only the global batch. **In progress, but
the qualitative result is already unambiguous.**

| arm | GAS | GBS | events | behaviour |
|---|---:|---|---:|---|
| G1 | 1 | 192 seqs | 2 in 60 steps | **recovered immediately, every time** |
| G8 | 8 | 1,536 seqs | 4 CONSECUTIVE (steps 2-5) | **STUCK: loss frozen at ~43** |

G1: loss returns to ~7 within one step of every event, across 60 steps.
G8: loss jumps to 43.05 at step 2 -- 3.3x above initialization -- and stays
there (43.05 / 43.03 / 42.99 / 43.02) while every optimizer step is skipped.
The tiny drift is data variation through frozen weights.

**Same dp. Only the batch changed.** This is the first reproduction of a
non-recovering state in this investigation, and it appeared as soon as GBS
grew 8x at constant dp.

**That inverts the documented trigger.** "The wall is LBS>1 AND dp_degree >
~186" has stood for months, but every run supporting it moved dp and GBS
together. This is the first experiment to separate them, and the failure
follows GBS.

**The magnitudes hold up under the strongest possible test.** Inside the stuck
state, `qk_q_absmax_local` reads **61.5** -- the same value as every healthy
run today and at dp=12 in the clean regime. `grad_absmax_local` reads 0
because the gradients were already zeroed. Nothing is large even while the
model is completely stuck. Overflow is not what this is.

**Caveats.** Four steps, one arm, and the events begin at step 2 where warmup
transients live. Unlike the depth result this is a QUALITATIVE change rather
than a count against a noisy baseline -- G1 never showed anything like it in
60 steps -- but G32 at production batch (6,144 seqs) is the confirmation, and
the depth arms taught me what a single unreplicated run is worth.

## GAS SWEEP COMPLETE: both large-batch arms blow up at the FIRST update

| arm | GBS | step 1 | step 2 | outcome |
|---|---|---:|---|---|
| G1 | 192 seqs | 12.95 | normal | 60 steps, 2 events, **recovered from both** |
| G8 | 1,536 seqs | 12.95 | **43.05 + NaN** | frozen all 8 steps |
| G32 | 6,144 seqs | 12.95 | **46.20 + NaN** | same, ran out of steps |

All three start identically at ~12.95. **Both large-batch arms jump to 43-46
and go non-finite at step 2 -- the first optimizer step.** G1 never does this
in 60 steps.

**My report metric was wrong and its verdict should be ignored.** It computed
a per-microbatch event rate and concluded "larger batches are MORE stable per
unit work" because G32's rate is 0.0156 -- but G32 ran TWO STEPS and blew up
on the second. It did not survive; it ran out of steps. A rate normalizer
assumes events are scattered occurrences; this is a single deterministic
blow-up at the first update, which a rate cannot represent. The verdict logic
was tested against synthetic *rate* patterns and passed all five, which is
exactly why it did not catch this: I tested the metric, not whether the metric
was the right one.

**This also weighs against the accumulation-exposure explanation.** G32 uses
32 microbatches per step and G8 uses 8. If a poisoned accumulation buffer were
the mechanism, G32 should be ~4x worse. The two are indistinguishable -- same
step, same jump, same freeze. Exposure predicts a difference that is not
there.

The `80b_noaccum_control.pbs` arm (job `12474428`) still decides it directly:
same 6,291,456 tokens/step as G8, delivered as ONE microbatch instead of
eight.

## Design caveat

Fewer layers is also a smaller model -- less memory pressure, different FSDP
sharding. Depth is the intended variable and per-layer shape is identical
across arms, but "fewer layers" and "smaller model" are not fully separable in
this design. The monotone 0/1/8 across three points, with the terminal failure
at the depth that fails in production, is strong; it is not airtight.
