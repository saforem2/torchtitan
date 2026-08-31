# The 80B NaN: an accelerating rate of transient non-finite gradients

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

## Design caveat

Fewer layers is also a smaller model -- less memory pressure, different FSDP
sharding. Depth is the intended variable and per-layer shape is identical
across arms, but "fewer layers" and "smaller model" are not fully separable in
this design. The monotone 0/1/8 across three points, with the terminal failure
at the depth that fails in production, is strong; it is not airtight.
