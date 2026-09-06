# 12474733: five clean steps that are not as clean as they look

**`Exit_status = 0`, 5 steps in 3h20m, zero non-finite events, 32N on
access-verified hosts, SophiaG lr=1e-6, GBS 25,165,824 (GAS 64).**

| step | loss | grad_norm |
|------|------|-----------|
| 1 | 12.95721 | 8.0676 |
| 2 | 12.94267 | 8.0293 |
| 3 | 12.91178 | 8.0689 |
| 4 | 12.86637 | 8.0323 |
| 5 | 12.80435 | 8.0753 |

Read as "loss descends, grad_norm flat, no NaN", this is an unremarkable
stability datapoint. The diagnostics say otherwise, and all three findings
below come from lines I had been skipping.

## 1. Clipping fires on 100% of steps

```
diag/clip_fired = 1.0          (every step)
diag/grad_norm_preclip  ~ 8.07
diag/grad_norm_postclip = 1.0
```

**The flat ~8.07 grad_norm is the PRE-clip value.** The optimizer sees 1.0 on
every step. So this run does not show that the 80B is stable at lr=1e-6; it
shows that clipping is absorbing a steady ~8x overshoot and the model never
experiences the raw gradient.

That matters for how the original failure should be read: if `8574385` was
also clipping continuously, its "flat grad_norm ~6.17 before the step-14 inf"
describes the clipped view, not the gradients that actually overflowed.

## 2. Most parameters are frozen, and thaw as the LR ramps

```
n_params_frozen:        464 -> 398 -> 369 -> 321      (of 759)
update_ratio_median:    0.0 -> 0.0 -> 6.5e-08 -> 9.2e-08
```

At step 1, **61% of parameters were not moving at all** and the MEDIAN
parameter received exactly zero update. This is consistent with a warmup LR of
2.5e-08 being below the resolution at which bf16 weights change -- the update
rounds away -- and the count falls monotonically as the LR climbs.

Worth knowing before reading any early-step result: for the first steps of a
warmup at this scale, most of the model is not training. `update_ratio_median
= 0.0` would also be the signature of a genuinely frozen layer (the bf16
RMSNorm freeze this metric was written for), so the two need distinguishing by
whether the count decreases as LR rises. Here it does.

## 3. One layer carries 44x the gradient of the next, stably

```
diag/layer_gradnorm_skew = 217.3, 216.9, 218.1, 217.1, 216.7
diag/top0_gradnorm       = 0.2504      diag/top1_gradnorm = 0.0057
```

The skew is `max/mean` and it does not drift -- five steps within 1.4 of each
other. A single tensor holds ~44x the gradient norm of the runner-up and ~217x
the mean.

The diagnostics module's own comment calls this "the single most diagnostic
number here: one exploding block raises max/mean long before the global norm
(which is an L2 over everything) moves enough to notice." Under a rising LR,
that block is the candidate for what goes first.

**This log does not name it.** `diag/top0_gradnorm_layer` is absent -- the job
launched before commit `3c17d3b55`, which added the companion key. So the most
promising lead available is a number without an address, which is precisely
the gap that fix closes. The next run names it.

## What this run does and does not establish

- **Does:** the 80B trains without non-finite gradients for 5 steps at
  lr=1e-6 under SophiaG at production GBS on this stack.
- **Does not:** show stability of the raw gradients, since clipping intervened
  on every step.
- **Does not** reproduce `8574385` -- its warmup was clamped to 40 steps, so
  its LR trajectory is ~116x the original's. See
  [`known-bugs/warmup-clamp-silently-voids-short-reproductions.md`](../guides/known-bugs/warmup-clamp-silently-voids-short-reproductions.md).

## Next

Rerun with the fixed instrumentation and read `diag/top0_gradnorm_layer` at
step 1. If the dominant block is identifiable at a safe LR, it can be watched
approaching the failure rather than autopsied after it.
