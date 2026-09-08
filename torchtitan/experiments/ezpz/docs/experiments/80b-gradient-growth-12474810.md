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
