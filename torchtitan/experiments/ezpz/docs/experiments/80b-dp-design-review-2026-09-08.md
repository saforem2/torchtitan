# Adversarial review of the dp experiment: what it caught, what it missed

Four independent critics (confounding / statistical power / mechanism /
feasibility) reviewed the dp-bracket design before nodes were spent. Recording
the outcome because the hit rate is instructive in both directions.

## The headline objection was already fixed -- and found the same way

> "The capture instrumentation CANNOT name a tensor, and I proved it by
> execution, not argument. `collect_param_stats` runs AFTER `clip_grad_norm_`
> has already applied clip_coef=0 ... Python's `max()` and `sorted(key=-v)`
> silently swallow NaN, so even pre-clip a NaN planted among 760 tensors
> surfaces ZERO non-finite keys."

**Correct for the code as of `3c17d3b55~1`. Already fixed in `3c17d3b55`.**
The critic identified exactly the two defects that commit addressed -- the
NaN-swallowing sort and the missing companion name key -- and identified them
by the same method, driving the function with a poisoned gradient rather than
reading it.

Verified against the current code at 760 tensors, both pre- and post-clip,
with the offender at indices 3/5/100/400/700/759:

```
PRE-clip   idx 3..700 : named=['diag/top0_gradnorm_layer']  n_layers_nonfinite=1.0
POST-clip  idx 5,400,759: total_norm=inf grad_absmax=0.0 top0=nan
                          named=['diag/top0_gradnorm_layer']
```

The fix works because non-finite values now sort FIRST
(`key=(0 if isfinite else 1, v), reverse=True`, `diagnostics/__init__.py:156`)
instead of being dropped, and the name is emitted as
`diag/topN_gradnorm_layer` (line 168) -- the "companion key" the old comment
promised and never wrote.

Two independent parties reaching the same defect by the same method is a
reasonable signal that both the defect and the method were real.

## The objection that was wrong, and why

> "Run the cheap arm first: job 12474403's L84 arm ran at dp=192 and logged 8
> non-finite events across 60 steps, at 1/32 the cost per step."

**`12474403` is a VOID run.** It is named in
[`80b-nan-what-we-know.md:137`](../guides/known-bugs/80b-nan-what-we-know.md)
as having run ~1000x past the LR ceiling. Its 8 "non-finite events" are
default-LR blowups, not the phenomenon under investigation. Firing the capture
on them would name a tensor for the wrong failure.

This is the same error made with `8540102` in the opposite direction: the
critic did not know the run was void, and cited it as evidence. Provenance of
a cited job matters as much as its numbers.

## The objections that stand

**1. This was never a ladder.** With 3 dp per node, GAS is integral only where
`6144 % (3N) == 0`, i.e. N in {1,2,4,8,16,32,64,128}. Between the done-and-clean
32N and the unreachable 128N there is exactly **one** rung. Calling four points
a bracket implied an interpolable threshold that the geometry does not support.
The run that went out is a single 96-vs-192 comparison, which is the honest
description.

**2. No checkpointing.** `12474803` runs `--checkpoint.no-enable`, so a node
loss returns nothing. Tolerable for a 75-minute run; it would have been a
serious flaw in the 21-hour GBS=6144 version that was under consideration, and
that is the version the critic was reviewing.

**3. Single runs are weak evidence against an intermittent failure.** Recorded
failures land at steps 9, 17, 18 and 37; SophiaG divergences at 1048/1176/1071.
One clean run at a given configuration is one draw. Any "safe" claim needs
either multiple seeds or a survival-analysis framing, and neither is in the
current design.

## Net

The review cost about ten minutes of wall time and returned one already-fixed
critical, one wrong recommendation resting on a void run, and three sound
methodological objections -- two of which (the non-ladder, the single-draw
weakness) directly changed how the results here are written up.
