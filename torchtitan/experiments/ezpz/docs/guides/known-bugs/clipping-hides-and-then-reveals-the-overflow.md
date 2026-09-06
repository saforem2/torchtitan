# Gradient clipping runs on every 80B step, and it changes what a NaN means

Two facts established 2026-09-06, both by running code rather than reading it.

## 1. Clipping fires on 100% of steps, so the reported grad_norm is pre-clip

Job `12474733`, 5 steps at SophiaG lr=1e-6, production GBS:

```
diag/clip_fired         = 1.0     (every step)
diag/grad_norm_preclip  ~ 8.07
diag/grad_norm_postclip = 1.0
```

`max_norm = 1.0` is the core default (`config/configs.py:66`) and
`agpt_80b` does not override it, so this holds for **every** 80B run,
including `8574385`.

**Consequence for the documented failure.** `8574385`'s signature is recorded
as "flat grad_norm ~6.17 before the step-14 inf". That 6.17 is the PRE-clip
norm. The optimizer saw 1.0 the whole time. So "grad_norm was flat and then it
exploded" describes a quantity the model never experienced -- the flatness is
partly an artifact of reporting the input to a clamp whose output was pinned.

Any claim about 80B stability that rests on grad_norm being steady has to say
which side of the clip it means.

## 2. One inf zeroes the whole model -- except the tensor that overflowed

`clip_grad_norm_` is called with `error_if_nonfinite` at its default of
`False` (`distributed/utils.py:571`), so an overflow does not raise. It
computes `scale = max_norm / (total_norm + eps)`, and with `total_norm = inf`
that scale is **0**. Driving it directly:

```
total_norm: inf

  0.weight: allzero=True  finite=True   absmax=0.0
  1.weight: allzero=False finite=False  absmax=nan    <- the poisoned tensor
  2.weight: allzero=True  finite=True   absmax=0.0
```

Every finite gradient is multiplied to exactly 0.0. The offending tensor
becomes **nan**, because `inf * 0 = nan`.

Two things follow.

**The step was already a no-op before the trainer skipped it.** The non-finite
branch in `trainer.py` logs "SKIPPING the optimizer step to avoid writing NaN
into the weights"; by that point clipping has zeroed everything anyway. The
skip is still correct -- it avoids optimizer-state updates and a nan
propagating through momentum -- but the weights were never at risk from the
gradients themselves. This is why an 80B run can hit a non-finite step and
*recover*, as `12474403`'s L72 arm did at step 56.

**Post-clip, the culprit is uniquely identifiable.** After clipping, exactly
one tensor is non-finite and every other is exactly 0.0. A capture that runs
AFTER `clip_grad_norm_` (which is where ours runs) can find the site by
looking for the only tensor that is not zero -- a stronger signal than
comparing gradient norms, and one that does not depend on the norms being
distinguishable.

## Why this was not obvious

`clip_fired` has been in the diagnostics all along. I reported "grad_norm flat
~8.07, zero non-finite, healthy" for five consecutive steps without reading
the line directly beneath it that said the number was being clamped 8x on
every one of them.

Related: [`warmup-clamp-silently-voids-short-reproductions.md`](warmup-clamp-silently-voids-short-reproductions.md)
-- the same shape, where the trainer banner reports a warmup the scheduler had
already overridden.
