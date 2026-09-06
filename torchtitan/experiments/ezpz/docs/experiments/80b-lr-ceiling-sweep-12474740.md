# 12474740: an accidental LR-ceiling sweep

**What it was meant to be:** a warmup-matched reproduction of `8574385`.
**What it is:** a 25-step ramp through the documented AdamW ceiling, with the
non-finite capture armed. Kept running on those grounds, not the original ones.

## How it got here

I set `--training.steps=25 --lr-scheduler.warmup-steps=4650` believing a long
warmup survives in a short run. It is the inverse: torchtitan clamps warmup to
`total_steps` whenever warmup exceeds it, so the run got warmup 25 and its LR
ramps 186x faster than the original. See
[`known-bugs/warmup-clamp-silently-voids-short-reproductions.md`](../guides/known-bugs/warmup-clamp-silently-voids-short-reproductions.md).

That makes it useless as a reproduction of `8574385` -- but it is not nothing.

## What it actually tests

With peak `1e-6` and warmup 25, `lr(n) = 1e-6 * n/25`:

| step | effective LR | |
|------|--------------|---|
| 1  | 4.000e-08 | |
| 10 | 4.000e-07 | |
| 18 | 7.200e-07 | |
| **19** | **7.600e-07** | **crosses the documented ~7.4e-7 AdamW ceiling** |
| 25 | 1.000e-06 | below the 1.36e-6 NaN onset |

So it sweeps from well under the ceiling to well over it, in one run, with
`--diagnostics --diagnostics-per-layer --diagnostics-interval=1` recording
per-layer gradient stats at every step.

**The question it answers:** does the 80B actually destabilise at the ceiling
the guide documents, under SophiaG? Both outcomes are informative:

- **Non-finite at/after step ~19** -- the ceiling is real and transfers to
  SophiaG. The capture then names the first tensor to go, which is the
  measurement that has never been taken.
- **Clean through step 25** -- the ~7.4e-7 ceiling is AdamW-specific and does
  not bound SophiaG, which would narrow what `8574385` could have been, since
  that run died at **3.87e-9** -- four orders of magnitude lower.

## What it is NOT

Not a reproduction of `8574385`. Nothing here licenses a claim about that job.
The genuine reproduction is `80b_capture_rescaled.pbs`, which rescales the peak
to `5.376344e-09` so `lr(n) = 5.376344e-09 * n/25` matches `1e-6 * n/4650`
exactly (verified to 2.2e-16).

## Status

Running as of 2026-09-06, 32N on access-verified hosts, GBS 25,165,824,
GAS 64. Watch for `NON-FINITE GRADIENT CAPTURE` lines and the step they land
on relative to 19.
