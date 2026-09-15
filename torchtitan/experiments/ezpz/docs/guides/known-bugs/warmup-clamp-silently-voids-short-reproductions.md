# A short run silently rewrites your LR schedule

**torchtitan clamps `warmup_steps` to `total_steps` and only WARNS.**
`torchtitan/components/optimizer/lr_scheduler.py:107-112`:

```
Warmup steps (4650) exceed total steps (40). Adjusting warmup steps to 40.
```

Nothing fails. The run proceeds, the flag you passed is still in the command
line, and `--lr-scheduler.warmup-steps=4650` appears in the log exactly as you
typed it. But the schedule the model actually sees is a 40-step ramp.

## Why this voids a reproduction

Under a linear warmup the effective LR at step N is `peak * N / warmup`. Shrink
the warmup and every step gets a different LR than the run you are copying:

| step | original (warmup 4650) | 40-step run | ratio |
|------|------------------------|-------------|-------|
| 1  | 2.151e-10 | 2.500e-08 | 116x |
| 10 | 2.151e-09 | 2.500e-07 | 116x |
| 18 | 3.871e-09 | 4.500e-07 | 116x |

Job `8574385` NaN'd at step 18. A 40-step "reproduction" reaches step 18 at
**116x** the original's LR -- so a NaN there is evidence about the fast ramp,
not about the bug. And since the failure being chased is an instability, the
confound points the same direction as the hypothesis, which is the worst case:
you would see what you expected and be wrong.

## The number this exposed

`8574385` died at an effective LR of **~3.9e-9** -- four orders of magnitude
BELOW the ~7.4e-7 ceiling `docs/guides/training/agpt_80b.md` documents. The
80B's real failure is not an over-large learning rate. That is why this
configuration, and not the `lr=8e-4` runs that were void on their own terms,
is the one worth instrumenting.

Reading the flag would never have shown this. It took reading the effective
schedule the flag produced.

## The banner lies on the very next line

Caught live, job `12474740`, two consecutive log lines:

```
[W][optimizer/lr_scheduler:108:build] Warmup steps (4650) exceed total steps (25). Adjusting warmup steps to 25.
[I][ezpz/trainer:852:__init__] Trainer is initialized with ... total steps 25 (warmup 4650)
```

**The trainer banner reports `warmup 4650` AFTER the scheduler clamped it to
25.** The banner prints the config object; the scheduler holds the schedule.
Reading the banner -- the natural place to check -- confirms the wrong number.

## Shortening a run makes the clamp WORSE, not better

The guard is `if warmup_steps > total_steps` (`lr_scheduler.py:105-112`), so
every short run is clamped and a shorter one is clamped harder:

| total_steps | warmup becomes | lr@18 | vs original |
|-------------|----------------|-------|-------------|
| 25   | 25   | 7.200e-07 | 186x |
| 40   | 40   | 4.500e-07 | 116x |
| 500  | 500  | 3.600e-08 | 9x   |
| 4650 | 4650 | 3.871e-09 | 1x (the original) |

I built a "matched" 25-step run believing a longer `warmup` flag would survive
if the run were short enough. It is the exact inverse. Matching the original
warmup needs `total_steps >= ~2325`, which at ~34 min/step is **~55 days**.

## The fix: rescale the peak, do not fight the clamp

The schedule is only a means to an LR. Under linear warmup
`lr(n) = peak * n / warmup`, so scaling the peak by the same factor as the
warmup reproduces the trajectory exactly:

| | peak | warmup | lr@18 |
|---|------|--------|-------|
| original `8574385` | 1.000000e-06 | 4650 | 3.8710e-09 |
| rescaled | **5.376344e-09** | 25 | 3.8710e-09 |

`5.376344e-09 = 1e-6 * 25/4650`. Verified equal at steps 1, 5, 10, 15, 18, 20,
25 to **2.2e-16**. Same trajectory in 25 steps instead of 4650. See
`80b_capture_rescaled.pbs`.

The nominal `lr` in that run's log reads `5.376344e-09`, not `1e-6`. Anything
checking for the original value would flag it as wrong, so the void guard
checks for the rescaled value instead.

## What to do

- **Keep the original warmup and shorten by `--training.steps` only** if the
  step you care about is still reachable. For the 80B at GBS 25,165,824 a step
  costs ~34 min, so 25 steps needs a 12h window to clear step 18 with margin.
- **Never truncate warmup to fit a walltime.** Reduce node count instead and
  move the compensation into GAS -- see
  `sunspot-home-mount-check-offlines-nodes.md`.
- **Make the analysis assert it.** `80b_capture_matched.pbs` greps its own log
  for `Adjusting warmup steps to` and prints VOID with the ratio, so a clamped
  run cannot be read as a reproduction later.

The general shape: a config that is *silently adjusted* rather than rejected
leaves a log line and a correct-looking command line behind. Grep the log for
what the framework decided, not just for what you asked. Related:
`project_silent_noop_exit_zero`.
