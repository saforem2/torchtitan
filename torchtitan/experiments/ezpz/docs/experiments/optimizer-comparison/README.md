# Fixed-batch optimizer comparison: AdamW vs Mano vs SophiaG

**Model:** agpt 30B (26.2B params), OLMo-2 tokenizer (100,352 vocab), seq 4096.
**Status:** LR finders launched 2026-08-23 (jobs 12473711/12/13).

## What this replaces, and why

The previous Mano run (64N, GBS=15360, jobs 12473683 + 12473698) was cancelled
because its result could not answer the question it was asked. Two defects,
both in how the LR was scheduled rather than in the optimizer:

1. **The LR was decaying throughout.** `decay_ratio=0.8` was inherited from
   the shared `agpt` config. `--training.steps` was chosen as a walltime
   CEILING, but the scheduler reads it as the schedule HORIZON, so a number
   picked for the timeout silently set the decay pace. At step 140 the LR was
   at 1.125e-04 -- **37.5% of the nominal 3e-4** -- so most of the run was not
   at the LR it was labelled with.

2. **The resume moved the LR discontinuously.** The continuation set
   `steps=400` where the parent had `steps=200`, which reshapes the schedule.
   At the resume point the LR jumped **1.83x upward** (1.406e-04 -> 2.578e-04).
   The loss bump at resume was first attributed to data-order; it was the LR.

Both are avoided below by holding the LR **constant after warmup**.

## Design

Every arm is identical except the optimizer and its LR:

| | |
|---|---|
| global batch | **GBS=960** (LBS 5 x 192 ranks x GAS 1), 3.93M tok/step |
| nodes | 16N per arm, **3 arms concurrently** = 48N |
| LR schedule | warmup 20 steps, then **constant** (`decay_ratio=0.0`) |
| step rate | ~42.7 s/step (measured 8N/GAS=1; per-rank throughput is flat 8N->64N) |

**Why GBS=960.** It sits near the AdamW baseline's GBS=480 -- the one existing
30B loss curve worth comparing against -- while still being a realistic shape.
It is deliberately NOT the 15360 production batch: three arms at that size
would not fit concurrently, and serial arms invite exactly the kind of drift
this experiment exists to rule out.

**Why constant LR.** It makes the optimizer the only variable and it is
resume-safe: a chained job cannot reshape the schedule, which is defect (2)
above. The cost is that final loss will be modestly worse than a decayed run,
so absolute numbers are NOT comparable to the decayed AdamW baseline. Only the
three arms are comparable to each other, which is the actual question.

## Phase 1: LR finders (running)

Each optimizer gets its own finder **at GBS=960**, because the suggested LR is
strongly batch- and optimizer-dependent. Measured for Mano alone:

| batch | suggested LR |
|---|---:|
| 2B config | 4.79e-03 |
| 30B, GBS=480 | 1.29e-04 |
| 80B, GBS=6144 | ~3e-06 |

Three orders of magnitude for the *same optimizer*. An inherited LR would turn
the comparison into "which optimizer got a luckier constant."

```bash
qsub -v OPT=adamw   -N lrf-adamw   lrfind_opt.pbs
qsub -v OPT=mano    -N lrf-mano    lrfind_opt.pbs
qsub -v OPT=sophiag -N lrf-sophiag lrfind_opt.pbs
```

One parameterized script, not three copies: three copies drift, and a fix
applied to one leaves the others measuring something else.

Sweep 1e-6 -> 1e-1 over 100 steps, warmup_fraction=0.1. The upper bound is
far above any plausible answer on purpose -- a sweep that never blows up
produces **no suggestion at all**, so bracketing divergence is required.

Expect SophiaG to find the ceiling: the 2026-07-03 80B SophiaG run NaN'd at
step 14 and burned ~12h. That is what the blow-up detection is for.

## Phase 2: comparison runs (not yet launched)

Each arm runs at its own finder-suggested LR, constant after warmup, to a
shared token budget. At 3.93M tok/step an 8h job yields ~2.3B tokens, so the
budget will need chaining -- and every chain link keeps the same constant LR,
which is the property that makes chaining safe here.

Comparisons are **per-token**, never per-step or per-wallclock.

## Reading the results

`rc=124` on a finder means the inner timeout fired before the sweep completed;
the result is incomplete, not a finding. The job summary prints the reported
GAS (must be 1) and the resolved optimizer name, so a config that silently
failed to take cannot pass unnoticed.
