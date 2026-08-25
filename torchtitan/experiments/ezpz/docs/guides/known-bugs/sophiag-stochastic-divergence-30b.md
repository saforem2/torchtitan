# SophiaG: a non-deterministic grad-norm blow-up at 30B

**Seen:** 2026-08-25, job 12473783 (30B, GBS=960, constant LR 3.55e-05).
**Status:** reproduced as NOT reproducible -- see the controlled re-run below.

## What happened

At step 1048 the SophiaG arm of the optimizer comparison exploded:

| step | loss | grad_norm |
|---:|---:|---:|
| 1047 | 3.967 | 0.39 |
| 1048 | 3.957 | **2.41** |
| 1049 | 3.996 | **89.9** |
| 1051 | 4.089 | 1,702 |
| **1057** | 7.015 | **100,611** |

Five orders of magnitude in nine steps. The loss curve gives almost no warning
-- 3.957 at step 1048 looks entirely normal, and grad_norm is the only signal.

It recovered slowly (loss back to ~4.3 by step 1127) but never regained its
pre-spike best of 3.951, so every point after 1048 measures "SophiaG after a
blow-up" rather than SophiaG.

**Not the machine.** AdamW and Mano were at grad_norm 0.58 and 0.30 at the same
wall-clock, on the same nodes, same data, same batch. Only SophiaG diverged.

## The controlled re-run

Forked from the clean pre-spike `step-1000` checkpoint carrying the FULL
optimizer state:

```bash
qsub -v OPT=sophiag,LR=3.55e-5,SEED=$PWD/checkpoints/agpt-30b-optcmp-sophiag/step-1000 \
     optcmp_rerun.pbs
```

`--checkpoint.no-initial-load-model-only` is the load-bearing flag.
`initial_load_model_only` defaults to **True**, which loads weights but drops
the optimizer state -- that would have made this "fresh SophiaG from step-1000
weights", which probably would not diverge either, but for a trivial reason.
Carrying SophiaG's Hessian estimate is what makes a negative result mean
something.

Same weights, same optimizer state, same LR. Only the data order and RNG
differ. It tracked the original to five decimals at step 1001, then:

| step | re-run grad_norm | original grad_norm | ratio |
|---:|---:|---:|---:|
| 1048 | 0.41 | 2.41 | 6x |
| 1049 | 0.34 | 89.9 | **263x** |
| 1051 | 0.43 | 1,702 | 3,937x |
| 1057 | 0.52 | **100,611** | **195,000x** |

It passed straight through, and by step 1058 was at loss 3.912 -- BELOW the
original's pre-spike best.

## What this means

The blow-up is **not deterministic**. Identical weights and optimizer state,
and floating-point nondeterminism alone was enough to avoid it entirely.

That is the worse of the two possible answers. A reproducible failure at a
known step can be predicted, bisected, and guarded. A knife-edge one means the
optimizer was sitting on an instability the whole time and whether it fires is
luck.

It is consistent with the Phase 1 LR finder: SophiaG had the **narrowest usable
LR band** of the three (blow-up at 3.55e-04, only 10x above its suggestion,
against Mano's wider margin).

**n=1.** One re-run passing shows the failure is not deterministic; it does NOT
establish a rate. Several more forks from the same seed would give one, and the
script now exists to launch them cheaply.

## What changed in the code

`nan_abort_consecutive` never fired, correctly and uselessly: every value in
that sequence is finite. A grad-norm runaway guard was added
(`grad_norm_abort`, commit aeb91745b) that compares against the run's own
trailing median rather than a constant, because healthy grad_norm differs ~10x
across these three optimizers and drifts down over training.

It skips warmup, and that is load-bearing rather than defensive: on the healthy
arms EVERY grad_norm above 20x sits at step <= 19 (adamw 5,6,7,11 up to 82.3;
mano 6..19 up to 74.3). Without the skip the guard aborts two good runs.

Validated against four real logs: fires at step 1049 (eight steps before the
peak), clean on adamw, mano, and the re-run.

```bash
--grad-norm-abort=20.0        # off by default (0.0)
```

## If you hit this again

1. `grad_norm`, not loss, is the early signal -- loss lags by several steps.
2. Check the other arms at the same wall-clock before blaming the machine.
3. The pre-spike checkpoint is usually intact; fork from it rather than
   restarting, and carry the optimizer state.
