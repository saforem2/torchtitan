# SophiaG: a RECURRENT grad-norm blow-up at 30B

**Seen:** 2026-08-25, job 12473783 (30B, GBS=960, constant LR 3.55e-05).
**Status:** RECURRENT. 2 of 2 runs from the same seed diverged, at different
steps. An earlier version of this document concluded the opposite -- see
"Correction" at the bottom for why that was wrong.

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

## The re-run diverged too, 128 steps later

The re-run cleared step 1048 and kept training normally for 128 more steps --
then blew up on its own schedule:

| step | loss | grad_norm |
|---:|---:|---:|
| 1173 | 3.722 | 0.67 |
| 1174 | 3.732 | **2.94** |
| 1176 | 3.747 | **37.4** |
| 1180 | 3.989 | 19,648 |
| 1182 | 4.280 | 85,081 |
| **1189** | -- | **204,016** |
| 1219 | 8.831 | 467 |

Same signature as the original: a few steps of quiet ramp (0.67 -> 2.9 -> 5.0),
then four orders of magnitude. The peak was TWICE the original's, and the loss
ended up worse (9.47 vs 8.16).

## It is a regime flip, not a spike (added 2026-08-25)

Quantified across the four arms, counting post-warmup steps (step > 20) whose
grad_norm exceeds 2.0:

| arm | steps | grad_norm > 2.0 | max grad_norm |
|---|---:|---:|---:|
| AdamW | 568 | **0** | **0.8** |
| Mano | 562 | **0** | **0.8** |
| SophiaG | 565 | 365 | 100,611 |
| SophiaG re-run | 379 | 135 | 204,017 |

The healthy optimizers never come within a factor of 2.5 of the threshold in
~1,100 combined steps. There is no overlap between the two populations.

Splitting SophiaG at its onset step shows the behavior is bimodal rather than
a degradation:

| window | steps with grad_norm > 2.0 | max |
|---|---:|---:|
| SophiaG, steps 21-1047 (pre-onset) | **0 / 147** | 0.921 |
| SophiaG, steps 1049+ (post-onset) | 364 / 417 (**87%**) | 100,611 |
| re-run, steps 1176+ (post-onset) | 131 / 204 (**64%**) | 204,017 |

Before onset SophiaG is indistinguishable from AdamW and Mano. After onset it
sits in a persistent high-gradient regime and does not leave it.

**Consequence: apparent "recovery" is an artifact.** Watching the loss alone,
the re-run looked like it was recovering twice (7.15 -> 4.56, then 5.50 -> 4.29).
Both were dips inside the bad regime, not returns to the good one -- grad_norm
stayed 30-315 throughout. Any read of these arms must use grad_norm, not loss.

**The onset is a discontinuity, not a ramp.** Arm 1 held grad_norm 0.35-0.52
for the 18 steps before onset, then went 0.39 -> 2.41 -> 89.9 -> 437 -> 1702
while the loss moved only 3.957 -> 4.089. A 4,000x gradient change under a
0.13-nat loss change is not the LR being slightly too high.

**This is why the LR-finder sweep did not predict it.** The SophiaG sweep
descends smoothly and monotonically across the whole low band -- 11.079 at
8.8e-6 down to 10.025 at 4.1e-5, no instability anywhere near the 3.55e-5 the
arms actually ran at, and the measured blow-up is 10x higher at 3.55e-4. A
static LR sweep probes the loss surface in the first ~100 steps; it cannot see
a state-dependent failure that arms after 1,000 steps of Hessian accumulation.

**Therefore lowering the LR is a weak fix.** The obvious next move -- restart
at ~1.2e-05 -- rests on the assumption that this is LR-driven. The evidence
above is against that: the failure is bimodal, state-dependent, and invisible
to the LR sweep. A low-LR arm may well delay onset (smaller steps accumulate
curvature error more slowly) but nothing here predicts it prevents it, and a
clean 2,000-step low-LR run would not prove much either -- onset was at 1,048
and 1,176, so a run that merely goes further is consistent with "delayed".

What would actually discriminate: instrument SophiaG's Hessian-estimate norm
(and its update clipping) per step and check whether the estimate degrades
monotonically before onset. If it does, the fix is in the Hessian update
(rho, the EMA, or the clipping), not the LR.

## What this means

The blow-up is **recurrent, not a one-off**: 2 of 2 runs from the same seed
diverged, with near-identical severity, at different steps (1048 and 1176).

That is worse than either earlier reading. It is not a rare event that luck
avoids, and it is not tied to a specific batch of data -- if it were, the
re-run would have fired at 1048 where the data order was nearly identical.
SophiaG at lr=3.55e-05 on this model **will** diverge; only the timing is
unpredictable.

Phase 1 did show SophiaG with the **narrowest usable LR band** of the three
(blow-up at 3.55e-04, 10x above its suggestion), which is why "the LR is too
close to the cliff, lower it" was the first reading. The regime-flip evidence
above argues against that: the sweep descends smoothly through the entire low
band with no instability near 3.55e-05, and the onset is a discontinuity under
a nearly flat loss rather than the gradual degradation a too-high LR produces.
Lower LR may delay onset without preventing it. Instrumenting the Hessian
estimate discriminates the two; a low-LR arm alone does not.

## Correction

The first version of this document concluded the blow-up was "not
deterministic" on the strength of the re-run clearing step 1048. That was a
badly scoped test: it watched a SPECIFIC STEP and the live tail stopped at
1085, so it could only ever answer "did it fire in the same place", not "does
it fire at all". The right question was the latter, and the answer is yes.

The lesson is about the experiment design, not the optimizer: when testing
whether a failure reproduces, run past the window you expect it in, and define
the stopping condition by the BEHAVIOUR (no divergence for N steps) rather
than by the step number where it happened last time.



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

Validated against four real logs. It fires on BOTH divergences -- step 1049 in
the original (8 steps before the peak) and step 1176 in the re-run (13 steps
before its 204,016 peak) -- and stays clean on adamw and mano. Given the
failure is recurrent rather than rare, this guard is not optional for any
SophiaG run at this LR.

```bash
--grad-norm-abort=20.0        # off by default (0.0)
```

## If you hit this again

1. `grad_norm`, not loss, is the early signal -- loss lags by several steps.
2. Check the other arms at the same wall-clock before blaming the machine.
3. The pre-spike checkpoint is usually intact; fork from it rather than
   restarting, and carry the optimizer state.
