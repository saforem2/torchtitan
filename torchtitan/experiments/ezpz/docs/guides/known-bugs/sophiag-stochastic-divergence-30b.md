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

| arm | post-warmup steps | grad_norm > 2.0 | max grad_norm |
|---|---:|---:|---:|
| AdamW | 1,984 | 54 (**2.7%**) | 14.6 |
| Mano | 1,983 | 128 (**6.5%**) | 30.0 |
| SophiaG | 1,524 | 573 (**37.6%**) | 100,611 |
| SophiaG re-run | 644 | 380 (**59.0%**) | 204,017 |

An earlier version of this table reported **0 excursions and max 0.8** for both
healthy arms. That was wrong: it was computed from a single chain link's
`train.log` rather than the whole chain, and each arm is six chained jobs, so
it missed every excursion outside that one window. Mano has reached 30.0.

The separation survives the correction but is quantitative, not absolute. The
healthy arms spend 3-7% of steps above 2.0 and peak in the tens; the SophiaG
arms spend 38-59% there and peak in the hundred-thousands -- a factor of
~3,000 in magnitude.

What actually discriminates them is the SHAPE of an excursion, not its
existence. Mano's largest late spike (3.84 at step 1944) ramped over six steps
-- 0.24, 0.39, 0.55, 1.03, 3.84 -- with the loss moving alongside it (2.89 ->
3.09), and was back under 1.0 within five steps: a hard batch, not a state
change. SophiaG's onsets are discontinuities under a nearly flat loss: 0.39 ->
2.41 -> 89.9 -> 437 -> 1702 while the loss moves only 3.957 -> 4.089.

The populations overlap in KIND but not in degree: every arm has excursions,
and SophiaG's are three orders of magnitude larger.

Splitting SophiaG at its first onset (step 1048), measured over the full
chain:

| window | steps with grad_norm > 2.0 | max |
|---|---:|---:|
| SophiaG, steps 21-1047 (pre-onset) | 184 / 1,027 (**17.9%**) | 75.2 |
| AdamW, same window | 54 / 1,027 (5.3%) | 14.6 |
| Mano, same window | 121 / 1,027 (11.8%) | 30.0 |
| SophiaG, steps 1049+ (post-onset) | 364 / 417 (**87%**) | 100,611 |
| re-run, steps 1176+ (post-onset) | 131 / 204 (**64%**) | 204,017 |

An earlier version of this document reported the pre-onset window as
**0 / 147, max 0.921** and concluded the behavior was bimodal -- a clean flip
from healthy-looking into a bad regime. That number came from reading ONE
chain link's `train.log` when each arm is six chained jobs, so it missed every
excursion outside that window.

On the full chain SophiaG is already the noisiest arm BEFORE its blow-up:
17.9% of pre-onset steps above 2.0 against AdamW's 5.3% and Mano's 11.8%, and
a pre-onset peak of 75.2 that neither healthy arm comes near. So this is
escalation from an elevated baseline, not a flip between two clean states --
and that elevated baseline is a usable early-warning signal, where waiting for
the discontinuity is not.

### It is recoverable, but not reliably -- read to the end of this section

Both SophiaG arms were left running well past their blow-ups to see whether
the high-gradient state ever ends. It does, for one of them. Excursion rate
per 100-step window (fraction of steps with grad_norm > 2.0):

| window | sophiag | sophiag re-run |
|---|---:|---:|
| 1000-1099 | 52% (max 100,611) | -- |
| 1100-1199 | 81% (max 497) | 28% (max 204,017) |
| 1200-1299 | 92% (max 9,613) | 68% (max 63,685) |
| 1300-1399 | 93% (max 9.9) | 55% (max 315) |
| 1400-1499 | 61% (max 419) | 88% (max 4,250) |
| 1500-1599 | 11% (max 89.7) | 98% (max 1,120) |
| 1600-1699 | **0% (max 0.7)** | 97% (max 1,023) |
| 1700-1799 | -- | 93% |

**sophiag recovered completely.** By step 1600 it is at 0 excursions with a
peak of 0.7 -- below Mano's lifetime average, ~600 steps after a 100,611 spike.
The decline is not monotone (it peaks at 93% in 1300-1399 before falling), so
the recovery is only visible once several windows are compared; any single
window would read as noise.

**sophiag re-run did not.** Same optimizer, same LR, same seed checkpoint,
same data; 600 steps past onset it is still at 93-98% and still spiking past
1,000.

This retires the "does not leave that regime" claim earlier versions of this
document made. That claim was made when the only evidence was ~400 post-onset
steps, all of which happened to be inside the bad window. Given another 200
steps, one arm walked out of it.

So the regime is **metastable, not absorbing** -- and, like onset itself,
whether an arm escapes appears to be luck. Two replicates from the same
checkpoint diverged at different steps; two replicates past divergence had
opposite recoveries. That is the same stochastic signature at both ends.

What does NOT change: the blow-up is recurrent (3/3 here, 4/4 including
the independent fresh run -- see the 2026-08-28 update), five orders of magnitude
beyond the healthy arms, and costs hundreds of steps of progress even when the
arm eventually recovers. A recoverable failure that burns 600 steps of a 2,500
step budget is still disqualifying for this comparison.

### And it relapses -- recovery is not an exit

Left running further, the re-run did briefly recover, then fell straight back
in. Per-100-step excursion rate (grad_norm > 2.0):

| window | sophiag | sophiag re-run |
|---|---:|---:|
| 1600 | 0% (max 0.7) | 97% (max 1,023) |
| 1700 | 0% (max 0.5) | 95% (max 29.5) |
| 1800 | 2% (max 7.0) | **4% (max 5.2)** |
| 1900 | 0% (max 0.9) | 93% (max 17,065) |
| 2000 | -- | **100% (max 27,524)** |

The re-run's window-1800 is a genuine quiet spell -- 4%, peak 5.2, materially
better than Mano's lifetime 6.5% -- and it does not hold. The next window is
back to 93% with a peak three orders of magnitude higher, and the one after is
100%.

So recovery is not a one-way exit. An arm can sit quiet for a hundred steps and
re-enter, which means **no finite quiet window proves an arm is out of it**.
The reading immediately above -- one arm "left the regime", the other did not
-- was taken at the moment the re-run happened to be in its quiet window, and
it did not survive another 200 steps. That is the second time a claim in this
section came from a window that happened to sit inside one phase.

`sophiag` has now held 0-2% across four consecutive windows with a peak of 7.0,
which is a far stronger basis than the single window the earlier claim rested
on -- but on this evidence it is a longer quiet spell, not proof of exit.

The practical consequence is unchanged and firmer: an optimizer that can
re-enter a 27,000-grad-norm state after appearing healthy for a hundred steps
is not usable for a production chain, recoverable or not.


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
above is against that: the failure is state-dependent and invisible to the LR
sweep. A low-LR arm may well delay onset (smaller steps accumulate
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

## Update 2026-08-28: 4/4, from an INDEPENDENT run, and there is no warning

Everything above was measured on three replicates that shared a checkpoint
lineage -- forks of one trajectory, so "3/3" was three samples of a single
draw. A fourth arm was run to test that: fresh from random init, its own
checkpoint dir and wandb group, `--debug.seed=1234` pinned
(`optcmp.pbs -v OPT=sophiag,TAG=fresh-seed1234,TSEED=1234`).

**It diverged too, at ~step 1550 -- a fourth distinct onset step.** The
escalation, monotone over five steps:

| step | loss | grad_norm |
|---|---|---|
| 1638 | 2.952 | 0.20 |
| 1639 | 2.954 | 4.95 |
| 1640 | 2.992 | 36.14 |
| 1641 | 3.034 | 98.98 |
| 1642 | 3.641 | **731.80** |
| 1643 | 4.445 | 98.72 |

A 3,600x gradient climb in four steps, loss driven 2.89 -> 4.45.

### This settles two things

**Divergence is a property of SophiaG at this scale, not of one lineage.** An
independent draw with RNG pinned still found its own onset step. Seed does not
prevent it, and data order is ruled out for good -- `dataloader.seed` was
already 42 in every arm, so it was never what differed.

**The pre-onset noise is NOT a precursor.** This is the genuinely new result
and it inverts a natural reading of the table above. The original arm ran
**17.9% of steps above 2.0 with max 75.2** for the 1,027 steps BEFORE its
blow-up, which looks like a warning sign you could monitor for. This arm ran
**1,529 comparable steps at 0.0%, max 0.31** -- nine consecutive clean 50-step
windows from step 650 to 1550, never exceeding 0.65 -- and diverged anyway.

Excursion rate by 50-step window (grad_norm > 2.0, steps at loss < 5.0):

| window | over 2.0 | rate | max gn | mean loss |
|---|---|---|---|---|
| 1450 | 0 | 0.0% | 0.23 | 2.9631 |
| 1500 | 0 | 0.0% | 0.23 | 2.9347 |
| 1550 | 3 | 6.0% | 37.01 | 2.9497 |
| 1600 | 10 | **20.0%** | **731.80** | 3.0540 |
| 1650 | 2 | 4.7% | 13.49 | 2.9752 |

So a healthy-looking SophiaG run tells you nothing. "It has survived N steps"
is not evidence of safety, and there is no signal to watch that buys warning.

Note the comparison window is steps whose **loss is already below 5.0**, not a
fixed step cutoff. A fresh init sits at loss ~12 with grad_norm 5-50 for its
first tens of steps; counting those makes any from-scratch run look like it
diverged at step 21. Unfiltered, this arm reports "162 excursions pre-onset" --
every one of them at loss >= 5.0, i.e. ordinary early training.

### It did not escape either -- and the dip at 1650 was a lull, not an exit

Left running past onset, the fresh arm went deeper rather than out. Excursion
rate by 50-step window (grad_norm > 2.0, steps at loss < 5.0):

| window | over 2.0 | rate | max gn | mean loss |
|---|---|---|---|---|
| 1550 | 3 | 6.0% | 37.01 | 2.9497 |
| 1600 | 10 | 20.0% | 731.80 | 3.0540 |
| 1650 | 2 | **4.0%** | 13.49 | 2.9710 |
| 1700 | 25 | **50.0%** | **4068.43** | 3.0387 |
| 1750 | 6/6 | **100.0%** | 436.47 | 3.5660 |

The 1650 window looked like recovery: rate down from 20% to 4%, peak magnitude
down from 731 to 13, mean loss back under 2.98. It was not. The next two
windows are 50% and 100%, with a peak of 4,068 -- five times anything the arm
had produced -- and mean loss finally climbing (2.97 -> 3.04 -> 3.57).

That is the same relapse shape the re-run showed (4% -> 93% -> 100%), now
reproduced in an independent arm. A single improving window inside this regime
is indistinguishable from an exit while you are in it; only the sequence
separates them. An earlier version of this document called a recovery on
exactly such a dip and had to retract it.

So the arm with the CLEANEST pre-onset history of any replicate -- 1,529 steps
at 0.0% -- both diverged and stayed diverged. Across four arms: 4/4 onset,
and of the three followed well past onset, one escaped and two did not.

### ...and then it escaped, completely

The relapse section above was written at window 1750 and reported the arm
"both diverged and stayed diverged". That was premature -- it escaped about
100 steps later, and more cleanly than any previous arm:

| window | over 2.0 | rate | max gn | mean loss |
|---|---|---|---|---|
| 1700 | 25/50 | 50.0% | 4068.43 | 3.0387 |
| 1750 | 31/42 | 73.8% | 6045.04 | 3.4676 |
| 1800 | 36/36 | **100.0%** | **16531.91** | 4.5410 |
| 1850 | 27/46 | 58.7% | 176.15 | 3.7745 |
| 1900 | 0/50 | **0.0%** | 0.97 | 3.0622 |
| 1950 | 0/50 | **0.0%** | 0.23 | 2.9242 |
| 2000 | 0/50 | **0.0%** | 0.62 | 2.8749 |
| 2050 | 0/33 | **0.0%** | 0.19 | 2.8535 |

From 100% of steps above grad_norm 2.0 with a peak of 16,532, to four
consecutive windows at zero with peaks under 1.0 -- inside ~100 steps. Mean
loss recovered 4.54 -> 2.85, which is BELOW the 2.89 it held before onset, so
the arm did not merely stabilise, it resumed making progress.

The full single-arm arc, all at a fixed seed:

1. 1,529 steps at 0.0% excursions, max 0.31 -- no precursor whatsoever
2. onset at ~1550, escalating 0.20 -> 4.95 -> 36.1 -> 99.0 -> 731.8 in 4 steps
3. a lull at window 1650 (4.0%) that reversed
4. deterioration to 100% of steps and gn 16,532, loss 2.89 -> 4.54
5. full escape by window 1900, loss back under pre-onset by 2050

**What distinguishes the real escape from the fake one at 1650:** the 1650
window was 4.0% with a max of 13.49 -- still spiking, just less often. The
escape is 0.0% with maxima collapsing 176 -> 0.97 -> 0.23 -> 0.62 -> 0.19 and
loss monotonically recovering. Rate alone is not enough; watch whether the
PEAK magnitude collapses and whether loss resumes descending.

Tally across four arms: 4/4 onset, and of the three followed well past onset,
**two escaped and one did not**.

### The escape was not permanent: a SECOND onset at ~4106

The escape section above is correct but incomplete as a conclusion. The arm
held the good state for ~2,200 steps -- reaching loss 2.4984, BELOW the
AdamW arm's final 2.51357 -- and then went again, from a completely quiet
state:

| step | loss | grad_norm |
|---|---|---|
| 4105 | 2.52557 | 0.0875 |
| 4106 | 2.50375 | 4.7431 |
| 4108 | 2.51449 | 0.5845 |
| 4109 | 2.86697 | 112.6056 |
| 4110 | 4.88487 | 177.2523 |

Loss 2.51 -> 4.88 in five steps. Same escalating shape as its first onset at
~1550.

The run-up is even quieter than the first time -- six consecutive 100-step
windows at exactly zero excursions, max never above 0.24:

| window | over 2.0 | rate | max gn | mean loss |
|---|---|---|---|---|
| 3400 | 0/100 | 0.0% | 0.20 | 2.5547 |
| 3500 | 0/100 | 0.0% | 0.22 | 2.5500 |
| 3600 | 0/100 | 0.0% | 0.24 | 2.5447 |
| 3700 | 0/100 | 0.0% | 0.22 | 2.5357 |
| 3800 | 0/100 | 0.0% | 0.17 | 2.5244 |
| 3900 | 0/100 | 0.0% | 0.17 | 2.5151 |
| 4000 | 0/100 | 0.0% | 1.84 | 2.5113 |
| 4100 | 3/11 | **27.3%** | **177.25** | 2.7608 |

**Two consequences.**

First, read "escaped" as "returned to the metastable good state", not as
"recovered". A SophiaG arm that has escaped is a SophiaG arm that can diverge
again, and this one did so while training better than the AdamW baseline.

Second, this is now TWO independent onsets in a single arm, each preceded by
nothing -- 1,529 clean steps before the first, ~2,200 before the second. The
no-precursor result no longer rests on one observation. There is no quiet
streak long enough to certify a SophiaG run as safe.

### The second onset was an order of magnitude milder -- severity is stochastic too

It resolved after ~2 intermittent windows rather than the first onset's four:

| window | over 2.0 | rate | max gn | mean loss |
|---|---|---|---|---|
| 4050 | 0/50 | 0.0% | 0.21 | 2.5097 |
| 4100 | 7/42 | **16.7%** | **177.25** | 2.6922 |
| 4150 | 0/50 | 0.0% | 0.20 | 2.5203 |
| 4200 | 4/50 | 8.0% | 15.81 | 2.5114 |
| 4250 | 0/50 | 0.0% | 0.34 | 2.5097 |
| 4300 | 0/50 | 0.0% | 0.19 | 2.4995 |

Spike, clean, partial, clean -- and mean loss never left 2.51 except in the
onset window itself. The arm came out at 2.4977, below the AdamW arm's final
2.51357.

Side by side, the same arm's two onsets:

| | first (~1550) | second (~4106) |
|---|---|---|
| peak grad_norm | 16,531.91 | 177.25 |
| worst window rate | 100% | 16.7% |
| peak mean loss | 4.5410 | 2.6922 |
| duration | ~4 windows | ~2, intermittent |
| clean steps before it | 1,529 | ~2,200 |

Same optimizer, same LR, same arm, and in both cases a precursor window of
exactly zero excursions -- yet two orders of magnitude apart in peak gradient.
**So severity is stochastic as well as timing.** A SophiaG blow-up may cost
you 100 steps or 400, and there is nothing in the run-up that predicts which.

A practical note for anyone reading a live run: the 4200 window read "1/39,
2.6%" while partial and "4/50, 8.0%" once full. Do not compare a partial bin
against complete ones -- wait for the bin to close.

### Method note

This arm was characterized three times in three readings -- "healthy", then
"an isolated transient", then "a clustered regime" -- each from a handful of
consecutive steps, and each optimistic read was wrong. The 50-step binned
table resolved it immediately. Bin first: a sequence of windows is the unit of
evidence here, never a run of steps. That is the same lesson the metastability
section above records, re-learned.

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
