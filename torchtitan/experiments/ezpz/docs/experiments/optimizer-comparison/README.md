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

## Phase 1 RESULTS (2026-08-23, jobs 12473714/15/16)

All three swept 1e-6 -> 1e-1 over 100 steps at GBS=960 on fineweb-edu, and all
three produced a real blow-up, so every suggestion is a measurement rather than
a sweep that ran out of range.

| arm | suggested LR | blow-up | min loss | at LR |
|---|---:|---:|---:|---:|
| AdamW | **3.05e-05** | 3.05e-04 | 8.8381 | 3.594e-04 |
| Mano | **5.61e-05** | 5.61e-04 | 8.9925 | 5.275e-04 |
| SophiaG | **3.55e-05** | 3.55e-04 | 9.3441 | 4.642e-04 |

Verified from the raw CSVs, not just the log summaries: 90 rows each, and every
row of every arm records `global_batch_size=960`, `world_size=192`. The fixed
batch is therefore a checked fact, not an assumption -- which matters because
the suggestion is batch-dependent and the arms are only comparable at one batch.

**The inherited placeholder was on the divergence cliff.** Both mano and sophiag
configs carried lr=3.0e-4 from the 2B competition configs:

| | value | vs suggested | vs blow-up |
|---|---:|---:|---:|
| placeholder | 3.0e-4 | 8.5x too high (sophiag) | only 1.18x below |

Running SophiaG there would have read as "SophiaG is unstable at 30B" when the
truth is "SophiaG was run at 8.5x its usable LR" -- the same shape as the
documented 2026-07-03 80B NaN. This is the concrete payoff of measuring per
optimizer per batch instead of inheriting a constant.

Note the three suggestions land within 1.8x of each other (3.05e-05 to
5.61e-05), which is much tighter than the three-orders-of-magnitude spread seen
ACROSS batch sizes. At a fixed batch the optimizer choice moves the usable LR
far less than the batch size does.

## Phase 2: comparison runs (LAUNCHED 2026-08-23)

Jobs 12473720 (adamw), 12473721 (mano), 12473722 (sophiag) -- 16N each, running
concurrently on 48N, each at its own Phase 1 LR, constant after a 20-step warmup.

```bash
qsub -v OPT=adamw,LR=3.05e-5   -N cmp-adamw   optcmp.pbs
qsub -v OPT=mano,LR=5.61e-5    -N cmp-mano    optcmp.pbs
qsub -v OPT=sophiag,LR=3.55e-5 -N cmp-sophiag optcmp.pbs
```

## Phase 2: design notes

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


## Phase 2 launch: five failures, and the process fix

Phase 2 took six attempts to leave the ground. Every failure was a
sub-minute crash on a 48-node allocation, and every one was a DIFFERENT layer
of the same cause: the 80th upstream sync landed between the Phase 1 finders
(which ran on the pre-merge tree and succeeded) and the first Phase 2 launch.

| # | jobs | failure | layer |
|---|---|---|---|
| 1 | 12473720-22 | `Unrecognized options: --training.seq-len, --training.local-batch-size, --training.global-batch-size` | CLI flags renamed by #4121 |
| 2 | 12473725-27 | `NotImplementedError: dataset='fineweb_edu_local' used the HF delegate` | dataloader class deleted by #4088 |
| 3 | 12473729-31 | `Unrecognized options: --dataloader.num-workers` | flag absent from `GrainDataLoader.Config` |
| 4 | 12473732-34 | `GrainDataLoader.__init__() missing 2 required keyword-only arguments` | dataloader kwargs stale in `trainer.py` |
| 5 | 12473735-37 | `NameError: name 'global_batch_size' is not defined` | stale name in a log line |

**Every one was reachable at 2 nodes.** Fixing forward at full scale, five
times, was the actual mistake -- each failure looked like an isolated bug, but
they were a cascade from one merge, so fixing the top layer only exposed the
next. The standing rule (smoke before large jobs) was skipped because the
finders had just succeeded; they had run on the pre-merge tree.

Two durable guards came out of it:

**`optcmp_smoke.pbs`** -- runs all three arms sequentially, 3 steps each, in one
2N allocation, printing a PASS/FAIL verdict per arm. One job proves every arm
reaches real training. Its GBS is deliberately not 960: 24 ranks cannot reach
the comparison batch, and the smoke tests the code path, not the numerics.

**An argument preflight inside `optcmp.pbs`** -- parses the exact argv on one
rank before allocating, and asserts the resulting batch (`max_context_length`
4096, LBS 5, GBS 960, GAS 1, `decay_ratio` 0.0) rather than only that the flags
parse. It deliberately stops at config construction and does NOT build the
Trainer, because that calls `set_device` and a login context has no GPU -- the
first version did exactly that and rejected valid arguments.

The batch assertions matter more than the flag check. A flag that parses but
yields a different GBS would not crash; it would silently train at the wrong
batch and invalidate the comparison against the Phase 1 LRs. Verified
negatively: the preflight rejects an unknown flag, a halved GBS, and
`decay_ratio` flipped back to 0.8.

`pyflakes` over `torchtitan/experiments/ezpz/` is the cheap catch for failure 5
and should be run after any upstream sync -- it reports zero undefined names in
`trainer.py` now.


## Phase 2 interim results (chain 1, ~1.95B tokens per arm)

![Loss and gradient norm, three optimizers at GBS=960](figures/optcmp_30b_gbs960.svg)

![Mano minus AdamW loss; below zero means Mano is ahead](figures/optcmp_30b_crossover.svg)

Regenerate with `python3 torchtitan/experiments/ezpz/scripts/plot_optcmp.py`.
It reads the arms' `train.log` files directly rather than W&B, so the figures
rebuild from a checkout with no network and no run-id bookkeeping. Chained
jobs re-log steps they resumed over, so the reader keys by step with
last-write-wins -- otherwise a resumed step would appear twice with the killed
run's value.


Jobs 12473743/44/45, 16N each, GBS=960, constant LR after a 20-step warmup,
each arm at its own Phase 1 LR. Loss / grad_norm at each 100-step mark:

| step | tokens | AdamW | Mano | SophiaG | Mano - AdamW |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.39B | **6.1018** | 6.3064 | 7.1746 | +0.2046 |
| 200 | 0.79B | **5.3382** | 5.4752 | 6.6477 | +0.1370 |
| 300 | 1.18B | 4.7682 | **4.6622** | 6.2720 | **-0.1060** |
| 400 | 1.57B | 4.4364 | **4.1891** | 5.9182 | -0.2473 |
| ~495 | 1.95B | 4.2057 | **3.8722** | 5.5492 | **-0.3335** |

**Mano overtakes AdamW between step 200 and 300** and the gap keeps widening
(+0.20 -> -0.33 nats). This is not a single-point wobble: the sign flips once,
monotonically, and holds across 200+ steps.

Per-100-step improvement over the last two intervals shows why -- Mano is
descending faster, not merely starting luckier:

| arm | 300->400 | 400->~495 |
|---|---:|---:|
| AdamW | 0.3318 | 0.2307 |
| **Mano** | **0.4731** | **0.3169** |
| SophiaG | 0.3538 | 0.3690 |

SophiaG is a clear third throughout (1.7 nats back) but has the flattest
decay in improvement rate, so its ordering versus the others is the least
settled of the three.

### What this is not yet

* **1.95B of a 10B budget.** The documented pattern from earlier competitions
  is "Mano/Muon win short runs, AdamW wins in the cosine decay phase" -- and
  these runs are constant-LR by design, so they have NO decay phase. This
  result speaks to the constant-LR regime only.
* **One seed per arm.** The crossover is large relative to the step-to-step
  noise, but no seed variance has been measured here.
* Absolute losses are NOT comparable to the decayed AdamW 30B baseline
  (2.115 at 3.93B tokens); only the three arms are comparable to each other.

### Interruptions (none affected the numbers)

Chain 1 was killed externally at 05:23-05:25 on 2026-08-24 -- all three arms
simultaneously, mid-line, on different nodes, with no catchable signal. PBS
recorded 6:09 walltime against an 8h request, and a maintenance reservation
(14:00 -> Tue 00:30) was scheduled the same day, so node drain is the likely
cause. All three had complete step-400 checkpoints, so chain 2
(12473753/54/55) resumes there; it is queued behind the reservation with an
estimated start of Tue 00:30.


## Phase 2 results at 7.37B tokens per arm (chains 1-7)

| step | tokens | AdamW | Mano | Mano - AdamW |
|---:|---:|---:|---:|---:|
| 1300 | 5.11B | 3.3155 | **3.0444** | -0.2711 |
| 1500 | 5.89B | 3.2141 | **3.0028** | -0.2113 |
| 1700 | 6.68B | 3.1416 | **2.9574** | -0.1842 |
| 1900 | 7.47B | 3.0596 | **2.8880** | -0.1715 |
| 2000 | 7.86B | 3.0627 | **2.9141** | -0.1485 |
| 2100 | 8.25B | 2.9894 | **2.8428** | -0.1466 |
| 2200 | 8.65B | 2.9912 | **2.8451** | -0.1461 |
| 2275 | 8.94B | 2.9490 | **2.8073** | -0.1417 |

**Mano leads throughout, and the gap has stopped closing.** It narrowed
quickly through 7.9B and has been nearly flat since:

| window | slope | implied crossover | gap at 10B |
|---|---:|---:|---:|
| 5.9-7.5B | +0.0252 nats/B | ~14.2B | -0.105 |
| **7.9-8.9B** | **+0.0056 nats/B** | **~34.6B** | **-0.137** |

Fitting only the recent window drops the closing rate by 4.5x. At the current
rate Mano finishes the 10B budget still ~0.14 nats ahead, and AdamW does not
catch it within any horizon this experiment can reach.

This is the fourth revision of this claim, and the pattern in the failures is
consistent enough to state plainly: every wrong version came from fitting a
trend across a window that included a regime change.

| at | claim | why it failed |
|---|---|---|
| 3.54B | "peaked, AdamW catches up ~6-7B" | 3 points off a fresh peak |
| ~5.2B | "no longer closing, reopening" | read noise (-0.271 -> -0.273) as reversal |
| 5.31B | "crossover well past budget" | rate fit across the noisy stretch |
| 7.37B | "crossover ~10.1B, at the budget end" | fit spanned the fast-closing phase that had already ended |

The honest statement now: **Mano wins this comparison at the 10B budget.** The
gap is -0.142 at 8.94B and closing at a rate that would need ~26B more tokens
to reach zero. That is a statement about this configuration -- constant LR, one
seed, no decay phase -- and the documented prior is that AdamW recovers ground
during cosine decay, which this experiment deliberately does not have.

SophiaG is excluded from this table. It diverged at step 1048 and every later
point measures a post-divergence trajectory. See below.

### SophiaG: a regime flip, 3 of 3

All three replicates forked from the clean `step-1000` checkpoint blew up, at
**three distinct steps: 1048, 1176, 1071**. Timing is random; the event is not.
Full analysis in
[`guides/known-bugs/sophiag-stochastic-divergence-30b.md`](../../guides/known-bugs/sophiag-stochastic-divergence-30b.md).

The failure is not a too-high LR:

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

Splitting SophiaG at its first onset (step 1048) over the full chain:

| window | steps > 2.0 | max |
|---|---:|---:|
| SophiaG, steps 21-1047 (pre-onset) | 184 / 1,027 (**17.9%**) | 75.2 |
| AdamW, same window | 54 / 1,027 (5.3%) | 14.6 |
| Mano, same window | 121 / 1,027 (11.8%) | 30.0 |

This retires the "bimodal" framing an earlier version of this document used.
That claim rested on the same single-log census -- it reported SophiaG as
0/147 above 2.0 before onset, i.e. indistinguishable from the healthy arms,
with a clean discontinuity into a bad regime. On the full chain SophiaG is
already the noisiest arm BEFORE its blow-up: 17.9% of pre-onset steps above
2.0 against 5.3% and 11.8%, and a pre-onset peak of 75.2 that neither healthy
arm approaches.

So the honest reading is escalation from an elevated baseline, not a flip
between two clean states. The blow-up is still real, still recurrent (3/3),
and still five orders of magnitude beyond anything the healthy arms do -- but
SophiaG was visibly the least stable arm the whole time, which is a better
early-warning signal than waiting for the discontinuity.

### It is recoverable -- one arm left the regime, the other did not

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

What does NOT change: the blow-up is recurrent (3/3), five orders of magnitude
beyond the healthy arms, and costs hundreds of steps of progress even when the
arm eventually recovers. A recoverable failure that burns 600 steps of a 2,500
step budget is still disqualifying for this comparison.

It ended at loss 4.193 -- back in its pre-blow-up range, and on a loss chart
alone indistinguishable from a recovered run. Its grad_norm at that same step
was 2.449, six times anything AdamW or Mano have reached. **Read these arms by
grad_norm, never by loss.**

The grad-norm guard caught replicate 3 at 53.49 (133x trailing median) and
stopped it at rc=0 in 55 minutes, against the ~8h each that replicates 1 and 2
burned. `nan_abort_consecutive` never fires here -- every value in the blow-up
is finite.

### Caveats unchanged

Constant LR by design, so no decay phase -- and the documented prior from
earlier competitions is exactly "Mano/Muon win short runs, AdamW wins in the
cosine decay phase." One seed per arm.

### Status

**Running.** 7.37B of the 10B budget per arm; jobs 12473867 (adamw) and
12473868 (mano) resumed from step-1800. SophiaG is retired from the comparison
-- all three replicates are post-divergence and no clean arm exists.

Checkpoint policy: each arm keeps step-400 (matched pre-divergence anchor),
step-1000 (the SophiaG divergence seed), and its newest resumable link.
Mid-chain links are deleted once superseded. `backup` is a timestamped `mv` and
reclaims nothing, so freeing quota requires actual deletion.

A walltime cut can leave an empty `step-N` directory with no `.metadata` (seen
on adamw/step-1881). The resume logic skips those by design, but they are worth
removing so the newest-directory heuristic stays honest.
