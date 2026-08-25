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


## Phase 2 results at 3.54B tokens per arm (chains 1-3)

Jobs 12473743/44/45 -> 12473753/54/55 -> 12473764/65/66, all resuming cleanly
from the previous link's checkpoint. Every arm ran to `rc=124`, the inner
timeout, which is the designed clean stop.

| step | tokens | AdamW | Mano | SophiaG | Mano - AdamW |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.39B | **6.1387** | 6.3146 | 7.1219 | +0.1759 |
| 300 | 1.18B | 4.7682 | **4.6622** | 6.2720 | -0.1060 |
| 500 | 1.97B | 4.2196 | **3.8716** | 5.5492 | -0.3479 |
| **600** | **2.36B** | 4.0272 | **3.6636** | 5.2602 | **-0.3636 (max)** |
| 700 | 2.75B | 3.8458 | **3.4978** | 4.9237 | -0.3479 |
| 800 | 3.15B | 3.7171 | **3.3764** | 4.6574 | -0.3407 |
| 900 | 3.54B | 3.5957 | **3.2748** | 4.3526 | -0.3209 |

**Mano still leads, but the gap has PEAKED and is now closing.** It crossed
AdamW between 0.79B and 1.18B, widened to -0.3636 nats at 2.36B, and has
narrowed every interval since. Per-100-step improvement now favours AdamW:

| arm | 600->700 | 700->800 | 800->900 |
|---|---:|---:|---:|
| **AdamW** | 0.1814 | **0.1287** | **0.1214** |
| Mano | 0.1658 | 0.1215 | 0.1016 |
| SophiaG | 0.3365 | 0.2663 | 0.3048 |

This is the earlier interim reading updated, not confirmed: at 1.95B the gap
was still widening and Mano was descending fastest. Both have since reversed.
Extrapolating the current rates, AdamW would catch Mano around ~6-7B tokens --
inside the planned 10B budget, so the ordering at the budget end is genuinely
open.

SophiaG remains a clear third (1.08 nats behind Mano) but is descending
FASTEST of the three and has been since ~2B. It started worst and has closed
from 1.51 to 1.08 nats behind. Whether that continues is the other open
question.

### Caveats unchanged

Constant LR by design, so no decay phase -- and the documented prior from
earlier competitions is exactly "Mano/Muon win short runs, AdamW wins in the
cosine decay phase." The crossover-then-reconvergence seen here is consistent
with that prior playing out even WITHOUT a decay phase. One seed per arm.

### Status

Stopped at 3.54B of the 10B budget: project 2297 hit its 22.0 TB hard quota
(27 checkpoints x ~290 GB = 7.5 TB of optcmp alone), so the next chain link
would fail at its first write. Continuing requires pruning checkpoints or a
quota increase; the newest checkpoint per arm (adamw/mano step-800, sophiag
step-900) is intact and resumable either way.
