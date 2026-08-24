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
