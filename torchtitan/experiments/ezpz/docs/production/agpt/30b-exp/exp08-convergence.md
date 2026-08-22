# exp08: does the 30B config actually train?

**Date:** 2026-08-19
**Jobs:** 12473304 (fresh, compiled), 12473384 (resume, compiled), 12473387 (resume, no compile)
**Config:** `agpt_30b_olmo2tok`, 8N, TP=1, LBS=5, seq=4096, full AC, olmo2 tokenizer

## Why

Everything through exp07 measured 12-20 steps of throughput. That cannot
distinguish a good config from one that diverges at step 400, and it never
touches the checkpoint path at all. Three questions:

1. does loss descend, and stay descending?
2. does grad_norm stay bounded?
3. does save -> resume -> continue actually work?

## 1 and 2: yes, cleanly

Job 12473304 ran 482 steps before hitting its 6h walltime (signal 15 -- a
clean walltime kill, not a crash).

| step | loss  |
|-----:|------:|
|    1 | 12.03 |
|   80 |  6.99 |
|  160 |  5.47 |
|  240 |  4.61 |
|  320 |  4.06 |
|  400 |  3.71 |
|  480 |  3.37 |

Monotone throughout, no plateau and no spikes. grad_norm stayed in the
0.7-1.0 band across the run. Memory was flat at 40.17GiB (62.79%) -- no
creep, which is what you want to see before committing to a long run.

Throughput held at the tuned numbers rather than degrading: 480-490 tps,
82.9-84.5 TFLOP/s, **27.8-28.4% MFU**, matching the 497 tps / 28.99%
measured in exp05 over 12 steps. The config does not get slower once it is
actually training.

## 3: the checkpoint round trip works

> **Superseded below.** This section's title used to end "but not
> under compile". Compiled resume works on the `partial_dtensor` pin --
> see [Compiled resume WORKS](#compiled-resume-works-on-partial_dtensor-job-12473515).
> The history is kept because the two wrong readings are the reusable part.

This is the part that needed two jobs to answer, because the first attempt
failed in a misleading way.

**Job 12473384** (resume, compiled) died before step 1:

```
Error: expected all tensors_saved_with_vc_check to be Tensors,
       got types: [<class 'torch.Tensor' ...
```

**CORRECTION (2026-08-20): neither resuming NOR my commit was the trigger.**

Superseded twice. The trigger is `compile + AC + model.parallelize()` under
`full_dtensor`, which the ezpz configs were pinned to. They are now pinned to
`partial_dtensor` (`b2ff09632`) and this config runs clean. Full analysis:
[agpt-full-dtensor-vc-check.md](../../../guides/known-bugs/agpt-full-dtensor-vc-check.md).

The intermediate (also wrong) reading is kept below, because the mistake --
diffing job scripts while the tree moved underneath -- is the reusable part.

**Intermediate correction, now also superseded: resuming was NOT the trigger.**

The original reading here was that resuming causes this, because diffing the
two job scripts showed the only functional difference was
`--checkpoint.interval` 250 -> 100. That comparison was sound but incomplete:
it compared the scripts and not the tree. The passing run (12473304) started
at 13:06; my blendcorpus `positions` commit landed at 13:38. The two jobs ran
different code.

What actually broke it: blendcorpus began yielding `positions` for every
consumer, not just the flex MoE configs it was written for. agpt subclasses
Decoder (`AgptModel -> Llama3Model -> Decoder`), so core forwards positions
into agpt as well. SDPA builds no mask but still receives positions for RoPE,
and `rope._maybe_wrap_positions` does
`DTensor.from_local(positions, x.device_mesh, ...)` when the query is a
DTensor -- putting a `DeviceMesh` in the saved-for-backward set, which is
exactly what the assertion names.

Confirmed by a user report of the same assertion on a **fresh** `agpt_20b`
run (`--checkpoint.no-enable`), which resume cannot explain, and reproduced
directly on `agpt_20b` at HEAD (job 12473391, 0/3 steps).

Fixed in `2fa5d123c` by gating emission behind an explicit
`dataloader.emit_positions` flag, default off, set only on the two flex MoE
configs.

Still true and independently useful:

- **The DCP load itself SUCCEEDED** -- "Finished loading the checkpoint in
  65.64 seconds". The checkpoint is valid and readable.
- The uncompiled resume below genuinely works and genuinely continues the
  trajectory, so the round trip is verified regardless of the above.

**Job 12473387** (resume, `--compile.no-enable`) works:

```
Finished loading the checkpoint in 61.68 seconds.
step: 251  loss:  4.54003
```

Step 251, loss 4.540 -- picking up from the step-250 checkpoint whose loss
was 4.61, not restarting from ~12. Loss continues descending (4.44 by step
263). **Save -> resume -> continue is verified.**

The cost of dropping compile, measured over the full run rather than off
the first step:

| arm | tps | memory |
|---|---:|---:|
| compiled (fresh) | 489 | 62.79% |
| uncompiled (resume) | 455 | 93.80% |

> **Do not quote 93.80% as a current 30B number.** It measures a path that
> was abandoned once compiled resume was shown to work -- the live chain runs
> compiled at 59.84%. This row is the historical cost of the workaround, not
> the occupancy of anything in production.

Throughput is only ~7% down, which is less than expected. **Memory is the
real cost: 93.80% vs 62.79%**, 31 points, leaving almost no headroom. At
that occupancy the run is one allocation spike away from a level_zero
failure, so uncompiled is not something to ship at this size even though it
is nearly as fast.

(An earlier reading of 174 tps was the first post-resume step, before
throughput settled -- not representative.)

> **CORRECTED 2026-08-22.** The paragraph that stood here said this run
> "ended at step 299 ... a post-resume SAVE has not been directly observed
> yet." Both halves are wrong. The log records **153 step lines, step 251
> (loss 4.54003) through step 403 (loss 3.67819)**, and **two completed
> saves** -- `Finished saving the checkpoint in 49.27 seconds` and
> `in 26.99 seconds`, at steps 300 and 400. PBS killed it on walltime
> (`7206` vs a `7200` limit), not at 299.
>
> The cause was timing, not analysis: this section was committed
> (`e3c43169d`) at 15:32 CDT while job `12473387` was still running, and it
> was never refreshed after the job ended at ~17:12. A doc written against a
> live run needs a second pass once that run stops.
>
> Two consequences. First, the post-resume SAVE was directly observed here,
> two days before job `12473515` was credited with closing that gap. Second,
> and more interesting: **the step-400 checkpoint that `12473476` resumed
> from was written by this UNCOMPILED job.** So the chain's compiled lineage
> begins from an uncompiled-written checkpoint -- a stronger cross-mode
> round-trip than this document claims anywhere. (Verified 2026-08-22 by
> reading the log directly on Sunspot at
> `outputs/logs/30b-converge/12473387.*/train.log`.)

This run was submitted with a 2h walltime, which is what ended it. The resume
LOAD, 153 steps of continued descent, and two post-resume SAVEs are all
verified. The step-250 checkpoint it loaded was itself written by the compiled
run, so the save path was never in doubt.

## Update 2026-08-20: past the old ceiling, on the repinned backend

Job 12473476 resumed from step-400 and is the first 30B run under the
`partial_dtensor` repin (`b2ff09632`), which also makes it the first
production-length validation of that change.

| step | loss |
|-----:|-----:|
| 401 | 3.698 |
| 441 | 3.523 |
| 481 | 3.332 |
| 521 | 3.208 |
| 541 | 3.127 |

Monotone, no plateau, grad_norm 0.48-0.79, memory flat at 59.84%. It is past
the previous 531-step ceiling with a step-500 checkpoint on disk, so the
save path works under the new pin too.

The config dump confirms `"spmd_backend": "partial_dtensor"` -- the same code
path every earlier trajectory ran under its old name `"default"`, so this is
continuity rather than a new regime.

## Long run: 471 steps, loss 3.698 -> 2.617 (job 12473476)

Resumed from step-400 and ran to step 871 of the 2000-step config.

| step | loss | grad_norm |
|-----:|-----:|----------:|
| 401 | 3.698 | -- |
| 501 | 3.265 | -- |
| 601 | 3.03 | -- |
| 750 | 2.755 | -- |
| 871 | **2.617** | 0.21 |

Monotone throughout, no plateau, no spikes. **Zero NaN/inf** in loss or
grad_norm across all 471 steps. Memory flat at 38.29GiB (59.84%) start to
finish. Seven checkpoints written (250 through 800).

grad_norm fell from ~0.7 early to ~0.21 by step 871, which is the expected
shape as the model settles rather than a sign of stalling -- loss is still
descending at the same rate at the end.

W&B: https://wandb.ai/aurora_gpt/agpt-30b/runs/ul9l3nd6

This is also the first production-length validation of the
`partial_dtensor` repin (`b2ff09632`).

### Why it stopped at 871 and not 2000

`rc=124` -- the script's own `timeout 20400` (5h40) fired. Not a crash, not
PBS walltime: when this script was derived from the 6h original I raised the
PBS walltime to 12h but left the inner `timeout` at its old value, so the run
was capped at less than half the allocation it had. Elapsed was 5h40:58
against `timeout 20400`, which matches to the second.

Scale the inner `timeout` with the PBS walltime when deriving these scripts.

## Compiled resume WORKS on partial_dtensor (job 12473515)

The section above concluded that compiled resume was the open item. That is
no longer true, and the reason is the same repin that fixed everything else:
the vc_check assertion is a `full_dtensor` failure, and the ezpz configs have
been pinned to `partial_dtensor` since `b2ff09632`.

Job 12473515 resumed from step-800 **compiled**, on the pin:

```
Loading the checkpoint from .../agpt-30b-olmo2tok-converge/step-800.
Finished loading the checkpoint in 40.38 seconds.
step: 801  loss:  2.71221
```

No vc_check. Loss 2.712 at step 801 continues the 2.617 the previous job left
at step 871 (the resume is from step-800, so a small step back is expected,
not a restart). It has since run 789 steps:

| step | loss | grad_norm |
|-----:|-----:|----------:|
| 801 | 2.71221 | 0.2647 |
| 900 | 2.58695 | 0.2782 |
| 1000 | 2.50868 | 0.1841 |
| 1100 | 2.39879 | 0.2459 |
| 1200 | 2.37702 | 0.2263 |
| 1300 | 2.43408 | 0.1998 |
| 1400 | 2.38148 | 0.1630 |
| 1500 | 2.32552 | 0.1403 |
| 1589 | **2.26308** | 0.1255 |

Zero NaN/inf across all 789 steps. Memory flat at 38.29GiB (59.84%), 489 tps,
28.3% MFU, unchanged from the fresh compiled run -- so the resume costs
nothing in throughput or occupancy, unlike the uncompiled workaround's 93.80%.

Three checkpoints written post-resume (1000, 1250, 1500), each in ~27 s. That
also closes the last gap in the round trip: a **post-resume SAVE** is now
directly observed, which the 2h job above could not reach.

The step-1300 row is worth naming because it reads like a regression and is
not. Loss oscillates in a +/-0.05 band about a descending mean while
grad_norm falls monotonically (0.26 -> 0.13). Divergence shows the opposite
signature -- grad_norm rising. Two arbitrary 50-step samples will disagree at
this amplitude; sample denser before calling a bump.

### Checkpoint interval is a capacity decision at this size

The preceding job (`30b_long2.pbs`) set `--checkpoint.interval=100` on the
reasoning that more restart points are cheap. At 30B they are not:

  294 G per checkpoint x 20 checkpoints (2000 steps / 100) = 5.9 T

against 1.5 T free at the time. It filled the filesystem and the run died
with `Errno 28` **during a checkpoint write**, not from any training fault.
The partial step-900 directory it left behind is harmless -- `_find_load_step`
(`components/checkpointer/dcp.py:640-684`) only counts a step directory that
holds `.metadata` or `model.safetensors.index.json`, so an incomplete one is
skipped rather than loaded.

The fix was `interval=250`, **not** `--checkpoint.keep-latest-k`. Deleting
history to buy space trades a permanent asset for a temporary one; widening
the interval only costs resume granularity. `keep-latest-k` stays 0.

Rule: at 30B, budget `interval` against free space before treating it as a
resume-granularity knob. 294 G x (steps / interval) must fit.

### Chain continuation

At 42 s/step the `timeout 41400` lands near step 1770, ~230 short of 2000, so
`30b_long4.pbs` is queued `afterany:12473515` as **12473545** (6h walltime,
`timeout 19800`, interval back to 100 for the short final stretch -- 3 writes,
~880 G against 6.0 T free).

Both clocks move together in that script. `30b_long.pbs` is the counterexample:
its PBS walltime went 6h -> 12h while the inner `timeout` stayed at 20400, and
it took an `rc=124` at step 871 with half its allocation unused.

## COMPLETE: 2000/2000 steps, 12.028 -> 2.115 (job 12473545)

The run finished. Not a timeout -- step 2000 of a 2000-step config, `rc=0`,
`Exit_status=0`, 2h58 of a 6h allocation.

Four jobs, one continuous trajectory, **zero NaN/inf in any of them**:

| job | steps | loss | note |
|---|---|---|---|
| 12473304 | 1 -> 482 | 12.028 -> 3.357 | fresh start |
| 12473476 | 401 -> 871 | 3.698 -> 2.617 | `rc=124`, inner timeout unscaled |
| 12473515 | 801 -> 1781 | 2.712 -> 2.246 | compiled resume (the THIRD -- see note) |
| 12473545 | 1751 -> 2000 | 2.248 -> **2.115** | finished it |

> **CORRECTED 2026-08-22.** `12473515` was labelled the "first COMPILED
> resume". It was the third. Counting `--compile.no-enable` occurrences in
> each chain job's log: `12473304` 0, `12473387` **3**, `12473476` 0,
> `12473515` 0, `12473545` 0. So `12473476` also resumed compiled (from
> step-400), and only `12473387` was the deliberate uncompiled arm.
>
> Note the trap that produced the wrong label: the dumped config block reads
> `"enable": true` in ALL FIVE jobs, including the uncompiled one -- it is the
> config DEFAULT, not the effective setting. Only the CLI flag count
> discriminates. Read the launch line, not the dumped config, when asking
> whether a run was compiled.

Nine checkpoints on disk (400 through 2000), 2.6 T total. Steady state held
to the end: 496 tps, 28.3% MFU, memory flat at 38.29GiB (59.84%), grad_norm
descending 0.26 -> 0.074.

W&B: https://wandb.ai/aurora_gpt/agpt-30b/runs/dso9al26

### What this settles

The three questions exp08 was opened to answer are all answered YES:

1. **Loss descends over hundreds of steps** -- 1999 of them, monotone in the
   mean, no plateau and no divergence.
2. **grad_norm stays bounded** -- 0.7-1.0 early, 0.074 at the end, falling
   throughout.
3. **Checkpoints round-trip** -- three separate resumes, the last two
   compiled, each continuing the trajectory rather than restarting it.

The chain also demonstrates the operational shape a long run needs: each job
resumes the previous one's last checkpoint, and the replayed steps between
that checkpoint and the previous job's exit are the cost of the interval
(31 steps at interval=250, 1 step at interval=100).

### The one process lesson worth repeating

Two of the four jobs ended on their inner `timeout`, and one of those --
12473476 -- ended EARLY because only the PBS walltime was raised (6h -> 12h)
while `timeout 20400` stayed put. It died at step 871 with half its
allocation unused. Scale both clocks together or the shorter one silently
wins.

## Verdict

**The 30B config trains to completion.** 2000/2000 steps, loss
12.028 -> 2.115, zero NaN, nine checkpoints, and a round trip verified end to
end **under compile**: save -> resume -> continue -> save again, with no
throughput or memory penalty.

The compiled-resume caveat this section used to carry is **retired**. It was a
`full_dtensor` failure, and the configs are pinned to `partial_dtensor`
(`b2ff09632`); job 12473515 resumed compiled on that pin and has run 789
steps. A chained run needs no special restart plan beyond scaling the inner
`timeout` with the PBS walltime.

What remains open is not convergence but capacity: at 294 G per checkpoint,
`--checkpoint.interval` has to be budgeted against free space (see above).

## Cross-references

- Throughput tuning that produced this config: [exp05](./exp05-2n-performance.md)
- Scaling behavior 2N-64N: [exp06](./exp06-scaling.md)
- Tokenizer choice: [exp07](./exp07-custom-tokenizer-feasibility.md)
