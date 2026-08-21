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

## 3: the checkpoint round trip works, but not under compile

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

Throughput is only ~7% down, which is less than expected. **Memory is the
real cost: 93.80% vs 62.79%**, 31 points, leaving almost no headroom. At
that occupancy the run is one allocation spike away from a level_zero
failure, so uncompiled is not something to ship at this size even though it
is nearly as fast.

(An earlier reading of 174 tps was the first post-resume step, before
throughput settled -- not representative.)

This run ended at step 299, one step short of its step-300 checkpoint,
because it was submitted with a 2h walltime. So the resume LOAD is verified
and 49 steps of continued descent are verified; a post-resume SAVE has not
been directly observed yet. The step-250 checkpoint it loaded was itself
written by the compiled run, so the save path is not in doubt -- but
confirming it end-to-end wants a longer walltime.

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

## Verdict

The 30B config trains and its checkpoints are sound. The open item is
narrow and well-localized: **compiled resume** hits the vc_check bug. Options
are (a) resume uncompiled for one interval then restart compiled, (b) find
the workaround, (c) wait for the upstream fix. This does not block using the
config, but it does mean a long chained run needs a plan for restarts.

## Cross-references

- Throughput tuning that produced this config: [exp05](./exp05-2n-performance.md)
- Scaling behavior 2N-64N: [exp06](./exp06-scaling.md)
- Tokenizer choice: [exp07](./exp07-custom-tokenizer-feasibility.md)
