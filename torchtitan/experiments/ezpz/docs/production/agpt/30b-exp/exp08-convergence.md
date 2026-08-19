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

That is the known compile + AC AOT autograd bug, appearing here on the
resume path. Two things worth being precise about:

- **The DCP load itself SUCCEEDED** -- "Finished loading the checkpoint in
  65.64 seconds". The checkpoint is valid and readable. The crash is strictly
  in the first compiled backward afterwards.
- **The config did not change.** Diffing the two job scripts shows the only
  functional difference is `--checkpoint.interval` 250 -> 100. Fresh-start
  compiled runs are fine (12473304 did 482 steps), so resuming is what trips
  it, not compile alone.

**Job 12473387** (resume, `--compile.no-enable`) works:

```
Finished loading the checkpoint in 61.68 seconds.
step: 251  loss:  4.54003
```

Step 251, loss 4.540 -- picking up from the step-250 checkpoint whose loss
was 4.61, not restarting from ~12. Loss continues descending (4.44 by step
263). **Save -> resume -> continue is verified.**

But uncompiled is not a production answer:

| arm | tps | memory |
|---|---:|---:|
| compiled (fresh) | 489 | 62.79% |
| uncompiled (resume) | 174 | 93.80% |

2.8x slower and 31 points more memory -- compile is doing real work here on
both axes. Uncompiled resume is a diagnostic, not a workaround to ship.

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
