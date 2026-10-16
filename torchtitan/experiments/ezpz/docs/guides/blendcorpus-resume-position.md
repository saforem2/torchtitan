# BlendCorpus restarts replay the data stream

Every restart of the 20B production chains re-read the BlendCorpus stream from
sample 0. The saved `dataloader.consumed_samples` is 0 in every checkpoint, and
the pinned clones never increment it. The fix is on `origin/ezpz` since
`0f7f0c7f53` (2026-09-21), but neither pinned clone has it. This page has the
evidence, the size of the replay, and the options for resuming the existing
chains. Written 2026-10-15.

Verified means I read the code or the file. Inferred means it follows from
those, but I did not observe it directly.

## Mechanism (verified)

`blendcorpus.data.data_samplers.MegatronPretrainingSampler` (the `single`
sampler, which every 20B config uses) iterates
`range(self.consumed_samples, self.total_samples)`. The start offset is the
constructor argument and nothing else. `BlendCorpusDataLoader.load_state_dict`
rebuilds the sampler at the `consumed_samples` it is handed. So a checkpoint
that stores 0 restarts the stream at 0.

Both pinned clones initialise `_consumed_samples = 0` and write it in
`state_dict()`, with no `+=` anywhere in the iteration path:

| Clone | HEAD | Increment in `__iter__` |
| --- | --- | --- |
| `runs/agpt-20b-n256/torchtitan-ezpz` | `086ff996` | none |
| `runs/agpt-20b-v2/torchtitan-ezpz` | `f1c20ee7` | none |
| `origin/ezpz` | `d4449ba0ec` | yes (`0f7f0c7f53`) |

## Saved counter (verified)

Read with `dcp.load({"dataloader.consumed_samples": None}, no_dist=True)` using
each clone's own `.venv/bin/python -I`:

| Chain | Steps read | `consumed_samples` |
| --- | --- | --- |
| 20B 256N | 100, 4500, 8800, 13800, 17600, 18600, 18700, 19400 | 0 |
| 20B 512N | 9000, 9800 | 0 |

A step-100 checkpoint of a run with no restart also stores 0, so the counter
was never incremented in any segment I sampled.

## Loss signature (verified numbers, inferred cause)

Steps 17601+ after a restart from `step-17600`:

| Job | Step 17601 loss |
| --- | --- |
| original run | 2.06619 |
| `8902320` restart | 2.04307 |
| `8902321` restart | 2.04307 |

The two restarts agree to 5 digits and differ from the original by 0.023. Same
checkpoint, same data position on every restart, different from the
uninterrupted position. `8929146` from `step-18700` gave 2.282 at step 18701
against 2.454 originally. That fits a reset to sample 0. It does not prove it:
I did not decode sample indices on the cluster.

## How much was replayed

Every `Training starts at step N` in the job logs and failover attempt logs
under `runs/agpt-20b-n256/`, `runs/agpt-20b-v2/` and
`torchtitan-umbrella-ezpz0292/logs/multi-autoretry-*` (verified). With no
counter, a segment that starts at step N reads stream positions
`0 .. (steps run in that segment)`. The most the model can have seen of the
stream is therefore the longest single segment.

| | 20B 256N (GBS 6144) | 20B 512N (GBS 12288) |
| --- | --- | --- |
| Resumes (attempts) | 36 | 73 |
| Distinct resume steps | 29 | 36 |
| Checkpoint head | 19,400 | 11,100 |
| Longest segment | 1,550 steps (8301 -> 9850) | 1,200 steps (7601 -> 8800) |
| Distinct stream positions, upper bound | 1,550 steps = 78.0B tokens | 1,200 steps = 120.8B tokens |
| Tokens trained at head | 976.4B | 1,117.4B |
| Replayed, lower bound | 17,850 steps (92.0%) | 9,900 steps (89.2%) |

Tokens use `seq-len=8192` (read from the job args). Segment lengths are
`last logged step - start step + 1` per log file, so they are good to a few
steps. The 512N chain also rolled back to `step-9000` on 2026-09-05 after
reaching 10,699, which counts as another segment. Full per-attempt table (109 rows; one attempt with no logged step is
omitted): [`blendcorpus-resume-restarts.tsv`](blendcorpus-resume-restarts.tsv).

The replay is inferred, not measured. It rests on three verified facts: the
sampler start offset, the 0 counters, and the restart steps. If the first
1,550 steps of the stream are the only data the 256N model has seen, the
trajectory is a ~12x multi-epoch run over 78B tokens. I have not tested that.

## Fix (this PR)

1. Count what the sampler consumes. The increment on `origin/ezpz` adds
   `global_batch_size` per yielded batch. A yield is one microbatch, so with
   gradient accumulation (GAS > 1) the counter overshoots by GAS. It now adds
   `micro_batch_size * bc_mpu.get_data_parallel_world_size()`, the same product
   the sampler uses. Identical at GAS = 1, which is both 20B chains (2 x 3072 =
   6144 and 2 x 6144 = 12288).
2. `--dataloader.resume-consumed-samples N` sets the start position for
   checkpoints whose stored counter is 0. A nonzero stored counter always wins
   and the flag is logged and ignored. That keeps an auto-retry chain, which
   re-passes the same args, correct once its first new checkpoint lands.
   `N` must be a multiple of `micro_batch_size * dp` or `load_state_dict`
   raises.
3. `tests/test_blendcorpus_resume_position.py` runs the real iteration and
   state-dict code against a fake sampler with `MegatronPretrainingSampler`
   semantics. It checks resume equals the continued stream, zero replays from
   0, GAS = 4 counts 4x microbatches, and the override rules. 9 tests pass.
   Against the pinned behaviour (no increment) 3 fail, and against the
   `global_batch_size` increment the GAS test fails.

Not covered: nothing here ran on Aurora. The fake sampler copies the library's
semantics, it is not the library.

## Choosing the position for existing checkpoints

This is a scientific call. The chains have trained on a small prefix repeatedly,
so any new position changes the data distribution under the optimizer, and the
loss will move. All three options keep the checkpoint's weights and optimizer
state.

Positions below are for the current heads (`step-19400` on 256N, `step-11100`
on 512N). All are multiples of `micro_batch_size * dp`.

| Option | 256N `N` | 512N `N` | What it does |
| --- | --- | --- | --- |
| A. Step-aligned | 119,193,600 (19,400 x 6144) | 136,396,800 (11,100 x 12288) | Position equals what an uninterrupted run would have. Skips the stream between the replayed prefix and the step-aligned point, which no model run has seen, and never revisits it. Later restarts stay consistent with step count. |
| B. First unseen | 9,523,200 (1,550 x 6144) | 14,745,600 (1,200 x 12288) | Continues immediately after the longest prefix any segment read. Uses the stream contiguously with no gap. Position lags the step count by ~17,850 steps (256N), which is harmless to the sampler but makes position and step disagree from here on. |
| C. Leave at 0 | omit the flag | omit the flag | Keeps replaying the prefix. Only reason is exact reproducibility of the existing curve. |

Consequences common to A and B: the model has seen the prefix about 12x, so
fresh data should raise the loss at the restart, then settle. I expect this,
I have not measured it. Eval numbers on checkpoints before and after the switch
are not comparable as a clean token-count curve.

B depends on the segment-length bound, which is good to a few steps. If you
pick B, round the position up to the next multiple and add a margin rather than
using the exact figure.

Check that `N` is below `len(dataset)`, the `single` sampler asserts
`consumed_samples < total_samples` and aborts before step 1 if not. Both
options are far under `train_iters * GBS`.

## Rolling it out

The pinned clones are old commits and do not have `resume_consumed_samples`.
Changing them is your decision. Two ways:

- Move a clone to a commit with this change. The pinned builders do not match
  `origin/ezpz` line for line (`0f7f0c7f53` changed the same file), so I did
  not try to patch them in place.
- Port the three edits to the clone's `blendcorpus_builder.py` by hand: the
  counter increment, the `resume_consumed_samples` config field, and the
  `load_state_dict` override. Untested on the pinned commits.

Then add `--dataloader.resume-consumed-samples=N` to the submit script for the
restart and watch the first log line after `Finished loading the checkpoint`:
`resuming the data stream at --dataloader.resume-consumed-samples=N instead of
0`. The next checkpoint should store `N + steps * GBS`. Verify with the `dcp.load`
read above before trusting a later restart.

Do not change running or queued jobs (`8929146` was live when I checked) on the
strength of this page. Apply the new position at the next chain submission.
