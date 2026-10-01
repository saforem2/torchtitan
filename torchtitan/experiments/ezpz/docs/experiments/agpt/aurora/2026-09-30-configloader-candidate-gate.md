# ConfigLoader production migration gate — Aurora seat-4

**Status:** seat 4 and 20B-512 passed; 20B-256 remains open.

## Candidate

| Field | Value |
|---|---|
| Final SHA | `e3520c057ae699d45c5fe3d17036f84b73dda88b` |
| Branch | `origin/sync/upstream-97e673b779` |
| Superseded | `fa91efb3f83fb4386cfb4d8d5660d203b7335494` (withdrawn after Sunspot hardware exposed migration defects) |
| Sunspot evidence | job `12479129`, exit 0: dense TP1, dense TP2 with the supported no-AC contract, and MoE each completed 3 finite optimizer updates |

Upstream replaced `ConfigManager`/Tyro with `ConfigLoader`, whose entire CLI is
`--module`, `--config`, repeatable `--override`, `--comm-backend`,
`--output-dir`, `--resume-step`, and `--print-config`. Every active Aurora
production seat launches with legacy dotted options, so compatibility depends
on the ezpz-local `LegacyConfigLoader`.

## Verified on the candidate SHA

Parser parity and the negative control pass both locally and on Aurora compute
nodes. The compute-node check inside the job prints
`CONFIGLOADER_PARITY_AND_NEGATIVE_CONTROL_PASS` and asserts:

- all three production seat geometries resolve to their exact production
  values (config flavor, SophiaG LR, token budgets, scheduler, checkpoint
  policy, validator);
- upstream `ConfigLoader` still rejects a representative legacy dotted option,
  so the gate exercises the compatibility layer rather than a permissive
  parser.

## Remaining gate

Seat 4 (`agpt-2b-stage2-dolmino-n256-gbs6144`) is the representative lineage:
complex RoPE, stage-2 Dolmino, latest complete checkpoint `step-41300`
(3,073 files, `.metadata` 108,980,210 bytes).

The job must, in order:

1. restore full state from canonical read-only `step-41300`;
2. complete finite updates `41301`–`41303`;
3. write a fresh full-state checkpoint into a job-unique sink;
4. resume that fresh checkpoint and complete finite step `41304`.

The canonical directory is exposed to the sink through a symlink, so no
production checkpoint is ever written.

## Attempt history

| Job | SHA | Outcome | First concrete cause |
|---|---|---|---|
| `8881737` | `fa91efb3f8` | cancelled while queued | candidate superseded before allocation; no log, no output |
| `8881769` | `e3520c057a` | failed before restore | DP shard 24 cannot evenly shard fused dimension 11008 (`FSDP does not support uneven sharding on dim 1`) |
| `8881958` | `e3520c057a` | failed before restore | 12,288 tokens/step is not divisible by 16 ranks × 512 tokens (8,192) |
| `8881991` | `e3520c057a` | semantic pass; wrapper exit 37 | Restored `step-41300`, completed finite updates `41301`–`41303`, wrote fresh full-state checkpoints, restored fresh `step-41303`, and completed finite step `41304` |

Both failures were defects in the diagnostic wrapper's topology arithmetic, not
evidence about `LegacyConfigLoader` or checkpoint compatibility. Neither run
reached checkpoint restore.

Job `8881991` satisfied the complete seat-4 contract. The historical restore
finished in 237.91 seconds. Loss/gradient pairs were `2.58528/3.4850`,
`2.54956/3.4599`, and `2.73049/3.5645` at steps `41301`–`41303`. Each fresh
checkpoint contains 17 files totaling 23,846,149,586 bytes, including an
840,433-byte `.metadata`. The second process restored `step-41303` in 6.76
seconds and completed step `41304` with loss `2.74505` and gradient norm
`3.5586`; its checkpoint has the same file/byte counts.

PBS recorded exit 37 because the wrapper searched for one space in
`Training starts at step 41304`, while the log contains two spaces after the
timestamp prefix. This postcondition was a false negative after all required
work and artifacts completed. Reconciled evidence is recorded in
`RECONCILED_VALIDATION` under the job artifact root.

## Next gates

The 20B-512 lineage passed in job `8882148`. The historical `step-11100`
restore completed in 2,051.75 seconds. Steps `11101`–`11103` had finite
loss/gradient pairs `2.59988/5.6665`, `2.82035/5.7522`, and `2.62790/5.2748`.
Each current-format checkpoint contains 33 files totaling 248,972,711,367
bytes, including 6,690,302 bytes of metadata. A second process restored fresh
`step-11103` in 36.85 seconds and completed finite step `11104` at
`2.68816/4.7086`, then wrote another checkpoint with matching counts. As in
seat 4, exit 37 was only the wrapper's one-space postcondition after all work
completed; `RECONCILED_VALIDATION` records the semantic pass.

The 20B-256 retry `8882210` remains. After it passes, run one three-seat
umbrella smoke with job-unique sinks. Only after every lineage passes should
the production umbrella move to current software.

A passing gate is a verified full-state restore, real finite optimizer updates,
nonempty current-format checkpoint metadata and shards, and a fresh-save
resume. Scheduler exit zero alone is not acceptance.
