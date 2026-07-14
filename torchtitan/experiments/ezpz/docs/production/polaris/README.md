# Production Training Runs -- Polaris (A100)

> **Living document** -- updated as jobs complete and new runs are submitted.
>
> Last updated: 2026-07-14

Polaris (NVIDIA A100-SXM4-40GB) production trajectories. Distinct from the
Aurora/Sunspot (Intel XPU) chains tracked in the
[parent production README](../README.md): different hardware, different
dataset (`dolma`), different job-ID range (`72xxxxx`), and a
Polaris-specific checkpoint mode (`async`; see Known Issues).

All runs use `scripts/submit_agpt_{2b,20b}_autoretry.sh` (native
`ezpz launch --auto-retry`), the `dolma` data list, and the SophiaG
optimizer at LR=2.28e-5. The script auto-detects `machine: polaris`.

**Jump to:** [Status at a glance](#status-at-a-glance) ·
[2B chain](#2b-agpt-2b-sophiag-dolma) ·
[20B chain](#20b-agpt-20b-sophiag-dolma-n128-gbs1024) ·
[Known issues](#known-issues-polaris-specific)

## Status at a glance

Tokens = `GBS x seq_len(8192) x step`. `dolma` corpus; no fixed token
target is assigned to these Polaris chains yet, so tokens are reported
absolute (not % of a budget).

| Trajectory | State | Persisted step | Live step | Loss | Tokens | Notes |
|------------|-------|---------------:|----------:|-----:|-------:|-------|
| **2B n128** (`gbs512`) | idle (last leg done) | **5,300** | 5,383 | **2.36** | ~22.6B | 53 ckpts; last leg 7228272 ran full 12h |
| **20B n128** (`gbs1024`) | **ADVANCING** (leg 4 Q) | **1,000** | 1,100 | **2.39** | ~11.5B | 3 full 12h legs done; leg 4 (7247525) Q, leg 5 (7252666) held. 10 complete ckpts |

Smaller/earlier 2B node-count variants also exist on disk (comparison
runs, not the canonical chain): n64 -> step ~1,970 / loss 2.57;
n32 -> step ~543 / loss 4.03.

---

## 2B (`agpt-2b-sophiag-dolma`)

Canonical chain: **n128, GBS=512**. Also ran n32 / n64 as scaling
comparators (same optimizer/dataset).

| Field | Value |
|-------|-------|
| Submit script | `scripts/submit_agpt_2b_autoretry.sh` |
| Optimizer / LR | SophiaG / 2.28e-5 |
| GBS | 512 (NHOSTS_TRAIN=128, LBS=1, GAS=1) |
| Seq len | 8192 |
| Checkpoint mode | `async` (script default for 2b) |
| Checkpoint interval | 100 |
| Checkpoint dir | `outputs/checkpoints/agpt-2b-sophiag-dolma-n128-gbs512` |

**Progress (n128 canonical):** step **~5,383**, loss **2.36**
(from 12.9 at init) -- deep into convergence. **53 checkpoints**
persisted (step-100 ... step-5300), latest ckpt step-5300
(2026-07-05). ~**22.6B tokens** consumed. grad_norm settled to ~0.16.
Throughput ~1,080 TPS/GPU, ~3.9% MFU (compile config as submitted).

The final full-walltime leg was **7228272** (large, 130 nodes, ran the
complete 12:00:45 window, step -> 5,383, saving 53 ckpts cleanly). No
current continuation is queued for the 2B chain (production focus moved
to bringing up 20B).

### Scaling comparators (not the canonical chain)

| Variant | Nodes | Step reached | Loss |
|---------|------:|-------------:|-----:|
| n64 | 64 | ~1,970 | 2.57 |
| n32 | 32 | ~543 | 4.03 |

---

## 20B (`agpt-20b-sophiag-dolma-n128-gbs1024`)

New chain brought up 2026-07-06/07. This is the current production focus.

| Field | Value |
|-------|-------|
| Submit script | `scripts/submit_agpt_20b_autoretry.sh` |
| Optimizer / LR | SophiaG / 2.28e-5 |
| GBS | **1024** (NHOSTS_TRAIN=128, **LBS=1, GAS=2**) |
| Seq len | 8192 |
| Activation checkpoint | full (20b config default) |
| Checkpoint mode | **`async`** (REQUIRED override on Polaris -- see Known Issues) |
| Checkpoint interval | 100 |
| Checkpoint dir | `outputs/checkpoints/agpt-20b-sophiag-dolma-n128-gbs1024` (~700 GB on disk) |

### Bring-up (2026-07-06)

Validated via a smoke campaign before the first prod submit (per
smoke-before-prod practice). Three real issues surfaced and were fixed:

1. **OOM at low node count.** 20B OOMs at 2N (model shard 13.1 GiB/rank
   @ 8 GPUs) but fits at scale (3.05 GiB @ 40 GPUs, ~0.24 GiB @ 512).
   Pure-FSDP shard scales with GPU count -> smoke at >=10N, not 2N.
2. **bit.ly 429 broke env bootstrap.** The submit script's fallback
   `curl https://bit.ly/ezpz-utils` was rate-limited (HTTP 429),
   leaving every ezpz function undefined -> cascade to `libcudart.so.12`
   + venv failure. Fixed by pre-creating `.ezpz-utils-cache/ezpz-utils.sh`
   in the repo root (script prefers the local cache over curl).
3. **Sync checkpoint deadlock at 512 ranks** (see Known Issues) -- the
   first prod attempt (7237697) hung in the step-100 checkpoint save.
   Fixed with `CHECKPOINT_ASYNC_MODE=async`.

Batch config: **LBS=1, GAS=2** on Polaris (LBS=2 OOMs A100-40GB). GAS=2
keeps the canonical per-GPU effective batch (matching Aurora's LBS=2)
so the SophiaG LR=2.28e-5 stays valid; GBS = 512 x 1 x 2 = 1024.

### Trajectory

| Leg | Job | Outcome | Steps | Loss |
|-----|-----|---------|------:|-----:|
| smokes | 7237663/68/80/90 | config validated | -- | -- |
| leg 1 (bad) | 7237697 | **killed** -- sync-ckpt deadlock @ step 100 | 0->~100 | (hung) |
| **leg 1** | 7237948 | ran full 12h, clean | 0 -> **400** | 12.95 -> **3.38** |
| **leg 2** | 7237949 | ran full 12h (resumed step-300) | 301 -> **705** | 4.14 -> **2.72** |
| **leg 3** | 7243413 | ran full 12h (resumed step-700) | 701 -> **1,100** | 2.71 -> **2.39** |
| leg 4 | 7247525 | **Q** (`afterany:7243413`, resumes step-1000) | -- | -- |
| leg 5 | 7252666 | held (`afterany:7247525`) | -- | -- |

**Persisted checkpoints:** 10 complete (step-100 ... step-1000, each
234 GB / 512 shards; 2.3 TB total on disk). Two incomplete/empty dirs
(step-400 from leg 1, step-1100 from leg 3) -- each from a walltime-kill
landing mid-async-save exactly on a checkpoint boundary, so the next leg
resumed from the previous complete checkpoint (~100 steps rework). No
corruption; the fallback is the checkpoint system behaving safely. Leg 2's
handoff was clean (step-700 flushed before the kill -> leg 3 resumed at
701, ~5 steps rework), showing the rework only happens when the wall
coincides with a save.

**Current state:** step **1,100** (persisted 1,000), loss **2.39**,
~**11.5B tokens**, memory 20.2 GiB (51%), ~140 TPS/GPU, ~6.7% MFU,
grad_norm ~0.10. Descent across legs: 12.95 -> 3.38 -> 2.72 -> 2.39.
**Milestone:** matched the mature 2B chain's loss (2.36) at ~half the
tokens (~11.5B vs ~22.6B) -- the expected larger-model token-efficiency
crossover.

---

## Known issues (Polaris-specific)

### Checkpoint async-mode is REQUIRED on Polaris (CUDA)

`submit_agpt_20b_autoretry.sh` defaults `checkpoint.async-mode` to
`disabled`. On Polaris (512 ranks) the **synchronous DCP save
deadlocks**: the step-100 save hung with GPUs at 0%, ranks spinning in a
collective, zero files written, no resumable checkpoint (job 7237697,
killed after ~3.4h). The 2b Polaris script already defaults to `async`,
which is why the 2b chain checkpoints fine.

**Fix:** pass `-v CHECKPOINT_ASYNC_MODE=async` on every Polaris 20B
submit. With async, the step-100 save staged in 0.55s and flushed 234 GB
in the background without blocking training.

**Do NOT change the script's `disabled` default.** The canonical
production scripts target Aurora/Sunspot (Intel XPU), where async
checkpointing is broken -- `disabled` is the correct XPU default. `async`
is a Polaris/CUDA-only per-invocation override.

### Walltime-kill leaves the final checkpoint incomplete

Each 12h leg is killed mid-step at the wall (`Exit_status = -29`). If the
kill lands during an async save, that checkpoint dir is left empty and
the next leg resumes from the previous complete one -- costing up to
`CKPT_INTERVAL` (=100) steps of rework per handoff. Tolerable and
self-recovering; reducible by lowering `CKPT_INTERVAL` if rework becomes
costly.

### Chain continuation

Each leg is chained with `qsub -W depend=afterany:<prev>`. Keep one
`+1` continuation held behind the newest job so training never runs dry
at a walltime handoff. Current depth: leg 2 running (7237949) + leg 3
held (7243413).
