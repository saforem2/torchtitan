# 20B 512N canonical chain: relaunch on native auto-retry (resume step-4400)

- **Date:** 2026-07-01
- **Machine:** Aurora
- **Status:** SUBMITTED -- head `8638793` (Q, select=522), cont `8638795`
  (H, `afterany:8638793`). Resumes the 20B 512N canonical chain from
  step-4400 on the modern `ezpz launch --auto-retry` stack.

## Why

The 20B 512N canonical chain had been frozen at **step-4400 since
2026-05-29**. Its latest attempt to advance was as `trainer-1` inside the
multi-model umbrella job `8568429` (the legacy `failover_lib.sh` wrapper
running 4 chains under one 1576N allocation). That trainer **died at init**
and never resumed:

- Bad node `x4410c0s0b0n0` killed ranks with `signal 11` on every attempt.
- The legacy failover scraper logged `no specific bad node identified --
  rotating one spare in blindly` each time: it could *count* the
  bad-node lines but could not *parse the hostname* out of the `died from
  signal 11` signature, so it swapped innocent `x4411` spares while the
  real culprit `x4410` stayed in the active set. All 3 retries (+ initial)
  exhausted -> `max retries (3) exhausted; giving up`.

This is a concrete instance of the blind-swap failure class from the
[restart-economics analysis](20260630-failover-restart-economics.md): the
`failover_lib.sh` bad-node regex can't extract a hostname from a
`signal 11` line. Native `ezpz launch --auto-retry` (>= 0.17.1) has a
better scraper, so the fix is to relaunch this chain on the native path.

## What it took (three obstacles, all resolved)

1. **Resume-format compatibility (pre-#3623).** The pinned
   `runs/agpt-20b-v2` clone is rolled back to pre-fused `1263e5a1`
   (nested optim state-dict). Confirmed *empirically* that
   `step-4400`'s DCP keys are nested
   (`optimizer.state...exp_avg`/`.hessian`, `qkv_linear.wq`, **no**
   `.fused`) -> loads cleanly with this clone's code. No migration shim
   needed. (See [pre-3623 known bug](../../../../guides/known-bugs) memory.)

2. **ezpz too old for auto-retry.** The clone venv had **ezpz 0.16.0**
   (no `--auto-retry` flag). Upgraded to **0.21.3** in place via
   `uvi` (`uv pip install --no-cache --link-mode=copy git+.../ezpz`).
   ezpz's core deps do not include torch (torch only in an optional
   extra + `override-dependencies sys_platform==never`), so the upgrade
   touched **only ezpz** -- `torch 2.13.0.dev20260428+xpu` unchanged,
   verified. `--auto-retry`/`--spare-nodes`/`--max-failover-retries` now
   present.

3. **Venv broadcast tarball.** The `.venv.tar.gz` still held 0.16.0. A
   naive `tar` rebuild on the login node crawled at ~210 KB/s (37,493
   small files off a busy Lustre login node = metadata-bound; a compute
   node was no faster). Because **only ezpz (2.4M) changed**, rebuilt via
   a **fast tmpfs surgical-swap** (debug job `8638674`): decompress the
   good 0.16 tarball into RAM-backed `/tmp` (28s), swap in the new ezpz
   (tiny), re-tar from tmpfs (16s), copy one 2.7G file back. Result:
   2.7G, **39,803 files** (matches reference), `ezpz-0.21.3.dist-info`
   verified inside. Old 0.16 tarball preserved as a backup (the
   `agpt-20b-n256` clone symlinks to this tarball; the upgrade is
   additive/back-compatible, old kept for revert).

## Smoke verification (2N, job 8638756)

Resumed step-4400 at 2N (throwaway: `CKPT_INTERVAL=999999`,
`VALIDATOR_ENABLE=0`, `TRAINING_STEPS=4402` -> loads + 1 step, never
saves, chain untouched):

- `Using [24/24] available "xpu" devices` (Intel Max 1550, 63.98GiB)
- `[auto-retry] attempt 1` -- ezpz 0.21.3 native failover active on compute
- `Finished loading the checkpoint in 567.11 seconds` (slow = DCP
  resharding 512-shards -> 2 nodes; native 512N load is faster)
- **`Training starts at step 4401`** -- resume CONFIRMED (not a fresh
  start from step 1)

### CKPT_DIR gotcha (caught by the smoke)

`job.dump_folder = ./outputs`, and `--checkpoint.folder` is resolved
relative to it. So the correct `CKPT_DIR` is the **bare**
`checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288` (which the
autoretry script computes by default) -> resolves to
`./outputs/checkpoints/...` = the real step-4400. A first (wrong) smoke
used `CKPT_DIR=outputs/checkpoints/...` -> `./outputs/outputs/...`
(empty) -> silently fresh-started from step 1. The umbrella job's bare
`checkpoints/...` was correct all along.

**Consequence for launch:** at `NHOSTS_TRAIN=512` the script default
already yields `...n512-gbs12288`, so the 512N launch needs **no
`CKPT_DIR` override**. Only sub-N smokes need an explicit `n512`
override (their default would be `n{smoke_N}`).

## Launch

From `runs/agpt-20b-v2/torchtitan-ezpz` (pinned clone, ezpz 0.21.3):

```bash
qsub -q prod -l select=522 -l walltime=12:00:00 -l filesystems=home:flare \
  -N agpt-20b-n512-autoretry-resume -v NHOSTS_TRAIN=512 \
  torchtitan/experiments/ezpz/scripts/submit_agpt_20b_autoretry.sh
# + afterany continuation (8638795)
```

- **Defaults** (unchanged): `LBS=2 GAS=1 TP=1` -> **GBS=12288** (matches
  the ckpt dir), `OPTIMIZER=sophiag LR=2.28e-5`, `TRAINING_STEPS=46429`
  (full 4.67T budget), validator on (`VALIDATOR_FREQ=100`),
  `CKPT_INTERVAL=100`, spares = 522-512 = 10 (`--spare-nodes auto`).
- **Resumes** step-4400 (verified), so it continues the canonical
  trajectory rather than restarting.

## What to watch

- Head `8638793` goes R and logs `Training starts at step 4401` (resume),
  then `step: 44xx` advancing with loss ~2.5 / grad_norm bounded.
- Native auto-retry now scrapes bad nodes properly -- if a node fails, it
  should swap the *correct* host (unlike the legacy blind-swap that
  stranded this chain).
- First checkpoint save at step-4500 into
  `outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288`.

## Cross-refs

- Restart economics / blind-swap class:
  [`20260630-failover-restart-economics.md`](20260630-failover-restart-economics.md)
- 80B launch (same day, native auto-retry):
  [`20260628-80b-sophiag-constant-lr-512-1024-2048.md`](20260628-80b-sophiag-constant-lr-512-1024-2048.md)
