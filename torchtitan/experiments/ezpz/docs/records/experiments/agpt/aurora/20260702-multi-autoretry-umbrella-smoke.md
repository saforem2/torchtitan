# Multi-chain umbrella on native auto-retry: diagnosis + smoke

- **Date:** 2026-07-02 (Aurora, ~02:40-03:45 UTC)
- **Status:** SMOKE PASSED. `smoke_multi_autoretry.sh` job `8639375`,
  2 trainers, `failed: 0 / 2`.

## Why

The production multi-chain umbrella `8568429`
(`submit_agpt_multi_aurora_venv_failover.sh`, legacy `failover_lib.sh`)
exited **3** after ~9.5h -- it lost **3 of 4** co-allocated chains:

| Trainer | rc | Root cause |
|---|---|---|
| 2b n512 | 143 | `rank 3864 died from signal 11` on x4409 -> `failover_lib.sh` blind-swap `max retries (3) exhausted` |
| 20b n512 | 143 | same bad-node/blind-swap class (relaunched separately as `8638793`) |
| 2b n256 | 0 | clean |
| 20b n256 | 127 | node x4208 **unreachable** (`ping failed ... No reply after 97s`) mid-run @ step-2,186 -> failover exhausted |

Common thread: every failure is the **legacy `failover_lib.sh` giving up on
recoverable bad-node events** -- the same blind-swap defect (scraper can't
parse the hostname out of a `died from signal 11` line, rotates innocent
spares) that stranded the 20B 512N chain. Native `ezpz launch --auto-retry`
(ezpz >= 0.17.1) has a better scraper, so the fix is to move the umbrella off
`failover_lib.sh`.

## Smoke: `smoke_multi_autoretry.sh` (native auto-retry umbrella)

> [!IMPORTANT]
> **VERDICT: native auto-retry umbrella PASSED (`failed: 0/2`, job 8639375).**
> Two co-allocated trainers, each launched via `ezpz launch --auto-retry`,
> ran clean to step 20 with auto-retry armed per trainer (`active=2 spare=1`),
> distinct node-slices/ports/CKPT_DIRs, one shared venv broadcast, and **zero
> cross-talk / zero error signals**. This is the validated replacement for the
> legacy `failover_lib.sh` umbrella (whose blind-swap lost 3/4 chains in
> `8568429`). Not yet promoted to production -- prod multi-venv (2B + 20B in
> one alloc) still needs the pre-stage-to-distinct-`/tmp/.venv-<model>` step.

A small, self-contained umbrella that mirrors the co-allocation pattern but
launches each trainer with **`ezpz launch --auto-retry`** directly.

Design decisions:
- **ONE venv broadcast for the whole allocation** (all trainers share the 2B
  venv), so there is no concurrent-yeet collision on the hardcoded
  `/tmp/.venv` (the single-job-per-alloc hazard that corrupts N simultaneous
  yeets). `ezpz launch --auto-retry` does NOT yeet -- it takes `--hostfile`
  + `--nproc` + `--spare-nodes` and runs against the already-staged venv.
- **Split `PBS_NODEFILE`** into per-trainer slices (active + per-trainer
  spare); fire N concurrent `ezpz launch --auto-retry` with distinct
  `MASTER_PORT`, `CKPT_DIR`, and a 15s stagger.
- `--checkpoint.async-mode=disabled` (the async `new_group(gloo)` path is
  XPU-broken on this torch build -- see the CPT sweep report).

**Run:** `qsub -l select=6 -v NUM_TRAINERS=2,TRAINER_NNODES=2,TRAINER_SPARES=1,TRAINER_STEPS=20`
(debug-scaling, 1h). 2 trainers x (2 active + 1 spare) = 6 nodes.

**Result (8639375):**
- `failed: 0 / 2` -- both trainers `OK` to step 20.
- Per trainer: `[auto-retry] attempt 1 -- active=2 hosts, spare=1`,
  `Training starts at step 1`, own nodefile slice, distinct SMOKE `CKPT_DIR`.
- **0 error signals** in the top log (no `No backend`/xpu, no SIGSEGV, no
  `Address already in use` port collision, no CheckpointManager `stager`
  crash). No cross-talk between the concurrent trainers.
- One venv broadcast served both; no `/tmp/.venv` corruption.

## Conclusion + next steps

The **native auto-retry umbrella pattern works** at small scale -- concurrent
co-allocated `ezpz launch --auto-retry` trainers run cleanly off a single venv
broadcast. This is the intended replacement for the legacy `failover_lib.sh`
umbrella that lost 3/4 chains.

Follow-ups:
- The **2B 512N chain (umbrella trainer-0) is still down** and not yet
  relaunched (only the 20B 512N was, as `8638793`). Relaunch it on
  `submit_agpt_2b_autoretry.sh` (same recipe as the 20B relaunch), or fold
  both 512N chains into a native auto-retry umbrella.
- Consider promoting `smoke_multi_autoretry.sh` into a production multi-chain
  umbrella (bigger slices, real CKPT_DIRs, per-model venvs) to replace
  `submit_agpt_multi_aurora_venv_failover.sh`. Multi-venv (2B + 20B in one
  alloc) still needs the pre-stage-to-distinct-`/tmp/.venv-<model>` approach
  the legacy umbrella uses, since `ezpz launch` alone won't broadcast.

## Production promotion: `submit_agpt_multi_autoretry.sh` (2026-07-06)

Promoted the smoke into a production 4-chain umbrella
`scripts/submit_agpt_multi_autoretry.sh` that advances all four canonical
chains (2B-512, 20B-512, 2B-256, 20B-256) in one allocation. Default layout
1536 train (512+512+256+256) + 4*SPARES(10) = `select=1576`.

Design (differs from the single-venv smoke):
- **Per-trainer `cd` to its OWN pinned clone.** torchtitan is imported from
  cwd (no editable install in site-packages), and the 20B clones are rolled
  back pre-#3623 for DCP-resume compat -- so the cwd is load-bearing, not
  cosmetic. Getting it wrong silently breaks a chain's resume.
- **Per-model venv pre-stage** to `/tmp/.venv-<model>`, deduped by RESOLVED
  tarball (20B-n256 symlinks 20B-v2's -> ONE broadcast). `ezpz launch
  --auto-retry` only READS the venv, so concurrent disjoint slices never
  collide on it.
- **Inline training command** = faithful copy of the
  `submit_agpt_{2b,20b}_autoretry.sh` flag set, so a bug in a pinned clone's
  own script cannot break the umbrella and there is no palsd cross-kill.

**Login-node dry-run (DRY_RUN=1, fixture 1576 hostfile): PASSED.** 4 disjoint
slices (522+522+266+266=1576), correct clones + ckpt dirs, venv dedup verified
(trainers 0+2 -> `/tmp/.venv-2b`, 1+3 -> `/tmp/.venv-20b`).

### Smoke `8648251` (MULTI_PROFILE=tiny, select=12) -- FAILED, then fixed

> [!IMPORTANT]
> The tiny smoke (4x 2B, throwaway ckpt dirs) came back **`failed: 4/4`** --
> and caught a real prod blocker before the 1576 submit. Root cause: the
> **`agpt-2b-v2` clone venv had ezpz 0.16.0**, which predates `--auto-retry`.
> Old ezpz does not recognize the flag, so `--auto-retry --spare-nodes 1`
> leaked PAST the `--` into `cmd_to_launch` and mpiexec rejected it
> (`--cpu-bind` usage dump, exit 1 in ~1s). The 20B-v2 venv was already
> 0.21.3, so only 2B was affected. (The original single-venv smoke `8639375`
> passed because it ran from the MAIN clone's venv, not the runs-clone
> tarballs.)

**Fix:** upgraded ezpz in the `agpt-2b-v2` venv `0.16.0 -> 0.21.5`
(`uv pip install --no-deps --no-cache 'ezpz @ git+https://github.com/saforem2/ezpz@main'`
-- torch build untouched, verified `2.13.0.dev20260428+xpu` intact and
`--auto-retry` now in `ezpz launch --help`). Rebuilt the `agpt-2b-v2`
`.venv.tar.gz` via the fast tmpfs surgical-swap (decompress good tarball in
`/dev/shm`, replace just the `ezpz` package + dist-info, re-tar) so the
broadcast `/tmp/.venv-2b` carries the new ezpz. Old tarball backed up
(`.venv.tar.gz-20260706-211636`).

## Cross-refs

- 20B 512N relaunch (same bad-node class):
  [`20260701-20b-512n-relaunch-autoretry.md`](20260701-20b-512n-relaunch-autoretry.md)
- Restart economics / blind-swap defect:
  [`20260630-failover-restart-economics.md`](20260630-failover-restart-economics.md)
- async/XPU checkpoint bug:
  [`20260701-2b-cpt-olmo-dolmino-sweep.md`](20260701-2b-cpt-olmo-dolmino-sweep.md)
