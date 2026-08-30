# Production Training — agpt 20B @ 256 nodes

> **Eval scores:** see [`docs/evals/agpt/20b/`](../../../../evals/agpt/20b/README.md)
> for the v2 lm-eval results.

## v2 — 20B @ 256N — SophiaG LR=2.28e-5 (fp32 master)

> Last updated: 2026-08-30
>
> Status: **NOT RUNNING.** The last leg to train this chain was umbrella
> `8773440` seat t2 on **2026-08-26** (**201 steps, loss 2.31488 ->
> 2.24577**); nothing on Aurora has trained since. Chain head **step-12,000**
> persisted (**604.0B tokens, 12.9%** of 4.67T, 137 step dirs verified on disk
> 2026-08-21) -- the 08-26 leg resumed from that head, and its resulting
> checkpoint head has not been re-verified on disk here. `8773440` used 5h13m
> of a 12h slot and exited `-14`. The successor umbrella `8784460` (2,098
> nodes, `large`) has been `Q` since 2026-08-26 13:22 UTC with 101h+ eligible
> and `score_boost=0`, with `8784462` held behind it; a ticket is drafted but
> NOT sent
> ([`ops/alcf-ticket-8784460-not-scheduling-20260830.md`](../../../../ops/alcf-ticket-8784460-not-scheduling-20260830.md)).
>
> Trajectory since the step-300 stall: `8505255`
> (sync mode) reached step-1,125; `8558548`/`8558549` carried it
> 1,101 -> 3,136; native-autoretry continuations (`8647385`, `8661054`)
> advanced it 3,101 -> 4,375; `8681340` (260N) resumed to step-6,000; a 16N
> `capacity`-queue bridge (`8703284`, run `v58n7vam`) then carried it
> 6,000 -> 6,800. **Bridge retired 2026-07-29** after it collided with a
> concurrently-scheduled umbrella trainer on the shared ckpt dir (corrupted
> step-6,200 + step-6,300, both quarantined; latest CLEAN ckpt AT THAT TIME =
> step-6,800).
> From 2026-08-05 on, the chain has been carried by the successive
> **5-chain ~2,098N umbrellas** (`8714502` resumed step-6,800, then `8714503`,
> `8744245`/`8744247`, `8756070`, `8760249`, `8764675`, and finally `8773440`
> on 08-26); a small `8698753` (260N) resume served as queue backfill.
> Individual + umbrella are mutually guarded (handoff-kill on umbrella start)
> so they never co-write the dir again. Per-umbrella step ranges are in the
> [dispatch log](../../../dispatch-log.md).
> NOTE: the run-ids for the 1,101->3,135 segment
> were originally recorded as the `ezpz.examples.test` preflight-smoke runs;
> corrected 2026-07-24 to the real `torchtitan.ezpz.train` run-ids (the
> metrics were always in the right project), so the charts now source them
> natively (no .o-log fallback).
> **Relocated 2026-06-12** to its own clone
> `agpt-20b-n256/` (metadata-only `mv`, to relieve Lustre dir-size on the
> `agpt-20b-v2/` subtree) so it can be re-armed independently of the
> canonical 512N chain.
>
> **Re-arm blocked then fixed (2026-06-14 → 16):** first two re-launches
> from the new clone (`8540345`, `8540346`) both died in <10s with
> `ModuleNotFoundError: No module named 'spmd_types'` — the symlinked
> `.venv.tar.gz` (2026-05-23) predated the spmd_types==0.2.1 install, so
> the broadcast `/tmp/.venv` lacked it. Fixed 2026-06-16: installed
> spmd_types into the venv (`--no-deps`, torch untouched), rebuilt the
> tarball, re-verified. Chain re-submitted as `8558548` (head, resumes
> step-1,100) + `8558549` (cont1, `afterany`).
>
> Independent trajectory from the canonical 512N chain
> ([n512/](../n512/README.md)) — different ckpt dir (`gbs6144` vs
> `gbs12288`), so it can't extend that chain — but useful as a per-token
> comparator at the same optimizer state.

| Field | Value |
|-------|-------|
| Clone | `/flare/AuroraGPT/foremans/runs/agpt-20b-n256/torchtitan-ezpz/` (relocated 2026-06-12 from `agpt-20b-v2/` to reduce Lustre dir-size pressure on the v2 subtree; `mv` was metadata-only on same FS, no data copy) |
| Submit script | [`scripts/submit_agpt_20b_aurora_venv_failover.sh`](../../../../../scripts/submit_agpt_20b_aurora_venv_failover.sh) (one script handles all v2 node counts via env vars) |
| Stack | torch 2.13 venv (yeet-env tarball mode; venv symlinked from `agpt-20b-v2/`) |
| Optimizer | SophiaG, LR=2.28e-5 |
| Compile | on |
| GBS | 6,144 (LBS=2) |
| Total steps | 92,859 |
| Total tokens | 4.67T |
| Checkpoint dir | `outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144` (under new clone) |
| Checkpoint interval | 100 steps, keep_latest_k=0 (keep all) |
| W&B | [r1yyxbmt](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/r1yyxbmt) |

### Loss / Throughput / MFU

![20B v2 256N Training](figures/production_20b_v2_256n.svg)

### Diagnostics

![20B v2 256N Diagnostics](figures/training_diagnostics_20b_v2_256n.svg)

### Tokens vs Wall Clock

![20B v2 256N Tokens vs Time](figures/tokens_vs_time_20b_v2_256n.svg)

### Progress

| Job ID | Date | Walltime | Steps | Loss (start → end) | TPS/GPU | MFU | Status |
|--------|------|---------:|------:|-------------------:|--------:|----:|--------|
| [`8463659`](#log-8463659) | 2026-05-04 | 12h | 1–364 | 12.96 → 4.61 | 21-410 (variable) | 1-20% (variable) | **NODE_FAIL** after step 364 (`shepherd died from signal 9` on `x4406c6s7b0n0`, exit -20). step-100/200/300 ckpts saved. |
| [`8470102`](#log-8470102) | 2026-05-08 | 12h | 300–~500 | 4.61 → ~5.5 | varies | varies | **Crashed** @ 3h15m (gloo TCP timeout `Connection closed by peer`, multiple ranks). step-400 ckpt saved. |
| [`8470103`](#log-8470103) | 2026-05-08 | 12h | 300–~500 | (resumed but) | — | — | **Crashed** @ 2h59m (also gloo TCP timeout). |
| [`8479581`](#log-8479581) | 2026-05-11 | 12h | 400–500 | (resumed) → 4.08 | 31-419 (variable) | 1.6-20.9% (variable) | **Crashed** @ 3h39m (also gloo TCP timeout, peer 10.115.83.2; exit 0). step-500 ckpt saved. |
| [`8479582`](#log-8479582) | 2026-05-11 | 12h | 500+ | — | — | — | Released, Q to resume from step-500 |
| [`8481646`](#log-8481646) | 2026-05-21 | 12h | 301-500 (logged) | 4.95 → 4.12 | ~405 (steady) | **~20%** | **Failover wrapper validated end-to-end**: attempt 1 ran 2h33m, hit gloo crash from `x4110c3s3b0n0`, wrapper auto-swapped in spare `x4114c7s4b0n0`. **No new ckpts persisted** — `step-400` ckpt dir is empty (May 8 stale from `8470102`) and `step-500` was never written (async save killed by walltime). Attempt 2 only had 98s of parent walltime left. The wrapper's swap+retry path is proven; the training-progress contribution is zero. See [failover writeup](../../../../experiments/agpt/aurora/20260521-failover-validated-8481646.md). |
| `8505255` | 2026-05-22 | 12h | 300 → **1,125** | 4.12 → **3.28** | ~480 (steady) | ~24% | **Broke the step-300 stall.** Sync-mode 12h dispatch, advanced step-300 → step-1,125 cleanly. Persisted step-400..1,100 (11 valid ckpts). `step-400.bak-20260523-095600` is the old empty placeholder, renamed. |
| `8540345` | 2026-06-14 | 12h | — | — | — | — | **Died <10s** (`ModuleNotFoundError: spmd_types`). First re-launch from the relocated `agpt-20b-n256/` clone; symlinked tarball predated the spmd_types install. All 4 failover attempts failed identically (venv bug, not bad nodes). No ckpts. |
| `8540346` | 2026-06-16 | 12h | — | — | — | — | Same `spmd_types` failure (cont1 of 8540345, auto-released). No ckpts. |
| `8558548` | 2026-06-16 | 12h | 1,100 → ? | — | — | — | **Re-submit after tarball fix** (spmd_types==0.2.1 installed + rebuilt 2026-06-16). Resumes from step-1,100. Q at submit. |
| `8558549` | — | 12h | (cont1) | — | — | — | Held (`afterany:8558548`). |
| `8647385`+ | 2026-07-06..23 | 12h | 3,100 -> 4,375 | ~3.2 -> ~2.5 | ~440 | ~22% | Continuations 8647385 (3101-3603), 8661054 (4201-4375) advanced the chain; interspersed with CCL/gloo crashes + resubmits (transient infra). |
| `8681340` | 2026-07-23 | 12h | 5,101 -> **6,000** | 2.485 -> 2.533 | 40-443 (variable) | ~22% | **Done.** Full 260N throughput, resumed the real chain from `step-5100` and wrote `step-5200`..`step-6000` (3,072 shards each). Was marked RUNNING here through 2026-08-30; it is long finished and has aged out of PBS history. |
| umbrellas `8698125`..`8764675` | 2026-07-28..08-20 | 12h-24h | 6,151 -> **12,000** | 2.41007 -> **2.43380** | ~370 (mean) | ~18.5% (mean) | Carried by the ~2,098N 5-chain umbrella seat t2 across eight dispatches, with the 16N `8703284` bridge and the 260N `8698753` backfill in between. TPS/MFU are means over the committed metric store, which covers 6,151-10,380 of this range. Per-dispatch step ranges: [dispatch log](../../../dispatch-log.md). |
| `8773440` t2 | 2026-08-26 | 12h | **201 steps** on top of the step-12,000 head | **2.31488 -> 2.24577** | -- | -- | **Last leg to train this chain.** Umbrella used 5h13m of a 12h slot, `Exit_status=-14`; three of five seats trained, both 512N seats never started. The seat was abandoned early as `stuck_pre_training` because ezpz 0.21's progress regex matched `step=` while torchtitan prints `step: ` -- fixed in ezpz 0.27.3, pinned by [`tests/failover/test_progress_marker_contract.py`](../../../../../tests/failover/test_progress_marker_contract.py). **Nothing has trained since.** |

**Latest checkpoint:** step-12,000 (137 step dirs, audited on disk 2026-08-21).
The 08-26 leg (`8773440` t2) trained 201 further steps on top of this head; its
resulting checkpoint head has not been re-verified on disk, so the last
disk-confirmed figure is the one quoted.

**Cumulative persisted steps:** 12,000 (disk-confirmed)

**Tokens consumed:** 12,000 x 6,144 x 8,192 = **604.0B tokens** (12.9% of 4.67T
target)

**Loss:** 2.24577 at the end of the 08-26 leg (`8773440` t2, last logged step).
The committed metric store here stops at step-10,380 / ~2.372 (`lc9oukel`); it
was exported 2026-08-17 and predates the 08-20 and 08-26 legs.

> **Corrected 2026-08-17** (and again 2026-08-30, see the status header).
> This page previously carried three different step
> counts -- 6,800 in the header, 9,400 here, and a loss quoted at step-8,334 --
> none of which matched the 10,300 on disk. The 8,334 figure was not a typo: it
> was the true head of the chain's *plotted* data, because three W&B runs
> (`djmhgmmq`, `t9vly2u8`, `lc9oukel`) carrying 2,699 steps were never
> registered in `trajectories.py`. Every chart, the live board, and the
> exported metric store stopped at 8,334 while training had reached 10,300 --
> the chain was under-reporting its own progress by roughly 25%. Registered in
> `777941c64`; see
> [unregistered W&B runs](../../../../guides/known-bugs/unregistered-wandb-runs.md).
>
> Note the checkpoints live in this chain's OWN clone under
> `/flare/AuroraGPT/foremans/runs/agpt-20b-n256/`, not the main repo tree.

## v2 -- 20B @ 256N -- job 8661913 (MISLABELED "constlr"; actually from-scratch)

> **Correction (2026-07-12):** this run was originally recorded here as a
> "constant-LR fork from the trained base." That was WRONG. Inspecting the
> resolved config from its own log shows it was a **fresh from-scratch run on
> the DEFAULT schedule**, not a fork and not constant-LR:
> `initial_load_path=null` (nothing loaded), `decay_ratio=0.8`,
> `warmup_steps=200`, `--optimizer.lr=2.28e-5`. Loss starts at **12.21 @ step-10
> and drops steeply** (12.21 -> 11.0 in 6 steps) -- the classic random-init
> warmup signature, not a continuation of the base (which was already ~2.7).
> The `-constlr` in the ckpt-dir name came from an intended config that never
> reached the command line.

- **Job:** `8661913` (256N, 6h, legacy failover wrapper). Ran 2026-07-11 21:59.
- **What it actually was:** a duplicate 20B-256 training run from random init on
  the standard warmup+linear-decay schedule (same optimizer/LR as the canonical
  chain), writing to a NEW dir `...-gbs6144-constlr`.
- **Result:** reached only **step-400** (still in early warmup, loss ~4.3) before
  it **silently hung after ~step 435** and sat idle until PBS walltime-killed it
  at 04:01. No NaN; MFU ~22%; checkpoints step-50..step-400 saved (~385s/save).
- **Disposition:** NOT resumed -- it would just duplicate the canonical 20B-256
  chain (step-4,200 at the time of that decision; step-12,000 now) at a worse
  loss. The mislabeled ckpt dir can be
  archived/removed. A REAL constant-LR experiment is now handled differently:
  the canonical chains hold LR flat via `DECAY_RATIO=0.0` in
  `submit_agpt_20b_autoretry.sh` (see 2026-07-12 journal entry).


### Logs

| Job ID | Path |
|--------|------|
| <a id="log-8463659"></a>`8463659` | `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n256-v2.o8463659` |
| <a id="log-8470102"></a>`8470102` | `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n256-v2-chain1.o8470102` |
| <a id="log-8470103"></a>`8470103` | `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n256-v2-chain2.o8470103` |
| <a id="log-8479581"></a>`8479581` | `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n256-v2-chain3.o8479581` |
| <a id="log-8479582"></a>`8479582` | `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n256-v2-chain4.o8479582` |
| <a id="log-8481646"></a>`8481646` | `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n256-v2-failover-chain1.o8481646` + `logs/failover-8481646/{attempt-1,attempt-2}.log` |
