# Production Training — agpt 80B

> Last updated: 2026-06-12

## 80B trajectory chart

**No live 80B production chart yet** — production has not begun (4N
smokes are ≤20 steps each, and 256N is currently blocked on the NaN
diagnosed [here](../../experiments/agpt/aurora/20260611-80b-n32-nan-diagnosis.md)).
The per-trajectory stub [`n4/README.md`](n4/README.md) holds the 4N
validation details, and the 256N production-fix recipe will be linked
here once `--debug.deterministic` n=64/128/256 sweep validates and a
production chain starts persisting ckpts.

For the cross-model view (2B + 20B together — 80B is not on it yet for
the reasons above), see [`../README.md`](../README.md).

## v2 status — 4N validated 2026-06-08; 256N still blocked

**Headline**: the 80B production stack was validated end-to-end on
Aurora on **2026-06-08** via an interactive 4N smoke that completed
through step-10 sync-checkpoint save cleanly
([n4/README.md](n4/README.md): 904 GB / 48 .distcp shards / .metadata —
matches the Sunspot 12468197 reference exactly). Required clearing 5
stacked bugs: repo 229 commits behind, broken venv symlink, blendcorpus
init-barrier deadlock at 4N+, missing `ZE_FLAT_DEVICE_HIERARCHY=FLAT`,
and an interactive-launch env block. The same software stack was then
tried at 256N.

**256N attempts (2026-06-08 / 06-09)**:

| Job ID | Date | LR | Result | Notes |
|--------|------|-----|--------|-------|
| 8530891 | 2026-06-08 | 1e-6 | **Loss NaN at step 2** | Step 1 clean (loss 12.94), step 2 grad_norm NaN, step 3+ loss NaN. Even at scheduler-clamped LR ≈ 1.8e-8, NaN persists. Open hypotheses: bf16 overflow at GBS=1536, TP=2 loss-reduction bug, fp32 second-moment overflow. |
| 8531345 | 2026-06-08 | 1e-7 | OOM / failover-exhausted | Retry with smaller LR; bad-node hit storm, never reached step 1. |
| 8531721 | 2026-06-09 | 1e-7 | `std::bad_alloc` at model construction | Ranks 401-528 (contiguous block on 11-12 nodes) — failover ran out of spares. Never reached step 1. |

### NaN onset is GBS-dependent — threshold is GBS ∈ (168, 372] (2026-06-24)

The 2026-06-24 multi-node functionality sweep (LR=1e-6, the *working*
4N recipe, q_BLNH fix in place) shows the NaN is **batch-size driven,
and the onset GBS is far lower than the 256N/GBS=1536 where we first
saw it**:

| Job ID | Nodes | GBS | Result | step-1 → step-3 |
|--------|-------|-----|--------|------------------|
| 12469486 | 28 | 168 | **clean 20 steps** | 12.949 → 12.856 (descends to 10.38 @ step 20) |
| 12469492 | 62 | 372 | **NaN at step 3** | step 1 loss 12.930 / grad_norm 7.91 (clean); step 2 loss 12.898 / **grad_norm NaN**; step 3+ loss NaN |

So a clean 80B run is reproducible at GBS=168 and breaks by GBS=372,
identical LR/recipe. Same signature as the 256N failure: **grad_norm
goes NaN one step before the loss does** — i.e. the blow-up is in the
gradient/optimizer path (grad reduction, clipping, or fp32 second-moment),
NOT the forward loss value (still finite the step grad_norm first NaNs).
This rules out the "bf16 overflow at huge GBS=1536" framing — it happens
at GBS=372 too — and points the diagnosis at the TP=2 grad path or
optimizer. The grad_norm-first ordering is the strongest lead.

**Status:** 80B production at GBS>168 is **blocked on this NaN**. The
GBS=372 repro is far cheaper to debug than 256N — next step is a
node-count sweep at fixed small GBS (does 62N @ GBS=168 also NaN? that
would isolate scale vs batch-size) plus instrumenting the grad path at
the step-2 grad_norm-NaN boundary. Diagnostic ideas in `Next steps`.

## Working 4N stack (Aurora + Sunspot)

The working 80B config has been replicated four times across the
two clusters with bit-equivalent numerics:

| Job ID | Cluster | Date | Stack | Loss (step 1 → 20) | MFU |
|--------|---------|------|-------|---------------------|-----|
| 12466025 | Aurora | 2026-05-05 | torch 2.13, pre-xccl-workaround | 12.98 → 10.46 | ~17.8% |
| 12467825 | Sunspot | 2026-06-02 | + xccl_split_group workaround (commit 8031d1d3a) | 12.94 → 10.39 | ~17.8% |
| 12468157 | Sunspot | 2026-06-06 | + 47th upstream sync (RoPE refactor PR #3458, mixed-optimizer PR #3269) | 12.97 → 10.41 | ~17.8% |
| 8530800 (r4 smoke) | Aurora | 2026-06-08 | + blendcorpus barrier fix + venv-symlink fix + `ZE_FLAT_DEVICE_HIERARCHY=FLAT` | step-10 ckpt saved (904 GB, 48 shards) | — |
| 12469486 (**28N**) | Sunspot | 2026-06-24 | + 57th/58th upstream sync (AC policy hierarchy, disable_loss_parallel removal, **q_BLNH attention fix** for TP>1 local_map) | 12.95 → 10.38 | ~18.7% |

The first four are 4N and land within ±0.08 nat of one another at
step 20 (88.94% peak memory). **`12469486` is the first multi-node
(28N) validation** -- loss 12.95 → 10.38 matches the 4N baseline
within noise at ~18.7% MFU / 65.86% memory, `step-20` ckpt saved.
It also flushed out a TP>1-only regression from the 57th sync: the
ezpz attention forks used bare `q/k/v` forward args while upstream's
`set_gqa_inner_attention_local_map` now keys `in_dst_shardings` by the
shape-suffixed `q_BLNH/k_BLNH/v_BLNH`; the local_map contract check
asserts under TP>1. Fixed in `74c7452f6`; `sync_smoke.sh` gained a TP=2
entry (`13c09ddf9`) so future syncs catch it. See
[../../../upstream-sync.md](../../../upstream-sync.md) 57th-sync
follow-up. See [n4/README.md](n4/README.md) for the 4N smokes.

**Working config**: AdamW LR=1e-6, TP=2, AC=full, compile=OFF,
fp32-master, sync checkpoint mode (`CHECKPOINT_ASYNC_MODE=disabled`).
Submit script:
[`scripts/submit_agpt_80b_aurora_venv_failover.sh`](../../../../scripts/submit_agpt_80b_aurora_venv_failover.sh).
Production clone: `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/`.

## Per-trajectory detail

- [n4/](n4/README.md) — 4N validation smokes (Aurora + Sunspot,
  proven-stable end-to-end including sync ckpt save 2026-06-08)
- [n512/](n512/README.md) — 80B 512N (not yet attempted — 256N still
  blocked)

## Next steps

1. **Diagnose the 256N NaN.** Possible angles:
   - Try TP=4 to halve effective per-replica GBS (current TP=2, GBS=1536)
   - Try with `--validator.enable` off to rule out validator-loss-path issues
   - Dump per-tensor stats at step 1 to localize which weight first goes NaN
   - Bisect on bf16-vs-fp32 master at 256N (4N is fp32-master, working)
2. **256N spare headroom**: every 256N retry has been killed by bad
   nodes long before reaching meaningful training. The wrapper rotates
   spares correctly but the hit-rate at 256N is high; try `select=296`
   (40 spares vs current 10).
3. **Replay 80B once 2B 512N stalls clear** so we're not competing for
   the same 256N+ slots in `small` while debugging.

## Historical

- Pre-2026-06-08 80B 256N production: 11+ failed dispatches starting
  2026-05-11. Failure-mode writeup:
  [20260524-80b-256n-sigsegv-cascade-8505222.md](../../../experiments/agpt/aurora/20260524-80b-256n-sigsegv-cascade-8505222.md).
  Distinct from the 80B 8N smoke failure (`blendcorpus` EOFError race —
  fixed by the init-barrier removal that landed in 2026-06-08's r4 smoke):
  [blendcorpus-eoferror-race.md](../../../guides/known-bugs/blendcorpus-eoferror-race.md).
- Historical v1 (NaN'd, bf16-master) runs:
  [../historical/v1-bf16/](../historical/v1-bf16/README.md).
