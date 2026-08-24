# 2B continued-pretraining (CPT): olmo x dolmino mixing-ratio sweep

- **Date:** 2026-07-01
- **Machine:** Aurora
- **Status:** PILOT LAUNCHED (2 of 3 mixes). Heads `8638977` (dolmino-100)
  + `8638978` (olmo50-dolmino50), each with an `afterany` continuation
  (`8638979`/`8638980`). All Q.

## Why

The 2B 256N base **completed** stage-1 at step-92,859 (4.674T tokens) and
its benchmarks are **plateaued** -- over the final ~635B tokens HellaSwag
moved 0.5595->0.5610 and ARC-Easy held 0.651+-0.003 (see
[`docs/records/evals/agpt/2b`](../../../evals/agpt/2b/README.md)). More olmo-mix
tokens cannot help a saturated 2B, so the next lever is **continued
pretraining on a different data distribution**. This sweep measures how
much a higher-quality mid-training mix (dolmino-mix-1124) lifts the
plateaued benchmarks, and at what **olmo:dolmino blend ratio**.

## Design

Fork the plateaued base (model weights only; fresh optimizer + LR
schedule + step counter) via `--checkpoint.initial-load-path`, then CPT
on each mix.

| Experiment | Mix | Head | Cont |
|---|---|---|---|
| dolmino-100 | `dolmino-mix-1124` | 8638977 | 8638979 |
| olmo50-dolmino50 | `olmo50-dolmino50` | 8638978 | 8638980 |
| olmo25-dolmino75 | `olmo25-dolmino75` | (fan out after pilot) | |
| control | olmo-100 | (existing plateaued tail, no run) | |

**Per run:** 256N, `LBS=2 GAS=1 -> GBS=6144` (held to the base's batch so
the LR=2.28e-5 calibration stays valid -- LR is batch-dependent), LR
2.28e-5 re-warm 200 steps then linear decay (`decay_ratio=0.8`), ~300B
tokens (`TRAINING_STEPS=5960`), validator on, distinct `CKPT_DIR`
(`checkpoints/agpt-2b-cpt-<mix>-n256-gbs6144`), `select=260` (256+4 spare).

**Data-lists** (built 2026-07-01, `utils/build_cpt_mixes.py`): the olmo
and dolmino source lists use different weight-sum conventions (2.717 vs
1.0), so each source block is renormalized to its target fraction before
concat -> each output sums to 1.0 and the weight column encodes the
olmo:dolmino ratio exactly (verified 0.50/0.50, 0.25/0.75; all `.bin`
present).

## Checkpoint isolation (no overwrite risk)

- The base (step-92,859) is **read-only** via `initial-load-path`, and
  lives in a **different clone** (`runs/agpt-2b-v2`) than where CPT writes
  (main clone `checkpoints/`).
- Each CPT run writes a **distinct, uniquely-named** `CKPT_DIR`
  (`agpt-2b-cpt-<mix>-*`) that collides with no existing chain
  (`agpt-2b-sophiag-*`, `agpt-20b-*`).
- `CKPT_KEEP_LATEST_K=0` hardcoded (no purging).
- **Continuations omit `initial-load-path`** -- they resume the CPT
  chain's own checkpoint (which has step-N by then), not re-fork the base.

## Fork mechanism verified (smoke 8638933, 2N)

- `initial-load-path` binds (config `initial_load_path` set) and loads
  step-92,859 in 35s with **zero missing/mismatch keys**.
- Loss descends 7.20 -> 5.93 over 5 warmup steps. Starting at 7.2 (not the
  base's plateau ~2.8) is the **informative CPT signal**: dolmino is a
  genuinely different distribution from olmo (higher perplexity for the
  olmo-trained model), which is exactly the gap CPT aims to close. The
  rapid drop confirms the weights loaded and are adapting (a fresh-random
  model would start ~10-11).
- The `initial_load_model_only=True has no effect...` warning is a benign
  `__post_init__` ordering artifact (the path is bound and used).

## Gotcha: async checkpointing is XPU-broken (must disable)

The **first** smoke (8638909) crashed at CheckpointManager init with
`RuntimeError: No backend type associated with device type xpu` in
`checkpoint.py:484` (`dist.new_group(backend="gloo")` -> torch probing
`_get_backend(device_id)` on an XPU device). This path only runs when
`async_mode` is ASYNC. The **2b autoretry script defaults
`CHECKPOINT_ASYNC_MODE=async`** (the 20b script defaults to `disabled`,
which is why the 20b relaunch smoke was fine). **All 2b CPT runs pass
`CHECKPOINT_ASYNC_MODE=disabled`** to skip the broken async gloo/XPU path.
TODO: fix the 2b script default (or the core async gloo path) for XPU.

## Script change

Added an `EXTRA_ARGS` env passthrough to `submit_agpt_2b_autoretry.sh`:
under PBS, `qsub` cannot forward trailing `"$@"` args to a `#PBS` script,
so CPT flags (`--checkpoint.initial-load-path`, `--lr-scheduler.*`) are
passed via `-v EXTRA_ARGS="..."` (whitespace-split, empty by default so
existing invocations are unaffected; interactive `"$@"` still works).

## What to watch

- Heads go R, log `Loading the checkpoint from .../step-92859` +
  `Training starts at step 1` (fresh CPT counter), loss starts elevated
  (dolmino ~7, olmo50 lower) and descends.
- As CPT ckpts land, eval each vs the **olmo-100 plateau baseline** at
  matched CPT-token counts (the flat tail is the control). The question:
  does the dolmino shift move HellaSwag/ARC/etc. off the plateau, and does
  more dolmino (100 vs 50 vs 25) help monotonically?

## Cross-refs

- Base eval + plateau: [`docs/records/evals/agpt/2b`](../../../evals/agpt/2b/README.md)
- Data-list builder: `torchtitan/experiments/ezpz/utils/build_cpt_mixes.py`
