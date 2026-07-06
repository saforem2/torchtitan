# Continued Pre-Training (CPT) — 2B olmo x dolmino mixing-ratio sweep

> **Living document** — updated as CPT runs complete and are evaluated.
>
> Last updated: 2026-07-06

## Motivation

The 2B 256N base **completed** stage-1 pretraining at step-92,859 (4.674T
tokens, 100%) and **plateaued**: over the final ~635B tokens its benchmarks
were dead flat (HellaSwag 0.560, ARC-Easy 0.651) and validation loss saturated
at **~2.80** on the olmo-mix-1124 distribution. See
[`docs/evals/agpt/2b`](../../evals/agpt/2b/README.md). More olmo-mix tokens
cannot help a saturated 2B, so the next lever is **continued pre-training on a
different, higher-quality data distribution.**

This sweep forks the plateaued base and continues training on varying blends of
olmo-mix-1124 x **dolmino-mix-1124** (the OLMo stage-2 high-quality
mid-training mix), to answer: *does a distribution shift move the plateau, and
does more dolmino help?*

## Result (pilot complete 2026-07-05)

![2B CPT loss curves](figures/cpt_loss.svg)

Both pilots forked the base (model-weights-only via
`--checkpoint.initial-load-path`, fresh optimizer + LR) and ran the full
~300B-token budget (step-5,960, 60 checkpoints each).

| Mix | CPT jobs | Start loss | End train loss | **Final val loss** | vs olmo-100 |
|-----|----------|-----------:|---------------:|-------------------:|------------:|
| **dolmino-100** | 8638977 + 8638979 | 7.27 | 2.480 | **2.492** | **-0.31** |
| **olmo50-dolmino50** | 8638978 + 8638980 | 7.24 | 2.590 | **2.601** | **-0.20** |
| olmo25-dolmino75 | (queued) | — | — | — | — |
| olmo-100 (control) | (base plateau) | — | ~2.80 | ~2.80 | 0 |

> [!WARNING]
> **The loss numbers are MISLEADING -- downstream eval (2026-07-06, job
> 8647850) shows this CPT recipe DEGRADES benchmarks.** dolmino-100
> step-5960: HellaSwag **0.486** (vs olmo-100 plateau 0.560, **-7.4pp**),
> ARC-Easy **0.547** (vs 0.651, **-10.4pp**), PIQA 0.687 (vs 0.733). The
> decline is monotonic over CPT steps, and olmo50-dolmino50 tanks
> **similarly** (HS ~0.49) -- i.e. the damage is ~independent of the dolmino
> ratio. Lower dolmino train-loss just reflects dolmino's lower-entropy
> (DCLM-heavy) distribution; the model over-specializes AWAY from the eval
> distribution. **Root cause hypothesis: the recipe re-warmed the converged
> base back to the FULL peak LR (2.28e-5), disrupting it (downstream
> forgetting).** The MDS reference ran continuous SophiaG @ 2.17e-5 and never
> re-warmed to peak for stage-2. **Gentler retry launched: dolmino-100 @
> LR=2e-6 constant, warmup=20 (job 8648114)** -- if it holds the plateau, the
> recipe was the culprit; if it also tanks, dolmino is wrong for downstream.

**Loss-only findings (do NOT imply capability gains -- see the eval warning above):**
1. **The distribution shift is real.** Both runs started at ~7.2 -- far above
   the base's 2.8 plateau -- because dolmino is genuinely out-of-distribution
   for the olmo-trained base.
2. **CPT beats the plateau ON LOSS** (but not on benchmarks): dolmino-100 to
   **2.49**, olmo50-dolmino50 to **2.60** vs olmo-100's 2.80.
3. **More dolmino = lower loss, monotonically.** dolmino-100 (2.49) <
   olmo50-dolmino50 (2.60) < olmo-100 (2.80). ~0.31 nats gained at 100%
   dolmino. The `olmo25-dolmino75` arm (data-list built) will fill in the
   ratio curve between 50 and 100.

> **Caveat:** these are training/validation-loss numbers on each run's *own*
> mixture, so cross-mix loss is not a perfectly apples-to-apples quality
> metric (dolmino may simply be lower-entropy). The decisive comparison is
> **downstream benchmark eval** of the CPT checkpoints vs the flat olmo-100
> tail -- pending (see Next steps).

## Method

- **Fork:** `--checkpoint.initial-load-path=<base step-92859>` +
  `--checkpoint.initial-load-model-only` (default) -> loads model weights only;
  fresh optimizer / LR schedule / step counter. The base's nested
  (pre-#3623) optimizer format is irrelevant (discarded).
- **Batch held:** 256N, LBS=2, GAS=1 -> **GBS=6144** (matches the base, so the
  LR=2.28e-5 calibration stays valid -- LR is batch-dependent).
- **Schedule:** LR 2.28e-5, re-warm 200 steps, then linear decay
  (`decay_ratio=0.8`). ~300B tokens (`TRAINING_STEPS=5960`).
- **Launcher:** `scripts/submit_agpt_2b_autoretry.sh` from the main clone
  (has `initial_load_path`), with CPT flags injected via `EXTRA_ARGS` and
  **`CHECKPOINT_ASYNC_MODE=disabled`** (the async `new_group(gloo)` path is
  XPU-broken on this torch build).
- **Data-lists** (`utils/build_cpt_mixes.py`): each source block renormalized
  to its target fraction so the blendcorpus weight column encodes the exact
  olmo:dolmino ratio (verified 0.50/0.50, 0.25/0.75; all `.bin` present).
- **Checkpoints:** `outputs/checkpoints/agpt-2b-cpt-<mix>-n256-gbs6144/`
  (note the `outputs/` prefix from `job.dump_folder=./outputs`). Base is
  read-only; each mix has a distinct dir (no collision); `keep-latest-k=0`.

## Token budget: pilot (300B) vs the reference stage-2 (2.4T)

The pilots run **~300B tokens** each -- this is a deliberately small **ratio
screen**, NOT a full stage-2. The MDS 2B reference recipe (which our v2 base
mirrors up to its stage-1 boundary) did a much larger stage-2:

| MDS stage | tokens | cumulative |
|-----------|-------:|-----------:|
| stage-1 (base pretrain) | 4.673T | 4.673T |
| **stage-2 (mid-training)** | **2.391T** | 7.064T |
| stage-3 (final anneal) | 0.706T | 7.770T |

Our completed v2 2B base = **4.674T = exactly the MDS stage-1 boundary**, so
the direct stage-2 analog is **~2.391T tokens** (~8x the 300B pilot). See
[`docs/evals/agpt/2b-mds`](../../evals/agpt/2b-mds/README.md).

## Plan: two-phase (screen cheap, scale the winner)

1. **Eval the 300B pilot checkpoints** (step-5,960 each: dolmino-100 +
   olmo50-dolmino50, + olmo25-dolmino75 once it runs) on the downstream
   benchmark suite, overlaid vs the olmo-100 flat tail -- the real quality
   test that decides the winning olmo:dolmino ratio.
2. **Fan out `olmo25-dolmino75`** (data-list ready) so the ratio screen covers
   0/50/75/100% dolmino.
3. **Scale ONLY the winning ratio to the full ~2.391T** (TRAINING_STEPS ~47,500
   at GBS=6144) -- the principled stage-2 match to the MDS reference. This
   two-phase approach (cheap 300B screen -> one expensive 2.4T run at the best
   mix) avoids paying the full stage-2 cost x3 across all ratios.

> **Why not run 2.4T up front:** at GBS=6144 a 2.391T run is ~47,500 steps
> (~5x the 512N 12h window -> multi-day, chained). Running that for 3 ratios
> before knowing which wins would be ~7T tokens of compute; the 300B screen
> costs ~1/8 of one and identifies the winner first.

## Cross-refs

- Launch/mechanism report:
  [`../../experiments/agpt/aurora/20260701-2b-cpt-olmo-dolmino-sweep.md`](../../experiments/agpt/aurora/20260701-2b-cpt-olmo-dolmino-sweep.md)
- Base plateau evidence: [`../../evals/agpt/2b`](../../evals/agpt/2b/README.md)
- Reproduce plot: `python3 docs/production/cpt/plot_cpt_loss.py --data-dir <tsv-dir>`
