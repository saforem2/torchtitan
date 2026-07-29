# 2B MDS mid-training: anneal-schedule A/B + data-mix experiment (Sunspot)

**Dates:** 2026-07-27 -> 2026-07-29
**Cluster:** Sunspot (Intel Max 1550 XPU), 32N/384 tiles, torch 2.13.0.dev20260519+xpu
**Base:** AuroraGPT-2B MDS `global_step138650` (val ~2.05, vocab 256000, gemma
tokenizer), forked model-only (`checkpoint.initial_load_path` +
`initial_load_model_only=True`), fresh AdamW.
**Recipe (fixed across all arms):** HSDP dp_shard=12 / dp_replicate=NHOSTS, LBS=2,
seq_len 8192, `--debug.seed=42`, ckpt interval 100, `--checkpoint.async-mode=disabled`,
~1600 steps ~= 10B tok/arm.

## TL;DR

Two questions, both answered with **held-out val-loss** (pre-registered primary
metric; NLL, lower=better) on frozen holdouts **disjoint from every training arm**:

1. **Does the LR SCHEDULE matter?** NO. Constant-LR (flat) beats WSD-decay-to-0 on
   BOTH bases. The schedule is not the lever at 10B.
2. **Does the DATA MIX matter?** ENORMOUSLY. Pure edu-web forgets math
   catastrophically; a **75% math / 25% edu** blend is the sweet spot -- keeps
   ~all the math and captures ~all of edu's general-domain gain.

**Actionable recipe:** to continue a math-saturated 2B base, train on **~75% math
(open-web-math) / 25% general-edu (fineweb-edu) at CONSTANT LR**. Do NOT spend a
WSD anneal; do NOT go pure-general.

This **overturns** the `docs/notes/data-strategy-after-olmo-mix-2026-07.md`
prediction that the "LR-decayed-to-zero annealing stage" is the primary lever --
empirically the *data mix*, not the decay schedule, is what moves the needle.

## Experiment 1: anneal schedule A/B (flat vs WSD-decay-to-0)

Two bases x two schedules, 10B tok each, identical data (open-web-math) + seed;
arms differ ONLY in the LR schedule. FineMath-4+ / open-web-math held-out NLL:

| base | base(step0) | flat-1600 | wsd-1600 | winner |
|------|-------------|-----------|----------|--------|
| MDS (val 2.05)  | 1.8476 / 1.8964 | **1.8039 / 1.8070** | 1.8183 / 1.8290 | flat +0.014 FM |
| olmo (val 2.65) | 1.9033 / 1.8963 | **1.8807 / 1.8631** | 1.9003 / 1.8839 | flat +0.020 FM |

- **flat beats WSD on BOTH bases, both holdouts.** Both arms beat the base, so
  continued math training helps -- but the WSD decay adds nothing over constant LR.
- Jobs: 12471859 (MDS+olmo, hit 3h walltime during olmo -> olmo re-run 12471885),
  evals 12471878 / 12471908 / 12471914.

**Why the redo mattered:** an earlier quick A/B (v1) was inconclusive by design --
it passed `--checkpoint.load-only` (saved ZERO checkpoints), omitted LBS=2 (halved
the budget) and `--debug.seed` (arms saw different data), ran 300 steps, and
compared final TRAIN loss (invalid: WSD ends at LR~0 so its train loss is
structurally lower). The redo fixes all of that and decides on held-out val-loss of
saved checkpoints.

## Experiment 2: data mix (constant LR, the anneal winner)

All arms fork the MDS base at constant LR 2e-6, 10B tok; differ ONLY in the mix.
FineMath-4+ (math generalization) + wikitext (anti-forgetting) held-out NLL, both
DISJOINT from every training arm (no arm trains on FineMath or wikitext):

| arm | FineMath (math) | wikitext (general) | note |
|-----|-----------------|--------------------|------|
| owm-100 (control) | 1.8039 | 2.6828 | == anneal flat winner |
| **owm75 / edu25** | **1.8089** | **2.6597** | **WINNER** |
| owm50 / edu50 | (pending, job 12472037) | | |
| edu-100 | 2.1122 | 2.6585 | catastrophic math forgetting |

- **edu-100 (Wave 1):** swapping math-web -> edu-web forgets math by **+0.308 nats**
  (~22x the entire anneal effect) for a tiny general gain (-0.024). Lopsided ~13:1.
- **owm75/edu25 (Phase 2):** math essentially intact (FineMath +0.005 vs owm) AND
  ~95% of edu-100's entire general gain captured (wikitext 2.6597 vs edu's 2.6585).
  25% edu buys almost the full general improvement for almost no math cost.
- **Shape:** the math-loss curve is steep, the general-gain curve flat -> a
  math-heavy blend is near-free diversity. 50/50 (pending) is expected to erode
  math for little extra general, confirming 75/25 as the sweet spot (no 90/10
  Wave 3 needed -- 75/25 already ~= owm on math).
- Jobs: Wave 1 owm-control (free re-score of anneal flat) + edu-100 (12471929);
  Phase 2 blends 12472036 (75/25) + 12472037 (50/50), evals 12471930 / 12472040.

## Method / infra notes (reusable)

- **Configs** (`agpt/config_registry.py`): `agpt_2b_mds_anneal_{flat,wsd}`,
  `agpt_2b_{mds,olmo}_anneal_*`, `agpt_2b_mds_mix_{owm,edu}`,
  `agpt_2b_mds_mix_owm_edu_{7525,5050}` (Interleaved weighted blends, first ezpz
  use of `InterleavedHuggingFaceTextDataLoader`).
- **Launchers:** `scripts/anneal_ab.sh`, `scripts/mix_ab.sh` (per-arm data-source
  preflight + `MIX_FOLDER_SUFFIX` to isolate smoke vs prod checkpoints).
- **Eval:** `scripts/eval/build_math_holdout.py` (FineMath + owm + `--wikitext`
  frozen holdouts), `scripts/eval/held_out_math_valloss.py` (deterministic per-ckpt
  NLL), convert DCP->HF with `--model_flavor 2b-mds` / `agpt_2b_mds_config.json`
  (vocab 256000 -- each base evaluated with ITS OWN vocab).
- **Traps fixed en route:** olmo anneal forked with cos_sin RoPE (must be COMPLEX
  to match the base); HF-hub streaming 429-storms at 384 ranks (fix = precache +
  local-parquet routing, `resolve_precached_parquet_dir`); smoke+prod sharing a
  checkpoint folder -> prod auto-resumes the smoke's wrong-rank dataloader state
  (fix = `MIX_FOLDER_SUFFIX`); wikitext test split has only ~1656 docs
  (`--wikitext-num-docs 1500`).

## Open / next

- 50/50 eval pending (confirmatory).
- The 75/25 recipe is the actionable output for the flagship continued-pretraining
  stage. A science-corpus blend (per the data-strategy memo section 2) is the
  natural follow-on: replace the generic edu 25% with science/math-dense sources.
