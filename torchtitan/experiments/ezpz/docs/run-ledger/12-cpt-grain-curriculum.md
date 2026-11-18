# 12 — Grain-Native CPT Curriculum (olmo / FineWeb-Edu / Cosmopedia Anneal)

> **Branch:** `ezpz-cpt111k-eval-sft` — covers `torchtitan/experiments/ezpz/` only.
> **One-line status:** New `agpt_2b_grain_cpt_anneal` (Warmup 70/20/10 olmo/fwe/cosmo -> anneal to 20/50/30 over 100B / 190734 steps, GBS=64) replaces static 60/40 `BlendCorpusDataLoader` to lift MMLU from **0.302->0.317 flat** (+0.015/135k, chance band); textbook/science injection is the only MMLU lever that has moved this base; `11-eval-matrix.md` + `12-cot-136800-eval.md` are the baselines.
> Every claim cites its source; inferences labeled `[INFERENCE]`.
> **No pre-tokenization required** — Grain tokenizes dynamically on-the-fly via `EZPZTokenizer` (~4.5k tps/tile).

---

## 1. What exactly we run

### 1.1 New config — `agpt_2b_grain_cpt_anneal`

| Field | Value | Source |
|---|---|---|
| Config fn | `agpt_2b_grain_cpt_anneal()` at `config_registry.py:1260-1300` (+ helper `agpt_2b_grain_cpt_anneal_at(progress)` and `_grain_anneal_weights(p)` at `config_registry.py:1151-1315`) | `config_registry.py` |
| Pattern | `_agpt_2b_mds_mix_blend` (`config_registry.py:970-1021`) — `GrainDataLoader(DatasetMixConfig(WeightedDataset x N))` with `ConcatThenSplitPackingConfig` per source, weights count post-packing 8192-token rows (token-weighted). | `config_registry.py:970-1021` `dataset.py:214-268` `packing.py:22-69` |
| Base | MDS 2B `global_step138650` via `_MDS_ANNEAL_BASE` (`_agpt_2b_mds_mix_base` -> `_agpt_2b_mds_anneal_base` at `config_registry.py:919-604`) with gentle constant LR 2e-6 warmup 20 (`SYNTHESIS` anneal winner, avoids dolmino re-warm shock at 2.28e-5). | `config_registry.py:570-627,919-928` `cpt/README.md:156-157` |
| Budget | 100B tokens / 524288 tok/step (GBS 64 * 8192) = **190734 steps** so this curriculum is drop-in against the live chain's budget math (`cpt_fineweb_edu_4n.sh:72-78`). | `cpt_fineweb_edu_4n.sh:72-78` `01-cpt-frontier.md:22-36` |
| Budget constant | `_GRAIN_CPT_ANNEAL_STEPS = 190734` at `config_registry.py:1146` | `config_registry.py:1146` |
| Dataloader | `GrainDataLoader` over `DatasetMixConfig` of three `ConcatThenSplitPackingConfig(SingleDatasetConfig(source=..., processor=TextProcessor))`. Each source tokenized inline by `EZPZTokenizer` (backend `hf`, vocab 256000, `tests/assets/hf/gemma-7b` or `assets/hf/gemma-7b`). See `01-cpt-frontier.md:120-123` for same tokenizer path on live CPT. | `loader.py:69-195` `text_datasets.py:41-66` `blendcorpus/build_tokenizer.py:9-53` `agpt/config_registry.py:366-370` |
| Sampling | **Weighted Interleaved** via Grain `DatasetMixConfig` (`grain.IterDataset.mix(children, weights=weights)`). Strict weighted sampling with thread prefetch; post-#4088 replacement for `InterleavedHuggingFaceTextDataLoader`. Document-level mixing would weight attempt-index incl. filtered attempts; post-packing rows weight tokens. | `dataset.py:214-268` `config_registry.py:975-980,1134-1139,1238` |
| Pre-tokenize | **None.** No `pretokenize_to` / `save_to_disk` / `--pretokenized_dataset` step. Grain is streaming Grain-native (same trainer `FaultTolerantTrainer` + `GrainDataLoader` as CPT/pretrain in `torchtitan/experiments/ezpz/train.py`, verified 2N/8N) and tokenizes at ~4.5k tps/tile. Contract forbids adding one. | `sft_data.py` header comment + assignment contract |
| Checkpoint | `checkpoints/agpt-2b-grain-cpt-anneal` (`agpt_2b_grain_cpt_anneal_at(p)` uses `checkpoints/agpt-2b-grain-cpt-anneal-p{100*p:03d}` to avoid collision). DCP resume via `FaultTolerantTrainer` + `GrainStreamingCheckpointManager` same as live CPT. | `config_registry.py:1258-1315` |

#### Data sources in the mix

| Slot | Dataset | Corpus | Grain source | How tokenized | Where / verified |
|---|---|---|---|---|---|
| olmo | `allenai/olmo-mix-1124` | Production pretrain (94.8% DCLM, `thread-a` Sec 2.1) | `HuggingFaceStreamingSource(path="allenai/olmo-mix-1124", split="train")` streaming raw text | `TextProcessor` + `EZPZTokenizer` inline | Aurora `data-lists/aurora/olmo-mix-1124.txt` on `/flare`; precache via `scripts/precache_hf_dataset.py` at `_OLMO_MIX_CACHE`; do NOT use gemma-tokenized `.bin/.idx` on `/gila` (wrong vocab/host). See `agpt/config_registry.py:1645-1672` for precache pattern. |
| fwe | `fineweb-edu-100BT` | Filtered educational web (pedagogical textbooks, explanations, Q&A) | `HuggingFaceRandomAccessSource(path="parquet", load_dataset_kwargs={"data_files": _fineweb_edu_shards()})` — 16 local parquet shards | Same `TextProcessor` path | **`/lus/tegu/projects/datasets/datasets/fineweb-edu-100BT`** `sample/100BT/*.parquet` — 140 shards, ~280GB, 726k rows/~1.4B tok each; 16-shard subset `_FINEWEB_EDU_DIR` / `_fineweb_edu_shards()` = ~22B tokens sorted+sliced byte-identical. Source `agpt/config_registry.py:1489-1520`. |
| cosmo | `HuggingFaceTB/cosmopedia` | Synthetic textbooks (`auto_math_text+khanacademy+openstax+stanford+wikihow`) | `HuggingFaceStreamingSource(path="HuggingFaceTB/cosmopedia", split="train")` streaming raw text | Same | HF `HuggingFaceTB/cosmopedia`; precache via `precache_hf_dataset.py` if 429 at scale (same 429 caution as olmo at 384 ranks, `datasets.py:65`). Gated-science corpora (`nvidia/Nemotron-CC-Math-v1` `4plus`) remain gated; this run does not need them. |
| (legacy) dolmino | — | DCLM redistrib `+6.88pp peS2o +1.97pp FLAN +1.36pp math` (`mmlu-letter-prior-at-chance.md`) | Blendcorpus data-lists via `build_cpt_mixes.py` (`olmo_src` ~2.717 weight sum → renormalized `scaled_block` to target fraction; writes `dolmino-mix-1124.txt`, `olmo50-dolmino50.txt`, `olmo25-dolmino75.txt`) | Old blendcorpus tokenizer (gemma `.bin/.idx`) | `torchtitan/experiments/ezpz/utils/build_cpt_mixes.py`. **Not used** by this Grain curriculum; documented as available + why rejected (degree tweak, 89% DCLM, same kind). Pilot bled -7.4pp HS / -10.4pp ARC-E at step-5960. |

### 1.2 Two-phase schedule — fractions and steps

Linear interpolation `w(p) = w0 + (w1 - w0) * p` where `p = step / 190734` in `[0, 1]`, clamped. All fractions sum to 1.0 at every `p`.

| Phase | Progress `p` | olmo | fineweb-edu | cosmopedia | Rationale (Sec. 3) |
|---|---|---:|---:|---:|---|
| **1 Warmup** | **0.00** | **0.70** | **0.20** | **0.10** | Preserve web diversity (94.8% DCLM commonsense anchors HS/ARC-E/PIQA); seed textbook Q&A (FWE pedagogical) + STEM textbooks (cosmo) without distribution shock. |
| mid | 0.25 | 0.575 | 0.275 | 0.15 | Gradual shift past dolmino shock horizon (P0 1000) while still majority olmo. |
| mid | 0.50 | 0.45 | 0.35 | 0.20 | Cross the live chain's terminal mix (40 OWM / 60 FWE) on aggregate educational fraction (fwe+cosmo = 0.55 vs 0.60 before, but now split textbook vs generic edu). |
| mid | 0.75 | 0.325 | 0.425 | 0.25 | Majority educational (FWE+cosmo 0.675); olmo floor still protects HS better than dolmino-100's -7.4pp case. |
| **2 Anneal end** | **1.00** | **0.20** | **0.50** | **0.30** | **80% educational+textbook** (0.50+0.30) — stronger than SmolLM2 60/40 FWE and FineWeb-Edu 37.5 MMLU operating point; 20% olmo floor so commonsense never fully leaves (dolmino precedent: pure shift forgets ARC-E monotonically `0.619 -> 0.547`). |

Formulas (`_grain_anneal_weights` at `config_registry.py:1151-1170`):

```
w_olmo(p)  = 0.70 + (0.20 - 0.70)*p = 0.70 - 0.50*p
w_fwe(p)   = 0.20 + (0.50 - 0.20)*p = 0.20 + 0.30*p
w_cosmo(p) = 0.10 + (0.30 - 0.10)*p = 0.10 + 0.20*p
```

Token budget illustration (100B total, token-weighted rows):

| Phase | Tokens | olmo | FWE | cosmo |
|---|---:|---:|---:|---:|
| Warmup slice 0-25% | 25B | 14.4-17.5B varying | 5.0-6.9B | 2.5-3.8B |
| Early 25-50% | 25B | 11.3-14.4B | 6.9-8.1B | 3.8-5.0B |
| Late 50-75% | 25B | 8.1-11.3B | 8.1-10.6B | 5.0-6.3B |
| Anneal end 75-100% | 25B | 5.0-8.1B | 10.6-12.5B | 6.3-7.5B |

Exact per-step fractions via `_grain_anneal_weights(p)` — not quantized to phase buckets above.

**How progress advances:** `agpt_2b_grain_cpt_anneal()` is Phase 1 Warmup at `p=0.0`. Resumption at anneal slices uses `agpt_2b_grain_cpt_anneal_at(progress=p)` with `p` from resume step / 190734; folder encodes `p` (`checkpoints/agpt-2b-grain-cpt-anneal-p{100*p:03d}`) to avoid collision. The trainer itself is stateless re: data weights — only `cfg.dataloader` changes across phases (same mechanism as the 60/40 FWE fork from `checkpointer.initial_load_path`). See `config_registry.py:1140-1315` + `09-gate-rl-probe.md` for `initial_load_path` fork semantics.

### 1.3 Commands

```bash
# Warmup (Phase 1) — drop-in against live CPT budget; no pre-tokenize step
python -m torchtitan.experiments.ezpz.train --module ezpz.agpt \
  --config agpt_2b_grain_cpt_anneal

# Mid-anneal resume / parallel anneal slice (e.g. p=0.5 == 45/35/20)
python -m torchtitan.experiments.ezpz.train --module ezpz.agpt \
  --config agpt_2b_grain_cpt_anneal_at --progress 0.5

# Verify imports without launching training (required by acceptance)
python -c "from torchtitan.experiments.ezpz.agpt.config_registry import agpt_2b_grain_cpt_anneal; print(agpt_2b_grain_cpt_anneal)"
python -c "from torchtitan.experiments.ezpz.agpt.config_registry import _grain_anneal_weights; print(_grain_anneal_weights(0.0), _grain_anneal_weights(1.0))"
```

PBS note: production CPT on Sunspot is `select=4 walltime=12:00:00 filesystems=tegu:home workq datascience` (`cpt_fineweb_edu_4n.sh:2-7`); Aurora production is `flare:home` / `A=AuroraGPT`. This config inherits `GBS=64 LBS=2 PPN=8 4N=32r seq 8192` from the agpt 2B base — same walltime/mesh as the 60/40 FWE run it supersedes. Do not change GBS without recomputing `190734 = 100B / (GBS*8192)`.

---

## 2. Why we run — MMLU 0.302->0.317 flat, need textbook/science injection

### 2.1 What the static 60/40 FWE run actually did

| Signal | 60/40 FWE CPT (`agpt_2b_mds_mix_owm_edu_6040`, `BlendCorpusDataLoader`, GBS=64) | Verdict |
|---|---|---|
| Budget | 0 -> 136800 / 190734 = **71.7% of 100B** (assignment snapshot `12-cot-136800-eval.md:10` is at 71.7%; ledger table at 01 Sec 1.2 row 9 is at 75.2% from earlier snapshot — both consistent via `100B/524288=190734` divisor) | Chain ran, budget math sound (`cpt_fineweb_edu_4n.sh:72-78`). 190700 vs 190734 is rounding convention across ledgers. |
| Train loss | 2.068 -> 1.856 (new low @10108) -> 1.93 @34298, flat-to-slightly-up within noise | Healthy (`01-cpt-frontier.md:186-210`); same entropy effect as dolmino 2.49 vs 2.80 but still misleading alone. |
| Val loss | **12.53 flat 110700->111200** | **Validator bug** (missing `global_valid_tokens` divisor, `01-cpt-frontier.md:212-243`); ignore as training signal. |
| Commonsense | HellaSwag/ARC-E/PIQA/ARC-C from 79-step ladder `outputs/evals/cpt-fwe/step-*/results/results_0shot*.json` (HF pruned 587G->5.7M but results kept) | To be gated at 150k per `11-eval-matrix.md` G1 (-2pp HS, -3pp ARC-E). |
| MMLU | **0.302 @1k -> 0.317 @136800 = +0.015 over 135k steps** (`outputs/evals/cpt-fwe/step-*/results`, HF ladder; `11-eval-matrix.md:4` and `12-cot-136800-eval.md:90-100`) | **Flat at chance.** Band 0.2295 (always-A) - 0.2689 (always-D) per `03-eval-harness.md:72-77`; constant-D 0.2689 beats all 2B. Letter-prior `p(A,B,C,D)=(0.37,0.32,0.20,0.11)` explains dispersion. 0.015/135k is within noise; no MMLU lift. |
| CoT | **0.0 at 6/6 raw-CPT probes** (1k/10k/100k/111k/125k/**136800**, format 0.0, 200/200 unclosed, mean_gen_len 1527 hitting max_tokens) jobs `12484888/12484889` (`Tegu HF 3.7G 9/9`) | CPT alone cannot induce `<think>/<answer>\boxed{}` envelope — SFT required (`SYNTHESIS` §1-5, `12-cot-136800-eval.md:80-111`). |
| Gate status | G1-G4 at 150k (HS -2pp, ARC-E -3pp, PIQA -2pp, ARC-C -2pp, MMLU >=0.240 no-regression, SFT probe >=B2 0.205 per `SYNTHESIS:225-239`) — not yet scored on FWE HF | Ladder through 143400 scored; 150k gate pending per `09-gate-rl-probe.md`. |

**Diagnosis:** the base is DCLM-saturated (olmo-mix-1124 is 94.8% DCLM) and 60/40 FWE was a *kind* change in the right direction (FWE is pedagogical web vs DCLM's generic quality classifier; academic peS2o/arxiv in olmo-mix is 2.0% research papers not pedagogy per `01-cpt-frontier.md:112-117`). But 60% FWE alone is **generic educational web** — it trades a small HellaSwag deficit for MMLU at 350B/1.7B but at 2B/100B it produced only +0.015. Dolmino (89% DCLM, `+6.88pp peS2o +1.97pp FLAN +1.36pp math`) already proved a *degree* tweak fails: loss down, HS -7.4pp ARC-E -10.4pp at step-5960 (`cpt/README.md:61-147`). Extending narrow SFT (B2v2 ep3/ep6 0.09/0.105 vs B2 0.205) proved the ceiling is data-distribution not more tokens on the same 7.4k (`01-cpt-frontier.md:136`). Only a second *kind* shift — textbook/science — remains.

### 2.2 Why textbook/science — external MMLU evidence (kind, not degree)

| Evidence | MMLU signal | Implication for THIS curriculum | Source |
|---|---|---|---|
| **FineWeb-Edu paper** arXiv:2406.17557 @350B 1.7B | FWE **37.5 vs DCLM 35.5 on MMLU**; DCLM 62.3 vs 60.1 on HS — small HS deficit buys MMLU | FWE's MMLU edge is real but generic edu alone insufficient at 2B/100B (+0.015). Must add **textbook density** on top (cosmopedia STEM) to exceed generic edu ceiling. | `01-cpt-frontier.md:126-129` `thread-a:300-305` |
| Same paper @38B tokens | **33.6% MMLU at 38B** where next-best needs **300B** (~8x token efficiency) | Pedagogical distribution is token-efficient early — justifies 20->50 FWE ramp, not flat 50 from step 0 (shock) | `01-cpt-frontier.md:128` |
| **SmolLM2** arXiv:2502.02737 Sec 4.3 stage-1 **60% FWE / 40% DCLM -> 33.6% MMLU** | Validated FWE operating point at stage-1 | Our end-state **50% FWE + 30% cosmopedia = 80% educational+textbook** exceeds 60% FWE — specialization stronger for post-pretrain. Warmup 70% olmo aligns with SmolLM2's 40% DCLM floor. | `01-cpt-frontier.md:130` `production/sft/fineweb-edu-60-40-kind-shift.md` |
| **OLMo-2 1B on olmo-mix-1124 base**, Dolmino mid-training arXiv:2501.00656 Table 9: **MMLU 26.9 -> 44.3 (+17.4)** | Strongest in-family proof: distribution shift on THIS exact base unlocks MMLU at 1B | A kind-shift *can* move MMLU at small scale — but dolmino was wrong kind (89% DCLM, same kind). This curriculum is right kind (DCLM 94.8% -> textbook 30%). | `01-cpt-frontier.md:131` `cpt/README.md:104-106` |
| Dolmino pilots on this codebase: **HS -7.4pp ARC-E -10.4pp, ratio-indep HS drop at step-1000** | Loss improved while benchmarks bled; HS drop was LR re-warm shock (2.28e-5), not ratio | Keep LR gentle constant 2e-6 (anneal winner) and retain 20% olmo floor so commonsense never fully leaves. Gentle retry at 1200 never evaluated — hypothesis still open, respected here. | `01-cpt-frontier.md:264-299` `cpt/README.md:61-147` `thread-a:283-286,328-329` |
| Cosmopedia corpus (auto_math_text, khanacademy, openstax, stanford, wikihow) | Synthetic STEM textbooks directly match **MMLU STEM cloze** format | 10->30 ramp: small at warmup (synthetic noise if too much early), dominant by anneal end (STEM MMLU needs textbook density). Same 75/25 science-slot logic as `config_registry.py:1060-1102` wave-3 blends but here annealed, not fixed. | `config_registry.py:1060-1102` (wave 3 doc) |

**Why THESE ratios (70/20/10 -> 20/50/30):**

- **70% olmo at warmup** preserves 94.8% DCLM diversity (HS/ARC-E anchors at 0.560/0.651, PIQA 0.733) and avoids the dolmino P0 shock (HS -6-7pp at step-1000 regardless of ratio). Cutting olmo to 40% from step 0 would repeat the shock with no diagnostic value.
- **20->50% FWE** (not flat 60) respects FineWeb-Edu's 8x-efficiency-at-38B evidence: efficient early, so ramp gradually — flat 60% from step 0 is unnecessary shock for same eventual MMLU per 38B->350B trend.
- **10->30% cosmopedia** (not 0->25 or 25 flat): at 10% the synthetic textbook signal is present but not dominant (avoids early synthetic-mode overfit); at 30% it supplies the STEM textbook Q&A density that generic FWE lacks (openstax biology/chemistry, khanacademy math/science, auto_math_text) and that MMLU STEM directly rewards.
- **20% olmo floor at anneal end** (not 0) because dolmino's ratio-dependent ARC-E bleed (`0.619->0.547` pure dolmino vs `~0.61` held at 50/50) proved commonsense forgets without olmo in the mix, and G1-G4 continue-gate requires HS/ARC-E within -2/-3pp to justify training to 190k.

---

## 3. Where logs/outputs live

### 3.1 Config and code

| Artifact | Path |
|---|---|
| Config fns | `torchtitan/experiments/ezpz/agpt/config_registry.py` — constants `_GRAIN_CPT_ANNEAL_STEPS=190734`, `_GRAIN_CPT_WARMUP_FRACTIONS=(0.70,0.20,0.10)`, `_GRAIN_CPT_END_FRACTIONS=(0.20,0.50,0.30)`, helpers `_grain_anneal_weights(p)` + `_grain_cpt_packed_{olmo,fwe,cosmo}()` + `_agpt_2b_grain_cpt_anneal_base(progress)` at `config_registry.py:1060-1315` |
| Base it forks | `global_step138650` DCP at `_MDS_ANNEAL_BASE` (`config_registry.py:552-558`, env `MDS_ANNEAL_BASE` override); olmo-mix-1124 256N chain at `_OLMO_ANNEAL_BASE` (`config_registry.py:1113-1119`) for olmo anneal A/B comparison |
| Data patterns | `config_registry.py:970-1021` (_agpt_2b_mds_mix_blend), `config_registry.py:1489-1625` (_fineweb_edu_shards, _use_fineweb_edu, _use_hf_streaming) |
| Dataset mixes (legacy) | `torchtitan/experiments/ezpz/utils/build_cpt_mixes.py` — olmo x dolmino blendcorpus builder (renormalizes olmo ~2.717 vs dolmino 1.0 sums; writes `dolmino-mix-1124.txt`, `olmo50-dolmino50.txt`, `olmo25-dolmino75.txt`) |
| Grain stack | `torchtitan/components/data/dataset.py` (DatasetMixConfig/WeightedDataset/SingleDatasetConfig), `packing.py` (ConcatThenSplitPackingConfig), `loader.py` (GrainDataLoader), `sources.py` (HuggingFaceStreamingSource/HuggingFaceRandomAccessSource), `text_datasets.py` (TextProcessor + EZPZTokenizer at backend `hf`) |

### 3.2 Datasets

| Dataset | Canonical path | Note |
|---|---|---|
| fineweb-edu-100BT | **`/lus/tegu/projects/datasets/datasets/fineweb-edu-100BT`** `sample/100BT/*.parquet` (140 shards, ~280GB, `text` column) | Verified on Sunspot. `agpt/config_registry.py:1500-1520` `_FINEWEB_EDU_DIR` + `_fineweb_edu_shards()`. Used at 16-shard subset (~22B tok, byte-identical sorted+sliced). `HF_HUB_OFFLINE=1` guard not needed — loaded via `HuggingFaceRandomAccessSource(path="parquet", load_dataset_kwargs={"data_files": ...})` which reads local parquet via `datasets`, not hub. |
| olmo-mix-1124 | `data-lists/aurora/olmo-mix-1124.txt` blendcorpus list on Aurora `/flare/AuroraGPT/datasets/.../data_fused_gemma_eod/...` and `allenai/olmo-mix-1124` HF streaming | Aurora `/flare` prod + HF streaming fallback; precache via `scripts/precache_hf_dataset.py` to `_OLMO_MIX_CACHE` (`$HOME/.cache/huggingface/hub/datasets--allenai--olmo-mix-1124/snapshots/`) if 429 at 384r. Do not use gemma-tokenized `.bin/.idx` — wrong vocab for Grain. |
| cosmopedia | `HuggingFaceTB/cosmopedia` streaming | Textbook corpus; same streaming pattern; precache if 429. Gated `nvidia/Nemotron-CC-Math-v1` is alternative 25% science slot (wave-3) but not required here. |
| dolmino-mix-1124 | Generated by `build_cpt_mixes.py <olmo_src> <dolmino_src> <out_dir>` from `data-lists/aurora/olmo-mix-1124.txt` + `data-lists/aurora/dolmino-6.5T.txt` | Blendcorpus data-lists (gemma `.bin/.idx` on `/gila`/`/flare`); legacy; not used by Grain curriculum. |

### 3.3 Training checkpoints and eval outputs (once launched)

| Store | Path | When exists |
|---|---|---|
| DCP (live) | `outputs/checkpoints/agpt-2b-grain-cpt-anneal/` and `checkpoints/agpt-2b-grain-cpt-anneal-p{100*p:03d}/` under `CKPT_DIR` (Tegu-preferring `cpt_fineweb_edu_4n.sh:83-89` logic or `PBS_O_WORKDIR`) | After `qsub` training starts; every `checkpointer.interval=500` steps + walltime-aware flush |
| DCP resume | Same — DCP is native torchtitan `pytorch_model_fsdp_0/__*.distcp + .metadata`; resume requires `CKPT_DIR` symlink fix `cpt_fineweb_edu_4n.sh:41-44` (`outputs/outputs/checkpoints` double-nest) | Always if training ever ran |
| HF export | `outputs/evals/cpt-fwe/step-$s/hf` via `hf_batch_convert_cpt_fwe.sh` (`flavor=2b-mds bfloat16`, `9/9` files, ~3.7G) or `hf_ladder_p0p3_generator.sh`; canonical on Tegu `.../tegu-.../outputs/evals/cpt-fwe/step-$s/hf` | After DCP->HF convert (1N `02:00:00 tegu:home`, `convert_to_hf.py:staged_export`, `ezpz_export.json` sentinel) |
| Eval JSON | `outputs/evals/cpt-fwe/step-*/results/results_0shot*.json` per step (79-step ladder prior; `5.7M` retained after HF prune 587G) + `outputs/evals/cot/cpt*-gsm8k-cot-*.jsonl` | After `submit_cot_eval.sh` / `convert_and_eval.sh` (Aurora preferred for MMLU `compute_topk_scores 63G`, `03-eval-harness.md`) |
| Logs | `qsub` stdout/err `*.o<jobid>` / `*.e<jobid>` at `PBS_O_WORKDIR`; `journal.md` + run-ledger `torchtitan/experiments/ezpz/docs/run-ledger/*.md` | At submission / after exits |
| W&B | `EzpzWandBLogger` per `config_registry.py:603` | Training config; dashboard per run |
| Baselines | `11-eval-matrix.md` (79-step ladder inventory + G1-G4 gates), `12-cot-136800-eval.md` (CoT 0.0 at 6/6 probes, MMLU 0.302->0.317), `01-cpt-frontier.md` (chain 0->136800/190734), `09-gate-rl-probe.md` (150k gate), `thread-a-cpt-trajectory.md` + `cpt/README.md` (dolmino precedent) | Already on disk |

---

## 4. Expected results

### 4.1 What we expect to see

| Horizon | Metric | Expected vs 60/40 FWE baseline (MMLU 0.302->0.317 flat, CoT 0.0 raw) | Why | Falsifier |
|---|---|---|---|---|
| P0 Shock 0-10k (olmo ~70->55% still majority) | HS/ARC-E within -1pp of TT-256N 0.560/0.651 | No dolmino-like shock (-7.4pp HS at 1k) because LR stays 2e-6 constant (not 2.28e-5 re-warm) and olmo still dominant | HS drop >3pp at 1k => distribution shock despite gentle LR |
| P1 10-75k (anneal 55->32.5% olmo) | **MMLU lifts off chance floor** — first reading above 0.2689 band (D beats all 2B) at p~0.4-0.6; target 0.28-0.32 by 75k | FWE alone at 350B/1.7B gave +2.0pp; combined textbook 20->25% starts to inject STEM cloze | MMLU still 0.30-0.32 at 75k (no lift vs 60/40 path) => anneal rate too slow or cosmopedia share too small |
| P2 75-150k (50% FWE + 25-30% cosmo, olmo 32.5->24%) | **MMLU 0.32-0.36** (best case toward OLMo-2 +17.4 evidence but conservative: 0.24->0.36 over 100B is ~50% of OLMo-2's move, scaled to 2B) while HS/ARC-E within G1-G4 continue band (HS -2pp / ARC-E -3pp) | 80% educational+textbook is past SmolLM2 60% operating point; 20% olmo floor prevents dolmino ARC-E bleed (0.547 vs 0.610) | MMLU <0.30 at 150k with HS/ARC-E inside band => kind-shift insufficient at 2B/100B; pivot to SFT breadth (broad-distill on last-good ckpt) per `09-gate-rl-probe.md` |
| Post-SFT on anneal ckpt | **SFT ceiling > B2 0.205** (target 0.265+ with format >=0.90, gen_len <=550, n_unclosed <=10; G4 threshold `SYNTHESIS:236`) | B2 on MDS base is 0.205; every alternative (B2v2 0.09/0.105, B3 0.05, B4a 0.02, B4b 0.065, narrow-on-CPT-17k 0.015) is below. CPT+FWE+MMLU lift moves ceiling; SFT is not optional (CPT alone 0.0 CoT). | SFT on 150k anneal ckpt still 0.09-0.15 (no lift) => CPT knowledge does not transfer to reasoning; try broad-distill (`distill_cot_mix` 91k->36.9k @8192) instead of narrow `gsm8k-r1cot` |

External anchors for targets: FineWeb-Edu paper 37.5 MMLU at 1.7B/350B and 33.6% at 38B (8x eff.); SmolLM2 60/40 FWE -> 33.6%; OLMo-2 1B Dolmino +17.4 (26.9->44.3 on same olmo-mix base) — but dolmino was wrong kind for us (89% DCLM degree tweak bled -7.4pp HS / -10.4pp ARC-E; our kind is 80% textbook at end). Conservative expectation is directional lift (above 0.2689 band) by 75k and meaningful lift (0.32+) by 150k, not immediate 44.

### 4.2 Gates and how to score them

Score every HF checkpoint via the **same ladder as `11-eval-matrix.md`** (P0 shock 1k/5k/10k, P1 25k/50k/75k, P2 100k/111k/125k/136k/143k/150k, P3 if continuing). Do NOT score every `200`-step HF from `hf_batch_convert_cpt_fwe.sh:58` `seq 74400 200 136800` — select only ladder points. MMLU on **Aurora** (Sunspot OOM `level_zero:40 63G` on `compute_topk_scores`, `03-eval-harness.md`), commonsense on either; CoT via `eval_cot_gsm8k.py` (vLLM dtype float32, `gpu_memory_utilization 0.70`, `enforce_eager=True`, `max_model_len 2048`, temp 0.0, `max_tokens 768`, stop `[1,107]`, format regex `<think>.*?</think>\s*<answer>.*?</answer>`). HS 0.560 -> >=0.540 and ARC-E 0.651 -> >=0.621 continue threshold at 150k per `SYNTHESIS:231-232` + `09-gate-rl-probe.md` G1.

---

## 5. Next steps

### 5.1 No pre-tokenize — explicitly

**There is no `--pretokenize_to` / `save_to_disk` / `--pretokenized_dataset` step for this curriculum.** Grain tokenizes dynamically via `EZPZTokenizer` (`backend="hf"`, ~4.5k tps/tile) inside each `SingleDatasetConfig(processor=TextProcessor)`. That is the contract: the assignment forbids adding a pre-tokenize stage, and none is needed because (a) the 2N/8N pretrain/CPT path is already verified on `GrainDataLoader + HuggingFaceStreamingSource + EZPZTokenizer + ConcatThenSplitPackingConfig` and (b) the 16-shard FWE local parquet selection is materialized small enough per source (~22B tok) to stream without pre-sharding at 100B budget. The TRL `train_sft.py --pretokenize_to` workaround (8.5h re-tokenize at init for 93M-row mixes, per-rank arrow-cache race) is TRL-specific and does **not** apply to `FaultTolerantTrainer + GrainDataLoader` (`sft_data.py:_cot_ok`). Do not add one.

### 5.2 To launch

```bash
# 1. Verify import (required by acceptance — must succeed, no VENV or network needed)
python -c "from torchtitan.experiments.ezpz.agpt.config_registry import agpt_2b_grain_cpt_anneal; cfg = agpt_2b_grain_cpt_anneal(); print(cfg.checkpointer.folder, cfg.dataloader.dataset.datasets[0].weight, cfg.dataloader.dataset.datasets[1].weight, cfg.dataloader.dataset.datasets[2].weight)"

# 2. (Sunspot, PBS) Warmup
qsub -l select=4 -l walltime=12:00:00 -l filesystems=tegu:home -q workq -A datascience -v NHOSTS_TRAIN=4 \
  torchtitan/experiments/ezpz/scripts/cpt_fineweb_edu_4n.sh  # replace config with agpt_2b_grain_cpt_anneal
# Or directly:
python -m torchtitan.experiments.ezpz.train --module ezpz.agpt --config agpt_2b_grain_cpt_anneal

# 3. HF convert + eval ladder (same as 02+SYNTHESIS E1-E3, only steps change)
# for s in 1000 5000 10000 25000 50000 75000 100000 111200 125000 136800 143400 150000; do ...
# See 02-hf-conversion.md Next steps + thread-a:588-621 one-shot ladder template;
# replace step list with anneal ladder above; flavor=2b-mds bfloat16; PBS 1N 02:00 tegu:home
```

### 5.3 If this curriculum lifts MMLU

SFT broad-distill (`distill_cot_mix` OpenThoughts-114k-decontam + OpenR1-Math-220k-decontam via `distill_cot_mix_streaming_dataset`, 91k->36.9k @8192 3.4G `cba5c64`, consumer `sft_distill_cot_4n.sh` `rl-monarch-locality-20261001` LD-first) on the anneal checkpoint that first clears the MMLU band — the only open SFT path adding new reasoning data per `04-sft-broad-distill.md`. Narrow `gsm8k-r1cot` (B2v2 ep3/ep6 0.09/0.105) already showed narrow SFT cannot move ceiling without broader data.

### 5.4 If it does not

Per `09-gate-rl-probe.md` decision tree: if G1-G2 (HS/ARC-E within -2/-3pp) hold but MMLU still flat at 150k, continue to 190k and re-gate with SFT probe (`SYNTHESIS:242`); if G1-G2 bleed, stop and fork SFT from last-good ladder point (not necessarily 150k) per `SYNTHESIS:244`; if everything fails, revert to MDS base per `SYNTHESIS:246`. Rank-3 broad-distill on MDS base remains independent lever (needs rank-2 LR sweep, `SYNTHESIS:38`).

---

## Provenance

- Static CPT baseline: `agpt_2b_mds_mix_owm_edu_6040` BlendCorpus 60/40 at step 136800/190734=71.7% MMLU 0.302->0.317 flat, 6/6 CoT 0.0 — `01-cpt-frontier.md` + `11-eval-matrix.md` + `12-cot-136800-eval.md` (jobs 12484888/12484889).
- New curriculum: `torchtitan/experiments/ezpz/agpt/config_registry.py:1060-1315` (`agpt_2b_grain_cpt_anneal`, `agpt_2b_grain_cpt_anneal_at`, `_grain_anneal_weights`, `_grain_cpt_packed_*`, 100B/190734 at GBS 64, 70/20/10->20/50/30 linear).
- MMLU evidence (FineWeb-Edu 37.5 vs 35.5, 33.6% at 38B ~8x eff.; SmolLM2 60/40; OLMo-2 +17.4; dolmino -7.4/-10.4): `01-cpt-frontier.md:124-136` + `thread-a:300-329` + `production/sft/fineweb-edu-60-40-kind-shift.md` + `cpt/README.md:61-147`.
- Datasets: fineweb-edu-100BT verified `/lus/tegu/projects/datasets/datasets/fineweb-edu-100BT`; olmo-mix-1124 `data-lists/aurora/olmo-mix-1124.txt` + `allenai/olmo-mix-1124` streaming + `_OLMO_MIX_CACHE`; cosmopedia `HuggingFaceTB/cosmopedia`; dolmino via `utils/build_cpt_mixes.py` (olmo sum ~2.717).
- Grain contract: `torchtitan/components/data/{loader,dataset,packing,sources,text_datasets}` + `experiments/ezpz/blendcorpus/build_tokenizer.py:EZPZTokenizer` (~4.5k tps/tile, no pre-tokenize).
- Band + harness: `03-eval-harness.md:72-77` (MMLU 0.2295-0.2689, `p=(0.37,0.32,0.20,0.11)`), `eval_cot_gsm8k.py` + `submit_cot_eval.sh` (vLLM 0.26 float32), `09-gate-rl-probe.md` (G1-G4 gates).
