# 12 — SFT MCQ Template Probe (Template Adherence Before Correctness)

> **Branch:** `ezpz-cpt111k-eval-sft` — covers `torchtitan/experiments/ezpz/` only.
> **One-line status:** CPT-FWE 136800 scores `format_hit_rate 0.0 / n_unclosed 200/200` on CoT and cannot emit any MCQ/MMLU response envelope — SFT is the only path to template adherence; this probe defines a Grain-streaming SFT (no pre-tokenize) that teaches the response template first, then correctness, before any RL.

---

## 1. Template failure — CPT cannot produce any envelope

### CoT envelope (GSM8K)

All six raw-CPT probes 1k->136800 score identically zero on `eval_cot_gsm8k.py` (`submit_cot_eval.sh`, vLLM 0.26 fp32, `frameworks/2026.1.0`, `temperature 0.0 max_tokens 768`):

| Step | CoT acc | `format_hit_rate` | `n_unclosed` | `mean_gen_len` | Source |
|------|--------:|------------------:|-------------:|---------------:|--------|
| 1k | 0.0 | **0.0** | **200** | ~1527 | `03-eval-harness.md:30` `12-cot-136800-eval.md:33` |
| 10k | 0.0 | **0.0** | **200** | ~1527 | same |
| 100k | 0.0 | **0.0** | **200** | ~1527 | same |
| 111k | 0.0 | **0.0** | **200** | ~1527 | same |
| 125k | 0.0 | **0.0** | **200** | ~1527 | same |
| **136800** | **0.0** | **0.0** | **200** | **1527** | **12484889** `12-cot-136800-eval.md:40` |

`format_hit_rate` = fraction matching `_FORMAT_RE = r"<think>.*?</think>\s*<answer>(.*?)</answer>"` (DOTALL+IGNORECASE) per `eval_cot_gsm8k.py:33-35` and `03-eval-harness.md:23`. `n_unclosed` = `finish_reason == "length"` (never emitted `</answer>`). CPT generations degenerate-loop to `max_tokens` — expected pre-SFT: CPT never saw `<think>/<answer>\boxed{}` (taught only in Stage-1 SFT `tulu_math_uc_mix -> checkpoint-900` and Stage-2 `gsm8k-r1cot -> checkpoint-93` per `03-eval-harness.md:195`).

This is the **correct null**, not a regression: `CPT 17k/73k/74k/111k all 0/200 degenerate loops n_unclosed=200 — expected, not regression` per `SYNTHESIS.md:219` and `thread-c-eval-harness.md:390-401 S 3.1`.

### MCQ / MMLU envelope

The same failure applies to MCQ. MMLU scoring uses two modes (`aggregate_eval_matrix.py:104-116`, `03-eval-harness.md:66-83`):

- **CF (cloze):** score `P(option | question)` ranked over 4 choices — no generation required, so CPT can emit a number.
- **MCF (multiple-choice format / response template):** the model must *generate* a letter response (`A`/`B`/`C`/`D`) in the expected template; accuracy requires template adherence.

Current ladder MMLU at chance confirms no MCF template benefit:

| Step | `MMLU_max = max(CF,MCF)` | Band `0.2295(A) - 0.2689(D)` | Source |
|------|--------------------------:|-----------------------------|--------|
| 1k | 0.302 | inside band | `12-cot-136800-eval.md:92` |
| 10k | 0.295 | inside band | same |
| 50k | 0.307 | inside band | same |
| 100k | 0.314 | inside band | same |
| **136800** | **0.317** | inside band | same |
| TT-256N SOTA 2B | 0.244 | — | `11-eval-matrix.md:46` |
| constant-D | **0.2689** | beats all 2B | `03-eval-harness.md:72-77` |

Delta **+0.015 / 135k steps** — flat at chance. Letter-prior `p(A,B,C,D) approx (0.37,0.32,0.20,0.11)` per `03-eval-harness.md:81-82` explains dispersion; `MCF null is not 0.25` (see `mmlu_letter_baseline.py`). With `format_hit_rate 0.0`, MCF cannot exceed CF by construction — the response template is absent. **Gating MMLU-MCF pre-SFT is a category error** (same as gating CPT on CoT, `SYNTHESIS.md:219` `06-hf-ladder-unpark.md:398-401`).

**Implication:** CPT lifts token-level knowledge (training loss `2.068 -> 1.856 -> 1.93` per `01-cpt-frontier.md:46-56`) but cannot induce any structured generation envelope. SFT is not polish — it is the *mechanism* that installs `<think>/<answer>\boxed{}` for CoT and `Answer: X` / boxed-letter for MMLU. Knowledge gains from CPT curriculum (Grain Annealing, `12-cpt-grain-curriculum.md`) only become *measurable* as correctness after SFT installs the template.

---

## 2. What we will run — SFT 1N template probe on CPT-136800 (or new Grain-CPT base)

### Base

| Choice | Path | When |
|--------|------|------|
| **Primary: CPT-136800 HF** | `/lus/tegu/projects/datasets/datasets/outputs/evals/cpt-fwe/step-136800/hf` (also `/lus/tegu/projects/datascience/foremans/outputs/evals/cpt-fwe/step-136800/hf` per Tegu routing; `12-cot-136800-eval.md:17` 3.7G 9/9 files, `flavor 2b-mds bfloat16`) | Now — DCP `step-136800` exists, HF `12484888` landed |
| **Upgrade: new Grain-CPT base** | Next Grain-curriculum checkpoint (Annealing schedule via `GrainDataLoader + HuggingFaceStreamingSource + ConcatThenSplitPackingConfig + EZPZTokenizer` per `12-cpt-grain-curriculum.md`; `config_registry.py:919-1057`) once it passes 150k G1-G4 gate | Only if new base beats 136800 on commonsense delta > 2pp on `outputs/evals/cpt-fwe/step-*/results/results_0shot*.json` |

Both use the same SFT recipe; only `initial_load_path` changes (model-only fork per `cpt/README.md` `FaultTolerantTrainer` pattern `config_registry.py:540-550`).

### SFT config — Grain streaming, no pre-tokenize

```
Trainer:  FaultTolerantTrainer + GrainDataLoader + EZPZTokenizer
          (same trainer as CPT, per sft_data.py:1-12; Grain tokenizes
           on-the-fly at ~4.5k tps/tile — no --pretokenize_to)
VENV:     rl-monarch-torch214-locality-20261001 + LD-first
          (.cache/xpu-ld-first/ per cpt_fineweb_edu_4n.sh:54-65)
PBS:      select=1, walltime=02:00:00, q=workq, A=datascience,
          filesystems=tegu:home  (1N Sunspot smoke)
LR:       2e-5 cosine, warmup 20, steps ~900 (1 epoch on probe mix)
Precision: bf16 fsdp full_shard, assistant_only_loss=True, packing=True
Seed:     42
```

Invocation (no new job created here — spec only):

```bash
# 1N smoke on CPT-136800 HF (model-only fork)
python -m torchtitan.experiments.ezpz.train \
  --module ezpz.agpt --config agpt_2b_mds_sft_mcq_template_probe_streaming

# Or explicit base override
MDS154391_SFT_BASE=/lus/tegu/.../step-136800/hf \
MDS154391_SFT_DCP=/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64/step-136800 \
  python -m torchtitan.experiments.ezpz.train --module ezpz.agpt --config agpt_2b_mds_sft_mcq_template_probe_streaming
```

Dataset builder: `torchtitan/experiments/ezpz/agpt/sft_data.py:sft_mcq_template_probe_streaming_dataset()` — see S 3 for composition. Config entry: `agpt_2b_mds_sft_mcq_template_probe_streaming()` in `torchtitan/experiments/ezpz/agpt/config_registry.py` (pattern per `tulu_math_uc_streaming_dataset()` / `distill_cot_mix_streaming_dataset()` via `GrainDataLoader.Config(dataset=ConcatThenSplitPackingConfig(dataset=DatasetMixConfig(...)))`).

### Why 1N first

Template adherence (format) is learnable in O(1k) steps — B2 two-stage `tulu->ckpt-900->gsm8k-r1cot` installed it at `format 0.985` in 900 steps per `04-sft-broad-distill.md`. A 1N probe (96 DP ranks at `LBS=2`, `num_tokens_per_train_step = 6144*1024` per `config_registry.py:739-742`) is sufficient to distinguish envelope learning from noise; correctness (accuracy > 0.15) follows only after format > 0.90.

---

## 3. Data composition — MCQ-aware mix fractions (Grain streaming, no pre-tokenize)

### Datasets available (TTE — verify on Sunspot login)

All paths under `/lus/tegu/projects/datasets/datasets/` — ssh-accessible; verify before qsub:

```bash
# Pre-tokenized TRL artifacts (legacy — NOT used here, Grain streams raw HF instead)
ls -lh /lus/tegu/projects/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json  # 91k->36.9k @8192 3.4G cba5c64
ls -lh /lus/tegu/projects/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192/dataset_info.json
ls -lh /lus/tegu/projects/datasets/datasets/agpt2b-metamath-gsm-distill-len2048-v1/dataset_info.json
du -sh /lus/tegu/projects/datasets/datasets/inkling-distill 2>&1 | head
# Raw HF streaming (used by Grain — no local TTE file needed; verify HF cache / outbound)
grep -rn "inkling-distill\|agpt2b-b3" torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | head
```

| TTE directory | Raw HF source(s) consumed by Grain | Role in this mix | Size hint |
|---------------|------------------------------------|------------------|-----------|
| `agpt2b-distill-cot-mix-len8192` | `open-thoughts/OpenThoughts-114k` + `open-r1/OpenR1-Math-220k` (50/50) via `sft_data.py:distill_cot_mix_streaming_dataset()` | **CoT reasoning template** — installs `<think>/<answer>\boxed{}` | 91k raw -> 36.9k packed @8192, 3.4G (pretok form; Grain streams raw HF, no 3.4G needed) |
| `agpt2b-b3-instruct-cot-mix-len8192` | `allenai/tulu-3-sft-mixture` + `open-r1/OpenR1-Math-220k` + `openai/gsm8k` + `HuggingFaceH4/ultrachat_200k` + `nvidia/OpenMathInstruct-2` (30/25/15/15/15) via `rl/datasets_sft.py:_build_b3_instruct_cot_mix()` | Prior balanced instruct+CoT recipe — reference for fraction calibration | 45% instruct / 55% math-CoT |
| `agpt2b-metamath-gsm-distill-len2048-v1` | `nvidia/OpenMathInstruct-2` + `openai/gsm8k` + distill traces | Math breadth | len2048 variant |
| `inkling-distill` | OpenThoughts/OpenR1 filtered traces (inkling harness) | Alternate CoT traces | — |

> **No pre-tokenize step:** Grain's `AgptChatProcessor` (`sft_data.py:181-280`) tokenizes with `EZPZTokenizer` (Gemma vocab 256000, `config_registry.py:499-501`) on-the-fly at ~4.5k tps/tile. `SingleDatasetConfig(pre_filters=(_has_usable_assistant_turn, _cot_ok), processor=AgptChatProcessor, post_filters=(not-None,))` plus `ConcatThenSplitPackingConfig` packing is the entire pipeline — identical to CPT Grain path (`torchtitan/components/data/{GrainDataLoader, SingleDatasetConfig, HuggingFaceStreamingSource, ConcatThenSplitPackingConfig}` per contract). Do not add `--pretokenize_to`.

### Proposed mix — 40 / 30 / 20 / 10

```
sft_mcq_template_probe_streaming_dataset() =
  DatasetMixConfig(datasets=(
    WeightedDataset(distill_cot_mix, weight=0.40),   # reasoning template
    WeightedDataset(tulu_math_uc_mix, weight=0.30),  # instruction-following
    WeightedDataset(mcq_mmlu_format,  weight=0.20),  # MCQ response template
    WeightedDataset(math_breadth,     weight=0.10),  # math (OpenMathInstruct-2)
  ))
  -> ConcatThenSplitPackingConfig(shuffle=True, repeat=True)
```

| Bucket | Weight | Grain source (HF id + `source` key) | What it teaches | Why this weight |
|--------|-------:|--------------------------------------|-----------------|----------------|
| **CoT-distill (reasoning template)** | **0.40** | `open-thoughts/OpenThoughts-114k` (`openthoughts`) + `open-r1/OpenR1-Math-220k` (`openr1`) via `_cot_source(..., _OPENR1_SUFFIX)` (`sft_data.py:37-53` envelope: `"\nReason step by step inside <think></think>, then give the final answer inside <answer>\\boxed{}</answer>."`) — same as `distill_cot_mix_streaming_dataset()` 50/50 split | `<think>/<answer>\boxed{}` envelope + long-form reasoning (median OpenR1 trace ~13k chars, `OPENR1_MAX_THINK_CHARS=16000` per `sft_data.py:51`) | **Largest bucket** because template adherence is the probe's primary gate (`format_hit_rate 0.0 -> 0.95`). CoT-distill is the only data with verified long traces that survive `_cot_ok` + decontam (`_gsm8k_decontam_ok` via `GSM8KDecontaminator` 13-gram per `sft_data.py:339-368`). B2's `0.985/282` was built on this signal; `B3_instruct_cot_mix` at 25% OpenR1 + 15% gsm8k-r1cot already collapsed when diluted. 0.40 restores headroom without starving other buckets. |
| **Instruction-following (tulu)** | **0.30** | `allenai/tulu-3-sft-mixture` (`tulu`) via `_source()` + `AgptChatProcessor` (`sft_data.py:283-303`); `tulu_math_uc_mix` pattern `tulu 0.65 / openmath 0.15 / ultrachat 0.20` per `sft_data.py:306` | General instruction following, short-answer discipline, multi-turn `user/model` boundaries (`<start_of_turn>user ... <end_of_turn>` per `sft_data.py:243-244`, assistant-only masking `IGNORE_INDEX` per `sft_data.py:274-275`) | Anchors MMLU's instruction parse: MMLU questions are tulu-like ("Answer with the letter..."). Tulu-3 already contains FLAN/OpenAssistant pretrain-flavored data (`_build_tulu_math_uc_mix:946-947`). 0.30 prevents instruction collapse while leaving 0.40 for CoT headroom. |
| **MCQ (mmlu-format)** | **0.20** | `cais/mmlu` (auxiliary_train split) or `allenai/tulu-3-sft-mixture` MMLU-derived subset formatted as MCQ with `A/B/C/D` letter answer — rendered via new `_mcq_messages()` -> `{"role":"user","content": question + choices + "\nAnswer with the letter."}` / `{"role":"assistant","content": "X"}` and `AgptChatProcessor` (same `<start_of_turn>model` envelope; supervision on single letter + `_END_OF_TURN`). `HuggingFaceStreamingSource(path="cais/mmlu", name="auxiliary_train", split="train")` | **MMLU response template** — the letter-level envelope that MCF scoring gates on. CF (ranking `P(option|question)`) needs no generation; MCF (generating `A/B/C/D`) cannot exceed chance without this template, which explains `12-cot-136800-eval.md:100` band. | **Smaller than CoT because MCQ template is short and low-entropy** (single token `A/B/C/D` vs 13k-char CoT trace) — it learns fast and over-weights cause letter-bias (`p(A)=0.37` prior per `03-eval-harness.md:81-82`). 0.20 is enough to lift MCF from `CF-only` to `template-aware` without memorizing MMLU distribution. Kept separate from tulu so ablation `w/o MCQ` isolates MCF lift. |
| **Math breadth** | **0.10** | `nvidia/OpenMathInstruct-2` (`openmath`) via `_source()` / `_openmath_messages()` (`sft_data.py:143-147` `problem -> generated_solution`) | Math correctness breadth beyond CoT (short-form numerical answers, mixed formatting) | Minimal weight because distill CoT already carries math reasoning; retained only to prevent math forgetting seen in `agpt_2b_mds_mix_edu` (FineMath `+0.308` when OWM dropped per `config_registry.py:960-962`). |

**Justification summary for MMLU:** MMLU correctness decomposes as `P(template) * P(knowledge | template)`. CPT improves the second factor (knowledge) but the first factor is zero pre-SFT (`format 0.0`, `n_unclosed 200`). This mix isolates that decomposition: MCQ bucket installs `P(template)` for `A/B/C/D`, CoT-distill installs it for `<think>/<answer>\boxed{}` (GSM8K), tulu installs general instruction parsing, math breadth prevents forgetting. Ablation ladder: (a) full 40/30/20/10, (b) w/o MCQ (40/50/0/10) to isolate MCF lift, (c) w/o CoT (0/50/20/30) to isolate CoT lift.

Grain wiring (no new infra — reuses existing abstractions):

```python
# sft_data.py — new builder (pattern per tulu_math_uc_streaming_dataset)
def sft_mcq_template_probe_streaming_dataset(
    cot_weight: float = 0.40,
    tulu_weight: float = 0.30,
    mcq_weight: float = 0.20,
    math_weight: float = 0.10,
) -> ConcatThenSplitPackingConfig: ...
```

Each bucket is a `WeightedDataset(dataset=SingleDatasetConfig(source=HuggingFaceStreamingSource.Config(...), pre_filters=..., processor=AgptChatProcessor.Config(source=...), post_filters=...), weight=...)` — identical to `tulu_math_uc_streaming_dataset()` and `distill_cot_mix_streaming_dataset()`.

---

## 4. Where logs / outputs land

### Output roots (on TTE — `/lus/tegu/...` = shared Tegu filesystem, verify on Sunspot)

| Artifact | Path | Current state |
|----------|------|---------------|
| **Existing SFT outputs (282G)** | `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/` | Present per assignment — prior `tulu_math_uc_mix` lineage: `agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900` + `checkpoint-900-hf` + `checkpoint-93-hf` (B2 `0.205/0.985/282`), plus B2v2 ep3/ep6 finals per `11-eval-matrix.md:23` |
| **Next SFT (this probe)** | `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt2b-sft-mcq-template-probe-1n/` (folder per `config_registry.py:_cot_sft_config(folder=...)` pattern, e.g. `checkpoints/agpt2b-mds-sft-mcq-template-probe` local + TTE mirror `outputs/sft/...`) | Not yet created — will be created by first qsub |
| **HF eval dirs (CPT ladder)** | `/lus/tegu/projects/datascience/foremans/outputs/evals/cpt-fwe/step-{1000..136800}/hf` and `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-*/hf` (Tegu routing per `hf_batch_convert_cpt_fwe.sh:35-36`) | 79-step ladder present; 136800 HF `12484888` 3.7G 9/9 files |
| **Commonsense results** | `outputs/evals/cpt-fwe/step-*/results/results_0shot*.json` per step 1k->136800 | Retained after HF prune 587G->5.7M per `12-cot-136800-eval.md:54` |
| **CoT eval JSONLs** | `/lus/tegu/projects/datascience/foremans/outputs/evals/cot/cpt*-gsm8k-cot-200.jsonl` and next `/lus/tegu/.../outputs/evals/cot/agpt2b-sft-mcq-probe-gsm8k-cot-200.jsonl` | Existing CPT probes + next SFT probe |

### Verify before qsub (Sunspot login)

```bash
ls -lh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/ | head
du -sh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft  # expect ~282G
ls -lh /lus/tegu/projects/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json
ls -lh /lus/tegu/projects/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192/dataset_info.json
ls -lh /lus/tegu/projects/datasets/datasets/agpt2b-metamath-gsm-distill-len2048-v1/dataset_info.json 2>&1 | head
ls -lh /lus/tegu/projects/datasets/datasets/inkling-distill 2>&1 | head
# TTE is shared; /home (~30T OST pool) and /lus/tegu are same Lustre (172.22.8.115:/tegu/su/) — both paths hit one pool per memory.
```

### Logs

- SFT `run.log` + W&B (`EzpzWandBLogger` per `config_registry.py:761`) + `trainer_state.json` under `outputs/sft/agpt2b-sft-mcq-template-probe-1n/`.
- Eval: stdout JSON `{"n":200, "format_hit_rate":..., "cot_accuracy":..., "mean_gen_len":..., "n_unclosed":...}` via `eval_cot_gsm8k.py:215-232` + per-example JSONL at `--out`; MMLU via `aggregate_eval_matrix.py` + `eval_cpt_fwe_commonsense_sweep.sh` per `11-eval-matrix.md:71-73`.

---

## 5. Why — template adherence before correctness

CPT alone cannot produce any structured generation — proven at six checkpoints 1k->136800, each `format 0.0 n_unclosed 200 mean_gen_len 1527` (`12-cot-136800-eval.md:33-40`). MMLU's +0.015/135k flat-at-chance (`12-cot-136800-eval.md:90-104`) is a knowledge-bottleneck symptom that only becomes actionable via SFT evaluation; loss and commonsense move orthogonally (dolmino precedent `loss 2.80->2.49 while HS -7.4pp` per `01-cpt-frontier.md:42`).

SFT is the **induction step**: it teaches the model to *emit* the response template that downstream metrics score. Without it, both CoT exact-match (`\boxed{}` extraction inside `<answer>`) and MMLU-MCF (letter generation) are zero by construction. Correctness (`cot_accuracy`, `MMLU acc`) is the product `P(template) * P(knowledge|template)` — the first factor must be established before the second is measurable.

Historical precedent (per `03-eval-harness.md` + `SYNTHESIS.md` + `thread-d-b2-autopsy.md`):

| SFT variant | CoT acc | Format | Outcome |
|-------------|--------:|-------:|---------|
| B2 two-stage `tulu_math_uc_mix (900) -> gsm8k-r1cot (93)` | **0.205** | **0.985** | Only deliverable |
| B2v2 `distill_cot` retry (ep3/ep6) | 0.09 / 0.105 | 0.86 / 0.91 | Format degraded with longer training |
| B3 cold-start `b3_instruct_cot_mix` single-stage | 0.05 | 0.86 | Dilution regression |
| B4a/B4b reweight | 0.02-0.065 | 0.26-0.86 | Envelope collapse |

Lesson: template adherence is fragile and requires dedicated mix design — the broad `b3_instruct_cot_mix` (30/25/15/15/15) diluted CoT below the threshold that B2 held. This probe restores CoT to **0.40** (above B3's 0.40 total CoT) and adds MCQ as a *separate* 0.20 bucket so MCF lift is isolable.

---

## 6. Expected results — gates and interpretation

All evals 200-problem GSM8K-CoT `temperature 0.0 max_tokens 768` fp32 vLLM (`frameworks/2026.1.0` + `pti-gpu`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`) per `03-eval-harness.md:23` `submit_cot_eval.sh`, plus MMLU `max(CF,MCF)+band` per `aggregate_eval_matrix.py` and commonsense E2 sweep.

### Primary gate — template adherence (must pass before correctness is read)

| Metric | Pre-SFT (136800 CPT) | Post-SFT target (this probe) | Source / threshold |
|--------|---------------------:|------------------------------:|--------------------|
| `format_hit_rate` | **0.0** (200/200 loops) | **>= 0.95** | B2 0.985, G4b threshold 0.90 per `SYNTHESIS.md:237` `03-eval-harness.md:310`; probe target 0.95 is mid-point B2->B2v2 |
| `n_unclosed` | **200** | **<= 10** | B2 ~0, G4c threshold 10 per `SYNTHESIS.md:238` `03-eval-harness.md:311`; `finish_reason == "length"` |
| `mean_gen_len` | **1527** (degenerate) | **<= 550** | B2 282, G4c 550 per `SYNTHESIS.md:238`; guards verbosity blow-up (B4a 1378) |

**Interpretation:**

- **Go (format >= 0.95, n_unclosed <= 10):** Template installed — proceed to correctness gate.
- **Pivot (0.70 <= format < 0.95):** Envelope present but leaky — increase CoT-distill weight (0.40->0.50) or extend 1N->2N at same LR before widening mix.
- **No-go (format < 0.70 or n_unclosed > 50):** Base too weak or mix too dilute — do not extend to correctness; try base swap (new Grain-CPT) or cold-start from `global_step138650` control per `SYNTHESIS.md:219` E5.

### Secondary gate — correctness (only meaningful after primary Go)

| Metric | Pre-SFT | SFT target | Threshold / SE |
|--------|--------:|-----------:|----------------|
| `cot_accuracy` (GSM8K-CoT) | 0.0 | **>= 0.15** | Must beat B2 0.205 by ~0.05 at 2 sigma (SE 0.028 @200 per `03-eval-harness.md:20` `11-eval-matrix.md:54`); 0.15 is floor for non-trivial reasoning |
| `MMLU_max = max(CF,MCF)` | 0.317 (flat) | **> 0.35** and **MCF > CF** | MCF must exceed CF to prove template lift is non-vacuous; band `0.2295-0.2689` per `03-eval-harness.md:72-77` must be annotated; letter-prior `p=(0.37,0.32,0.20,0.11)` explains prior dispersion |
| `MMLU MCF lift` (ablation) | MCF approx CF | **MCF - CF >= 0.03** | Isolates MCQ bucket effect; `w/o MCQ` control (40/50/0/10) should show CF-only level |

**Why correctness second:** `cot_accuracy` and `MMLU acc` are zero or at chance whenever `P(template)=0` regardless of knowledge — reading them pre-template conflates two bottlenecks. The doc's ordering (template -> correctness -> RL) mirrors `thread-c-eval-harness.md:384-401 S 3.1` "CPT 0/200 is category error."

### Guardrail — no verbosity blow-up

`mean_gen_len` and `n_unclosed` are health signals per `thread-b-broad-distill.md:552-560` `thread-c-eval-harness.md:589-591`: B2v2 `282->416` + `0->22` unclosed preceded format degradation; B4a `1378/133` flagged envelope loss. Cap at `550/10`.

---

## 7. Next steps — pipeline position (after CPT curriculum, before RL)

```
CPT (Grain Annealing curriculum)  --->  SFT (this probe)  --->  RL (GRPO)
   knowledge lift (loss/NLL)          template adherence       reward hacking
   MMLU +0.015/135k -> needs          format 0.0 -> 0.95      only after SFT
   dynamic curriculum                 then correctness          CoT > 0.15
```

This probe sits **after CPT curriculum, before RL** — see `12-cot-136800-eval.md:114-122` pipeline and `09-gate-rl-probe.md` `05-infra-rl-storage.md`.

1. **CPT curriculum (parallel):** Continue Grain-Annealing CPT per `12-cpt-grain-curriculum.md` (replaces static 60/40 `BlendCorpusDataLoader` `agpt_2b_mds_mix_owm_edu_6040` with Annealing schedule via `GrainDataLoader + EZPZTokenizer` at ~4.5k tps/tile). Gate at 150k on G1-G4 (HS -2pp, ARC-E -3pp, SFT probe >= B2 0.205) per `11-eval-matrix.md:36` `SYNTHESIS.md:30` before 190k continuation.

2. **SFT probe (this doc):** 1N Grain-streaming SFT on 136800 (or new Grain-CPT base if G1-G4 passes). Gate on `format >= 0.95` first, then `cot_accuracy >= 0.15` + `MMLU MCF>CF`. Ablations `w/o MCQ` and `w/o CoT` isolate MCF vs CoT lift. Artifacts `outputs/sft/agpt2b-sft-mcq-template-probe-1n/` on TTE.

3. **RL GRPO — only after SFT Go:** GRPO on strong cold-start only (`CoT > 0.15`, `group_size 8-16`, LR 5e-6-1e-5, `generator.model_dtype=float32` per `05-infra-rl-storage.md` `rl/multihost_grpo_e2e.pbs` `rl/docs/production/rl/plans/cot.md:317`). Prior GRPO on B2 held format 1.000 but acc `0.205->0.215` noise + reward-hacking risk (`05-infra-rl-storage.md`). Launching RL before template adherence amplifies degeneracy (format 0.0 -> no verifiable reward).

4. **No training job created in this doc** — this is a plan and data-availability record. qsub only after TTE paths verified on Sunspot login (see S 3-4 `ls` commands). Next SFT directory reuses the existing TTE `outputs/sft/` 282G prefix.

---

## 8. Provenance

- CPT failure: jobs `12484888`/`12484889`, `12-cot-136800-eval.md:1-42`, `03-eval-harness.md:30-44,58,195`, `SYNTHESIS.md:219`, `thread-c-eval-harness.md:248-249,384-401`.
- MMLU band + letter-prior: `03-eval-harness.md:72-77,81-82`, `mmlu_letter_baseline.py`, `aggregate_eval_matrix.py:10-23,90-116`.
- Datasets: `rl/datasets_sft.py:986-1031` (`b3_instruct_cot_mix`), `sft_data.py:306-336,422-492` (`tulu_math_uc`, `distill_cot_mix`, `gsm8k_plus_distill`), `config_registry.py:699-868` (Grain SFT configs), TTE paths `/lus/tegu/projects/datasets/datasets/{agpt2b-distill-cot-mix-len8192,agpt2b-b3-instruct-cot-mix-len8192,agpt2b-metamath-gsm-distill-len2048-v1,inkling-distill}` per assignment (verify via `ls` on Sunspot; TTE = `/lus/tegu` shared Lustre 172.22.8.115:/tegu/su/, 282G existing `outputs/sft/`).
- Grain contract: `torchtitan/components/data/{GrainDataLoader, SingleDatasetConfig, HuggingFaceStreamingSource, ConcatThenSplitPackingConfig}` + `EZPZTokenizer` (`torchtitan/experiments/ezpz/blendcorpus/build_tokenizer.py`), `sft_data.py:181-280` `AgptChatProcessor` (assistant-only `IGNORE_INDEX` masking, `<start_of_turn>` boundaries), `config_registry.py:725-763` (`GrainStreamingCheckpointManager` pattern).
- Pipeline: `12-cot-136800-eval.md:114-122` CPT->SFT->RL, `09-gate-rl-probe.md`, `05-infra-rl-storage.md` (VENV map `xpu-torch216` CPT / `rl-monarch-locality-20261001` SFT-RL / `frameworks/2026.1.0` eval), `12-cpt-grain-curriculum.md` (Grain CPT curriculum this probe follows).
