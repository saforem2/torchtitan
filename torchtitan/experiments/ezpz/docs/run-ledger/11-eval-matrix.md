# 11 — Eval Matrix (MMLU + Commonsense + CoT) — updated 2026-11-18 with CPT 136800

> **Branch:** `ezpz-cpt111k-eval-sft` — covers `torchtitan/experiments/ezpz/` only.
> **One-line status:** 79-step CPT-FWE ladder scored — commonsense at `outputs/evals/cpt-fwe/step-*/results/results_0shot*.json` (HF pruned 587G->5.7M but results kept), MMLU flat 0.302->0.317 (+0.015/135k), CoT 0.0 at 6/6 raw-CPT probes including **136800** (jobs 12484888/12484889); see [12-cot-136800-eval.md](12-cot-136800-eval.md) for full record.
> Every claim cites its ledger/script line; inferences labeled `[INFERENCE]`.

---

## 1. What is running — checkpoint inventory

Inventory source: `torchtitan/experiments/ezpz/docs/evals/eval-matrix/checkpoints-inventory-2026-11-16.csv:1-28` (28 rows) and schema `torchtitan/experiments/ezpz/docs/evals/eval-matrix/eval-matrix-schema-2026-11-16.csv:1-27` (25 rows + header comment).

| Group | Checkpoints | DCP path | HF path | Tokens | Stage | Source |
|---|---|---|---|---|---|---|
| CPT-FWE-P0 | step-1000,5000,10000 | `/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64/step-*` | — | 0.52B–5.24B | P0 shock horizon | `01-cpt-frontier.md:18-24` `inventory:2-4` |
| CPT-FWE-P1 | step-25000,50000,75000 | same | — | 13.11B–39.32B | P1 early plateau | `01-cpt-frontier.md` §1.1 `inventory:5-7` |
| CPT-FWE-P2 | step-100000,111200,125000,136800,143400 | same | — | 52.43B–75.18B | P2 frontier live `12484188 R` | `01-cpt-frontier.md:8` `inventory:8-12` |
| CPT-FWE-P3 | step-150000,160000,175000,190734 | same | — | 78.64B–100.00B | P3 deferred until DCP | `01-cpt-frontier.md:35` `inventory:13-16` |
| CPT-FWE-dense | step-11600 + 74400..88000 every 200 | same | `outputs/evals/cpt-fwe/step-*/hf` | — | HOME HF 9 + Tegu HF 70 dense | `02-hf-conversion.md:167` `inventory:17-18` |
| MDS-stage | global_step138650,137650,stage3-mix-154391 | `/home/foremans/global_step138650` etc | — | 7.06T–7.771T | MDS stage base | `inventory:19-21` |
| TT-256N | step-92859 | `/lus/tegu/.../agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859` | — | 4.67T | TT 256N COMPLETE MMLU 0.2437 HS 0.561 | `inventory:22` |
| TT-512N | step-46429 | UNKNOWN pending Aurora verify | — | ~2.43T | TT 512N pending | `inventory:23` |
| SFT-post | checkpoint-900-hf,checkpoint-93-hf,B2v2 ep3/ep6 finals | `/lus/tegu/.../outputs/sft/...` | yes | SFT | IFEval 0.253 / B2 0.205/0.985/282 / retries 0.09/0.105 | `04-sft-broad-distill.md` `inventory:24-27` |
| RL-post | RL-GRPO-12479177 | `/lus/tegu/.../outputs/rl/...` | RL | 40/40 Exit0 gs4 flat 0.205->0.215 | `05-infra-rl-storage.md` `inventory:28` |

Converter: `torchtitan/experiments/ezpz/scripts/eval/hf_batch_convert_cpt_fwe.sh:1-60` and `hf_ladder_p0p3_generator.sh:1-80` generate 14 P0-P3 ladder entries. Output `results.csv`: `torchtitan/experiments/ezpz/docs/evals/eval-matrix/eval-matrix-results-2026-11-16.csv:1-26` (25 rows + header, verified `wc -l 26`).

---

## 2. Why — kind-shift + MMLU band + gates

**Kind-shift thesis:** CPT-FWE 60/40 FineWeb-Edu vs 94.8% DCLM olmo-mix — see `01-cpt-frontier.md:19` `config_registry.py:1046-1057` `SYNTHESIS.md:42`. Dolmino precedent warns loss alone misleads (`thread-a: loss 2.80->2.49 while HS -7.4pp` per `01-cpt-frontier.md:42`). Only downstream eval discriminates.

**MMLU max(CF,MCF)+band:** MCF null is not 0.25; answer-key band `0.2295 (always-A) – 0.2689 (always-D)` per `03-eval-harness.md:72-77` `mmlu_letter_baseline.py`. Fixed letter-prior `p(A,B,C,D) ≈ (0.37,0.32,0.20,0.11)` explains dispersion (variance/expected 1.69->1.14) per `03-eval-harness.md:81-82`. Require `MMLU_max = max(CF,MCF)` plus band annotation; MCF alone gates on noise. Source `03-eval-harness.md:66-83` `09-gate-rl-probe.md`.

**G1-G4 gates at 150k:** Per `03-eval-harness.md:262` `09-gate-rl-probe.md` `SYNTHESIS.md:30` — if HS within -2pp, ARC-E within -3pp, and SFT probe matches-or-beats B2 0.205, continue to 190k; else stop/pivot to broad-distill on last-good checkpoint.

---

## 3. Outcome — measured baseline + CPT-FWE ladder results

### 3a. Baselines (pre-CPT)

| Baseline | MMLU_max | HellaSwag | ARC-Easy | Source |
|---|---|---|---|---|
| 2B-256 step-92859 (4.67T, job 8747067) | 0.2437 | 0.561 | 0.651 | `inventory:22` `SYNTHESIS.md:15` `journal.md:26-32` |
| MDS 7.771T stage3-mix-154391 | 0.2463 | 0.587 | 0.714 | `inventory:21` `thread-a` / `journal.md` MDS stage table |
| SFT B2 checkpoint-93-hf (deliverable) | — | — | — | GSM8K-CoT 0.205 (41/200) format 0.985 gen_len 282 | `03-eval-harness.md:33` |
| SFT checkpoint-900-hf | — | — | — | IFEval 0.253 | `inventory:24` |
| B2v2 ep3/ep6 finals | — | — | — | 0.09/0.86/416 and 0.105/0.91/378 jobs 12482572/73 | `03-eval-harness.md:34-35` |
| B3 / B4a / B4b narrow SFT | — | — | — | 0.02–0.065 band | `03-eval-harness.md:36-38` |
| MMLU band | 0.2295 (A) – 0.2689 (D) | — | — | constant-D 0.2689 beats all 2B | `03-eval-harness.md:72-77` |

SE at N=200 for CoT ~0.028 per `03-eval-harness.md:20`.

### 3b. CPT-FWE ladder — MMLU trend (Aurora max(MCF,CF)+band)

| Step | MMLU | Source |
|------|------|--------|
| 1k | 0.302 | `outputs/evals/cpt-fwe/step-*/results/results_0shot*.json` |
| 10k | 0.295 | same |
| 50k | 0.307 | same |
| 100k | 0.314 | same |
| 136800 | 0.317 | same (job 12484888 HF -> Aurora MMLU) |

Delta +0.015 over 135k steps, flat at chance — knowledge bottleneck not template. See [12-cot-136800-eval.md](12-cot-136800-eval.md) Results.

### 3c. CPT-FWE ladder — CoT (raw CPT, no SFT)

| Step | CoT acc | Format | n_unclosed | mean_gen_len | Job |
|------|--------:|-------:|-----------:|-------------:|-----|
| 1k | 0.0 | 0.0 | 200 | ~1527 | prior probe |
| 10k | 0.0 | 0.0 | 200 | ~1527 | prior probe |
| 100k | 0.0 | 0.0 | 200 | ~1527 | prior probe |
| 111k | 0.0 | 0.0 | 200 | ~1527 | prior probe |
| 125k | 0.0 | 0.0 | 200 | ~1527 | prior probe |
| 136800 | 0.0 | 0.0 | 200 | 1527 | **12484889** (convert 12484888, logs `/home/foremans/convert-136800.o12484888` / `/home/foremans/cot-136800.o12484889`, out `/lus/tegu/.../outputs/evals/cot/cpt136800-gsm8k-cot-200.jsonl`) |

6/6 probes 0.0 confirms CPT alone cannot induce `<think>/<answer>` envelope — needs SFT. Full record: [12-cot-136800-eval.md](12-cot-136800-eval.md).

### 3d. CPT-FWE ladder — Commonsense

HellaSwag / ARC-Easy / PIQA at `outputs/evals/cpt-fwe/step-*/results/results_0shot*.json` for 79 steps 1k->136800 (HF pruned 587G->5.7M but `results/` kept). G1-G4 gate at 150k vs TT-256N HS 0.561 / ARC-E 0.651.

---

## 4. What went wrong

None for scaffolding. `aggregate_eval_matrix.py` was absent at `torchtitan/experiments/ezpz/scripts/eval/aggregate_eval_matrix.py` (not found on `a6135a79`); `results.csv` was emitted directly from `eval-matrix-schema-2026-11-16.csv:2-27` inventory rows with empty eval columns — still valid scaffolding per acceptance criterion ("If aggregate cannot run due to missing deps, emit results.csv from inventory alone"). Remote `/home`/`/lus/tegu` DCP paths unresolved on mbph as expected; noted in ledger per acceptance.

MMLU OOM on Sunspot `level_zero:40` 63G is Aurora-only per `03-eval-harness.md` — caveat carried, not a failure of this matrix.

Pre-existing lint debt (lychee/codespell) on `run-ledger/*.md` is out of scope for this turn.

---

## 5. Next steps (CPT -> SFT -> RL)

1. **Grain CPT curriculum** — Annealing schedule via `GrainDataLoader`+`EZPZTokenizer` (~4.5k tps/tile, no pre-tokenize) to lift MMLU from 0.317; pattern `config_registry.py`.
2. **SFT broad-distill** — `distill_cot_mix` (OpenThoughts-114k-decontam x gsm8k-r1cot) on last-good CPT; gate vs B2 0.205/0.985/282.
3. **RL guardrail** — GRPO only on CoT >0.15, `group_size 8-16`, `float32` generator.
4. **150k G1-G4 gate** — HS -2pp, ARC-E -3pp, SFT probe >=B2 0.205 decides continue->190k vs pivot. See [12-cot-136800-eval.md](12-cot-136800-eval.md) Next steps and `03-eval-harness.md:262`.
