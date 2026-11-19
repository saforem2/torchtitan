# agpt-2B post-training plan: best general instruct model

> **Goal:** produce the best post-trained agpt-2B *instruct* model — not just
> the best GSM8K-CoT score. COT is one capability, not the objective.
> Ground truth: [POST-TRAINING-2B.md](POST-TRAINING-2B.md) (B2 = 0.205 CoT,
> checkpoint-900 = IFEval 0.253). All numbers quoted from run docs.

## Target definition

**Primary: one unified instruct model** (IFEval + MMLU + GSM8K balanced).
Fallback: two specialized exports (checkpoint-900 general + B2 CoT)
if unified underperforms on either suite. Decision: after phase 2 eval.

Deliverables:
- `checkpoint-900-hf` — today's general baseline (IFEval 0.253, base-LM intact).
- `checkpoint-93` (B2) — CoT specialist (0.205 CoT, format 0.985, two-stage only).
- Unified candidate — tbd from phases below.

## Eval suite (promotion rule = weighted, not single-metric)

| Suite | Harness | Shots | Gate |
|---|---|---|---|
| IFEval | `ifeval` | 0 | strict prompt acc — primary instruct signal (SFT's +8pp win lives here) |
| MMLU | `mmlu` | 5 | knowledge — do NOT gate on MCF variant (constant-D 0.2689 > all 2B is harness prior) |
| GSM8K | `gsm8k` (5-shot) + CoT variant | 5 + CoT | math — report both, promote on aggregate |
| ARC-C | `arc_challenge` | 25 | reasoning |
| HellaSwag | `hellaswag` | 0 | thermometer only — v1 retired, not competitive |

Infra: vLLM **fp32** (bf16 gibberish on agpt-2b), current harness for all
re-scores (B2 0.165 on preserved `ep3-ckpt93-b2` vs old 0.205 harness variance).

## 3-phase experiments

### Phase 1 — Audit + baseline lock (now, parallel)
- Re-score `ckpt-900` and `ckpt-93` on current harness (same 200-prob GSM8K split).
- Verify pretok gate: `/tegu/datasets/agpt2b-distill-cot-mix-len8192` needs
  `dataset_info.json` + `[decontam]` counts. Prior NPROC 4/2 segfault; NPROC=1 queued.
- Root-cause SFT-on-CPT: same agpt2b_b2v2_distill_cot_2n.sh (LR 2e-5, 3ep, GBS 384)
  logged only 1 loss entry at ckpt-6 — check `logging_steps` vs `save_strategy=epoch`.

### Phase 2 — Two independent levers (no narrow SFT, run in parallel)
> **Update 2026-11-18:** CPT step 136800 (71.7% of 100B, jobs 12484888/12484889) confirms CPT alone cannot induce CoT — **0.0** CoT (format 0.0, 200/200 unclosed, mean_gen_len 1527) at 6/6 probes 1k->136800, with MMLU flat 0.302->0.317 (+0.015/135k). Knowledge bottleneck, not template. Full record: `torchtitan/experiments/ezpz/docs/run-ledger/12-cot-136800-eval.md`. Implication: phase 2 CPT lever must use Grain Annealing curriculum to lift MMLU; CoT requires SFT (CPT->SFT->RL pipeline).
- **Broad CoT data:** `distill_cot_mix` (OpenThoughts-114k-decontam × gsm8k-r1cot,
  same envelope) via pretokenized path — only open path under proven two-stage
  structure. Train `sft_distill_cot_4n.sh` on gated dataset → eval vs B2.

### Phase 3 — RL as guardrail (only on strong cold-start)
GRPO on B2 (format held 1.000 for 100 steps, 0.205→0.215 noise). Raise
`group_size 8–16`, LR 5e-6–1e-5, `generator.model_dtype=float32`.
Do not start RL on CPT-derived SFT until CoT > 0.15 — avoids CoT-stage-2
reward-hacking (reward 0.118→0.217 while acc 0.16→0.105).

## Guardrails (already proven)

- Cap full-mix SFT at O(1000) steps or drop LR — ckpt-8672 forgot hellaswag
  0.593→0.273 / arc_easy 0.694→0.298 when cosine kept LR >1e-5 for ~4000 steps.
- **No single-stage rebuilds.** B3 0.05 / B4a 0.02 / B4b 0.065 / B2v2 0.09–0.105
  all failed vs B2 0.205. Structure beats mix.
- CKPT_DIR must be `${CKPT_DIR:-default}` (fixed `ecb15198b`) — CPT lane was
  clobbering B2's `ep3` output.
