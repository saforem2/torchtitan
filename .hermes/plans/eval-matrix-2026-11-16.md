# Eval Matrix Plan — AGPT-2B across MDS / TT-256N / TT-512N / CPT-FWE + post-training

**Goal (user 2026-11-16):** One joint table `{MDS 1,2,3 | TT-256N stage-wise | TT-512N stage-wise | CPT-FWE ladder} x {MMLU MCF+CF+max | HS/ARC-E/PIQA/ARC-C | GSM8K-CoT pre/post-SFT} + delta per CPT/SFT/RL experiment`

> Sources: [Inventory CSV](./checkpoints-inventory-2026-11-16.csv) (717 DCP dirs; tokens = step*524288 verified) · [Results schema CSV](./eval-matrix-schema-2026-11-16.csv) (canonical `results.csv` header)

## 1. Measured baseline (harness verified: Llama-3.2-1B 0.3121 / 8B 0.6530 same path)

| model | tokens | MMLU MCF 5-shot | HS | ARC-E | gsm8k |
|---|---|---:|---|---|---|
| 2B-256 COMPLETE TT step-46429 | 4.67T | 0.2437 | 0.561 | 0.651 | 0.016 |
| MDS dolmino 7.06T | 7.06T | 0.2413 | 0.49 | 0.547 | — |
| MDS stage3-mix 154391 | 7.771T | 0.2463 | 0.587 | 0.714 | 0.0167 |
| MDS stage3 math/code 154391 | 7.771T | 0.2591 | 0.42 | — | 0.0303 |
| B2 ckpt-93 (only SFT that worked) | — | — | — | — | 0.205 fmt 0.985 gen 282 |
| B2v2/B3/B4/B4a re-tries | — | — | — | — | 0.02-0.105 |

Every MMLU 0.24-0.26 inside constant-letter band 0.2295-0.2689 (always-D 0.2689 beats all). Letter prior p~(0.37,0.32,0.20,0.11) R2 0.64-0.76 — no lift at any scale/mix.

## 2. Unmeasured gaps

1. MDS 1/2/3 stage-wise + post-trained: checkpoints exist (~/global_step138650, optimizer-experiments/stage3-mix/step-154391, stage3/step-154391) but never scored stage-wise. Only B2 lineage scored.
2. TT-256N/512N stage-wise + post-trained: 512N 46429 and 256N 92859 scored once (job 8747067), no ladder, SFT dirs on Tegu unevaluated. CPT-FWE has 717 DCP dirs 1000..143400 on /home + 9 HOME HF + 70 Tegu HF 74400..88000 but only 11600/17000 have CoT jsonl.
3. CPT/SFT/RL delta: dolmino impact quantified (HS -7.4pp 0.560->0.486, ARC-E -10.4pp at 8647850); FWE impact unscored; SFT only via B2 vs failures (healthy loss 0.8069 hid halved eval); RL 40/40 Exit 0 at 12479177 but gs=4 =41% dead groups flat 0.205->0.215.

## 3. Constraints

- MMLU compute_topk_scores 63G OOM on Sunspot -> Aurora-only workq 48h
- Commonsense HS/ARC-E/PIQA/ARC-C + GSM8K-CoT vLLM fp32 LIMIT 200 temp0.0 max_tokens768 -> Sunspot workq 1N 00:40-02:00 frameworks/2026.1.0
- DCP->HF: convert_to_hf --model_flavor 2b-mds --export_dtype bfloat16 + staged_export .lock via hf_batch_convert_cpt_fwe.sh select=1 02:00 tegu:home sentinel ezpz_export.json idempotent
- CoT pre-SFT 0/200 fmt 0 expected (no <think>/<answer>\boxed{}), gate only post-SFT

## 4. Plan — parallel with live CPT 12483964 R 143400/190734=75.2%

1. Inventory: enumerate DCP roots for asks 1-2 -> checkpoints.csv
2. HF fills: P0 1000,5000,10000 / P1 25000,50000,75000 / P2 100000,111200,125000,136800,143400 / P3 160000,175000,190734 -> 14 HF 1N 2h each to Tegu outputs/evals/cpt-fwe/step-$s/hf (re-run qsub hf_batch_convert_cpt_fwe.sh skips 79 done)
3. E2 sweep: per HF lm-eval 0.4.10 HF backend hellaswag acc_norm 0-shot base0.560 gate0.540, arc_easy 0.651->0.621, piqa 0.733->0.713, arc_challenge 25-shot ~0.31->0.290, mmlu+mmlu_continuation 5-shot -> max(CF,MCF) band 0.2295-0.2689 + letter-prior fit. Run for every row in (1).
4. GSM8K-CoT SFT probe: B2 recipe ckpt-900 -> gsm8k-r1cot 93 steps 3ep LR2e-5 cosine GBS384 seq2048 + submit_cot_eval.sh LIMIT200 vs B2 0.205/0.985/282 (N=200 SE0.028, +0.06=2sigma). Same probe on MDS/TT stage bases if G1-G3 held.

Deliverable: [results.csv ← schema](./eval-matrix-schema-2026-11-16.csv) ([Inventory](./checkpoints-inventory-2026-11-16.csv)) rows=checkpoints x cols={MMLU_MCF, MMLU_CF, max, HS, ARC-E, PIQA, ARC-C, GSM8K_pre, GSM8K_post, format, gen_len, unclosed} + delta columns per experiment = answers asks 1/2/3.

Cost: ~14 HF*1N2h + ~20 evals*1N1h + Aurora MMLU 48h per batch, all 1N workq parallel with CPT, no extra CPT beyond budgeted ~2.5 hops 4N/12h.

Next: stage step 1 inventory + P0 converts now.
