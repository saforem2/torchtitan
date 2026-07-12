# SFT on the completed v2 2B base (step-92,859): conversion + transfer + smoke

**Date:** 2026-07-06
**Machine:** Sunspot (SFT) + Aurora (checkpoint conversion)
**Purpose:** Run the proven `tulu_math_uc_mix` SFT recipe on the COMPLETED v2
2B production base (step-92,859 = 4.674T tokens) for the first time. All prior
SFT (and therefore all GRPO) used the older `AuroraGPT-2B-sophiag-gs138650`
base (an AuroraGPT-v1-lineage checkpoint), not the completed v2 256N base.

## Why

The 2B v2 256N production base pretraining completed 2026-06-29 at step-92,859
(4.674T tokens, 100%). It had never been SFT'd. This run produces the first
SFT on the actual completed production base, so downstream GRPO/alignment can
build on the final pretrained model rather than the older lineage.

## Blocker + approach

The v2 base lives only on Aurora `/flare` as a 24 GB DCP checkpoint; Sunspot
had no copy, no HF conversion existed, and the machines have no cross-mounts /
direct link. Prep chain: convert on Aurora (CPU, cheap) -> transfer the ~4 GB
HF result Aurora -> Mac -> Sunspot -> smoke on Sunspot -> (pending) full 32N.

## 1. Conversion (Aurora login node, 2026-07-06)

- DCP source:
  `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859`
  (24 GB, complete).
- Converter: `eval/convert_to_hf.py` (standard, not legacy -- v2 uses the
  `qkv_linear` FQN layout), CPU-only single process, `--model_name
  experiments.ezpz.agpt --model_flavor 2b --export_dtype bfloat16`. ~10 min.
- Assembled: `config.json` (from `eval/configs/agpt_2b_config.json`) + 4 gemma-7b
  tokenizer files. Output: `model-00001-of-00001.safetensors` (3.97 GB) + index
  + config + tokenizer = 7 files, 3.8 GB.
- Pre-flight (Aurora): `model_type=llama vocab=256128 layers=12 tok_ok=True`.

## 2. Transfer (Aurora -> Mac -> Sunspot)

- No cross-mounts / no direct Aurora<->Sunspot link; routed through the local
  Mac via the two ControlMaster sockets. `scp -r` each leg (~10 min each).
- Landed on Sunspot repo root as `AuroraGPT-2B-v2-256n-step92859-hf`.
- Integrity: safetensors byte-size `3973170627` matches Aurora exactly; all 7
  files present.
- Pre-flight (Sunspot, rl-vllm venv transformers):
  `model_type=llama vocab=256128 layers=12 heads=16 kv_heads=4`,
  `GemmaTokenizer` -> **PREFLIGHT_OK** (no silent Qwen-0.6B fallback).

## 3. Smoke (Sunspot 2N, job 12470086)

Script: `rl/scripts/sft/_smoke_agpt2b_v2_256n_tulu_mix_n2.sh` (copy of the
gs138650 smoke with BASE_MODEL -> absolute new-base path + fresh unique
CKPT_DIR; mix-spec dataset arg `tulu-3-sft-mixture:0.65,metamathqa:0.15,
ultrachat-200k:0.20` preserved). 10 steps, `--max_train_samples 50000`,
`--save_strategy no`.

**PASSED (2026-07-06).** All 10 steps + `Training complete.`, clean exit.
wandb: `https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/xudygzxr`
(*sunny-tree*).

| step | loss | grad_norm | mean_token_accuracy |
|---|---|---|---|
| 1 | 2.076 | 6.97 | 0.578 |
| 2 | 1.937 | 7.39 | 0.625 |
| 4 | 1.777 | 6.82 | 0.622 |
| 6 | 1.686 | 6.67 | 0.625 |
| 8 | 1.778 | 6.98 | 0.619 |
| 10 | 1.767 | 6.25 | 0.604 |

Loss descends 2.08 -> ~1.7; token accuracy ~0.60-0.63; grad_norm stable
~6-9. Verifications: **0 occurrences of `Qwen`** in the log (no silent
fallback -- the accuracy level alone rules out the 0.6B model), **0 barrier/
SIGABRT** (the metamathqa mix built without the OpenMathInstruct-2 oneCCL
barrier crash), model resolved to the new base path, `gemma` chat_template
fallback injected (same as the gs138650 run). The first-step loss (~2.08) is
higher than the gs138650 smoke's (~1.16) because this is a different
pretrained base seeing the chat format cold -- expected, and it descends
normally.

Note: transformers 5.x emits a benign `fix_mistral_regex` deprecation warning
for the gemma tokenizer on every rank. Identical to what the gs138650
production SFT saw (same gemma-7b tokenizer files); not a blocker.

## Next (pending go-ahead)

Full 32N SFT on Sunspot: copy `aurora2b_tulu_mix_32n_gbs6144.sh`, swap
BASE_MODEL + fresh CKPT_DIR, recipe otherwise identical (GBS=6144, 3 epochs,
lr 2e-5, `--auto-retry`, select=36). Produces the first production SFT
checkpoint on the completed v2 base, then a proper trajectory page under
`docs/production/sft/agpt/2b-v2-256n/`.
