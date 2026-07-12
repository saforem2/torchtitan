# SFT recipe: agpt-2b-v2-256n-step92859 x tulu_math_uc_mix

> **Last updated: 2026-07-06.**
> **Status: in progress.** First production SFT on the COMPLETED v2 2B base
> (step-92,859 = 4.674T tokens, 256N chain). Full 32N run is job **12470088**
> (Sunspot, launched 2026-07-06). Prior production SFT used the older
> `AuroraGPT-2B-sophiag-gs138650` base; this one uses the actual completed v2
> production base.

## Quick reference

| Field | Value |
|-------|-------|
| Base model | `agpt-2b-v2-256n-step92859` (completed v2 256N base, 4.674T tokens, 1.99B params, llama arch) |
| Base source | Aurora DCP `agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859` -> HF-converted -> staged to Sunspot as `AuroraGPT-2B-v2-256n-step92859-hf` |
| Recipe | mix-spec `tulu-3-sft-mixture:0.65 + metamathqa:0.15 + ultrachat-200k:0.20` (metamathqa swapped for OpenMathInstruct-2 to dodge the XPU oneCCL barrier) |
| Trainer | TRL `SFTTrainer` (HF Trainer base, FSDP full_shard) |
| Sequence length | 1024, `packing=True`, `assistant_only_loss=True` |
| Hyperparameters | LR 2e-5 (cosine to 0), bf16, AdamW (TRL default), 3 epochs |
| Scale | 32 nodes x 12 ranks = 384 ranks, GBS = 6144 (bsz 2 x gas 8) |
| Submit script | [`rl/scripts/sft/agpt2b_v2_256n_tulu_mix_32n_gbs6144.sh`](../../../../../../rl/scripts/sft/agpt2b_v2_256n_tulu_mix_32n_gbs6144.sh) |
| Output dir | `outputs/sft/agpt-2b-v2-256n-tulu-mix-32n-gbs6144/` |
| Expected steps | ~729 (3 epochs at GBS=6144 over the metamathqa-swap mix) |

## Prep + validation

- Conversion, transfer, and 2N smoke are documented in the
  [prep report](../../../../../experiments/agpt/sunspot/2026-07-06-sft-2b-v2-256n-base-prep.md).
- 2N smoke (job 12470086): 10 steps, loss 2.08 -> 1.7, mean_token_accuracy
  ~0.60, 0 Qwen fallback, 0 barrier crash, clean exit. wandb:
  `https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/xudygzxr`.

## Run status

| Job | Date | Steps | Loss | Notes |
|-----|------|------:|-----:|-------|
| 12470088 | 2026-07-06 -> | ~100+/729 (in progress) | 1.35 -> 0.95 | Full 32N, select=36 (32+4 spare), --auto-retry. Fresh CKPT_DIR (resume coerced to None -> clean start on v2 base, NOT resuming old 729). **checkpoint-100 saved** at 16:20; trajectory persists. mean_token_accuracy 0.68 -> 0.757. |

<!-- RUN-PROGRESS -->

### Early trajectory (job 12470088)

First ~100 steps on the v2 base (2N-smoke-consistent descent, at production
GBS=6144): loss 1.35 -> 0.95, mean_token_accuracy 0.68 -> 0.757. First
checkpoint (step-100) saved cleanly via `save_fsdp_model`. Runs to ~729 steps
(3 epochs), saving every 100 with `--save-total-limit 8` and auto-retry
bad-node protection. Compare: the gs138650 SFT started ~1.16 (this v2 base
starts higher because it is a different pretrained model seeing the chat
format cold) and finished 0.77 over 729 steps.

## Comparison to the gs138650 SFT

The prior production SFT
([tulu_math_uc_mix on gs138650](../../2b-mds/tulu_math_uc_mix/README.md))
reached final loss 0.77 over 729 steps. This run applies the identical recipe
to the completed v2 base; the eval comparison (base-LM benchmarks + downstream
GRPO signal) will show whether SFT on the completed 4.674T base beats SFT on
the older lineage.
