# Production GRPO

> GRPO-tuned checkpoints derived from SFT'd or pre-trained AuroraGPT
> models. Each entry is a recipe + checkpoint pair: a specific task
> applied to a specific starting model.

## Index

| Base | Recipe | Status | Reward (last 10 mean) | Steps | Checkpoint | Trajectory |
|---|---|---|---:|---:|---|---|
| `aurora2b-sft-tulu-mix-step729` | [sum_digits arithmetic](aurora2b/sft_arithmetic/README.md) | **complete** | 1.26 (acc 0.76) | 1000 | `outputs/grpo/aurora2b-sft-arithmetic-8n/checkpoint-1000/` | 8N, sum_digits, lr=1e-6, 2026-06-11 |
| `aurora2b-sft-tulu-mix-step729` | [arithmetic (cross-node vLLM)](../trl.md#current-status-2026-07-06) | **works** (path validated) | -- | -- | `outputs/grpo/aurora2b-sft-arithmetic-vllm-xnode/` | cross-node vLLM server-mode validated 2026-07-06 (jobs 12469976 / 12470083); see grpo-on-xpu-status |
| `agpt-2b-gs138650-sft-fullmix-step900` | sum_digits arithmetic (vLLM server-mode) | **works** (early) | acc 0.31->0.74 (~20 steps) | ~20 (walltime) | `outputs/grpo/fullmix-900-arithmetic-vllm-xnode/` | 10N cross-node vLLM, job 12470959, 2026-07-18; validates the full-mix SFT **ckpt-900** deliverable as a GRPO start. Hit 6h walltime (not converged). See [SFT evals](../../sft/agpt/2b-mds/tulu_math_uc_mix_full/evals/README.md) |
| `agpt-2b-gs138650-sft-fullmix-step900` | [alphabet_sort tuning sweep](beat-v5-sweep.md) (Monarch+TorchStore+vLLM, LoRA) | **study** | ~0.25 (char-ratio p1) | 100 | (LoRA adapters, not promoted) | v5/w1/w2/w3: no config lever beats the ~0.25 reward-shape ceiling; 2026-07-20 |
| `agpt-2b-gs138650-sft-fullmix-step900` | [ceiling-attack: shaped reward](ceiling-attack.md) (Monarch+TorchStore+vLLM, LoRA) | **breaks ceiling** | **0.667** (char-ratio p1, +168%) | 100 | (LoRA adapters, not promoted) | componentized reward (format+completeness+order); job 12471056, 2026-07-20 |
