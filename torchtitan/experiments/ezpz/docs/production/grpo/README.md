# Production GRPO

> GRPO-tuned checkpoints derived from SFT'd or pre-trained AuroraGPT
> models. Each entry is a recipe + checkpoint pair: a specific task
> applied to a specific starting model.

## Index

| Base | Recipe | Status | Reward (last 10 mean) | Steps | Checkpoint | Trajectory |
|---|---|---|---:|---:|---|---|
| `aurora2b-sft-tulu-mix-step729` | [sum_digits arithmetic](aurora2b/sft_arithmetic/README.md) | **complete** | 1.26 (acc 0.76) | 1000 | `outputs/grpo/aurora2b-sft-arithmetic-8n/checkpoint-1000/` | 8N, sum_digits, lr=1e-6, 2026-06-11 |
| `aurora2b-sft-tulu-mix-step729` | [arithmetic (multi-node vLLM)](../../rl/grpo-on-xpu-status.md#update-2026-07-01-multi-node-cross-node-vllm-grpo-works) | **in progress** | -- | 1000 (target) | `outputs/grpo/aurora2b-sft-arithmetic-vllm-xnode/` | 10N, cross-node vLLM server-mode, job 12469978, 2026-07-01 |
