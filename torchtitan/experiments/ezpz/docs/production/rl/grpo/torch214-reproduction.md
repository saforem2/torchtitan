# Torch 2.14 Monarch GRPO reproduction

> Status: **running** (2026-09-22). This page records a clean-room rerun of the
> `monarch.md` easy-task result and the shaped-reward ceiling attack on the
> current TorchTitan stack. Interim values are not final reproduction claims.

## Scope

Two 100-step AuroraGPT-2B GRPO+LoRA recipes are being rerun on Sunspot:

1. `rl_grpo_lora_agpt_2b_easy`: easy alphabet-sort task, LoRA rank 8,
   learning rate `2e-5`, linear character-ratio reward.
2. `rl_grpo_lora_agpt_2b_shaped`: the same easy task, LoRA rank 32,
   learning rate `5e-5`, componentized format/completeness/order reward.

The goals are to test the historical easy-task reward rise (`0.167 -> ~0.26`)
and the shaped run's final `0.667` result after rescoring completions with the
same character-ratio(power=1) metric used by the baseline.

## Provenance

| Item | Value |
|---|---|
| TorchTitan commit | `c5f8deac6d5ffd356eca9d12364ac5f7a607f2d9` |
| SFT source checkpoint | `outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900/` |
| DCP source | 96 intact `.distcp` model shards plus `.metadata` |
| Regenerated HF export | `/lus/tegu/projects/datascience/foremans/artifacts/agpt-2b-gs138650-sft-fullmix-step900-hf` |
| `model.safetensors` SHA-256 | `001cccc6331bca9c6695cb559e923d54640b845a755fe5191667ea2a6925b58e` |
| Metadata source | Original `/home/foremans/global_step138650` config and tokenizer |
| Template change | Documented Gemma `<start_of_turn>/<end_of_turn>` chat template only |
| Python | 3.12.12 |
| PyTorch | `2.14.0+xpu` |
| torchvision | `0.30.0.dev20260921+xpu` |
| Triton XPU | 3.8.0 |
| vLLM | `0.29.1rc1.dev481+gdc479a2d8.xpu` |
| vLLM XPU kernels | `0.1.15.dev30+g0bfb372.d20260922` |
| torchmonarch | 0.6.0 |
| ezpz | 0.27.3, installed from `git+https://github.com/saforem2/ezpz` |

The deleted historical `checkpoint-900-hf` symlinks were not used. The HF
weights were regenerated from the surviving sharded SFT checkpoint with
`accelerate merge-weights`; safetensors loading, key enumeration, tokenizer
loading, and chat rendering were validated before submission.

## Runs

| Recipe | PBS job | Node | Output directory | Current state |
|---|---:|---|---|---|
| easy baseline | `12478404` | `x1922c1s3b0n0` | `/lus/tegu/projects/datascience/foremans/reproductions/grpo-easy-c5f8deac6-12478404` | running |
| shaped ceiling attack | `12478405` | `x1922c1s5b0n0` | `/lus/tegu/projects/datascience/foremans/reproductions/grpo-shaped-c5f8deac6-12478405` | running |

Both use one Sunspot node, two XPU tiles (one trainer and one vLLM generator),
`ezpz launch --auto-retry`, FP32 generation, offline cached data, and strict
inner return-code propagation.

## Validated precursor smoke

Job `12478403` completed one optimizer step with PBS `Exit_status=0`. It covered
Monarch actor creation, vLLM XPU initialization, TorchStore synchronization,
rollout, compiled forward/backward, AdamW, checkpoint save, post-step weight
synchronization, and clean shutdown. Its intentionally retained zero-variance
cold-start batch had zero loss and gradient, so it validates the path rather
than learning.

## Interim observations

At step 4, both production reruns are healthy and have nonzero gradients:

| Recipe | Step rewards observed | Steady-state trainer throughput | Gradient norm range |
|---|---|---:|---:|
| easy | `0.081, 0.16, 0.17, 0.12` | about `1,795-1,850 tok/s` | `0.010-0.026` |
| shaped | `0.14, 0.12, 0.14, 0.15` | about `1,803-1,861 tok/s` | `0.030-0.050` |

This already reproduces the historical Monarch infrastructure throughput of
approximately 1,765 tok/s. It does **not** yet establish either final reward
claim; both jobs must finish and the shaped completions must be cross-scored.

## Acceptance criteria

The reproduction is complete only after all of the following:

- both PBS jobs finish with `Exit_status=0`;
- all 100 optimizer steps are present with nonzero training activity;
- easy-task reward history and last-window mean are extracted;
- shaped-reward history and last-window mean are extracted;
- shaped completions are rescored offline with character-ratio(power=1);
- final checkpoints, logs, code commit, model hash, and any deviations are
  recorded here.
