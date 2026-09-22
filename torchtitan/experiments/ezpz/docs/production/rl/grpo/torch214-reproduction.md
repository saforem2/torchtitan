# Torch 2.14 Monarch GRPO reproduction

> Status: **stopped after semantic rollout audit failed** (2026-09-22). This page records a clean-room rerun of the
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
| easy baseline | `12478404` | `x1922c1s3b0n0` | `/lus/tegu/projects/datascience/foremans/reproductions/grpo-easy-c5f8deac6-12478404` | stopped by operator; PBS exit 143 |
| shaped ceiling attack | `12478405` | `x1922c1s5b0n0` | `/lus/tegu/projects/datascience/foremans/reproductions/grpo-shaped-c5f8deac6-12478405` | stopped by operator; PBS exit 143 |

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

## Semantic rollout audit and stop decision

At step 4, both production reruns were mechanically healthy and had nonzero gradients:

| Recipe | Step rewards observed | Steady-state trainer throughput | Gradient norm range |
|---|---|---:|---:|
| easy | `0.081, 0.16, 0.17, 0.12` | about `1,795-1,850 tok/s` | `0.010-0.026` |
| shaped | `0.14, 0.12, 0.14, 0.15` | about `1,803-1,861 tok/s` | `0.030-0.050` |

This reproduces the historical Monarch infrastructure throughput of
approximately 1,765 tok/s, but an explicit audit of `rollout_samples.jsonl`
failed semantic-quality acceptance. Both jobs were stopped at 38m53s rather
than spending the remaining allocation optimizing an exploitable reward.

| Recipe | Rollouts inspected | Hit length limit | Copied demo names | Mean completion characters |
|---|---:|---:|---:|---:|
| easy | 836 | 100% | 94.4% | 2,373.6 |
| shaped | 852 | 100% | 94.2% | 2,223.0 |

The completions repeatedly copied `QuinnRivera/OmarSaito/PiaValdez`, emitted
multiple or malformed `<alphabetical_sorted>` blocks, repeated
`<end_of_turn>`, and often degraded into unrelated prose or character soup.
The shaped reward assigned values as high as 0.8-0.9 to some malformed,
repetitive outputs. Reward and throughput therefore cannot be treated as
evidence of a successful learning-result reproduction.

These were single-node, two-tile AGPT-2B runs. Historical records document a
10-node AGPT-2B cross-node vLLM run (job `12470959`), but its raw completions
have not yet been re-audited under this semantic criterion. No AGPT-20B
Monarch/GRPO rollout run or archived 20B rollout JSONL was found; existing
AGPT-20B records cover training, scaling, and evaluation rather than this RL
generation path.

### Forced-Gloo TorchStore A/B

Job `12478413` forced TorchStore's data transport to Gloo with
`TORCHTITAN_TORCHSTORE_TRANSPORT=gloo` and disabled TorchStore XCCL, while
leaving TorchTitan's XPU training collectives unchanged. The run used commit
`4f16c6a9b1d9369550e428412e9a96eb6aed21f5` and the same model hash above.
It completed with PBS `Exit_status=0`; TorchStore moved the approximately 8 GB
state at 2.39 GB/s on put and 2.56 GB/s on get, and training produced nonzero
loss and gradient.

The four captured rollouts nevertheless remained malformed and all reached
the 128-token limit. They copied the demonstration names, repeated malformed
blocks and `<end_of_turn>`, or answered with irrelevant Python code. Forced
Gloo therefore did **not** fix generation quality. This rules out the
same-host SharedMemory-versus-Gloo transport choice as the cause, but does not
yet rule out a bug above the byte transport layer (state-dict adaptation,
parameter-name mapping, dtype/layout conversion, or model/tokenizer lineage).
Because automatic TorchStore selection prefers SharedMemory on a colocated
single-node topology, a true cross-node XCCL-versus-Gloo comparison remains a
separate test.

### Known-good Qwen3-0.6B control

To distinguish a platform-wide vLLM/TorchStore fault from an AGPT-specific
problem, the official `Qwen/Qwen3-0.6B` checkpoint was tested with identical
alphabet-sort prompts. The staged `model.safetensors` SHA-256 was
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

Direct vLLM job `12478419` used the XPU Triton attention backend and exited 0.
Seven of eight outputs were exact, correctly sorted tagged blocks; the eighth
had the correct sorted names but omitted only the closing tag. There was no
repetition or gibberish.

TorchTitan/TorchStore job `12478425` loaded the same checkpoint through the
TorchTitan Qwen adapter, forced `TransportType.Gloo`, synchronized weights into
the generator, generated rollouts, ran backward and AdamW, saved a checkpoint,
and exited 0. All four recorded rollouts terminated normally and were coherent
tagged answers. Two preserved a concatenated name exactly; two split its first
and last name over adjacent lines, but none copied the demonstration names,
looped, or produced character soup. Step 1 reported reward 0.70, loss -0.0010,
and gradient norm 0.36.

This control demonstrates that vLLM XPU and TorchStore Gloo can preserve a
known-good model through the complete RL path. The AGPT failure is therefore
model/checkpoint/adapter-specific rather than a general Gloo byte-transport
failure. The next isolation step is pre-sync versus post-sync AGPT parameter
and logit parity across HF, TorchTitan, and the registered vLLM wrapper.

## Acceptance criteria

Any corrected reproduction is complete only after all of the following:

- both PBS jobs finish with `Exit_status=0`;
- all 100 optimizer steps are present with nonzero training activity;
- easy-task reward history and last-window mean are extracted;
- shaped-reward history and last-window mean are extracted;
- shaped completions are rescored offline with character-ratio(power=1);
- final checkpoints, logs, code commit, model hash, and any deviations are
  recorded here.
- sampled raw completions pass an explicit non-gibberish and task-compliance
  audit before scaling to multi-node or AGPT-20B.
