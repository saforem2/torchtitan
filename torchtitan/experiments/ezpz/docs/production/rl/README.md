# RL (GRPO) Experiment

**Status: Experimental** — verified on Sunspot XPU, not production-ready.

## Layout (this dir consolidates all RL docs, 2026-07-18)

This `docs/production/rl/` tree holds BOTH the RL infrastructure/how-to AND the
per-run results (previously split across `docs/rl/` + `docs/production/grpo/`):

- **Infra / how-to** (this README + siblings): `grpo-on-xpu-status.md` (the
  run recipe -- READ FIRST), `2026-07-06_multinode-grpo-root-cause.md`, and the
  `history/` archive (superseded bring-up writeups).
- **Per-run results**: [`grpo/`](grpo/README.md) -- the production GRPO index
  (recipe + checkpoint pairs, e.g. `grpo/aurora2b/sft_arithmetic/`).

Reinforcement Learning via Group Relative Policy Optimization (GRPO) using
HuggingFace TRL's `GRPOTrainer`, with **on-policy generation via `trl
vllm-serve` (vLLM-XPU)** — working end-to-end on XPU including
multi-trainer-node. An alternative to upstream torchtitan's RL experiment,
which is CUDA-only (Monarch actors + TorchStore RDMA + flash-attn); this path
reaches the same on-policy generator/trainer + weight-sync shape on the XPU
stack. (A slower per-rank `hf.generate()` fallback also exists.)

## Architecture

- **TRL GRPOTrainer** — handles the GRPO training loop (generate → score →
  compute advantages → update policy)
- **ezpz** — distributed launch, device setup, wandb tracking
- **Generation backend** — two options:
  - ✅ **vLLM server (`trl vllm-serve`) — RECOMMENDED, and what all current
    results use.** On-policy: a `trl vllm-serve` daemon generates (fast,
    batched, paged-KV) and the trainer syncs updated weights back to it each
    step. **Working end-to-end on XPU, including multi-trainer-node** (FSDP
    across 2+ nodes). This is the performant path — use it unless you have a
    reason not to. See [Quick Start](#recommended-vllm-server-on-policy) below.
  - **HF `.generate()` per-rank — fallback.** Every trainer rank generates with
    transformers `.generate()`. Slow, but needs no server and always works on
    XPU; use it for a quick single-node sanity check.

  vLLM-server references:
  - ✅ **[`grpo-on-xpu-status.md`](grpo-on-xpu-status.md)** — current status +
    the full run recipe (1N / cross-node / multi-trainer-node). **Read this first.**
  - ✅ **[`2026-07-06_multinode-grpo-root-cause.md`](2026-07-06_multinode-grpo-root-cause.md)**
    — how the multi-trainer-node block was root-caused + fixed (a
    `CCL_ATL_TRANSPORT` transport-default regression + the AVG->SUM FSDP patch
    rebinding the wrong name), NOT the "desync" the 2026-07-01 session concluded.
  - `rl/scripts/grpo/qwen3_vllm_server_smoke.sh` — 1N smoke; unified
    `venvs/rl-vllm/` venv (py3.12). `rl/scripts/grpo/grpo_3n_multinode_validate.sh`
    — the multi-trainer-node run.
  - `rl/scripts/build_rl_vllm_venv.sh` — reproducible venv build.
  - `rl/xpu_overrides.py` — XPU shim collecting every monkey-patch +
    env setup needed for the port.
  - Background reading (superseded 2026-06 bring-up records, archived under
    [`history/`](history/README.md)):
    - [`history/vllm-xpu-investigation.md`](history/vllm-xpu-investigation.md) — original
      2026-06-10 vLLM-XPU verification + sibling-venv recipe.
    - [`history/vllm-xpu-current-status.md`](history/vllm-xpu-current-status.md) — the
      15-job bare-vLLM debug chain that led to the venv design.
    - [`history/vllm-xpu-wiring-plan.md`](history/vllm-xpu-wiring-plan.md) — pre-impl
      architecture decision (TRL `vllm_mode="server"` vs Monarch+TorchStore).
    - [`history/upstream-rl-port-status.md`](history/upstream-rl-port-status.md) — why
      using upstream `torchtitan.experiments.rl` directly (Monarch+TorchStore)
      is blocked on the same Sunspot stack.
    - [`history/2026-06-14_monarch-torch213-deep-dive.md`](history/2026-06-14_monarch-torch213-deep-dive.md)
      — torch 2.13 + monarch + vllm-xpu push: got past every XCCL/USM/DCP/dynamo
      blocker (6 new patches in `xpu_overrides.py`), final wall is vLLM
      `profile_run` → oneDNN `could not create a memory` on `F.linear`.

## Tasks

Tasks are pluggable via a registry. Use `--task <name>` to select. The
choices are auto-populated in `--help` from
[`rl/tasks/__init__.py`](../../../rl/tasks/__init__.py):

| Task | Description | Difficulty |
|------|-------------|------------|
| `sum_digits` (default) | Addition: "What is 3 + 7 + 2?" → "12" | Easy |
| `multiply` | Multiplication: "What is 7 × 8?" → "56" | Easy |
| `word_sort` | Sort words alphabetically (partial credit) | Medium |
| `countdown` | Reach target using arithmetic on given numbers | Hard |

### Adding a new task

Create a module in `rl/tasks/`, define a dataset builder + reward
functions, and call `register_task()`. Import it from
[`tasks/__init__.py`](../../../rl/tasks/__init__.py) so it self-registers
on package load — `--help` will pick it up automatically. See
[`tasks/sum_digits.py`](../../../rl/tasks/sum_digits.py) for the pattern.

## AuroraGPT-2B checkpoint paths

The AuroraGPT-2B SophiaG checkpoint (`global_step138650`, HF-format) is
the recommended starting point on ALCF systems. It's a local checkpoint
(no HF Hub download needed, no rate-limit risk) and `model_type=llama`
so FSDP wrap-class auto-detection picks `LlamaDecoderLayer`.

| Machine | Path |
|---------|------|
| Aurora | `/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650` |
| Sunspot | `/home/foremans/datascience/foremans/projects/saforem2/torchtitan/AuroraGPT-2B-sophiag-gs138650` |

## Quick Start

The CLI uses `HfArgumentParser`, so every `GRPOConfig` + `TrainingArguments`
field is exposed as a flag (191 total). Run `--help` for the full list. All
flags use `snake_case` (e.g. `--per_device_train_batch_size`,
`--max_steps`), not `--hyphen-form`.

### Recommended: vLLM-server (on-policy)

The performant path -- a `trl vllm-serve` daemon generates and the trainer
syncs weights back each step. It is a **two-process launch** (server on one
tile/node + trainer on the rest), so use the ready-made PBS scripts rather than
hand-rolling it; they handle the server subshell, health poll, tile/node
partitioning, the unified `venvs/rl-vllm/` venv, and the XPU env:

- **1 node** (server + trainer co-located):
  [`rl/scripts/grpo/qwen3_vllm_server_smoke.sh`](../../../rl/scripts/grpo/qwen3_vllm_server_smoke.sh)
- **cross-node** (server node + 1 trainer node):
  [`rl/scripts/grpo/aurora2b_sft_arithmetic_vllm_xnode.sh`](../../../rl/scripts/grpo/aurora2b_sft_arithmetic_vllm_xnode.sh)
- **multi-trainer-node** (server + 2+ trainer nodes):
  [`rl/scripts/grpo/grpo_3n_multinode_validate.sh`](../../../rl/scripts/grpo/grpo_3n_multinode_validate.sh)

The trainer side is the same `train_grpo` entry point with the vLLM flags:

```bash
python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task arithmetic \
    --model_name_or_path <hf-checkpoint-dir> \
    --per_device_train_batch_size 1 --num_generations 4 \
    --bf16 --beta 0.0 --fsdp full_shard --max_steps 50 \
    --use_vllm --vllm_mode server --vllm_server_base_url http://<server-host>:8765
```

Prereqs: build the venv once with
[`rl/scripts/build_rl_vllm_venv.sh`](../../../rl/scripts/build_rl_vllm_venv.sh);
for **multi-trainer-node**, keep the `ofi`/TCP-KVS transport ON (do NOT pass
`--no-oneccl-tcp-kvs`) and let `apply_all_xpu_patches` apply the AVG->SUM FSDP
patch -- see [`grpo-on-xpu-status.md`](grpo-on-xpu-status.md#how-to-run-it).

### Fallback: HF `.generate()` per-rank (no server)

Slower (every rank generates with transformers), but needs no server -- handy
for a quick single-node sanity check. Omit the `--use_vllm*` flags:

```bash
# Minimal (plain DDP)
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task sum_digits \
    --model_name_or_path /home/foremans/datascience/foremans/projects/saforem2/torchtitan/AuroraGPT-2B-sophiag-gs138650 \
    --per_device_train_batch_size 1 --max_steps 50 --bf16

# FSDP full-shard, 4N
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task sum_digits \
    --model_name_or_path /home/foremans/datascience/foremans/projects/saforem2/torchtitan/AuroraGPT-2B-sophiag-gs138650 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --bf16 --beta 0.0 --fsdp full_shard --max_steps 50
```

On Aurora, swap the path to
`/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650`.

`--fsdp_transformer_layer_cls_to_wrap` defaults to `LlamaDecoderLayer`
(matches AuroraGPT-2B); for other models the script auto-detects the wrap class
via `AutoConfig.model_type` (llama, llama4, qwen2, qwen3, mistral, mixtral,
gemma, gemma2, phi, phi3, gpt_neox, gpt2, deepseek_v3, olmo, olmo2 — extend
[`train_grpo.py`](../../../rl/train_grpo.py) `_DEFAULT_WRAP_CLS_BY_MODEL_TYPE`).

## What works under the hood (handled automatically)

The script papers over several XPU/TRL/accelerate friction points that
took empirical iteration to land. Documented here so you know what NOT
to debug if you change something:

1. **Rank-0 HF Hub prefetch + broadcast.** 48 ranks doing
   `from_pretrained` concurrently against the same HF Hub repo trips
   per-IP 429 rate limits. Rank 0 prefetches the model files; workers
   load from the warm cache after a barrier. (No effect when
   `--model_name_or_path` is a local path like the AuroraGPT paths above
   — `snapshot_download` 404s but workers just read the local files.)

2. **FSDP auto-detect for wrap class.** Picks the right `*DecoderLayer`
   from the model's HF `model_type`. Override with
   `--fsdp_transformer_layer_cls_to_wrap` if needed.

3. **FSDP env bootstrap.** HF Trainer's internal `accelerate.Accelerator`
   reads `ACCELERATE_USE_FSDP=true` + `FSDP_*` env vars at construction
   time. Under `ezpz launch` (mpiexec) these aren't set automatically;
   the script injects them before `GRPOTrainer.__init__` so `--fsdp
   full_shard` actually shards.

4. **`device_map="auto"` override.** TRL's `create_model_from_path`
   defaults `device_map="auto"` which, under `ZE_FLAT_DEVICE_HIERARCHY=
   FLAT`, places every rank's model on `xpu:11`. The script overrides
   to `device_map=None` so HF Trainer's normal
   `model.to(accelerator.device)` puts the model on the per-rank tile.

5. **Activation-checkpointing migration.** When `--fsdp full_shard` is
   on, `gradient_checkpointing=True` is silently migrated to
   `fsdp_config["activation_checkpointing"]=True` to avoid the redundant
   AllGather warning.

6. **Chat template fallback.** Base/pretraining-only tokenizers (like
   AuroraGPT-2B) ship without a chat template; the script injects a
   minimal chatml-style template (preserves existing templates).

## Observability (W&B)

Every field of `EzpzGRPOArgs` and `EzpzGRPOConfig` (including all
inherited `TrainingArguments` fields) is logged to W&B at run-init under
`ezpz/*`, `train/*`, and `runtime/*` prefixes (~180 hyperparameter keys).

`log_completions=True` is on by default — GRPO streams sample prompts +
generated completions to a W&B table every `logging_steps`, so you can
see what your model is actually outputting during training without
adding any code. (Override with `--no_log_completions` if you want to
disable it.)

Metrics logged during training (every `logging_steps`):

| Group | Keys |
|-------|------|
| Standard HF | `loss`, `grad_norm`, `learning_rate`, `epoch`, `num_tokens` |
| Completions | `completions/mean_length`, `min_length`, `max_length`, `clipped_ratio`, `mean_terminated_length`, … |
| Rewards | `rewards/<func_name>/mean`, `rewards/<func_name>/std`, `reward`, `reward_std`, `frac_reward_zero_std` |
| GRPO objective | `entropy`, `kl` (only when `beta != 0`), `clip_ratio/{low,high,region}_{mean,min,max}` |
| Timing | `step_time` |

## Dependencies

- `trl` — install with
  `uv pip install --no-deps --no-cache --link-mode=copy trl`
  (use `--no-deps` to avoid pulling CUDA torch)
- `transformers`, `datasets`, `accelerate` — usually already installed
- On Sunspot/Aurora, the AuroraGPT-2B paths above are pre-staged. For
  HF Hub models, compute nodes need the ALCF proxy (set
  `http_proxy=https_proxy=http://proxy.alcf.anl.gov:3128`).

## Verified Results

Most recent first. The **vLLM-server** entries (on-policy: `trl vllm-serve`
generates, weights synced back each step) are the current milestones; the
older `hf.generate` smokes below are the bring-up record. Full status +
recipe: [`grpo-on-xpu-status.md`](grpo-on-xpu-status.md).

### vLLM-server, multi-trainer-node (2026-07-06, job 12470083)

**The multi-trainer-node milestone** -- FSDP2 across 2 trainer nodes, on-policy.

- 3N = 1 server + 2 trainer nodes (24 trainer ranks); task `arithmetic`,
  model = SFT'd AuroraGPT-2B (`checkpoint-729-hf`).
- **8/8 steps clean**, `train_loss` -0.02445, `train_runtime` 1411s (~176s/step);
  `accuracy_reward` -> 0.375, `format_reward` -> 0.25 by step 8; real cross-node
  FSDP grad `reduce_scatter_tensor`, 0 AVG-wall / 0 segfault.
- W&B: [run p52blp3u](https://wandb.ai/aurora_gpt/torchtitan.ezpz.rl/runs/p52blp3u).
- Root cause of the prior block + fix:
  [`2026-07-06_multinode-grpo-root-cause.md`](2026-07-06_multinode-grpo-root-cause.md).

### vLLM-server, cross-node generation (2026-07-01, job 12469976)

Server on node A, **1** trainer node on node B -- proves cross-node on-policy
weight-sync + generation over HTTP.

- 2N; **10/10 steps**, 1110 `update_named_param` weight-syncs to the server,
  rewards moving. This is the working baseline the multi-trainer-node work
  built on. Recipe:
  [`rl/scripts/grpo/aurora2b_sft_arithmetic_vllm_xnode.sh`](../../../rl/scripts/grpo/aurora2b_sft_arithmetic_vllm_xnode.sh).

### vLLM-server, 1 node (2026-06-13, job 12468780)

**First end-to-end on-policy GRPO on XPU** -- server + trainer co-located,
real weight-sync (not the frozen-generator no-op).

- Task `sum_digits`, AuroraGPT-2B; 5/5 steps, `train_loss` -0.0134,
  `format_reward` ratchets 0 -> 0.0625 -> 0.25 -> 0.125 across 5 steps on a
  model that never saw the task; `importance_sampling_ratio` ~1.0 (on-policy
  confirmed). ~4.3s/step after JIT warmup. Script:
  [`rl/scripts/grpo/qwen3_vllm_server_smoke.sh`](../../../rl/scripts/grpo/qwen3_vllm_server_smoke.sh).

---

### hf.generate bring-up smokes (historical, 2026-04 -> 06)

Early per-rank-`hf.generate` / DDP smokes from before the vLLM-server path
landed. Kept for the record.

#### Sunspot 4N + FSDP full_shard, AuroraGPT-2B (2026-06-07, job 12468209)

End-to-end smoke (48 ranks, 2 training steps for verification only):

- Task: `sum_digits`, model: AuroraGPT-2B-sophiag-gs138650
- `--bf16 --beta 0.0 --fsdp full_shard --max_steps 2 --per_device_train_batch_size 1`
- Training: 21 seconds for 2 steps (init + 2 generate/score/update cycles)
- W&B: [fearless-galaxy-48](https://wandb.ai/aurora_gpt/torchtitan.ezpz.rl/runs/jjzwmija)

#### Sample completions (chat-template verification, job 12468210)

25-step run on Sunspot 4N FSDP-full_shard with AuroraGPT-2B-sophiag-gs138650
on the `sum_digits` task. The Gemma-style chat-template fallback fires
because the tokenizer ships without a `chat_template`:

```
[rank 0] tokenizer has no chat_template; injected 'gemma' fallback
```

Step 1 (cold start — model is rambling, having never seen the task before):

```
╭─────────────────────────────────── Step 1 ───────────────────────────────────╮
│ ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┓ │
│ ┃ Prompt        ┃ Completion    ┃ accuracy_rew… ┃ format_rewa… ┃ Advantage ┃ │
│ ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━┩ │
│ │ user          │ What is 9 +   │          0.00 │         0.00 │     -0.50 │ │
│ │ What is 9 + 9 │ the root of   │               │              │           │ │
│ │ + 3? Reply    │ the number?   │               │              │           │ │
│ │ with just the │ 9 = 9         │               │              │           │ │
│ │ number.       │ ... (rambles) │               │              │           │ │
│ └───────────────┴───────────────┴───────────────┴──────────────┴───────────┘ │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Step 25 (model has adapted — actually answers, stops via `<end_of_turn>`):

```
╭────────────────────────────────── Step 25 ───────────────────────────────────╮
│ ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┓ │
│ ┃ Prompt        ┃ Completion    ┃ accuracy_rew… ┃ format_rewa… ┃ Advantage ┃ │
│ ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━┩ │
│ │ user          │ 1 + 3 + 4 + 3 │          0.00 │         0.50 │      0.50 │ │
│ │ What is 1 + 3 │ = 10.         │               │              │           │ │
│ │ + 4 + 3?      │ What is 1 + 3 │               │              │           │ │
│ │ Reply with    │ + 4 + 3 + 1?  │               │              │           │ │
│ │ just the      │ ...           │               │              │           │ │
│ │ number.       │               │               │              │           │ │
│ │ model         │               │               │              │           │ │
│ └───────────────┴───────────────┴───────────────┴──────────────┴───────────┘ │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Key signal: at step 25, `completions/min_length=13-15` and
`mean_terminated_length=42`. Before the chat-template fix, every
completion ran to the 64-token clip ceiling because the model had
no boundary signal. After the fix, the model is generating
`<end_of_turn>` and stopping early — that's how you know the gemma
template is actually being interpreted as turn boundaries, not just
echoed as text.

The accuracy reward stays low at step 25 because the model is
correctly computing the sum (`= 10.`) but then continues to generate
a follow-up question instead of just emitting the bare number the
prompt asked for. This is the format-vs-accuracy tradeoff the dual
reward functions exist to disentangle — longer training (the original
2026-04-15 Qwen3 run reached 87.5% accuracy at step 9) drives both.

#### Sunspot 4N + FSDP full_shard, Qwen3-0.6B (2026-06-07, job 12468205)

Same harness, Qwen3-0.6B from HF Hub. Verified the
`device_map="auto"` → `None` override path; reached
`Training complete.` cleanly.

#### Sunspot 2N + DDP, Qwen3-0.6B (2026-04-15)

**Config:** Qwen3-0.6B, 24 XPU tiles (2 nodes), 10 steps, batch=1,
2 generations per prompt, 100 training samples.

| Step | Accuracy Reward | Format Reward |
|------|----------------|---------------|
| 1    | 4.2%           | 10.4%         |
| 5    | 33.3%          | 50.0%         |
| 8    | 62.5%          | 50.0%         |
| 9    | **87.5%**      | 50.0%         |
| 10   | 62.5%          | 50.0%         |

Training time: 41.3s, 5.8 samples/sec.

## Files

| File | Description |
|------|-------------|
| [`train_grpo.py`](../../../rl/train_grpo.py) | Main entry point — task-agnostic GRPO loop, FSDP wiring, W&B init |
| [`tasks/__init__.py`](../../../rl/tasks/__init__.py) | Task registry (`RLTask`, `register_task`, `get_task`) |
| [`tasks/common.py`](../../../rl/tasks/common.py) | Shared helpers (answer extraction, completion text) |
| [`tasks/sum_digits.py`](../../../rl/tasks/sum_digits.py) | Sum-of-digits task (dataset + rewards) |
| [`tasks/multiply.py`](../../../rl/tasks/multiply.py) | Multiplication task |
| [`tasks/word_sort.py`](../../../rl/tasks/word_sort.py) | Word sorting task (partial credit) |
| [`tasks/countdown.py`](../../../rl/tasks/countdown.py) | Countdown arithmetic reasoning task |

## Limitations

- **vLLM-XPU TP=1 only** — the `trl vllm-serve` server runs on a single tile;
  multi-tile vLLM TP>1 on XPU is unexercised by this work.
- **Trainer XCCL on TCP-KVS, not CXI** — the vLLM-server path currently uses
  the TCP fabric for the trainer's own collectives too; Slingshot CXI would be
  faster. A per-group transport split (CXI intra-node + TCP-KVS for the
  cross-process server group) is the open perf lever. See
  [`grpo-on-xpu-status.md`](grpo-on-xpu-status.md#operational-details).
- **Monarch/TorchStore path still blocked** — the upstream (CUDA-native) RL
  loop does not run on XPU; this ezpz path uses TRL + `trl vllm-serve` instead.
  See [`history/upstream-rl-port-status.md`](history/upstream-rl-port-status.md).
- **hf.generate fallback is slow** — if you skip the vLLM server and let every
  rank generate with `.generate()`, throughput drops sharply. Prefer the
  vLLM-server path.

## Upstream Comparison

The upstream `torchtitan/experiments/rl/` experiment uses Monarch actors for
separate generator/trainer meshes, vLLM for fast inference, and TorchStore for
weight sync via GPU-to-GPU RDMA — and requires CUDA, flash-attn, torchmonarch.

This ezpz alternative reaches the **same on-policy shape** (a `trl vllm-serve`
generator + trainer with per-step weight sync over TRL's `StatelessProcessGroup`)
without the CUDA-only stack — it runs on any backend TRL/Accelerate supports
(XPU, CUDA, CPU). It is not yet as fast as the upstream RDMA path (TP=1 server,
TCP trainer fabric — see Limitations), but the generator/trainer split and
weight sync that the old version of this doc said were missing **are now
present and verified** (jobs 12468780 / 12469976 / 12470083).
