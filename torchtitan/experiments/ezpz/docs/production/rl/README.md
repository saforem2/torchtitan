# RL (GRPO) on Intel XPU

Reinforcement learning (GRPO) for AuroraGPT / Qwen3 on ALCF XPU systems
(Sunspot / Aurora). This is the **status hub** -- it routes to the per-path docs.

## Two frameworks

There are two independent GRPO stacks. TRL is the mature, multi-node-validated
path; the Monarch stack is the upstream `torchtitan.experiments.rl` engine,
recently vendored into our repo and verified.

| # | Path | Framework | Generation | Entry point | venv | Status | Doc |
|---|------|-----------|------------|-------------|------|--------|-----|
| A2 | **TRL + vLLM-server** | TRL `GRPOTrainer` | `trl vllm-serve` (on-policy, weight-sync) | `rl.train_grpo` `--use_vllm --vllm_mode server` | `rl-vllm` | **works** 1N / cross-node / multi-trainer-node | [`trl.md`](trl.md) |
| A1 | TRL + `.generate()` | TRL `GRPOTrainer` | HF `.generate()` per-rank | `rl.train_grpo` (no `--use_vllm`) | `rl-vllm` | works (slow fallback) | [`trl.md`](trl.md) |
| B | **Monarch + TorchStore + vLLM** | upstream `experiments.rl` | vLLM (Monarch actor) | `rl.train_upstream` | `rl-grpo-lora` | **works** (2-tile verified, vendored) | [`monarch.md`](monarch.md) |

A1 and A2 are the same TRL `GRPOTrainer` with two generation backends (toggle
`--use_vllm`); B is a separate architecture.

### Which to use
- **Production / multi-node GRPO today: TRL + vLLM-server (A2)** -- the validated,
  scaled path. See [`trl.md`](trl.md).
- **Fast local sanity check with no server: TRL `.generate()` (A1)** -- same entry
  point, drop the `--use_vllm` flags.
- **Upstream Monarch+TorchStore+vLLM (B)** -- now runs from our repo on 2 tiles
  (agpt-2b GRPO+LoRA); the on-policy RDMA-weight-store architecture. See
  [`monarch.md`](monarch.md).

## Quick start

```bash
# A2 -- TRL + vLLM-server (recommended). Use the ready-made PBS scripts (they handle
# the server subshell, health poll, tile/node partitioning, venv, XPU env):
bash torchtitan/experiments/ezpz/rl/scripts/grpo/qwen3_vllm_server_smoke.sh   # 1N
# details + cross-node / multi-trainer-node scripts: trl.md

# B -- Monarch + TorchStore + vLLM (agpt-2b GRPO+LoRA), from the REPO ROOT:
bash torchtitan/experiments/ezpz/rl/scripts/grpo/agpt2b_grpo.sh
# details: monarch.md
```

## Tasks (TRL path)

Pluggable via `--task <name>` (registry in
[`rl/tasks/__init__.py`](../../../rl/tasks/__init__.py)):

| Task | Description | Difficulty |
|------|-------------|------------|
| `sum_digits` (default) | Addition: "What is 3 + 7 + 2?" -> "12" | Easy |
| `multiply` | Multiplication: "What is 7 x 8?" -> "56" | Easy |
| `word_sort` | Sort words alphabetically (partial credit) | Medium |
| `countdown` | Reach target using arithmetic on given numbers | Hard |

Add one: define a dataset + reward fns in `rl/tasks/`, call `register_task()`,
import it from `tasks/__init__.py`. See `rl/tasks/sum_digits.py`. (The Monarch
path B uses the upstream `alphabet_sort` task via the `alphabet_sort_agpt` overlay.)

## AuroraGPT-2B checkpoint paths

The SophiaG `global_step138650` HF checkpoint is the recommended start (local,
no HF-Hub download; `model_type=llama` -> FSDP wraps `LlamaDecoderLayer`):

| Machine | Path |
|---------|------|
| Aurora | `/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650` |
| Sunspot | `/home/foremans/datascience/foremans/projects/saforem2/torchtitan/AuroraGPT-2B-sophiag-gs138650` |

The SFT deliverable used by path B is `outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf`.

## Layout

```
docs/production/rl/
  README.md      <- this hub
  trl.md         <- TRL GRPOTrainer (vLLM-server + .generate fallback): status, run, stack
  monarch.md     <- Monarch + TorchStore + vLLM (vendored, verified): status, run, arch
  grpo/          <- per-run GRPO results + charts (aurora2b sft_arithmetic, ...)
  2026-07-06_multinode-grpo-root-cause.md   <- TRL multi-node root-cause (evidence)
  history/       <- superseded investigations + bring-up chronology + prior reproductions
```

## Verified results (headline)

- **TRL + vLLM-server:** 1 node (job 12468780), cross-node generation (12469976),
  multi-trainer-node 3N/24-rank (12470083) -- all works. Detail + evidence:
  [`trl.md`](trl.md), [`grpo/`](grpo/README.md).
- **Monarch + TorchStore + vLLM:** agpt-2b GRPO+LoRA 2-tile, Train steps fire
  ~1765 tok/s, reward 0.16->0.26 on the easy task (job 12471049) -- runs from our
  repo, zero core edits. Detail: [`monarch.md`](monarch.md).

## Dependencies

Two venvs (both py3.12, torch 2.12+xpu, triton-xpu 3.7.1):
- `venvs/rl-vllm` (A1/A2): vllm + vllm-xpu-kernels + trl. Build:
  [`rl/scripts/build_rl_vllm_venv.sh`](../../../rl/scripts/build_rl_vllm_venv.sh).
- `venvs/rl-grpo-lora` (B): + torchstore/monarch (from source) + ezpz. Build:
  [`rl/scripts/build_rl_grpo_lora_venv.sh`](../../../rl/scripts/build_rl_grpo_lora_venv.sh).

NEVER `pip install` torch or its deps without `--no-deps` (silently replaces the
XPU torch with a CUDA build).
