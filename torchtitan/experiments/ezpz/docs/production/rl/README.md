# RL (GRPO) on Intel XPU

> [!IMPORTANT]
> **Current production instructions:** use
> [Production RL with Monarch, TorchStore, and vLLM on XPU](monarch.md).
> It contains the validated Sunspot two-host contract and the Aurora
> `next-eval` runtime/preflight requirements. The TRL and July-era pages below
> remain available for historical reproduction but are not the default runbook.

> **Last updated: 2026-09-25.** The newest production-path validation is the
> two-host Monarch/TorchStore/vLLM job `12478711`. The result proves runtime
> integration, not model-quality improvement. Historical result tables remain
> below for context.

> [!IMPORTANT]
> **For the 2B model's post-training results, start at
> [AuroraGPT-2B post-training status](../POST-TRAINING-2B.md)** -- SFT and RL
> together, with what each actually bought.
>
> Short version for quality: **mechanically successful GRPO has not yet moved
> held-out accuracy on the metric we care about.** The current multi-host gate
> validates actors, policy transfer, generation, optimization, checkpoints, and
> shutdown only. Quality promotion still requires a separate paired semantic
> evaluation.
>
> What RL *has* delivered: a genuinely solved task (sum_digits, accuracy_reward
> ~0.4 -> ~0.9 over 1000 steps), a **+168%** reward-shaping result, and
> **RL-as-an-SFT-probe** -- convergence speed cleanly separates checkpoints.
> No LoRA adapter has been promoted to production.
>
> Two findings worth reading before starting a run: the alphabet_sort plateau
> was the **reward function**, not capacity ([ceiling-attack](grpo/ceiling-attack.md));
> and reward-hacking was caught in the act -- reward rose while eval accuracy
> fell -- caused by a **weak cold-start**, not by GRPO ([cot.md](plans/cot.md)).

Reinforcement learning (GRPO) for AuroraGPT / Qwen3 on ALCF XPU systems
(Sunspot / Aurora). This is the **status hub** -- it routes to the per-path docs.

## Current production path

The current path is upstream-style TorchTitan RL with Monarch actors,
TorchStore policy synchronization, and vLLM generation. Sunspot has passed the
full two-host gate; Aurora `next-eval` has a defined runtime contract but still
requires its own multi-host RL hardware gate before production use.

Start here: [`monarch.md`](monarch.md).

## Historical frameworks and compatibility paths

Two older framework views remain documented. TRL is a legacy compatibility path
that was multi-node-validated in July 2026. The Monarch page records the older
single-host bring-up and reward studies. Neither replaces the current
production runbook linked above.

> [!NOTE]
> See [GRPO+LoRA on XPU: agpt-2b (Llama) port for SFT checkpoint-900](history/grpo-lora-agpt2b-repro.md)

| # | Path | Framework | Generation | Entry point | venv | Status | Doc |
|---|------|-----------|------------|-------------|------|--------|-----|
| Current | **Monarch + TorchStore + vLLM** | upstream TorchTitan RL | vLLM actor + TorchStore/Gloo | multi-host SPMD entry point | current machine-specific runtime | **Sunspot two-host validated; Aurora next-eval gate pending** | [`monarch.md`](monarch.md) |
| A2 | TRL + vLLM-server | TRL `GRPOTrainer` | `trl vllm-serve` (on-policy, weight-sync) | `rl.train_grpo` `--use_vllm --vllm_mode server` | `rl-vllm` | **deprecated for new production; historical validation retained** | [`trl.md`](trl.md) |
| A1 | TRL + `.generate()` | TRL `GRPOTrainer` | HF `.generate()` per-rank | `rl.train_grpo` (no `--use_vllm`) | `rl-vllm` | works (slow fallback) | [`trl.md`](trl.md) |
| B (historical) | Monarch + TorchStore + vLLM | older vendored bring-up | vLLM (Monarch actor) | `rl.train_upstream` | `rl-grpo-lora` | one-host/two-tile historical result | [`monarch.md`](monarch.md) |

A1 and A2 are the same TRL `GRPOTrainer` with two generation backends (toggle
`--use_vllm`); B is a separate architecture.

### Which to use
- **Production / multi-host GRPO today:** follow
  [`monarch.md`](monarch.md). The committed
  Sunspot launcher is the validated reference; Aurora `next-eval` requires the
  separate runtime and hardware gate documented there.
- **Historical TRL reproduction:** use [`trl.md`](trl.md) only when explicitly
  reproducing the legacy server-mode path.
- **Fast local sanity check with no server: TRL `.generate()` (A1)** -- same entry
  point, drop the `--use_vllm` flags.
- **Older single-host Monarch studies:** see [`monarch.md`](monarch.md) for
  reward experiments and bring-up details, not current launch instructions.

## Quick start

```bash
# Current two-host Sunspot production validation:
qsub -v EXPECTED_COMMIT="$(git rev-parse HEAD)" \
  torchtitan/experiments/ezpz/rl/scripts/grpo/agpt2b_multihost_torchstore_validate.pbs

# Aurora next-eval uses a different runtime/archive contract. Do not port this
# PBS header or venv path verbatim; follow monarch.md.
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

The SFT deliverable used by path B is
`outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf`.

> **That consolidated export is no longer on disk** (checked 2026-08-30 --
> no `checkpoint-*-hf` remains under that output dir). The sharded
> `checkpoint-900/` is intact (23 GB, 96 `.distcp` shards), so re-run
> `rl/scripts/consolidate_sft_ckpt.sh` before launching anything that
> expects the `-hf` path. See
> [SFT evals](../sft/agpt/2b-mds/tulu_math_uc_mix_full/evals/README.md#recommendation).

## Layout

```
docs/production/rl/
  README.md      <- this hub
  monarch.md <- current production runbook
  trl.md         <- deprecated TRL reproduction path
  monarch.md     <- historical single-host Monarch bring-up/results
  grpo/          <- per-run GRPO results + charts (aurora2b sft_arithmetic, ...)
  2026-07-06_multinode-grpo-root-cause.md   <- TRL multi-node root-cause (evidence)
  history/       <- superseded investigations + bring-up chronology + prior reproductions
```

## Verified results (headline)

- **Current Monarch + TorchStore + vLLM:** two physical Sunspot hosts, trainer
  and generator on distinct hosts, Gloo policy transfer, three finite GRPO
  updates, pre/post generation, policy versions 0→3, checkpoints 1/2/3, and
  clean exit (job `12478711`). Production contract:
  [`monarch.md`](monarch.md).
- **TRL + vLLM-server:** 1 node (job 12468780), cross-node generation (12469976),
  multi-trainer-node 3N/24-rank (12470083) -- all works. Detail + evidence:
  [`trl.md`](trl.md), [`grpo/`](grpo/README.md).
- **Monarch + TorchStore + vLLM:** agpt-2b GRPO+LoRA 2-tile, Train steps fire
  ~1765 tok/s, reward 0.16->0.26 on the easy task (job 12471049) -- runs from our
  repo, zero core edits. Historical detail: [`monarch.md`](monarch.md).

## Historical dependency snapshots

The following venvs reproduce the older TRL and single-host Monarch studies;
they are not the Aurora `next-eval` production runtime:
- `venvs/rl-vllm` (A1/A2): vllm + vllm-xpu-kernels + trl. Build:
  [`rl/scripts/build_rl_vllm_venv.sh`](../../../rl/scripts/build_rl_vllm_venv.sh).
- `venvs/rl-grpo-lora` (B): + torchstore/monarch (from source) + ezpz. Build:
  [`rl/scripts/build_rl_grpo_lora_venv.sh`](../../../rl/scripts/build_rl_grpo_lora_venv.sh).

NEVER `pip install` torch or its deps without `--no-deps` (silently replaces the
XPU torch with a CUDA build).

## Plans

- [Teaching agpt-2b chain-of-thought reasoning](plans/cot.md) -- staged R1-style recipe (cold-start CoT-SFT -> GRPO-RLVR), applies the reward-shaping result
