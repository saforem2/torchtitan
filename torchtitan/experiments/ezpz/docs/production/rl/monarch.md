# GRPO+LoRA on XPU: Monarch + TorchStore + vLLM

**Current status doc** for the UPSTREAM `torchtitan.experiments.rl` RL path
(Monarch actors + TorchStore weight store + vLLM generation), vendored into
**our** repo under `experiments/ezpz/` and verified end-to-end on Sunspot XPU.
For TRL-based GRPO see [`trl.md`](trl.md); for the bring-up chronology and
superseded investigations see [`history/`](history/README.md).

## Status: WORKS, vendored + verified (2026-07-19)

GRPO+LoRA on AuroraGPT-2B (the SFT checkpoint-900 deliverable) runs end-to-end
from **our** repo -- no dependence on external fork checkouts on the invocation --
driving the current-upstream `experiments/rl` engine with **our** `ezpz.agpt`
model, on 2 XPU tiles:

```
Train | Step: 1/2/3   ~1765 tok/s (steady state)
reward_mean ~0.16   nonzero ~53%   max 1.0
```

(job 12471049; matches the fork-based v5 baseline, so the vendored path has the
same learning dynamics.) A 100-step easy-task run shows a rising reward curve
(0.167 -> ~0.26); see [Reward study](#reward-study).

**Design guarantee:** the entire XPU enablement is a THIN overlay under
`experiments/ezpz/` -- **zero edits to `experiments/rl/` (upstream) or
`torchtitan/models/**` (core).** All XPU compatibility is applied as runtime
monkeypatches from `ezpz/rl/xpu_overrides.py` (assert-and-fail-loud if an upstream
target is renamed).

## Architecture

`experiments/rl/` IS current upstream torchtitan (it ships the RL engine:
controller, actors, vLLM wrapper, TorchStore weight-sync). We do NOT copy it.
Instead a thin ezpz overlay drives it:

| Piece | Role |
|---|---|
| `ezpz/rl/train_upstream.py` | Entry point. Applies `apply_all_xpu_patches()` BEFORE importing `rl/`, then delegates to the upstream `Controller` (spawn -> setup_async -> run). |
| `ezpz/rl/xpu_overrides.py` | The XPU patch suite (per-actor via the spawn `_bootstrap`): `init_distributed` PALS injection, XPU attention backend, FSDP2 AVG->SUM, `current_stream` deferral, `create_block_mask` `separate_full_blocks` strip, etc. |
| `ezpz/rl/alphabet_sort_agpt/` | The overlay config module: `rl_grpo_lora_agpt_2b()` + `_easy()`. Backs the upstream engine with `ezpz.agpt.model_registry("2b-rl", converters=[LoRA, LMHeadCast])`. |
| `ezpz/agpt/` | Our AuroraGPT model (`model_registry`, `parallelize`). Gained `converters=`, `skip_dp`, and the `2b-rl` config for RL. |

The upstream engine consumes our `FaultTolerantModelSpec` (a `ModelSpec` subclass)
directly, and `--module` accepts an arbitrary dotted path, so the overlay needs no
registry edits.

## How to run it

Build the venv once, stage the checkpoint, then run from the **repo root**:

```bash
# 1. build venvs/rl-grpo-lora (torchstore/monarch/vllm from source + ezpz + XPU patches)
bash torchtitan/experiments/ezpz/rl/scripts/build_rl_grpo_lora_venv.sh

# 2. stage ckpt-900: symlink weights + inject the gemma chat_template into a COPIED
#    tokenizer (never mutates the SFT deliverable)
bash torchtitan/experiments/ezpz/rl/scripts/grpo/stage_agpt2b.sh

# 3. GRPO on 2 tiles (COMPOSITE), from the repo root
bash torchtitan/experiments/ezpz/rl/scripts/grpo/agpt2b_grpo.sh
#   CONFIG=rl_grpo_lora_agpt_2b        -> stock task, short smoke
#   CONFIG=rl_grpo_lora_agpt_2b_easy   -> 1 turn / <=3 names + lr 2e-5 (the learning config, default)
```

The launcher runs the entry point:

```bash
python -m torchtitan.experiments.ezpz.rl.train_upstream \
    --module torchtitan.experiments.ezpz.rl.alphabet_sort_agpt \
    --config rl_grpo_lora_agpt_2b_easy \
    --hf_assets_path <staged ckpt-900 dir> \
    --async-loop.training-sample-builder.no-drop-zero-std-reward-groups
```

### CRITICAL: run from the repo root

The `rl-grpo-lora` venv's editable `torchtitan` maps to the external build fork.
Running from the **repo root** makes our cwd-local `torchtitan` win, so both
`experiments.ezpz` (overlay + agpt) AND `experiments.rl` (current-upstream engine)
resolve from our repo, while monarch/torchstore/vllm come from the venv. A neutral
cwd picks the fork -> `ModuleNotFoundError: torchtitan.experiments.ezpz`. The
launcher `cd`s to the repo root for this reason.

## Baked-in settings (learned from the runs)

The overlay config hard-codes the settings that make agpt-2b GRPO actually work
and learn on XPU (why, in [history/grpo-lora-agpt2b-repro.md](history/grpo-lora-agpt2b-repro.md)):

| Setting | Value | Why |
|---|---|---|
| `generator.model_dtype` | `float32` | **THE coherence fix.** bf16 corrupts agpt-2b generation through vLLM (large vocab/ffn -> rounding flips greedy argmax); fp32 is coherent. |
| reward `similarity_power` | `1` (linear) | stock `**4` starves GRPO of gradient; linear gives partial-credit variance that trains. |
| few-shot format env | one-shot | model rarely emits `<alphabetical_sorted>` unprompted; one-shot (bare lines, distinct example) raises the hit rate for dense signal. |
| attention | `flex` | RL generator asserts varlen|flex; varlen lacks an XPU flash kernel; flex + `max_autotune=False` works (matches the proven runs). |

## Reward study (task difficulty is the lever, not LR)

A controlled 3-way GRPO+LoRA comparison (all fp32 + few-shot + linear reward):

| Run | Change | Reward trend | Verdict |
|---|---|---|---|
| default task, lr 2e-5 | -- | flat ~0.13 (format 50->82%, no reward gain) | null |
| **easy task (1 turn, <=3 names)**, lr 2e-5 | task difficulty | **0.17 -> 0.26 rising** | **learns** |
| default task, lr 5e-5 | higher LR | flat/down ~0.13 | null |

GRPO improves the policy when the task is in the model's reach; the null default-task
run (format hit 82% with zero reward gain) shows format-optimization is a dead end --
only sort-quality moves reward. Live dashboards (kitcat + ambivalent):
`rl/scripts/grpo_lora_agpt2b_patches/rl_dash3.py`.

## The XPU integration bugs (all fixed as runtime patches, not core edits)

Found + fixed while vendoring; each is a patch in `xpu_overrides.py` or a config
choice, keeping `experiments/rl/` and core untouched:

1. `torch.accelerator.get_memory_info` returns free=0 on XPU (vLLM refused to
   start). Upstream vLLM's `get_mem_info_wrapper` already fixes it; a documented
   fallback patch (`patch_vllm_xpu_mem.py`) covers kernel builds where it doesn't.
2. RL generator asserts varlen|flex attention -> `2b-rl` uses flex.
3. FlexAttention `max_autotune` -> XPU OUT_OF_RESOURCES -> overlay disables it.
4. Generation `torch.cuda.current_stream()` -> "not compiled with CUDA": stale
   ezpz patch stripped it; current vLLM wraps it dynamo-safe in a partial, so the
   patch now DEFERS to upstream.
5. Trainer `create_block_mask(separate_full_blocks=...)` TypeError (torch XPU
   wheel lacks the kwarg; core passes it unconditionally) -> wrap the module-global
   `_compiled_create_block_mask` to strip it.

## Dependencies

External venv (`venvs/rl-grpo-lora`, py3.12; built by the script above):
torch 2.12+xpu, triton-xpu 3.7.1, vllm (from source) + vllm-xpu-kernels 0.1.10,
torchstore + monarch (songhappy forks @ xpu-upstream, from source),
saforem2/ezpz (`--no-deps`). Clones live outside the repo in `~/rl-repro/`.

## References

- **Full vendoring writeup + the bug-by-bug repro:** [history/grpo-lora-agpt2b-repro.md](history/grpo-lora-agpt2b-repro.md)
- **The original Qwen3-0.6B reproduction (the USM-wall crossing):** [history/grpo-lora-xpu-repro.md](history/grpo-lora-xpu-repro.md)
- **Why the earlier Monarch port stalled (superseded):** [history/upstream-rl-port-status.md](history/upstream-rl-port-status.md), [history/2026-06-14_monarch-torch213-deep-dive.md](history/2026-06-14_monarch-torch213-deep-dive.md)
