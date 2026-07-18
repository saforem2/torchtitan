# GRPO+LoRA on Intel XPU: Sunspot reproduction (Monarch + TorchStore + vLLM)

**Goal:** reproduce songhappy/torchtitan@rl `GRPO_LORA_XPU.md` on Sunspot -- the
UPSTREAM `torchtitan.experiments.rl` path (Monarch actors + TorchStore weight
store + vLLM generation), the architecture we ABANDONED for TRL vllm-serve
because of the USM/PMIx trainer-side wall (see
[history/upstream-rl-port-status.md](history/upstream-rl-port-status.md)). The
fork claims the full path works on Borealis; this tests whether it crosses our
wall on Sunspot.

**Status:** Stage 0-2 PASSED (2026-07-18). venv builds + all imports clean.
Stage 3 (component smokes incl. the Monarch-spawned XCCL allreduce = the USM
wall) is next -- the actual experiment.

## Build (Stages 0-2) -- DONE

Script: [`rl/scripts/build_rl_grpo_lora_venv.sh`](../../rl/scripts/build_rl_grpo_lora_venv.sh)
-> `venvs/rl-grpo-lora/` (py3.12). Clones in `~/rl-repro/` (outside repo).

**Stack (built + imports OK on the UAN):**

| Component | Version / source | Notes |
|---|---|---|
| python | 3.12.12 (uv) | |
| torch | 2.12.0+xpu | + torchaudio 2.11, torchvision 0.27 |
| triton-xpu | 3.7.1 | intel symbols OK |
| torchstore | editable, songhappy fork @ `xpu-upstream` (`03f588e`) | plain setuptools build |
| torchmonarch | 0.6.0.dev0, editable, songhappy fork @ `xpu-upstream` (`7d23347e`) | Rust/setuptools-rust build (gcc-14 + cargo 1.94 + protoc 30.2) |
| vllm | 0.23.1rc1 editable, vllm-project @ `main` | from-source C++ compile with gcc-14 |
| vllm-xpu-kernels | 0.1.10 | GitHub release wheel (matches torch 2.12) |
| torchtitan (fork) | 0.2.2, editable, songhappy fork @ `rl` | has GRPO_LORA_XPU.md + rl_grpo_lora config |
| transformers / datasets | 5.9.0 / 4.7.0 | recipe pins |

**Critical XPU guards (from `build_rl_vllm_venv.sh`, preserved):**
- `impi-rt` / `oneccl` / `oneccl-devel` uninstalled from venv -> confirmed
  absent (else in-venv libccl shadows the system oneCCL that knows Sunspot's USM
  allocator -> the XCCL "invalid usm pointer" wall).
- vanilla `triton` (pulled by xgrammar) uninstalled, triton-xpu reinstalled.

## Borealis -> Sunspot port: bugs fixed (all committed)

The recipe was written for Borealis + conda + gcc-13.3. Porting to Sunspot +
uv + gcc-14 surfaced 7 fixes, each committed to the build script:

1. **clone URLs** -- the `xpu-upstream` branches for torchstore (#171) + monarch
   (#4307) live on the **songhappy forks**, not the meta-pytorch base repos.
2. **force gcc-14** -- the login profile exports `CC=icx/CXX=icpx`; a
   `${CC:-gcc-14}` default respected it. Forced gcc-14 unconditionally (recipe
   used gcc; avoid clang/gcc ABI surprises). gcc-14.2.0 (Sunspot has no gcc-13).
3. **monarch path** -- Python project is at the repo ROOT (Rust project), not
   `python/` as the recipe said; `-e monarch` + cargo on PATH.
4. **`uv venv --clear`** -- idempotent re-runs (set -e + leftover venv killed re-launch).
5. **setuptools-rust** -- `--no-build-isolation` needs monarch's build backend present.
6. **protoc** -- monarch's `tracing-perfetto-sdk-schema` Rust crate needs protoc
   at build; Sunspot bare shell has none. Added `protoc-wheel-0` (libprotoc 30.2).
7. **setuptools_scm** -- vLLM's build backend, same `--no-build-isolation` class.

## Known-benign warnings at import (on the UAN, no XPU)
- `torch ... xpu False 0`, `vllm._C not found`, `Triton 0 active drivers` -- all
  because the import gate ran on the login node (no GPU). Re-verify on a compute
  node.
- `monarch._rust_bindings.rdma not available` -- expected; Borealis/Sunspot use
  shared-memory transport, not RDMA (recipe confirms).

## Stage 3 (next) -- component smokes on a compute node
- TorchStore roundtrip (push/pull state dict).
- Monarch 2-actor mesh sees XPU.
- bare vLLM generation on Qwen3-0.6B (cached at
  `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B`).
- **The Monarch-spawned XCCL allreduce -- our historical USM/PMIx wall.** If it
  fails here the same way, the fork's fix does not cross it on Sunspot -> STOP +
  document (a VALID outcome). If it passes, proceed to Stage 4 (the GRPO+LoRA run).

## Stage 4 (final) -- the run
`python3 -m torchtitan.experiments.rl.train --module alphabet_sort
--config rl_grpo_lora_qwen3_0_6b --hf_assets_path=<qwen3>` with the recipe env
(`ZE_AFFINITY_MASK=0,1,2,3`, `FI_PROVIDER=tcp`, `CCL_ATL_OFI_PROVIDER=tcp`,
`TORCHINDUCTOR_MAX_AUTOTUNE=0`, `VLLM_ENABLE_V1_MULTIPROCESSING=1`, HF offline).
Success = rising reward (their milestone `mean_r~0.6`) + gen throughput near
their ~53 tok/s (DP=2) / ~4,140 tok/s figures.
