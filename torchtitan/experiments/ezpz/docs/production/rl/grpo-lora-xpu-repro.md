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

## Stage 3 -- component smokes on a compute node (2026-07-18, node x1922c6s0b0n0)

Ran on a 1N debug alloc. Runtime env fix required (bug #8): the nested
compute-node shell drops the module env, so `libccl.so.1` was not found ->
torch import failed. Fixed by explicitly wiring
`CCL_ROOT=/opt/aurora/26.26.0/oneapi/ccl/latest` +
`LD_LIBRARY_PATH=$CCL_ROOT/lib:/opt/aurora/26.26.0/oneapi/2025.3/lib:...` (the
system oneCCL the in-venv-oneccl-uninstall relies on).

| Smoke | Result |
|---|---|
| S3.1 imports + XPU visible | **PASS** -- `torch 2.12.0+xpu xpu True 4` on the node (the UAN `vllm._C` / `Triton 0 drivers` warnings were just no-GPU-on-login, as expected; on-device they resolve). |
| S3.2 torchstore roundtrip (`torchstore_rl.py`) | **PASS** -- a Monarch `Generator` actor pulled a state_dict via the CPU-staged shared-memory transport; weights materialized `device='xpu:0'`. |
| S3.3 monarch 2-actor XPU mesh (`monarch_smoke.py`) | **PASS** -- `VERDICT: monarch smoke passed` (spawn + Gloo put/get roundtrip). |
| S3.4 **xccl-XPU transport** (`test_xccl_xpu.py::test_xccl_put_get`) | **USM WALL CROSSED (with a teardown caveat)** -- see below. |

### S3.4 -- the USM/PMIx wall test (the whole point)

`test_xccl_put_get` spawns Monarch Writer+Reader actors and moves an
`device="xpu"` tensor over the **xccl** transport (`TransportType.XCCL` was
selected, not rejected). The historical wall was
`ccl_check_usm_pointers: invalid usm pointer type: unknown` at the first XCCL
collective from a Monarch-fork-spawned rank.

**That error did NOT occur.** The put/get completed and BOTH assertions ran
before the failure point:
```
src = await writer.put.call(key)       # xpu tensor -> xccl
got = await reader.get.call(key)       # xccl -> xpu tensor
assert "xpu" in got["device"]          # (passed -- above the failing line)
assert abs(got["checksum"] - src["checksum"]) < 1e-3   # (passed)
await controller.teardown.call()       # <-- line 82: the ONLY failure
```
The failure is a Monarch **`SupervisionError` during `controller.teardown()`**
(`torchstore/controller.py:245`, `Endpoint call StorageVolumes_....res...`) --
a shutdown/supervision-ordering crash in cleanup, NOT the USM collective.

**Interpretation:** songhappy's fork (the `xccl.py` transport + the
`shared_memory.py`/`torchcomms` USM no-op patches) **crosses our USM/PMIx wall
on Sunspot** -- the xccl xpu-tensor transfer between Monarch-spawned actors
works. This is the answer the effort was chasing: the upstream
Monarch+TorchStore RL path is viable on Sunspot at the transport level.

**Caveat / open item:** the actor-teardown `SupervisionError` is a real (if
softer) issue -- needs a look before it can be called production-clean (it may
be benign at process exit, or a genuine shutdown-ordering bug). Does NOT block
Stage 4 (a full run does its own lifecycle), but should be understood.

### Bugs 8 (this stage)
8. **libccl.so.1 not found on compute node** -- nested non-login shell drops the
   module env; wire CCL_ROOT + LD_LIBRARY_PATH explicitly (system oneCCL at
   /opt/aurora/26.26.0/oneapi/{ccl/latest,2025.3}/lib).

Next: **Stage 4** -- the actual GRPO+LoRA run
(`--module alphabet_sort --config rl_grpo_lora_qwen3_0_6b`), which exercises the
full generate->score->train->weight-sync loop end to end.

## Stage 4 -- full GRPO+LoRA run (2026-07-19): BLOCKED at trainer-actor CCL init

Ran the recipe entry point on a compute node (5-step smoke first, not the full
200): `python -m torchtitan.experiments.rl.train --module alphabet_sort
--config rl_grpo_lora_qwen3_0_6b`. Got materially further each attempt; NOT
blocked by the USM wall (that stayed crossed). Two fixes landed, one blocker
remains.

### Fixed (bugs 9-10)
9. **`ModuleNotFoundError: renderers`** -- fork rl/rollout imports `renderers`
   (PrimeIntellect). Added `renderers @ git` to the build (Step 5).
10. **cwd shadowing** -- the launcher `cd`-ed into our repo, so our repo's
    `torchtitan/` won on `sys.path` and shadowed the editable FORK torchtitan
    (which has alphabet_sort + LoRA). Fix: run from a neutral cwd
    (`~/rl-repro/run`). Confirmed: from a neutral dir, `import torchtitan` ->
    `~/rl-repro/torchtitan-fork/...`.

After those, the pipeline reached: controller init, actor layout
(`1 generator x 2 GPUs + 2 trainer GPUs = 4 total`), TorchStore strategy init.

### BLOCKER: trainer SetupActor sig-11 on oneCCL init (Monarch-spawned, no KVS)

The FSDP trainer `SetupActor` dies with `Killed(sig=11)` during oneCCL init.
Root cause (from the CCL_WARN trail):
```
CCL_ATL_TRANSPORT changed to be ofi (default:mpi)      # our export applied
CCL_PROCESS_LAUNCHER changed to be none                 # applied
could not get local_idx/count from environment variables
OFI transport was not initialized, fallback to MPI transport
Killed(sig=11)
```
Even with `CCL_ATL_TRANSPORT=ofi`, OFI can't initialize -> MPI fallback ->
the documented `atl_mpi::create_comm_id` segfault (see grpo-on-xpu-status.md).

**Why:** architecture mismatch. The WORKING TRL-vllm-serve xnode GRPO script
spawns trainer ranks via `ezpz launch` (mpiexec) WITH a shared
`CCL_KVS_IP_PORT` rendezvous -- that provides `local_idx/count` + the KVS OFI
needs. The GRPO+LoRA recipe spawns the trainer via **Monarch actors** (not
mpiexec), so no mpiexec-provided rank env / KVS reaches the actor -> OFI init
fails. The recipe's Borealis env (`env-3.sh` + Monarch bootstrap) evidently
wired this; porting the CCL/KVS rendezvous through Monarch's actor bootstrap on
Sunspot is the remaining work.

### Next-session plan (bounded)
- Wire `CCL_KVS_IP_PORT` (+ `CCL_LOCAL_RANK`/`CCL_LOCAL_SIZE` or the
  `local_idx/count` CCL expects) into the Monarch trainer-actor bootstrap
  (proc_mesh SetupActor env), not just the shell. Check how songhappy's monarch
  `proc_mesh.py` XPU patch sets per-actor CCL env.
- Alternative: check whether the recipe expects `CCL_ATL_TRANSPORT=mpi` to
  actually WORK on Borealis (i.e. the mpi-transport segfault is Sunspot-specific
  and the fix is making OFI init succeed, not avoiding mpi).
- Then re-run the 5-step smoke; success = rising reward on alphabet_sort.

### Standing verdict (unchanged by this blocker)
The core question is ANSWERED: **the USM wall is crossed on Sunspot** (Stage 3).
The Stage-4 blocker is a CCL-transport/KVS-rendezvous init issue in the
Monarch-spawned trainer, NOT the USM pointer wall -- a different, more tractable
problem. Build (Stages 0-2) + component smokes (Stage 3) reproduce; the full
end-to-end loop (Stage 4) is one CCL-init fix away.
