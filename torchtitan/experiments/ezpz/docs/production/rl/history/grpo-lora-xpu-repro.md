# GRPO+LoRA on Intel XPU: Sunspot reproduction (Monarch + TorchStore + vLLM)

**Goal:** reproduce songhappy/torchtitan@rl `GRPO_LORA_XPU.md` on Sunspot -- the
UPSTREAM `torchtitan.experiments.rl` path (Monarch actors + TorchStore weight
store + vLLM generation), the architecture we ABANDONED for TRL vllm-serve
because of the USM/PMIx trainer-side wall (see
[history/upstream-rl-port-status.md](history/upstream-rl-port-status.md)). The
fork claimed the full path works on Borealis; the question was whether it crosses
our wall on Sunspot.

## STATUS: REPRODUCED (2026-07-19)

The full GRPO+LoRA async RL loop runs end-to-end on Sunspot XPU with real
gradient steps (job 12471019, 2-tile COMPOSITE, Qwen3-0.6B, alphabet_sort):

```
Train | Step: 1   tokens_per_second_full_step:   85.6   (first-step torch.compile warmup)
Train | Step: 2   tokens_per_second_full_step: 3256.5
Train | Step: 3   tokens_per_second_full_step: 3272.9
RUN3 EXIT rc=0
```

Per `GRPO_LORA_XPU.md`, "success = if you see step metrics." We see them. The
per-step perf breakdown confirms the whole async machinery: `fwd_bwd` ratio,
near-zero `blocking_generator_pull_model_state_dict` /
`blocking_trainer_push_model_state_dict` (the TorchStore weight sync), generator
inflight requests. `reward/_mean = 0` across the 3 steps is EXPECTED -- untrained
Qwen3-0.6B cannot do alphabet-sort yet and 3 steps is far short of the recipe's
`mean_r ~ 0.6` many-step milestone. The reproduction target was the *machinery*,
and it runs at ~3,270 tok/s.

**The USM/PMIx wall that made us abandon this path is crossed.** Every layer
works: build, xccl collectives, per-actor XPU isolation, vLLM generation,
TorchStore weight sync, GRPO+LoRA training steps.

### Final scorecard -- all green

| Piece | Status |
|-------|--------|
| Build (torchstore / monarch / vllm / fork, all from source) | REPRODUCED |
| USM wall (xccl xpu-tensor transfer between Monarch actors) | CROSSED |
| xccl collectives (libfabric fix) | WORKS |
| Per-actor XPU isolation (COMPOSITE topology, disjoint masks) | WORKS |
| vLLM generator init + KV cache (mem_utils fix) | WORKS |
| TorchStore weight sync round-trip | WORKS |
| Rollout generation + scoring + grouping | WORKS |
| **GRPO+LoRA train steps fire (~3,270 tok/s)** | **REPRODUCED** |

## How to run it (2-tile smoke)

Build once with
[`rl/scripts/build_rl_grpo_lora_venv.sh`](../../rl/scripts/build_rl_grpo_lora_venv.sh)
-> `venvs/rl-grpo-lora/` (py3.12); clones land in `~/rl-repro/` (outside the
repo). Then, on a compute node, run from a **NEUTRAL cwd** (`~/rl-repro/run`) so
the editable fork wins on `sys.path` over the main-repo `experiments/rl` copy:

```bash
# module env
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export CCL_ROOT=/opt/aurora/26.26.0/oneapi/ccl/latest
# libfabric MUST be on LD_LIBRARY_PATH (else oneCCL OFI init fails -> MPI
# fallback -> atl_mpi segfault; see root cause "bug #11" below)
export LD_LIBRARY_PATH=/opt/cray/libfabric/2.3.1/lib64:/opt/aurora/26.26.0/oneapi/2025.3/opt/mpi/libfabric/lib:$CCL_ROOT/lib:/opt/aurora/26.26.0/oneapi/2025.3/lib:$LD_LIBRARY_PATH
# COMPOSITE: node shows 6 whole GPUs; mask 0,1 -> two physically distinct GPUs so
# the per-actor bootstrap masks (trainer->0, generator->1) never collide.
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu" ZE_AFFINITY_MASK=0,1
export FI_PROVIDER=tcp CCL_ATL_TRANSPORT=ofi CCL_ATL_OFI_PROVIDER=tcp
export TORCHINDUCTOR_MAX_AUTOTUNE=0 VLLM_ENABLE_V1_MULTIPROCESSING=1
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

cd ~/rl-repro/run
PY=/lus/tegu/.../torchtitan/venvs/rl-grpo-lora/bin/python
QWEN=~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<snap>
$PY -u -m torchtitan.experiments.rl.train \
    --module alphabet_sort --config rl_grpo_lora_qwen3_0_6b \
    --hf_assets_path="$QWEN" \
    --async-loop.num-training-steps=3 \
    --async-loop.num-groups-per-train-step=4 \
    --async-loop.training-sample-builder.no-drop-zero-std-reward-groups \
    --trainer.parallelism.data-parallel-shard-degree=1 \
    --generator.parallelism.data-parallel-degree=1 \
    --generator.parallelism.tensor-parallel-degree=1 \
    --generator.gpu-memory-limit=0.70 \
    --generator.sampling.max-tokens=700
```

### Two config overrides are required for a step to fire

Both are properties of the recipe's own `rl_grpo_lora_qwen3_0_6b` config, and
NEITHER is XPU-specific:

1. `--generator.sampling.max-tokens=700` -- the config ships `100`, which
   truncates ~99.7% of completions before they close the `<alphabetical_sorted>`
   block (`RewardAlphabetSort` -> 0). Cosmetic here (the untrained base model
   scores 0 either way) but needed so completions finish rather than pile up as
   `truncated_length`.
2. `--async-loop.training-sample-builder.no-drop-zero-std-reward-groups` -- the
   `TrainingSampleBuilder` defaults `drop_zero_std_reward_groups=True`, and the
   LoRA config never overrides it (its sibling `*_varlen` configs set it False).
   With the default, every all-zero-reward group (i.e. every group at cold start)
   is dropped -> no batch is ever assembled -> no train step. **This is the one
   that actually blocked the step**; max-tokens alone (run2) never fired one.
   `--async-loop.num-groups-per-train-step=4` (from 8) just makes a step assemble
   sooner on a single tile.

**tyro gotcha:** booleans are flag pairs (`--...no-drop-zero-std-reward-groups`),
NOT `--...drop-zero-std-reward-groups=False` (that gives `Unrecognized options:
False`); flag names use dashes.

### Two experiment-fork code patches are required

Both live in `~/rl-repro/` (outside the repo), applied to the from-source builds:

1. **`vllm/utils/mem_utils.py`, `MemorySnapshot.measure()`** -- on XPU, source
   free/total from `torch.xpu.mem_get_info(device.index)` instead of
   `torch.accelerator.get_memory_info(device)`. The generic accelerator API is
   broken on Intel XPU (returns `free=0`); vLLM's memory check then refuses to
   start. This was the real "bug #12" (see root causes). One `if device.type ==
   'xpu'` branch.
2. **`libccl.so.1` runtime wiring** -- handled by the `CCL_ROOT` +
   `LD_LIBRARY_PATH` exports above (the nested compute-node shell drops the
   module env; the in-venv `oneccl` uninstall means we depend on the system
   oneCCL being on the path).

Diagnostic scaffolding used during the hunt (SHOULD be stripped for a clean
fork): an `EZPZMEM` warning in `vllm/v1/worker/xpu_worker.py` and an
`actor_mask.log` block in the fork's `train.py` `_bootstrap`. `ZES_ENABLE_SYSMAN=1`
in the launcher is harmless but NOT required (the `mem_get_info` API reads free
correctly without it -- verified).

## Build stack

Script produces `venvs/rl-grpo-lora/` (py3.12). From-source EDITABLE installs
(not wheels) for the four repos:

| Component | Version / source | Notes |
|---|---|---|
| python | 3.12.12 (uv) | |
| torch | 2.12.0+xpu | + torchaudio 2.11, torchvision 0.27 |
| triton-xpu | 3.7.1 | intel symbols OK |
| torchstore | editable, songhappy fork @ `xpu-upstream` (`03f588e`) | plain setuptools build |
| torchmonarch | 0.6.0.dev0, editable, songhappy fork @ `xpu-upstream` (`7d23347e`) | Rust/setuptools-rust (gcc-14 + cargo 1.94 + protoc 30.2) |
| vllm | 0.23.1rc1 editable, vllm-project @ `main` | from-source C++ compile with gcc-14 |
| vllm-xpu-kernels | 0.1.10 | GitHub release wheel (matches torch 2.12) |
| torchtitan (fork) | 0.2.2, editable, songhappy fork @ `rl` | has GRPO_LORA_XPU.md + rl_grpo_lora config |
| transformers / datasets | 5.9.0 / 4.7.0 | recipe pins |

**Critical XPU guards (preserved from `build_rl_vllm_venv.sh`):**
- `impi-rt` / `oneccl` / `oneccl-devel` uninstalled from the venv (else in-venv
  libccl shadows the system oneCCL that knows Sunspot's USM allocator -> the
  historical XCCL "invalid usm pointer" wall).
- vanilla `triton` (pulled by xgrammar) uninstalled, triton-xpu reinstalled.

## Root causes (the bugs that mattered)

### USM/PMIx wall -- CROSSED (Stage 3, 2026-07-18, node x1922c6s0b0n0)

`test_xccl_xpu.py::test_xccl_put_get` spawns Monarch Writer+Reader actors and
moves a `device="xpu"` tensor over the **xccl** transport. The historical wall
was `ccl_check_usm_pointers: invalid usm pointer type: unknown` at the first XCCL
collective from a Monarch-fork-spawned rank. **That error did not occur** -- the
put/get completed and both the device (`"xpu" in got["device"]`) and checksum
asserts passed. The only failure was a Monarch `SupervisionError` during
`controller.teardown()` (a shutdown-ordering issue in cleanup, not the
collective). songhappy's fork (the `xccl.py` transport + `shared_memory.py` /
`torchcomms` USM no-op patches) crosses the wall on Sunspot at the transport
level. (Component smokes S3.1 imports/XPU-visible, S3.2 torchstore roundtrip, S3.3
monarch 2-actor mesh all PASSED alongside it.)

### Bug #11 -- xccl SIGSEGV = libfabric.so not on LD_LIBRARY_PATH (FIXED)

Root-caused with a minimal torchrun 2-rank xccl allreduce probe +
`CCL_LOG_LEVEL=info`: even with `CCL_ATL_TRANSPORT=ofi`, OFI could not load
`libfabric.so` -> fell back to MPI transport -> `atl_mpi::create_comm_id`
SIGSEGV. NOT Monarch-specific (a plain torchrun hit it too), NOT the USM wall.
Fix: add `/opt/cray/libfabric/2.3.1/lib64` (+
`/opt/aurora/26.26.0/oneapi/2025.3/opt/mpi/libfabric/lib`) to `LD_LIBRARY_PATH`.
Verified: `torchrun --nproc_per_node=2 ... all_reduce -> OK`, rc=0.

### Bug #12 -- torch.accelerator.get_memory_info returns free=0 on XPU (FIXED)

The generator's vLLM `init_device` failed with `ValueError: Free memory on device
xpu:0 (0.0/121.57 GiB) ... less than desired GPU memory utilization`. Instrumented
at the exact failing site (`vllm/utils/mem_utils.py` `MemorySnapshot.measure()`),
same process / same instant / generator actor on mask=1:

```
EZPZMEM dev=xpu:0 ZE_AFFINITY_MASK=1 ZE_FLAT=COMPOSITE
  mem_get_info          = (free=127.51, total=127.97)   <- CORRECT
  accel_get_memory_info = (free=0.00,   total=121.57)   <- BROKEN
```

`torch.accelerator.get_memory_info()` is broken on Intel XPU (returns free=0,
total=sum-of-tiles); `torch.xpu.mem_get_info()` reads correctly. vLLM's
`MemorySnapshot` uses the generic accelerator API. Fix: XPU branch in
`mem_utils.py` using `torch.xpu.mem_get_info` (see recipe above). This took three
wrong theories to reach -- see the superseded chronology; the short version is
that the per-actor masks were always disjoint (an actor-mask log proved it) and a
standalone probe showed free reads fine, so the only difference had to be the API
vLLM chose.

### Borealis -> Sunspot port bugs 1-8 (build-time, all fixed in the build script)

1. clone URLs = songhappy forks (torchstore #171 / monarch #4307 `xpu-upstream`),
   not the meta-pytorch base repos.
2. force gcc-14 unconditionally (login profile exports CC=icx/icpx; Sunspot has
   no gcc-13). gcc-14.2.0.
3. monarch Python project is at the repo ROOT (Rust project), not `python/`;
   `-e monarch` + `~/.cargo/bin` on PATH.
4. `uv venv --clear` for idempotent re-runs.
5. `setuptools-rust` build dep (monarch, `--no-build-isolation`).
6. `protoc-wheel-0` (monarch's `tracing-perfetto-sdk-schema` crate; libprotoc 30.2).
7. `setuptools_scm` (vLLM build backend, same `--no-build-isolation` class).
8. `libccl.so.1` not found on compute node -- nested non-login shell drops the
   module env; wire `CCL_ROOT` + `LD_LIBRARY_PATH` explicitly.
   (renderers dep + neutral-cwd sys.path shadowing also resolved.)

---

## Appendix: superseded theories (chronology, kept for the learning)

These sections are HISTORICAL. Each represents a hypothesis that was later
disproven; none reflects the current status (see STATUS: REPRODUCED at the top).
The value here is the debugging path, and the self-inflicted harness mistakes
worth not repeating.

- **"BLOCKED at trainer-actor CCL init" (early 2026-07-19).** First Stage-4
  theory: the trainer `SetupActor` sig-11 on oneCCL init was blamed on
  Monarch-spawn lacking an mpiexec KVS rendezvous. WRONG -- it was bug #11
  (libfabric missing from `LD_LIBRARY_PATH` -> OFI init fail -> MPI-fallback
  segfault). Fixed and xccl verified with a plain torchrun probe.

- **"generator/trainer tile-placement collision" (2026-07-19 AM).** Second
  theory: under `ZE_FLAT_DEVICE_HIERARCHY=FLAT` the node shows 12 tiles and both
  actors were thought to land on physical `xpu:0`. Switching to `COMPOSITE` (6
  whole GPUs) was the right move for a clean topology, but the collision theory
  itself was WRONG -- the per-actor `actor_mask.log` proved the masks were always
  disjoint (trainer `ZE_AFFINITY_MASK=0`, generator `=1`).

- **"ZES_ENABLE_SYSMAN" theory (2026-07-19).** Third theory: the `free=0` read
  was blamed on Intel Sysman being off. Setting `ZES_ENABLE_SYSMAN=1` in the
  launcher, then in the per-actor bootstrap, did NOT fix it. A dedicated mem-API
  probe (2 APIs x parent/spawn x ZES on/off) showed free reads correctly WITH AND
  WITHOUT ZES -- disproving it and pointing at the API itself (bug #12).

- **Self-inflicted harness mistakes (worth not repeating).** Several early
  "Stage 4 failures" were not the code: (a) a double-launch collision (relaunching
  on a node with a still-running instance -> `stop called twice` panic); (b)
  over-aggressive `grep`/`tail` log filtering that hid the real error; (c) `set
  -e` + `module load` (Lmod exits nonzero) killing the launcher before any output
  -- the "no `set -euo` in PBS scripts" rule; (d) `nohup` through nested ssh not
  surviving -> use `setsid ... </dev/null & disown`. Clearing all of these was
  what finally produced clean, interpretable runs.

- **"PIPELINE GREEN but no train step" (2026-07-19).** After the mem_utils fix the
  full loop ran (vLLM gen ~3000 tok/s, TorchStore put 1.78 / get 11.21 GB/s,
  validation) but no step fired. Diagnosed from `rollout_samples.jsonl`:
  7,580/7,602 = `truncated_length` at `max_tokens=100`, and (the real gate)
  `drop_zero_std_reward_groups=True` dropping every zero-variance cold-start
  group. Both lifted -> steps fire (STATUS above).
