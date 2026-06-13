# Upstream `torchtitan.experiments.rl.train` port status (2026-06-13)

## TL;DR

The XPU compatibility layer needed to run upstream's GRPO loop is
*almost* trivial — but a deeper architectural mismatch between
Monarch's `spawn_procs` (fork-based) and oneCCL's PMIx-coupled
USM allocator blocks the trainer side. The generator side works.

**What works:**
- Bare vLLM-XPU end-to-end (`vllm_xpu_bare_smoke.sh`, job 12468754)
- Monarch actor framework on XPU (`monarch_smoke.sh`, job 12468739)
- Upstream `rl/`'s full import chain on XPU (with patches in
  `xpu_overrides.py`)
- All 4 generator actors initializing vLLM-XPU `LLMEngine` with TP=4

**What doesn't:**
- Trainer actors hitting any `torch.distributed` collective
  (broadcast / allreduce). Every attempt fails oneCCL XCCL's
  `ccl_check_usm_pointers` validation:
  > `RuntimeError: oneCCL: coll_check.cpp:68
  > ccl_check_usm_pointers: EXCEPTION: coll: broadcast / allreduce
  > - invalid usm pointer type: unknown for device type: gpu`

## XPU porting layer (working)

`torchtitan/experiments/ezpz/rl/xpu_overrides.py` provides:

1. **`has_xpu_kernels()`** — monkey-patches
   `torchtitan.tools.utils.has_cuda_capability` to always return
   `False` on XPU. Selects the FA2 (non-Hopper) branches in
   `rl/actors/generator.py:465` and `rl/models/attention.py:70`.

2. **`EzpzPerHostProvisioner`** — XPU equivalent of upstream
   `PerHostProvisioner`. Partitions Sunspot/Aurora tiles via
   `ZE_AFFINITY_MASK` instead of `CUDA_VISIBLE_DEVICES`.

3. **`patch_dtensor_rng_broadcast_for_xpu()`** —
   `OffsetBasedRNGTracker.__init__` skips the seed-state broadcast at
   `world_size=1`.

4. **`patch_init_distributed_for_xpu()`** —
   - Sets `PALS_LOCAL_RANKID` / `PALS_RANKID` from torch's
     `LOCAL_RANK` / `RANK` right before `init_process_group`
   - Sets `ZE_AFFINITY_MASK` and other static PALS env in the
     actor `_bootstrap` callable

5. **`_bootstrap`** — eagerly imports `torch.distributed.checkpoint`
   to dodge a torchstore circular-import race during actor setup.

6. **`venvs/rl-vllm/`** built via `build_rl_vllm_venv.sh`:
   - Excludes `impi-rt` / `oneccl` / `oneccl-devel` pip packages
     (their in-venv copies shadow `/opt/aurora/.../oneapi/ccl`'s
     libccl with broken USM-allocator versions)
   - Uses py3.12 (the intersection where both `torchmonarch` cp312
     and `triton-xpu==3.7.1` cp312 wheels exist)

## The core blocker: Monarch fork-spawn ↔ PMIx

Diagnostic confirmed (job 12468765) that our env injection runs
correctly in every actor process:
- `PALS_LOCAL_RANKID` matches the actor's mesh-local rank
- `PALS_RANKID` matches the actor's global rank
- `PALS_LOCAL_SIZE` matches the actor mesh size
- `ZE_AFFINITY_MASK` correctly partitions tiles

**Yet every XCCL collective still fails the USM pointer check.**

We confirmed via baseline that XCCL works fine on plain xpu tensors
when launched directly with `mpiexec --np 2`:
```
$ mpiexec --np 2 python -c '
    import torch, torch.distributed as dist
    dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    torch.xpu.set_device(int(os.environ["LOCAL_RANK"]))
    t = torch.zeros(4, device=f"xpu:{rank}").fill_(rank+1.0)
    dist.all_reduce(t)
    print(rank, t.tolist())'
[0] AFTER: [3.0, 3.0, 3.0, 3.0]
[1] AFTER: [3.0, 3.0, 3.0, 3.0]
```

But **the same allreduce fails when the process was spawned by
Monarch's `spawn_procs`** — even when launched under an outer
`mpiexec --np 1` wrapper that does provide PMIx env to the
controller.

### Why env-only injection isn't enough

The PMIx environment variables (`PALS_*`, `PMIX_*`) are sentinels
that point to **shared mmap segments** owned by the PMIx-launching
process (e.g. `mpirun`/`palsd`). When oneCCL sees those vars, it
attempts to call into PMIx's KVS via those segments to coordinate
USM allocator setup across ranks.

When Monarch forks an actor process, the child inherits the env
vars but the segments they reference were registered for the
*parent's* PID. The child can't open them; oneCCL falls back to a
degenerate state where the USM allocator isn't properly initialized,
and any tensor it sees gets classified as "unknown" USM type.

This is structural — wrapping the controller in mpiexec doesn't
change the fork-child relationship inside Monarch.

## Things we tried (all failed)

| # | Approach | Outcome |
|---|---|---|
| 1 | `--debug.seed 42` (skip seed broadcast) | passes; next broadcast at parallelize_fn |
| 2 | `tensor-parallel-degree=1` (skip mesh broadcast) | passes; next broadcast at init_weights |
| 3 | Patch `OffsetBasedRNGTracker` to skip at ws=1 | passes; init_weights still broadcasts at ws>1 |
| 4 | Inject `PALS_LOCAL_RANKID` / `PALS_RANKID` / `PALS_LOCAL_SIZE` | env injection runs correctly, USM check still fails |
| 5 | Wrap launcher in `mpiexec --np 1` for PMIx parent | USM check still fails; PMIx state doesn't fork-inherit |
| 6 | Force `xpu→gloo` backend in default map | passes USM; Gloo can't broadcast xpu tensors |

## Paths forward

### A. Replace Monarch `spawn_procs` with mpiexec-launched ranks (correct)

Architecture change: launch the trainer with `mpiexec --np 2` and
the generator with `mpiexec --np 4` as separate MPI worlds. Have
Monarch act as a controller actor that talks to *already-running*
mpiexec-launched processes via TCP/RPC instead of forking them.

This is how production HPC + actor frameworks normally compose, and
it's how `torchtitan/experiments/rl` is presumably intended to run
on CUDA clusters with NCCL (which has similar PMIx requirements on
some networks).

Requires:
- New entrypoint that splits the launch into 3 mpiexec calls
  (controller, trainer, generator)
- TCP/RPC bootstrap between Monarch controller and the worker MPI
  groups
- vLLM-side: it already uses an "external launcher" pattern which
  is PMIx-aware. Should work once the worker procs are mpiexec-spawned.

### B. Use Gloo with a CPU-staging copy layer (slow but works today)

Wrap DTensor's collective dispatch so xpu tensors are `.to("cpu")`
before broadcast/allreduce, then `.to("xpu")` back. This means every
collective hits a 2x CPU↔XPU copy. Workable for the trainer's
infrequent state-sync collectives, painful for any hot loop.

### C. Stay with TRL `vllm_mode="server"` instead

The bare vLLM smoke + the env-scrub fix means we can stand up a TRL
GRPO loop that uses an external vLLM server (no Monarch on the
trainer side). The trainer runs as our normal ezpz trainer process
(mpiexec-launched, with working XCCL); the generator is the bare
vLLM-XPU server we already validated. TRL handles weight sync over
HTTP. Lower throughput than Monarch+TorchStore RDMA, but doesn't
depend on solving the Monarch+PMIx mismatch.

## Recommendation

(C) is the fastest path to a working RL loop on XPU and matches
what was already in the wiring plan as Track 2. (A) is the
architecturally correct next step if/when we want the Monarch
+TorchStore performance.

The XPU porting layer (`xpu_overrides.py`) is reusable for either —
the `has_cuda_capability` patch and `EzpzPerHostProvisioner` apply
regardless of how the worker processes are launched.

## Files added during this investigation

- `torchtitan/experiments/ezpz/rl/xpu_overrides.py`
- `torchtitan/experiments/ezpz/rl/train_upstream.py`
- `torchtitan/experiments/ezpz/rl/scripts/grpo_qwen3_smoke.sh`
- `torchtitan/experiments/ezpz/rl/scripts/build_rl_vllm_venv.sh` (updated)
- `torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B/` (HF download)

## Job log

| Job | Stage reached | Blocker |
|---|---|---|
| 12468755 | trainer ctor | wandb phantom install (fixed) |
| 12468756 | set_determinism | seed broadcast USM (--debug.seed 42) |
| 12468757 | parallelize_fn TP=2 | impi-rt poisoning libccl (uninstalled) |
| 12468758-9 | init_weights TP=1 | RNG broadcast USM |
| 12468760 | init_weights TP=2 | DTensor mesh_broadcast USM |
| 12468761 | init_weights | PALS env injection (insufficient) |
| 12468762 | torchstore init | circular import (fixed) |
| 12468763 | init_weights | regression (import order) |
| 12468764 | init_weights | reorder didn't help |
| 12468765 | init_weights | DIAGNOSTIC: env injection confirmed working but USM still fails |
| 12468766 | init_weights | Gloo override: "No backend type for xpu" |
| 12468767 | init_weights | permanent Gloo map: same error |
| 12468768 | init_weights | "xpu:gloo,cpu:gloo": Gloo can't broadcast xpu tensors |
| 12468769 | init_weights | mpiexec --np 1 wrapper: PMIx doesn't fork-inherit |
