# vLLM-XPU + Monarch RL infra status (as of 2026-06-13)

## Summary

| Component | Status | Notes |
|---|---|---|
| `venvs/rl-actors/` venv build | ✅ done | py3.13 + torch 2.12+xpu + vllm 0.22 + monarch + torchstore + TRL 1.6, all imports clean |
| Monarch actors on XPU | ✅ working | Job `12468739`: 2-rank `spawn_procs` confirmed, both ranks see `xpu_count=12` from inside actor |
| TorchStore on XPU (Gloo transport) | ✅ importable | Full round-trip not yet smoked; transport class loads cleanly |
| vLLM-XPU bare engine init (single-tile) | ❌ blocked | `MPIDI_GPU_init_mpl_global` segfault inside MPI bootstrap; happens at every np=1 launch attempt |
| TRL `vllm_mode="server"` | ❌ blocked downstream | TRL wraps vLLM with `multiprocessing.spawn`, hits the same bootstrap path |
| ezpz `EzpzVLLMGenerator` skeleton | scaffolded | `rl/actors/ezpz_generator.py` documents the 5 upstream override points; awaiting working vllm-xpu base |

## What works

The **Monarch framework itself** is fully functional on Sunspot XPU.
Job `12468739` ran a 2-actor smoke (`this_host().spawn_procs({"gpus": 2})`)
that:
1. Echoed messages between controller and two spawned actors.
2. Each actor independently called `torch.xpu.device_count()` and
   reported 12 — confirming the actor processes correctly inherit
   XPU visibility.
3. Imported `TorchStore.LocalRankStrategy(default_transport_type=TransportType.Gloo)`
   for the non-CUDA path.

This unblocks the question "can the upstream actor pattern work on
XPU at all?" — yes.

## What's blocked

**vLLM 0.22.1 cannot complete EngineCore init at single rank on the
current Sunspot stack.** Specifically:

- vLLM's `init_worker_distributed_environment` defaults to
  `backend="xccl"` on XPU (hardcoded at
  `vllm/platforms/xpu.py:40 dist_backend: str = "xccl"  # xccl only`).
- The XCCL `init_process_group` triggers `c10d::ProcessGroupXCCL::allreduce`
  on the world group even at `world_size=1`.
- That `allreduce` calls into oneCCL's MPI transport (also tried the
  OFI transport — fails for a different reason: missing libpsm2/libucp).
- The MPI transport's `MPID_Init` calls `MPIDI_GPU_init_mpl_global` (cray-mpich
  or intel-mpi's GPU detection) which **segfaults**.

The 2026-06-10
[`vllm-xpu-investigation.md`](vllm-xpu-investigation.md)
verification ran the same vLLM 0.22.1 version and worked end-to-end.
Why it worked then and not now is **partially diagnosed** after the
job 12468750 replay:

- ✅ **Eliminated**: rl-actors venv (py3.13) ABI binding. The
  vllm-test (py3.14) replay using the exact original recipe
  failed identically.
- ✅ **Eliminated**: dtype/max_model_len/gpu_mem_util kwargs. The
  replay used the original kwargs (bfloat16, 2048, 0.85) and
  failed the same way.
- ✅ **Eliminated**: `CCL_*` env overrides. Job 12468751 removed all
  of them and let oneCCL pick the default OFI transport. Got a
  different but still-fatal failure:
  `RuntimeError: oneCCL: atl_ofi_comm.cpp:232 init_transport:
  EXCEPTION: failed to initialize ATL`. So both transports are broken
  in our invocation context: MPI silently segfaults at
  `MPIDI_GPU_init_mpl_global`, OFI explicitly fails to init ATL.
- 🔍 **Remaining suspect**: invocation context. The 2026-06-10 run
  was *interactive* on an existing eval allocation. Every failing
  job today has been a PBS-direct submit, where vLLM's `EngineCore`
  multiprocessing.spawn forks into a new process that has no PMIx
  context inherited from any prior `mpiexec`. Direct test would be:
  ssh into an active compute allocation, source `ezpz_setup_env`,
  run `venvs/vllm-test/bin/python vllm_xpu_vllmtest_replay.py`
  directly (no `ezpz launch`, no `mpiexec`).

## Debug chain so far (jobs 12468737..12468749)

| Job | Symptom | Fix attempted | Result |
|---|---|---|---|
| 12468737 | TRL `vllm_serve`: `current_platform.device_type` empty in worker | n/a — diagnostic | revealed TRL 1.5.1 + vllm 0.22 ABI mismatch |
| 12468740 | bare vllm: `ModuleNotFoundError: xgrammar` | `uv pip install xgrammar` | next error |
| 12468741 | repeated old error (xgrammar install hadn't propagated) | retry | next error |
| 12468742 | `FileNotFoundError: '<stdin>'` from heredoc | move to real `.py` file | next error |
| 12468743 | `oneCCL atl_ofi`: `libpsm2.so.2` / `libucp.so.0` not found | `export CCL_ATL_TRANSPORT=mpi` | switched to MPI failure |
| 12468744 | `MPIR_pmi_init(197): PMIX_Init returned -25` (no MPI bootstrap) | launch via `ezpz launch --np 1` | next error |
| 12468745 | `ezpz: command not found` (path issue) | use absolute `.venv/bin/ezpz` | next error |
| 12468746 | `ZMQError: ipc path > 107 chars` | `TMPDIR=/tmp/vllm-$USER` (short path) | got past ZMQ, hit MPI segfault |
| 12468747 | `MPIDI_GPU_init_mpl_global` segfault in MPI bootstrap | n/a — root cause | failed |
| 12468748 | same segfault, even with monkey-patched `XPUPlatform.dist_backend="gloo"` | torch's XCCL fires anyway since tensors are on XPU | failed |
| 12468749 | `ezpz launch --np 2 -ppn 2` to test single-rank hypothesis | **same segfault** at both ranks — kills the "np=1 is the bug" theory | failed |
| 12468750 | Replay 2026-06-10 recipe verbatim from `venvs/vllm-test/` (py3.14) | **same failure mode** — EngineCore subprocess dies silently after `CCL_WARN| value of CCL_OP_SYNC changed to be 1` + `CCL_PROCESS_LAUNCHER changed to be pmix`. Kills "rl-actors venv (py3.13 ABI) is the bug" theory. | failed |
| 12468751 | Drop all `CCL_*` env overrides — let oneCCL pick defaults (which on Sunspot is OFI transport) | **different failure**: `RuntimeError: oneCCL: atl_ofi_comm.cpp:232 init_transport: EXCEPTION: failed to initialize ATL`. Both transports broken: MPI silently segfaults, OFI explicitly fails to init. | failed |

## Diagnosis

The segfault is at `MPIDI_GPU_init_mpl_global` (MPICH's GPU
device detection). It runs when:
1. MPI starts up via PMIx.
2. PMIx detects we have a GPU and tries to enumerate device topology.
3. Something in that enumeration touches XPU state that the
   current driver/runtime doesn't expose the way MPICH expects.

The monkey-patch to `XPUPlatform.dist_backend` doesn't help because
the failure is below vLLM's level — torch itself routes XPU tensors
through `ProcessGroupXCCL`, which always uses oneCCL → MPI.
You cannot ask torch to use Gloo for an XPU tensor (Gloo is
CPU-only).

## Paths forward

1. **Multi-rank launch (np ≥ 2)**. The MPI segfault might be specific
   to the `np=1` case (production training runs np=96+ daily without
   issue). Try `ezpz launch --np 2 -ppn 2` with vLLM doing TP=1 — if
   it works, the single-rank case is the bug, not the XPU stack
   broadly.
2. **Restore an older oneCCL / oneAPI module load.** The 2026-06-10
   verification used the same `oneapi/release/2025.3.1` module name,
   but the underlying packages might have shifted. Check
   `/opt/aurora/26.26.0` vs older snapshots if available.
3. **File against ALCF.** This is genuinely a system-level regression
   if 2026-06-10 worked and 2026-06-13 doesn't with identical user
   stack.
4. **Skip vLLM entirely.** ezpz/rl's current HF-`.generate()` path
   works; the Monarch+TorchStore infrastructure we just stood up can
   still be useful for distributing trainer + reward scoring without
   vLLM.

## Files added this session

- `venvs/rl-actors/` — new sibling venv, py3.13 + full RL stack.
- `rl/scripts/monarch_smoke.{py,sh}` — Monarch framework smoke. PASSING.
- `rl/scripts/vllm_xpu_bare_smoke.{py,sh}` — vLLM bare smoke. FAILING
  at MPI bootstrap; see debug chain above.
- `rl/scripts/vllm_serve_xpu.sh`, `vllm_serve_smoke.sh`,
  `grpo/aurora2b_sft_arithmetic_8n_vllm.sh` — server-mode plumbing
  (depends on vllm-xpu working; currently blocked).
- `rl/actors/__init__.py`, `rl/actors/ezpz_generator.py` — skeleton
  for the Monarch+vLLM actor adaptation. Awaits a working
  vllm-xpu base.
- `docs/rl/vllm-xpu-wiring-plan.md` — architecture decision doc.
- `docs/rl/vllm-xpu-current-status.md` — this file.

## Recommended next steps

1. ~~**np=2 hypothesis**~~ — **refuted** (12468749).
2. ~~**rl-actors venv ABI hypothesis**~~ — **refuted** (12468750).
3. ~~**Test without CCL_* env overrides**~~ — **refuted** (12468751);
   different failure mode but still fatal. Both transports broken.
4. **Test interactive invocation** — only remaining variable. Needs
   SSH from login node into an active 8N allocation:
   ```bash
   ssh x1921c3s0b0n0 'bash --login -c "
       cd <repo>
       module load oneapi/release/2025.3.1 hdf5 pti-gpu
       export ZE_FLAT_DEVICE_HIERARCHY=FLAT
       export ONEAPI_DEVICE_SELECTOR=opencl:gpu\;level_zero:gpu
       export TMPDIR=/tmp/vllm-\$USER
       mkdir -p \$TMPDIR
       ./venvs/vllm-test/bin/python torchtitan/experiments/ezpz/rl/scripts/vllm_xpu_vllmtest_replay.py
   "'
   ```
   If it works, the smoke needs to land on a node that already has
   `mpiexec` running (e.g. wrapper around an existing training job)
   rather than PBS-direct. If it fails, the OS/runtime stack has
   shifted in some way I haven't fingerprinted and we should file
   to ALCF with the OFI failure as primary signal.

   **Note (2026-06-13)**: I tried running this test but the auto-mode
   classifier blocks SSH-to-shared-compute-node without explicit
   per-action user authorization. Pending user permission.
5. While the vLLM side is blocked, the Monarch+TorchStore framework
   is usable independently — could start prototyping
   `EzpzPolicyTrainer` against the existing HF-`.generate()` flow
   to validate the actor pattern works for our trainer side.
