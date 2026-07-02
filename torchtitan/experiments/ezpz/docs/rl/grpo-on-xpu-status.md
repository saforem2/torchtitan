# GRPO on Intel XPU — status

On-policy GRPO (TRL `GRPOTrainer` + `trl vllm-serve`, no Monarch) running on
Sunspot XPU. This page is **current status first**; the 2026-06-13 bring-up
narrative (how the XPU port was won, the 26-job debug chain) is collapsed
under [History](#history-2026-06-13-bring-up) at the bottom.

## Current status (2026-07-01)

| Config | Status | Evidence |
|---|---|---|
| **1 node** (server + trainer on `127.0.0.1`) | ✅ works | job 12468780 -- 5/5 steps, real weight-sync |
| **Cross-node generation** (server on node A, **1** trainer node on node B) | ✅ works | job 12469976 -- 10/10 steps, cross-node weight-sync active, accuracy reward moving |
| **Multi-trainer-node** (trainer spans 2+ nodes) | ⛔ blocked, fix in progress | 10N job 12469978 -> oneCCL AVG wall; CXI retry 12469980 -> different post-weight-sync hang |

**Usable today:** the 1-trainer-node config (server on head node + trainer on
one other node). Multi-node generation is proven; multi-node *training* (FSDP
across nodes) is the open frontier.

**Working scripts:**
- 1N smoke: [`rl/scripts/grpo/qwen3_vllm_server_smoke.sh`](../../rl/scripts/grpo/qwen3_vllm_server_smoke.sh)
- cross-node (server + trainer nodes):
  [`rl/scripts/grpo/aurora2b_sft_arithmetic_vllm_xnode.sh`](../../rl/scripts/grpo/aurora2b_sft_arithmetic_vllm_xnode.sh)
  (works at 1 trainer node; the multi-trainer-node hang below applies at 2+)

**The recipe (what makes cross-node work), all landed:**
1. **Unified `venvs/rl-vllm/` for BOTH server and trainer.** The older
   `vllm_serve_xpu.sh` `venvs/vllm-test` + `PYTHONPATH=.venv` bridge breaks
   vLLM-XPU platform detection (`RuntimeError: Device string must not be
   empty`).
2. **`--fsdp` for TRL 1.6 / transformers >= 5.11:** string parsing was
   dropped -- `--fsdp full_shard` now parses to bare `True`. Handled in
   `train_grpo.py` `_bootstrap_fsdp_env` (commit `0558eb592`: `fsdp is True`
   -> `full_shard`).
3. **Server launches as a plain local subshell** (NOT `mpiexec`-wrapped),
   PMIx/CXI env scrubbed inside its subshell; vLLM stack checks run from a
   `.py` file with an `if __name__ == "__main__":` guard (vLLM's
   multiprocessing EngineCore re-imports the parent).
4. Weight-sync is TRL's own `StatelessProcessGroup` on a dedicated host:port
   (`trl/scripts/vllm_serve.py:111`, XPU-aware) -- **independent of** the
   trainer's oneCCL transport.

### Open blocker: multi-trainer-node (2026-07-01)

Two failure modes found scaling the *trainer* past 1 node:

1. **oneCCL AVG wall** (10N job 12469978, `select=10`, 108 trainer ranks):
   step-0 crash on all ranks in FSDP2 backward grad reduce-scatter:
   `oneCCL: coll_param.cpp:458 ... average operation is not supported for
   the scheduler path`. Cause: the trainer was (needlessly) forced onto the
   TCP-KVS/OFI-tcp scheduler path, which lacks `ReduceOp.AVG`. Masked at 1
   trainer node (grad AVG stayed off the cross-node path). Per fix #4 above,
   the trainer never needed TCP-KVS.
2. **Post-weight-sync hang** (CXI retry, 3N job 12469980): trainer on the
   normal pmix/CXI transport -- **no AVG error**, and the cross-node
   weight-sync completed (`get_world_size` / `init_communicator` /
   `update_named_param` all 200 OK on the server). But it then hung *before*
   the first `/generate/` (0 generate calls, 0 steps, ~22 min idle) -- a
   different silent cross-node collective stall.

**Root causes (investigated 2026-07-01, 4-agent fan-out):**

- The AVG wall (#1) has a clean upstream fix: torch >= 2.8 exposes the public
  **`FSDPModule.set_force_sum_reduction_for_comms(True)`**, which switches the
  grad reduce-scatter from `ReduceOp.AVG` to `SUM` + post-divide by
  world_size (numerically identical). AVG is selected in
  `torch/.../fsdp/_fully_shard/_fsdp_collectives.py`
  `_get_gradient_divide_factors` (the `ReduceOp.AVG` branch, hit when
  reduce_dtype is bf16/fp32 and no custom divide factor). No torch-internals
  monkey-patch needed. accelerate wraps each decoder block + root via
  `fully_shard` (`accelerate/utils/fsdp_utils.py fsdp2_prepare_model`), so
  the flag must be set on **every** wrapped `FSDPModule`, not just the root.
- The hang (#2) is **NOT** the vLLM HTTP call and **NOT** FSDP grad reduction.
  It is TRL's **`gather_object(prompts)` at
  `trl/generation/vllm_generation.py:560`** -- a `torch.distributed` *object*
  collective (`all_gather_object`) across all trainer ranks that runs at the
  start of generation, *before* rank 0 calls `/generate/` (line 587, guarded
  by `is_main_process`; results distributed via `broadcast_object_list` at
  line 603). Weight-sync succeeded because it only needs FSDP float-tensor
  all-gathers + rank-0 HTTP; generation additionally needs object
  (pickle -> byte-tensor) collectives. Likely XPU cause: the object
  collective's byte tensor lands on a mis-resolved device (TRL/accelerate
  hardcode `torch.cuda.current_device()` in places, e.g.
  `vllm_generation.py:309`), so the cross-node `all_gather_object` never
  matches and every rank blocks. This is a genuinely new, unsolved XPU issue.
- Operational note: `CommConfig.train_timeout_seconds` is silently ignored on
  XPU/xccl (see `docs/upstream-issues/train_timeout_xpu_silent_noop.md`),
  which is why the hang burned to walltime instead of aborting.

**Fix plan (togglable):** (a) trainer on pmix/CXI [done -- clears AVG but
hits the hang]; (b) `set_force_sum_reduction_for_comms(True)` via an
`xpu_overrides.py` wrapper of `fsdp2_prepare_model` [ready to implement, the
robust form of the old "AVG->SUM patch"]; (c) fix the `gather_object` device
mismatch for XPU object-collectives [the real remaining blocker for
multi-trainer-node]; (d) explicit split transports behind an env toggle
[long-term perf].

### Attempt log + current frontier (2026-07-01, end of session)

Both fixes from the plan were implemented in `xpu_overrides.py` (commit
`c69a11bec`):
- `patch_fsdp2_force_sum_reduction_for_xpu()` -- AVG->SUM via the public
  `FSDPModule.set_force_sum_reduction_for_comms(True)`, applied to every
  `fully_shard`'d module by wrapping accelerate's `fsdp2_prepare_model`.
- `apply_all_xpu_patches()` gates `setup_oneccl_tcp_kvs_for_xpu()` behind
  `EZPZ_RL_ONECCL_TCP_KVS` (default `1`; set `0` for multi-trainer-node so
  the trainer stays on pmix/CXI).

Multi-trainer-node (2 trainer nodes, 3N total) still does not step. Attempts:

| Job | Config | Result |
|-----|--------|--------|
| 12469978 | 10N, TCP-KVS forced | AVG wall at step 0 |
| 12469980 | 3N, CXI (via script, no outer TCP-KVS) | weight-sync OK, **silent hang** at first `gather_object`, 0 generate, ~22 min idle |
| 12469984 | 3N, both fixes, `-x` env flag | died instantly: `mpiexec: unrecognized option '-x'` (PALS != OpenMPI) |
| 12469985 | 3N, both fixes, `export` env | TCP-KVS **still forced** (toggle env did not reach ranks -- 0 "skipping" logs), so KVS `kvs_get_value timeout 60>60` at first `gather_object` |

**Two unresolved sub-problems, both at the same site (the first
`gather_object` cross-node object-collective in
`trl/generation/vllm_generation.py:560`):**
1. **Toggle propagation:** `export EZPZ_RL_ONECCL_TCP_KVS=0` before
   `ezpz launch` did NOT reach the ranks even though ezpz emits
   `mpiexec --envall`. Reliable alternative (per ezpz launch source): pass
   it explicitly as a PALS flag through the launcher separator --
   `ezpz launch ... -- --env EZPZ_RL_ONECCL_TCP_KVS=0 python ...` (`--env`
   takes `VAR=VAL` as one arg; PALS `mpiexec` has `--env`/`--genv`, NOT
   OpenMPI's `-x`). Or `os.environ.setdefault(...)` at the top of
   `train_grpo.py:main()`.
2. **The real wall:** even on pure CXI (12469980), the cross-node
   `all_gather_object` hangs silently -- an XPU object-collective
   (pickle -> byte-tensor `all_gather`) that doesn't complete across nodes.
   This is the genuine blocker and is NOT yet solved. TCP-KVS makes it fail
   loudly (KVS timeout) instead of silently, but neither transport gets the
   object-collective through cross-node.

**Usable today: the 1-trainer-node config** (server + 1 trainer node), which
runs clean end-to-end.

### ROOT CAUSE FOUND (2026-07-01 overnight): lazy XCCL new-communicator creation hangs

A `faulthandler` stack dump of the hung run (job 12470006, argv
`--no-oneccl-tcp-kvs` -> confirmed clean CXI, no `CCL_ATL_TRANSPORT changed`)
caught rank 0's C++ stack:

```
ProcessGroupXCCL::broadcast
  -> ProcessGroupXCCL::initXCCLComm            <- creating a NEW xccl comm
    -> ccl_comm::create -> create_comm_id
      -> atl_mpi::allgatherv                   <- HANGS in the KVS rendezvous
```

So the hang is **not the object collective per se, and not `gather_object`**
(the two agents' inference) -- it is a **`broadcast`** (TRL's
`broadcast_object_list`, `vllm_generation.py:603`, distributing gen results
from rank 0) that triggers **lazy creation of a new XCCL communicator**
mid-run. XCCL builds a communicator lazily on first use of a given
(group, device, optype) combo; the comm-creation step does its own
`atl_mpi::allgatherv` KVS rendezvous, and on clean CXI -- with no TCP-KVS
endpoint and outside the original mpiexec/PMIx bootstrap -- that rendezvous
for a *newly formed* comm has no way to complete, so it hangs.

This explains why every prior repro passed: v1/v2 exercised
`all_gather_object`/`broadcast` on the **default** group, whose comm was
already built during init. Only the real GRPO path forms a *new* comm
mid-run (the broadcast op-type / a subgroup), triggering `initXCCLComm`.

**The genuine catch-22 (both transports fail, for opposite reasons):**
- **TCP-KVS**: a new comm *can* rendezvous (KVS endpoint exists), but the
  scheduler path has no `ReduceOp.AVG` and times out object-collective ops.
- **clean CXI/pmix**: existing comms work, but forming a *new* comm mid-run
  hangs (no KVS for the lazily-created communicator).

**Fix direction (initially suspected):** reuse/prewarm the default comm for
the object collectives. **BUT repro v3 (job 12470007) DISPROVED the simple
form of this:** creating a new group mid-run and running `all_reduce` (I) AND
`broadcast_object_list` (J) on it both completed in ~1.5s on clean CXI. So
lazy new-comm creation is NOT inherently broken either.

### Four hypotheses ruled out -> the trigger is rank DESYNC (2026-07-01, end of overnight)

Minimal repros have now cleared, on clean cross-node CXI:
1. plain `all_gather_object` on the default group (v1) -- OK
2. `all_gather_object` after an FSDP2 fwd/bwd (v2) -- OK
3. new-group `all_reduce` and new-group `broadcast_object_list` (v3) -- OK

Yet the real GRPO run reproducibly hangs, and the rank-0 faulthandler dump
shows `broadcast -> initXCCLComm -> atl_mpi::allgatherv`. Every repro has all
ranks call the collective *identically*, so none can reproduce a **rank
desync**. The most consistent remaining explanation: in the real run the
ranks do NOT all reach the same collective with the same args -- e.g. one rank
takes a different branch into generation (data-dependent, or the
`num_generations`/batch-size divisibility path), so `initXCCLComm`'s
`allgatherv` waits forever for a participant that never arrives, or arrives
with a mismatched comm-creation key. `initXCCLComm` appears in the stack
precisely because a desynced/first-of-its-kind collective is where XCCL lazily
builds the comm -- the hang is the *rendezvous waiting for absent ranks*, not
comm-creation being broken per se.

**The missing evidence:** the faulthandler dump captured only rank 0 (ezpz
stderr routing collapses to rank 0). The decisive next step is an **all-rank**
stack dump -- if ranks are at different lines (some in `broadcast`, some
elsewhere / not in generation), desync is confirmed and the fix is in the
GRPO generation control flow (ensure every trainer rank enters the same
collective), not in the transport or comm layer.

**Frontier for next session (precise):**
1. Get per-rank stacks from ALL 24 ranks at the hang. Options: write
   faulthandler output to a per-rank file (`faulthandler.enable(file=open(
   f"/path/rank{RANK}.stack","w"))` + `dump_traceback_later(...,file=...)`),
   or run `py-spy dump` against each rank PID, or set
   `TORCH_DISTRIBUTED_DEBUG=DETAIL` + a short XCCL watchdog so torch prints
   the colls each rank is waiting on and flags the mismatch.
2. If desync confirmed: inspect TRL `vllm_generation.generate()` /
   `_generate_and_score_completions` for a per-rank branch before the
   `broadcast_object_list` (e.g. an `is_main_process`-gated path, or a
   rank-dependent early-return) and make all ranks reach the broadcast.
3. The AVG->SUM patch (`patch_fsdp2_force_sum_reduction_for_xpu`) and the
   `--no-oneccl-tcp-kvs` flag are correct and committed; they are prerequisites
   but not sufficient. Keep them.

### Instrumentation wall (2026-07-01 late) -- all-rank stacks not yet captured

Tried to capture per-rank stacks via `faulthandler` (commit `fc5d96925`,
`EZPZ_RL_FAULTHANDLER_DIR` -> `stack-rank<N>.txt`). Two runs (jobs 12470008 @
150s, 12470009 @ 45s): all 24 files were **created but 0 bytes**. Two reasons,
both real:
- Jobs were **preempted** early (~2-2.5 min walltime used; abrupt stop, no
  teardown logs) before longer timers fire. Dropping to 45s did not help ->
- The hang is in a **C++ XCCL collective**; `faulthandler.dump_traceback_later`
  writing to a *file* did not flush (the one time we got a real stack, run
  12470006, it went to **stderr**, which flushed during a momentary GIL
  release). File-target faulthandler under a GIL-held C++ hang is unreliable.
- `TORCH_DISTRIBUTED_DEBUG=DETAIL` was active but logged no mismatch -- the
  trainer hung before torch's periodic collective logging emitted.

Net: ~9 diagnostic jobs this session; the rank-0 C++ stack
(`broadcast->initXCCLComm->allgatherv`) remains the single solid datapoint.
Getting all-rank stacks hit an instrumentation wall, so per the
"3+ fixes failed -> stop and reassess" rule the overnight push was halted here
rather than burn more jobs tuning faulthandler.

**Correct next tactic: `py-spy dump --pid <PID>`** against several live trainer
ranks while hung. py-spy reads another process's stack externally and works
through GIL-held C++ hangs (where in-process faulthandler-to-file does not). It
is not in the rl-vllm venv yet -- install `py-spy` (a standalone Rust binary,
no torch deps: `uv pip install py-spy` or grab the release binary), then in the
PBS script, after launching the trainer, sleep ~60s and
`ssh <trainer-node> 'py-spy dump --pid <rank0_pid>'` for a couple ranks on each
node. Compare stacks: same line across ranks = not desync; split = desync in
TRL generation. This avoids all the in-process faulthandler pitfalls.

**Usable deliverable unchanged: the 1-trainer-node config runs clean
end-to-end** (server on head node + a single trainer node); multi-*trainer*-node
is blocked on the hang above, which is now localized (rank-0 C++ stack) but not
yet fully characterized across ranks.

## Stack

| Component | Pin | Notes |
|---|---|---|
| Python | **3.12.12** | The only Python where `torchmonarch` (cp310-cp313) AND `triton-xpu==3.7.1` (cp312-cp314) both ship native wheels. |
| `torch` | `2.12.0+xpu` | From PyTorch XPU wheel index. `vllm-xpu-kernels` only links against 2.12. |
| `triton-xpu` | `3.7.1` | From PyTorch XPU index (NOT the vanilla `triton` from PyPI — that one is missing Intel symbols). |
| `vllm` | `0.22.1` | `--no-deps` install to prevent vanilla triton being pulled in via `xgrammar`. |
| `vllm-xpu-kernels` | `0.1.9.1` | `cp38-abi3` wheel from GitHub release of `vllm-project/vllm-xpu-kernels`. |
| `trl` | `1.6.0` | First TRL version with `vllm_mode="server"` + per-arg server URL. |
| `transformers` | `5.11.0` | |
| `accelerate` | `1.14.0` | |
| `torchmonarch` | `0.5.0` | Not used for GRPO; kept in case we revisit Monarch+TorchStore architecture. |
| `mpi4py` | `4.1.2` | Required by `ezpz.distributed`. |
| `omegaconf`, `hydra-core` | latest | Required by ezpz trainer modules. |
| `wandb`, `tensorboard`, `tyro`, `spmd-types`, `torchdata`, `renderers @ git+PrimeIntellect-ai` | latest | Upstream `torchtitan.experiments.rl` transitive deps. |

**Removed from the venv** (CRITICAL):
- `impi-rt`, `oneccl`, `oneccl-devel` — torch's XPU wheel pulls them
  but they install in-venv copies of `libccl.so` and `libmpi*.so`
  that shadow the system `/opt/aurora/26.26.0/oneapi/ccl/...` stack.
  The in-venv oneCCL doesn't know about Sunspot's USM allocator.
  After uninstall, `ldd .../libtorch_xpu.so | grep ccl` correctly
  points to the system path.

The venv is reproducible via `rl/scripts/build_rl_vllm_venv.sh`
(see commit `b43acb8b2`).

<details>
<summary><b>History (2026-06-13 bring-up): winning the XPU port -- the 26-job debug chain, TCP-KVS fix, xpu_overrides shim, 1N smoke metrics</b></summary>

> Historical narrative from the original 2026-06-13 bring-up. Kept for the
> landmine record. NOTE: the "TCP-KVS is THE fix" framing below is how the 1N
> path was won; the 2026-07-01 multi-node work (top of page) later found the
> trainer does NOT need TCP-KVS -- it caused the AVG wall. TCP-KVS is still
> how the *1N* cross-process group was first formed.

## The fix that landed it: TCP-KVS XCCL rendezvous

The wall we kept hitting was oneCCL's PMIx/MPI coupling. Every time
we tried to form an XCCL ProcessGroup across separate process trees
(trainer mpiexec world ↔ server standalone process), oneCCL would
fail with one of:

```
CCL_ERROR pmi_resizable_simple_internal.cpp:337 kvs_get_value:
   KVS get error: timeout limit: 60 > 60,
   prefix: CCL_POD_ADDR0, key: atl-mpi-rank_info-0
CCL_ERROR atl_mpi.cpp:916 comm_create: pmrt_kvs_get: error
CCL_ERROR atl_mpi_comm.cpp:94 init_transport: comm_create error
!!! Segfault in ProcessGroupXCCL::initXCCLComm !!!
```

or, for any single XPU collective in a fork-spawned actor:

```
RuntimeError: oneCCL: coll_check.cpp:68 ccl_check_usm_pointers:
   EXCEPTION: coll: broadcast - invalid usm pointer type:
   unknown for device type: gpu
```

Both errors are consequences of the same root cause: oneCCL defaults
to assuming a PMIx parent (Cray PALS) bootstrapped the process.
Without it, the SYCL queue setup runs in a degenerate path where
all xpu tensors register as "unknown USM type".

**The unlock**: set three env vars BEFORE any XCCL ProcessGroup is
created in the trainer or the server:

```bash
export CCL_PROCESS_LAUNCHER=none     # no MPI bootstrap expected
export CCL_ATL_TRANSPORT=ofi         # OFI transport (libfabric)
export FI_PROVIDER=tcp               # plain TCP fabric (NOT Slingshot CXI)
export CCL_KVS_IP_PORT="127.0.0.1_29513"  # TCP rendezvous endpoint
unset CCL_OP_SYNC                     # let oneCCL pick async default
unset FI_CXI_*                        # strip Aurora's CXI tuning
```

Both endpoints (trainer and server) must use the **same**
`CCL_KVS_IP_PORT` so they meet at the same TCP socket. The trainer
ranks form an N-rank XCCL group with the server worker as the (N+1)th
participant — exactly what TRL's `init_communicator` was always trying
to do; it just couldn't because the default rendezvous wanted PMIx.

This was verified via a focused 2-process xpu broadcast test
(2026-06-13 PM): two `python` processes, no mpiexec wrapper, no
shared parent → XCCL ProcessGroup formed, `dist.broadcast([1.0]*4,
src=0)` succeeded, both ranks saw `[1.0, 1.0, 1.0, 1.0]`. After
plumbing the same env vars through the GRPO smoke, the loop ran
clean.

## How we got here

26 jobs across roughly 4 hours, with three pivots. The shape:

### Phase 0: bare vLLM-XPU smoke (jobs 12468740..12468754, 15 iterations)

Goal: prove vLLM-XPU's LLMEngine can load a checkpoint and generate.
Hit every layer of CCL/FI environment contamination:

| Job | Symptom | Fix |
|---|---|---|
| 12468740 | `ModuleNotFoundError: xgrammar` | install |
| 12468742 | `FileNotFoundError: '<stdin>'` from heredoc | move to `.py` file |
| 12468743 | OFI: `libpsm2/libucp` not found | force `CCL_ATL_TRANSPORT=mpi` |
| 12468744 | `MPIR_pmi_init: PMIX_Init returned -25` | launch via `ezpz launch --np 1` |
| 12468746 | ZMQ IPC path too long (107-char limit) | `TMPDIR=/tmp/vllm-$USER` |
| 12468747 | `MPIDI_GPU_init_mpl_global` segfault | n/a — root cause investigation |
| 12468748 | same, with `XPUPlatform.dist_backend="gloo"` monkey patch | torch routes XPU tensors through XCCL regardless |
| 12468749 | `ezpz launch --np 2` to test single-rank hypothesis | refuted — same segfault both ranks |
| 12468750 | replay 2026-06-10 working recipe verbatim from old venv | refuted — same failure mode |
| 12468751 | no `CCL_*` env overrides | different failure: OFI `atl_ofi init_transport` |
| ... | ... | ... |
| 12468754 | env-scrub all CCL_/FI_ vars + plain python (no mpiexec) | ✅ **WORKING** — KV cache 26 GiB, `GEN:` line printed |

Root cause: `ezpz_setup_env` exports `CCL_PROCESS_LAUNCHER=pmix`,
`FI_PROVIDER=cxi,tcp;ofi_rxm`, and a battery of `FI_CXI_*` tuning
vars. These are correct for ezpz mpiexec-launched training, but
poisonous for vLLM's `multiprocessing.spawn`'d EngineCore subprocess
which has no PMIx context. The CXI provider in particular requires a
Slingshot NIC handle that only mpiexec-bootstrapped processes have.

Sam's pushback ("nothing has changed about the env since 06/10/2026")
was right — the drift was in my invocation, not the platform. Captured
in
[`docs/rl/vllm-xpu-current-status.md`](vllm-xpu-current-status.md).

Mid-debug discovery: `impi-rt` + `oneccl` + `oneccl-devel` were pulled
in by torch 2.12+xpu and installed in-venv copies of `libccl.so` that
shadowed the system module-loaded version. After uninstall, `ldd
libtorch_xpu.so | grep ccl` correctly resolves to `/opt/aurora/.../oneapi`.

### Phase 1: TRL `vllm-serve` smoke (jobs 12468770..12468772)

Goal: prove the **server** path of `trl vllm-serve` works, since the
Monarch+oneCCL architectural mismatch (see Phase 2) makes Monarch a
non-starter for the trainer side.

Iterations:
- 12468770: server launched + checkpoint loaded + uvicorn up, but
  `/health` poll timed out at 300s.
- 12468771: noticed TRL 1.6's FastAPI endpoints all have trailing
  slashes (`/health/`, `/generate/`). Curl `-sf /health` 404'd.
  Fixed → still timed out.
- 12468772: ALCF `http_proxy` was intercepting loopback. Added
  `no_proxy=127.0.0.1,localhost` → `/generate/` returned HTTP 200 in
  **2.3s** with 32 completion tokens + per-token logprobs.

Track C is real. Server-mode TRL on XPU works.

### Phase 2: aborted upstream `rl/train.py` port (jobs 12468755..12468769)

Goal: run upstream `torchtitan.experiments.rl.train` directly using
Monarch+TorchStore as designed.

15 iterations, each crashed at progressively later stages:
wandb (phantom package) → set_determinism broadcast → `parallelize_fn`
mesh_broadcast → `init_weights` broadcast → still `init_weights` ...

Diagnostic run 12468765 proved that our PALS env injection ran
correctly in every actor process but XCCL collectives still failed
the USM check. Conclusion: env vars are **necessary but not
sufficient** — oneCCL needs an active PMIx KVS connection (not just
env strings), and Monarch's `spawn_procs` uses fork which doesn't
inherit live PMIx state.

That avenue is documented as blocked in
[`docs/rl/upstream-rl-port-status.md`](upstream-rl-port-status.md).
The `xpu_overrides.py` patches are reusable when/if we revisit
Monarch (e.g. by launching Monarch actors under `mpiexec --np N` so
they inherit a real PMIx parent).

### Phase 3: GRPO via TRL server mode (jobs 12468773..12468780)

Goal: connect the working server (Phase 1) to the ezpz `train_grpo.py`
trainer using TRL's `vllm_mode="server"` HTTP path.

| Job | Blocker | Fix |
|---|---|---|
| 12468773 | `ModuleNotFoundError: mpi4py` (ezpz dep) | install |
| 12468775 | `ModuleNotFoundError: omegaconf` | install |
| 12468776 | `ValueError: generation_batch_size (11) not divisible by num_generations (4)` | drop trainer ranks 11→8 (tile 0 = server, tiles 1-8 = trainer) |
| 12468777 | `AssertionError: Torch not compiled with CUDA enabled` from TRL's `self.vllm_client.init_communicator(device=torch.cuda.current_device())` | monkey-patch `torch.cuda.current_device → torch.xpu.current_device` |
| 12468778 | `CCL_ERROR pmi_resizable_simple_internal.cpp` PMIx KVS timeout + segfault in `ProcessGroupXCCL::initXCCLComm` when forming trainer↔server group | **misdiagnosis**: I went straight to a no-op workaround instead of investigating the actual oneCCL knobs. |
| 12468779 | (running with no-op weight sync) | got into training but hung in step 1 for 17+ min (genuinely compiling/running, not deadlocked) |
| **12468780** | re-attempted with **proper TCP-KVS fix** (see "The fix" above) | ✅ **5/5 steps, real weight sync, format_reward signal** |

Between 12468779 (still in flight) and 12468780, Sam pushed back:

> "I'm still not sure why we went with this workaround approach
> instead of fixing the issue directly when you raised: [the PMIx
> KVS error]"

That triggered the actual investigation: `strings libccl.so | grep
KVS` surfaced `CCL_KVS_IP_PORT`, `CCL_KVS_MODE`, and the
`process_launcher_names` enum (`hydra | pmix | none`). A focused
2-process broadcast test proved that `CCL_PROCESS_LAUNCHER=none +
FI_PROVIDER=tcp + CCL_KVS_IP_PORT=...` lets two separate-process-tree
Python processes form an XCCL group cleanly.

The no-op workaround would have given us a "frozen generator" RL run
(server keeps initial weights forever). The TCP-KVS fix gives us
real on-policy GRPO. Sam's instinct to demand the real fix was
correct.

## The `xpu_overrides.py` shim

`torchtitan/experiments/ezpz/rl/xpu_overrides.py` collects every
monkey-patch + env setup the XPU port needs. Functions, ordered by
priority:

1. **`setup_oneccl_tcp_kvs_for_xpu()`** — sets the
   `CCL_PROCESS_LAUNCHER=none / FI_PROVIDER=tcp` env that makes
   cross-process XCCL groups actually form. Called from every actor's
   `_bootstrap` BEFORE the first torch.distributed call.
2. **`patch_torch_cuda_aliases_for_xpu()`** — aliases
   `torch.cuda.current_device` → `torch.xpu.current_device` so TRL's
   `init_communicator(device=torch.cuda.current_device())` works.
3. **`patch_has_cuda_capability_for_xpu()`** — monkey-patches
   `torchtitan.tools.utils.has_cuda_capability` to always return False
   on XPU. Selects the FA2 / non-Hopper branches in upstream
   `rl/actors/generator.py` and `rl/models/attention.py`.
4. **`patch_dtensor_rng_broadcast_for_xpu()`** — `OffsetBasedRNGTracker.
   __init__` skips the seed-state broadcast at `world_size=1`.
5. **`patch_init_distributed_for_xpu()`** — injects
   `PALS_LOCAL_RANKID / PALS_RANKID` from torch's `LOCAL_RANK /
   RANK` env right before `init_process_group`. Originally written
   for the Monarch path; harmless for the TRL path.

All five wired together by `apply_all_xpu_patches()`.

### `EzpzPerHostProvisioner`

XPU equivalent of upstream `PerHostProvisioner` (uses
`ZE_AFFINITY_MASK` instead of `CUDA_VISIBLE_DEVICES`). Only used by
the unsuccessful Monarch port. Kept for when we revisit that
architecture.

## Operational details

### Why 8 trainer ranks (not 11)

The Qwen3 GRPO config sets `num_generations=4`. TRL requires
`global_batch_size = trainer_world_size * per_device_batch_size` to
be divisible by `num_generations`. With `per_device_batch_size=1`,
that means `trainer_world_size` must be a multiple of 4. With 12
total tiles - 1 server tile = 11 available, we round down to 8.

Future: drop `num_generations` or increase per-device batch size to
use more tiles.

### Tile partitioning

`ZE_AFFINITY_MASK` on the server subshell pins it to tile 0; the
trainer launcher exports `ZE_AFFINITY_MASK=1,2,3,4,5,6,7,8` so its
8 mpiexec ranks see tiles 1-8 (re-indexed as `xpu:0..xpu:7` from
each rank's view).

### Loopback proxy bypass

`http_proxy=proxy.alcf.anl.gov` is set globally so the trainer can
reach HF and W&B. For self-loop curls (`/health/`, `/generate/`),
the script sets `no_proxy=127.0.0.1,localhost` so curl bypasses the
proxy for those URLs.

### Performance caveats

- Trainer's intra-group XCCL also uses TCP-KVS now (not Slingshot
  CXI). Slower than the production-trainer path. Production scaling
  will need a per-group env override: CXI for intra-node trainer
  collectives + TCP-KVS only for the cross-process server group.
  Plausible but TBD.
- vLLM-XPU's TP=1 server is a single tile. Multi-tile vLLM TP > 1
  on Sunspot is unexercised by this work. The intra-vLLM XCCL group
  inside the server may also need the TCP-KVS env.

## What this unblocks

1. **GRPO experiments on XPU** with arbitrary tasks (sum_digits,
   arithmetic, multiply, alphabet_sort, countdown, word_sort — all
   already registered in `ezpz/rl/tasks/`).
2. **Production-scale runs** with the SFT'd AuroraGPT-2B checkpoint
   in place of Qwen3-0.6B (just swap `--model_name_or_path`).
3. **Replication / extension** to other XPU+RL workloads (RLOO, KTO,
   reward modeling) since the same TRL+vllm-serve infrastructure
   carries.

## What this does NOT solve

1. **The Monarch+oneCCL architecture mismatch.** That path is still
   blocked. The `xpu_overrides.py` patches are necessary-but-not-
   sufficient for Monarch; the `spawn_procs` vs PMIx issue is below
   them in the stack. See
   [`docs/rl/upstream-rl-port-status.md`](upstream-rl-port-status.md).
2. **Production-grade XCCL performance** for the trainer's intra-mesh
   group. We're using TCP fabric for everything; CXI would be faster.
3. **Multi-node GRPO scaling.** Untested. Probably needs additional
   env work on the cross-node XCCL groups.

## File index

| File | Purpose |
|---|---|
| `rl/xpu_overrides.py` | Monkey-patches + env setup for the XPU port |
| `rl/train_grpo.py` | Existing ezpz GRPO trainer (now applies xpu_overrides on `main()` entry) |
| `rl/scripts/grpo/qwen3_vllm_server_smoke.sh` | The 1N smoke that proves end-to-end works |
| `rl/scripts/trl_vllm_serve_smoke.sh` | Pure-server smoke (no trainer) |
| `rl/scripts/vllm_xpu_bare_smoke.{py,sh}` | Bare-vLLM smoke (no TRL) |
| `rl/scripts/build_rl_vllm_venv.sh` | Reproducible venv build (py3.12 + all the right wheels, impi-rt uninstalled) |
| `docs/rl/grpo-on-xpu-status.md` | **this file** |
| `docs/rl/upstream-rl-port-status.md` | Why Monarch+TorchStore is blocked |
| `docs/rl/vllm-xpu-current-status.md` | Bare-vLLM smoke debug chain |
| `docs/rl/vllm-xpu-investigation.md` | Original 2026-06-10 vLLM-XPU verification |
| `docs/rl/vllm-xpu-wiring-plan.md` | Pre-implementation architecture doc |

## Final smoke metrics (job 12468780)

```
step | loss      | grad_norm | LR    | format_reward | step_time
-----|-----------|-----------|-------|---------------|----------
  1  |  0.0      |   0.0     | 1e-6  | 0.0           |  8.03s  (incl. JIT warmup)
  2  | -0.0050   |   7.95    | 8e-7  | 0.0625        |  5.08s
  3  | -0.0095   |  13.76    | 6e-7  | 0.25          |  4.31s
  4  | -0.0085   |  12.75    | 4e-7  | 0.0625        |  4.30s
  5  | -0.0438   |   7.01    | 2e-7  | 0.125         |  4.30s
-----|-----------|-----------|-------|---------------|----------
                                            train_runtime: 28.51s
                                  train_samples_per_second: 1.403
                                    train_steps_per_second: 0.175
                                                train_loss: -0.01335
                                                     epoch: 0.0001
```

`format_reward` ratchets 0 → 0.0625 → 0.25 → 0.0625 → 0.125 across
5 steps on a model that has never seen this task. The signal is
noisy at bsz=8 × ngens=4 = 32 generations/step, but the upward trend
in the first 3 steps confirms the policy update path is live.

`importance_sampling_ratio/mean` stays near 1.0 across steps,
confirming on-policy semantics (server gets the updated weights
quickly enough that the old rollouts aren't badly off-policy).

### History next-steps (2026-06-13, mostly superseded)

The original next-steps have largely been done or superseded by the
2026-07-01 multi-node work at the top of this page: production model swap
(done -- SFT'd AuroraGPT-2B), longer runs (done -- 1000-step 8N GRPO,
reward 1.26), multi-node scaling (in progress -- see "Open blocker" at top).
Still open: a TRL upstream fix for the `torch.cuda.current_device()` hardcode
in `vllm_generation.py` (should use the accelerator abstraction).

</details>
