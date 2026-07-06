# Multi-trainer-node GRPO on XPU: root cause (2026-07-06)

**TL;DR.** The "multi-trainer-node hang" documented over the 2026-07-01
overnight session was **not a rank desync** and **not a silent
object-collective stall**. It was two independent, sequential failures, both
now root-caused from evidence already on disk (no new diagnostic jobs were
needed to find them):

1. **A rank-0 SIGSEGV in `atl_mpi::create_comm_id`** -- every overnight 3N run
   had the trainer's `CCL_ATL_TRANSPORT` **unset**, which selects oneCCL's
   `mpi` default. `atl_mpi` cannot form the trainer-rank0 <-> vLLM-server
   `ProcessGroupXCCL` across two separate `mpiexec` worlds. The other 23 ranks
   then blocked in the main-PG all-gather waiting for the dead rank 0 -- which
   was misread as a "desync hang."
2. **The committed AVG->SUM FSDP patch never ran** -- it rebound the wrong
   name, so once the transport crash was removed the run hit the oneCCL AVG
   wall exactly as before the patch. Fixed by rebinding at every `from`-import
   site.

## How the overnight conclusion went wrong

The overnight session chased a "clean CXI" theory and passed
`--no-oneccl-tcp-kvs` to strip the TCP-KVS forcing. But
`setup_oneccl_tcp_kvs_for_xpu()` bundles **three** exports:
`CCL_PROCESS_LAUNCHER=none`, `CCL_ATL_TRANSPORT=ofi`, `FI_PROVIDER=tcp`.
Turning it off did not select CXI -- it left `CCL_ATL_TRANSPORT` unset, so
oneCCL fell back to its `mpi` default. So the overnight runs never actually
tested multi-trainer-node on a working transport; they crashed earlier, on a
self-inflicted transport regression.

The single rank-0 faulthandler stack that was captured
(`broadcast -> initXCCLComm -> atl_mpi::allgatherv`) is real, but it is the
signature of `atl_mpi` **failing to create a comm**, not of a collective
waiting on absent ranks. The "four hypotheses ruled out -> desync" chain in
`grpo-on-xpu-status.md` over-fit to that one partial stack.

## The evidence (all from overnight job 12470009 + the 2N smoke 12469976)

The overnight faulthandler run **did** produce full per-rank stacks (contrary
to the "0-byte files" note -- that was the earlier jobs; 12470009 wrote 24 x
~25 KB stack files, dumping 4 times at 45/90/135/180 s). The last pre-crash
dump shows all 24 ranks in TRL's FSDP2->vLLM weight-sync
(`vllm_generation.py:_sync_fsdp2_params_to_vllm`), split exactly as the loop
dictates:

- **rank 0** at `vllm_client.py:507` = `self.communicator.broadcast(weights,
  root=self.rank)` -- pushing a param to the server.
- **ranks 1-23** at `_api.py:727 full_tensor()` -> `all_gather_tensor` --
  unsharding the *next* param, blocked waiting for rank 0.

`trainer.log:3554` is the smoking gun:

```
|CCL_ERROR| atl_mpi_comm.cpp:94 init_transport: comm_create error
!!!!!!! Segfault encountered !!!!!!!
  atl_mpi::allgatherv(...)
  atl_base_comm::create_comm_id()
  ccl_comm::ccl_comm(...)
  ccl_comm::create(...)
  c10d::ProcessGroupXCCL::initXCCLComm(...)
```

Then run.log: `rank 0 died from signal 11 and dumped core`, and the launcher
SIGTERM'd the rest (exit 143). Total runtime ~278 s -- not a 5-hour hang.

Decomposition across every relevant run -- one controlling variable:

| Run | Nodes | trainer `CCL_ATL_TRANSPORT` | Outcome |
|---|---|---|---|
| 12469976 | 2N (1 trainer node) | **ofi** | trained: 1110 weight-syncs, 10 loss steps, rewards moving |
| 12469985 | 3N (KVS nominally on) | unset -> **mpi** | SIGSEGV `atl_mpi::create_comm_id` |
| 12470003-009 | 3N (`--no-oneccl-tcp-kvs`) | unset -> **mpi** | SIGSEGV `atl_mpi::create_comm_id` |

This is exactly the pre-TCP-KVS failure the bring-up history already documents
(`atl_mpi_comm.cpp:94 ... comm_create error` -> segfault in
`ProcessGroupXCCL::initXCCLComm`). The 1N path solved it by forcing
`ofi`/TCP-KVS; the overnight work removed that unknowingly.

## The second bug: the SUM patch rebound the wrong name

With the transport fixed (`ofi`/KVS on), 3N job **12470080** got **past** the
`atl_mpi` crash for the first time -- it reached `Starting GRPO training` and
FSDP backward, with no `comm_create error` and no SIGSEGV. It then hit the
oneCCL AVG wall on all 24 ranks:

```
oneCCL: coll_param.cpp:458 validate: EXCEPTION:
   average operation is not supported for the scheduler path
```

`patch_fsdp2_force_sum_reduction_for_xpu()` exists to prevent exactly this
(convert FSDP2's `ReduceOp.AVG` reduce-scatter to `SUM`+divide via the public
`FSDPModule.set_force_sum_reduction_for_comms(True)`). Its unconditional stderr
banner (`forced ReduceOp.SUM+divide on N FSDP2 modules`) was **completely
absent** -> the wrapper never executed.

Root cause: the patch rebound only
`accelerate.utils.fsdp_utils.fsdp2_prepare_model`. But
`accelerate/accelerator.py:88` does `from .utils import fsdp2_prepare_model` at
import, and the real call site (`Accelerator.prepare_model` ->
`fsdp2_prepare_model(self, model)`, `accelerator.py:1731`) reads *that*
module-local binding. Classic `from x import y` monkeypatch miss: rebinding the
source module leaves the already-imported alias pointing at the original.

Fix (commit `be2e7f112`): after installing the wrapper, walk `sys.modules` and
rebind every module whose `fsdp2_prepare_model` is the original. Verified in
isolation that `accelerate.accelerator.fsdp2_prepare_model` then resolves to
the wrapper; confirmed torch 2.12.0+xpu exposes
`FSDPModule.set_force_sum_reduction_for_comms`.

## Confirmation run

**Job 12470083 (3N: 1 server node + 2 trainer nodes = 24 trainer ranks), HEAD
`407657fca`, both fixes active. RESULT: multi-trainer-node GRPO steps.**

Sequence, all clean:
1. Server healthy (60 s, warm cache).
2. `forced ReduceOp.SUM+divide on 17 FSDP2 modules` printed by **all 24 ranks**
   (16 decoder blocks + root) -- the AVG->SUM patch now engages.
3. Cross-node weight-sync: **111 `update_named_param`** broadcasts to the vLLM
   server (the exact op that SIGSEGV'd on the `mpi` transport) -- no crash.
4. Generation: `/generate/` 200 OK -- past the `gather_object` collective the
   overnight doc feared was a silent XPU deadlock (also refuted).
5. FSDP2 backward: step 0's cross-node grad **`reduce_scatter_tensor`** is
   slow (~3 min) and tripped the 120 s faulthandler safety net -- but the
   all-rank dump showed **24/24 ranks at the identical `reduce_scatter_tensor`
   line**. That is the definitive desync test: all ranks in the SAME collective
   = a slow collective, NOT a desync. It completed.
6. **First step landed:**

```
{'loss': '-0.02836', 'grad_norm': '17.24', 'learning_rate': '1e-06',
 'num_tokens': '2132', 'completions/mean_length': '64',
 'rewards/accuracy_reward/mean': '0.3333', 'rewards/accuracy_reward/std': '0.4815',
 'rewards/format_reward/mean': '0.1667'}
```

Real `accuracy_reward` (0.333) and `format_reward` (0.167) on the arithmetic
task, across 2 trainer nodes. The multi-trainer-node blocker is cleared.

The all-rank `reduce_scatter_tensor` stack (24/24 identical) is the evidence the
overnight session lacked -- it directly **refutes the desync hypothesis** that
`grpo-on-xpu-status.md` had converged on.

**Full run: all 8/8 steps completed cleanly** (`Training complete.`,
`train_runtime` 1411 s, `train_loss` -0.02445, ~176 s/step, AVG_wall=0,
segfault=0):

| step | loss | grad_norm | LR | accuracy_reward | format_reward |
|---|---|---|---|---|---|
| 1 | -0.028 | 17.24 | 1.00e-6 | 0.333 | 0.167 |
| 2 | -0.113 | 12.62 | 8.75e-7 | 0.333 | 0.25 |
| 3 | -0.056 | 30.94 | 7.50e-7 | 0.208 | - |
| 4 | -0.133 | 12.16 | 6.25e-7 | 0.25 | - |
| 5 | 0.047 | 10.60 | 5.00e-7 | 0.208 | - |
| 6 | -0.044 | 14.29 | 3.75e-7 | 0.333 | - |
| 7 | 0.106 | 24.18 | 2.50e-7 | 0.25 | - |
| 8 | 0.025 | 19.47 | 1.25e-7 | 0.375 | - |

`accuracy_reward` ends at 0.375 (step 8); the signal is noisy at
24 ranks x 4 generations = 96 gens/step over only 8 steps, but the path is
live and the run is stable. Steps are ~176 s each -- slow, because everything
(trainer intra-mesh FSDP collectives + the cross-world weight-sync) is on the
TCP fabric, not Slingshot CXI. Per-group transport tuning (CXI for the trainer
mesh, TCP-KVS only for the server group) is the obvious next perf lever.

Teardown note: after `Training complete.` the trainer lingers in atexit/wandb
sync + communicator close before the launcher returns the
`GRPO_3N_VALIDATE_VERDICT` line; training itself is done at step 8.

## Why this matters

- The multi-trainer-node blocker was **two ordinary bugs** (a transport-default
  regression + a monkeypatch-scoping bug), not a novel XPU object-collective
  pathology. The `grpo-on-xpu-status.md` "catch-22" / "desync" framing is
  superseded by this page.
- The correct multi-trainer-node recipe is: **`ofi`/TCP-KVS transport ON**
  (forms the cross-world weight-sync PG without `atl_mpi`) **+ the AVG->SUM
  patch** (avoids the oneCCL scheduler-path AVG gap in FSDP backward). Both are
  now in `xpu_overrides.py` / the shared xnode script defaults.
- Method note: the answer was already on disk. Re-reading the overnight
  faulthandler dumps (Phase 1 of systematic debugging) beat spending a fresh
  py-spy job to regenerate the same evidence.

## Artifacts

- Script: `rl/scripts/grpo/grpo_3n_multinode_validate.sh`
- Fix: `rl/xpu_overrides.py` `patch_fsdp2_force_sum_reduction_for_xpu()` (commit
  `be2e7f112`)
- Overnight evidence: `logs/grpo-3n-cxi-test-12470009/` (stacks + trainer.log),
  `logs/grpo-2n-xnode-smoke-12469976/` (working 2N baseline)
