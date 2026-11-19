# 20B 512N cannot save: DCP save-plan scatter_object_list segfaults at 6144 ranks

> Status: open (2026-11-19). Handoff write-up; issues are disabled on this repo.

## Problem

The 20B 512N chain (`agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9000-fixedpos`) trains but cannot save. Every checkpoint save at 6144 ranks segfaults inside DCP's save-plan exchange, nothing is written, and the head has been `step-10800` since mid-November.

| Job | Save | Outcome |
|---|---|---|
| `9076330` (2026-11-14/15) | `step-10900` | 4x segfault |
| umbrella `9094039` (2026-11-16/17) | `step-10900` | 2x segfault, ~40 min after `Saving the checkpoint` |
| per-chain `9093715` (2026-11-18/19, 12h) | `step-10810` | 5 attempts, each `signal 11` in `dcp_save`; `step-10810/` exists without `.metadata` |

Stack from `9093715` attempt 4 (48 `Fatal Python error: Segmentation fault` in one log):

```
torch/distributed/distributed_c10d.py:4778 in scatter
torch/distributed/distributed_c10d.py:4169 in scatter_object_list
torch/distributed/checkpoint/utils.py:162 in scatter_object
torch/distributed/checkpoint/utils.py:219 in reduce_scatter
torch/distributed/checkpoint/state_dict_saver.py:487 in _save_state_dict
torch/distributed/checkpoint/state_dict_saver.py:196 in save
torchtitan/components/checkpoint.py:496 in dcp_save
torchtitan/components/checkpoint.py:608 in save
torchtitan/experiments/ezpz/trainer.py:709 in train
...
x4118c6s3b0n0: rank 26 died from signal 11
```

So the crash is in the coordinator scattering per-rank save plans back to 6144 ranks (`scatter_object_list` over the default oneCCL/MPICH group). Earlier logs also show MPICH `PMPI_Wait` / oneCCL frames. The chain's `.metadata` is ~1.25 GB, so the plans being pickled and scattered are large. The September stack saved the same plan successfully.

## Where

- Checkout: `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-20b512-fixedpos-20261029` (pinned `dc6c150`), `frameworks/2026.1.0` (torch `2.13.0a0`).
- Logs: `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/logs/multi-autoretry-9093715/trainer-0-20b-n512-attempt-*.console.log` (segfault at ~line 136012 of attempt 4).
- Chain page: `docs/production/agpt/20b/n512/README.md`; umbrella notes in `docs/production/2026-11-16-umbrella-9094039.md`.

## Reproducer

Full repro is the production save (6144 ranks, ~40 min to crash). A cheaper isolating repro:

1. Standalone script under `ezpz launch`: init the same backend (xccl, `CCL_OP_SYNC=1`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`), build a list of per-rank Python objects of size S on rank 0, call `dist.scatter_object_list` to all ranks. Sweep S (10 MB, 100 MB, 500 MB, 1.25 GB total) and N (64N, 128N, 256N, 512N). Expect a segfault above some size x ranks. Run on `debug-scaling` up to 256N first.
2. Repeat the failing point with a gloo group (`dist.new_group(backend="gloo")`) passed as `group=` to confirm the crash is in the xccl/MPICH path.
3. Optional: DCP save of a dummy state dict with many small tensors per rank (to inflate the plan) at the failing scale.

## Suggested plan

1. **Run the isolating repro** to find the failing size/scale and whether gloo avoids it.
2. **Route DCP planning through gloo:** pass `process_group=<gloo group>` to `dcp.save` via an ezpz-side override in `experiments/ezpz/` (core must not change). The plan exchange is CPU objects, so gloo is the natural backend; tensor writes are local file I/O and do not depend on the group. Caveat: gloo at 6144 ranks has failed before, in a `Connection closed by peer` cascade under async-save write pressure ([known-issues.md](../known-issues.md), "`--checkpoint.async-mode=async` kills the cluster at 20B 512N+"). A synchronous planning-only exchange is a different load, but test it at 512N before relying on it.
3. **Shrink the plan:** check whether this torch has `DefaultSavePlanner(enable_plan_caching=True)` (caches the global plan after the first save so later saves skip the exchange) and whether dedup can be relaxed. With the plan cached, only the first save pays the exchange.
4. **Do not switch to async save:** `checkpoint.async_mode=async` is known to cascade to failure at 20B 512N+ on XPU; sync (`CHECKPOINT_ASYNC_MODE=disabled`) is the production default.
5. **If torch-side:** file upstream with the standalone repro (scatter_object_list segfault at N ranks / S bytes on xccl).
6. Validate at 512N: the chain saves `step-10900`, then reloads it and trains (`resume` check).

Done when: 20B-512 writes a complete `step-10900` with `.metadata` at 6144 ranks and resumes from it.
