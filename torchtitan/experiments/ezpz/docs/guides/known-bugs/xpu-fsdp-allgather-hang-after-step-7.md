# XPU: intermittent FSDP all-gather hang after step 7 (2026-11-11)

Status: **open**. Not caused by ezpz code. Do not "fix" by unsetting
`CCL_OP_SYNC`; see below.

## Symptom

`agpt_debugmodel`, TP=1, 2 Sunspot nodes (24 ranks), venv
`xpu-torch216-sync-5a7a622f16` (torch `2.16.0.dev20261015+xpu`, oneAPI
`2026.1.0`, MPICH `5.0.0.aurora_test`). Steps 1-7 log normally, then the run
stops with no error. MPICH prints `3 pending requests in pool 0` and
`PMPI_Wait` backtraces under `libccl` every few minutes until walltime.

Reproduces on unmodified `ezpz` (`b9dd4eb7`), on different node pairs, with
and without W&B. Not a bad node: job 12483306 passed and 12483314 hung on the
same two nodes.

## Where it hangs

py-spy on all 24 ranks during a hang (job 12483329): **every rank** is in the
same call, the FSDP parameter all-gather in the step-8 forward:

```
all_gather_single (torch/distributed/distributed_c10d.py:5487)
DefaultAllGather.__call__ (fsdp/_fully_shard/_fsdp_collectives.py:115)
foreach_all_gather -> FSDPParamGroup.unshard -> pre_forward
Decoder.forward (torchtitan/models/common/decoder.py:254)
```

No rank is elsewhere, so this is not a rank desync in Python; the collective
is entered by all ranks and never completes inside oneCCL.

## Runtime A/B (job 12483330, 4 ten-step runs each)

| variant | hung |
|---|---|
| default (`CCL_OP_SYNC=1`, MPI transport) | 1/4 |
| `CCL_ATL_TRANSPORT=ofi` | 4/4 (two before step 1) |
| `CCL_OP_SYNC` unset | 0/4 |

Across all default-env runs that day: 5 of 10 hung.

`CCL_OP_SYNC` unset cleared it here, but
[moe-ep-a2a-degrades-with-size.md](moe-ep-a2a-degrades-with-size.md) measured
the opposite for MoE expert parallelism: unset deadlocks `moe_2b_ep` before
step 1 (job 12473471). Production uses `CCL_OP_SYNC=1`, so the smoke keeps
it. 0/4 is also weak evidence at a ~40% hang rate (13% chance by luck).

## What the smoke does about it

`sync_smoke_alcf.sh` runs 6 steps per config (below the hang point) and puts
every config under a timeout (`SMOKE_CONFIG_TIMEOUT`, default 900 s), so a
hang shows up as `rc=124 (timed out after N steps)` instead of eating the
job.

## Next

- More samples of `CCL_OP_SYNC` unset on dense configs, and a check on
  whether `CCL_OP_SYNC=1` is still needed for MoE EP on the current stack.
- Whether the step number is meaningful (always the step-8 forward so far).
