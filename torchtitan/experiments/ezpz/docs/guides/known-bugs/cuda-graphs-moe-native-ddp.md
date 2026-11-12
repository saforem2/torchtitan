# CUDA: default CUDA graph capture breaks ezpz MoE and native DDP (2026-11-12)

Status: **fixed** in `bae44ff5` (ezpz trainer drops to eager with a warning).
Affects CUDA systems (Polaris, Sirius) only; XPU never takes the capture path.

## Symptom

Since the 2026-10-15 sync (`5a7a622f16`), upstream's training engine captures
forward/backward in a CUDA graph by default (`training.disable_cuda_graphs`
defaults to `False`; `torchtitan/distributed/cuda_graph.py`). Capture starts
after `NUM_CUDA_GRAPH_WARMUP_STEPS`, so a run trains eagerly for the warm-up
steps and then dies.

Polaris job `7839433`, 2 nodes, 8 ranks, torch `2.15.0+cu132`, tree `ezpz` +
#77 #74 #76 #78:

| Config | Result |
|---|---|
| `agpt_debugmodel`, FSDP, TP=1 | passes 6 steps under capture (`10.880` -> `10.107`) |
| `moe_debugmodel` | steps 1-2, then `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned` |
| `agpt_debugmodel`, native DDP + ZeRO-1 | `CUDA error: operation not permitted when stream is capturing` |

## Where

- MoE: `experiments/ezpz/moe/experts.py` `_run_experts_for_loop`, reached from
  `routed_experts.py` `_run_backend`. The for-loop backend moves per-expert
  token counts to the host. Upstream's own docstring says only HybridEP with
  `non_blocking_capacity_factor` is capture-safe.
- Native DDP: `torch.nn.parallel.DistributedDataParallel._pre_forward`, called
  from `experiments/ezpz/trainer.py` `_non_pp_forward_backward_microbatch`.

## Fix

`trainer.py` `maybe_disable_cuda_graphs`, called from
`_initialize_forward_backward`. On CUDA with graphs enabled it sets
`training.disable_cuda_graphs=True` and logs

```
Disabling CUDA graphs: <reason> cannot run under capture. Pass --training.disable-cuda-graphs to silence this.
```

when native DDP is on or any MoE layer uses an expert backend other than
`grouped_mm` (`for_loop` is the automatic fallback below SM90, so every ezpz
MoE run on Polaris A100s). It raises instead if the config explicitly asks
for `optim.enable_cuda_graph` or `training.cuda_graph_per_accumulation_group`.
Dense FSDP and `grouped_mm` MoE keep capture. Unit tests:
`tests/test_cuda_graph_guard.py`.

Polaris job `7839729`, same tree plus the fix, CUDA graphs left on
(`SMOKE_CUDA_GRAPHS=1`):

| Config | Guard | Result |
|---|---|---|
| `agpt_debugmodel`, FSDP | no warning, captured | `10.880` -> `10.107` |
| `moe_debugmodel` | `MoE expert backend ['for_loop']` | `12.990` -> `11.069`, 6 steps |
| native DDP + ZeRO-1 resume | `native DDP` | `RESUME_OK`, diff `0.0041` |

Losses match the eager run `7839481`.

`--training.disable-cuda-graphs` still forces eager everywhere.
