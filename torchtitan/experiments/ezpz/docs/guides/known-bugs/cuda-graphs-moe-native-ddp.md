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

Upstream #5353 (2026-11-12 sync) made capture backend-agnostic and renamed the
fields: `training.disable_cuda_graphs` -> `training.disable_accelerator_graphs`,
`optim.enable_cuda_graph` -> `optim.enable_accelerator_graph`,
`training.cuda_graph_per_accumulation_group` ->
`training.accelerator_graph_per_microbatch_group`. The guard was renamed with
them; the jobs below ran the CUDA-named version.

`trainer.py` `maybe_disable_accelerator_graphs`, called from
`_initialize_forward_backward`. With graphs enabled it sets
`training.disable_accelerator_graphs=True` and logs

```
Disabling accelerator graphs: <reasons>. Pass --training.disable-accelerator-graphs to silence this.
```

when native DDP is on, any MoE layer uses an expert backend other than
`grouped_mm` (`for_loop` is the automatic fallback below SM90, so every ezpz
MoE run on Polaris A100s), or the device is XPU (ezpz XPU graphs stay opt-in
through `EZPZ_XPU_GRAPHS=1`). It raises instead if the config explicitly asks
for `optim.enable_accelerator_graph` or
`training.accelerator_graph_per_microbatch_group`. Dense FSDP and `grouped_mm`
MoE keep capture. Unit tests: `tests/test_accelerator_graph_guard.py`.

Polaris job `7839729`, same tree plus the fix, CUDA graphs left on
(`SMOKE_CUDA_GRAPHS=1`):

| Config | Guard | Result |
|---|---|---|
| `agpt_debugmodel`, FSDP | no warning, captured | `10.880` -> `10.107` |
| `moe_debugmodel` | `MoE expert backend ['for_loop']` | `12.990` -> `11.069`, 6 steps |
| native DDP + ZeRO-1 resume | `native DDP` | `RESUME_OK`, diff `0.0041` |

Losses match the eager run `7839481`.

`--training.disable-accelerator-graphs` still forces eager everywhere.
