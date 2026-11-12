# CUDA: default CUDA graph capture breaks ezpz MoE and native DDP (2026-11-12)

Status: **open**. Affects CUDA systems (Polaris, Sirius) only; XPU never takes
the capture path.

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

## Workaround

`--training.disable-cuda-graphs`. `scripts/sync_smoke_alcf.sh` passes it on
CUDA unless `SMOKE_CUDA_GRAPHS=1`.

## Fix options

- Validate at startup: reject (or auto-disable with a warning) CUDA graphs when
  native DDP or a host-syncing MoE backend is configured.
- Default `disable_cuda_graphs=True` in the ezpz config registries.

Neither is done. Check what Polaris/Sirius production launchers pass before
choosing.
