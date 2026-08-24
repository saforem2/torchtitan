# GRPO on Intel XPU: TRL `GRPOTrainer`

On-policy GRPO via **TRL `GRPOTrainer`** on Sunspot XPU (no Monarch). Two
generation backends: **`trl vllm-serve`** (recommended, on-policy) with a
**`.generate()` per-rank** fallback. For the Monarch + TorchStore + vLLM path
see [`monarch.md`](monarch.md); overview in [`README.md`](README.md).

**Current status only** -- the bring-up chronology and the (refuted) 2026-07-01
desync investigation live in [`history/`](history/README.md).

## Generation backends (3 tiers)

| Tier | What | Status |
|---|---|---|
| `hf.generate` per-rank | Every trainer rank generates with HF transformers `.generate()`. Slow, but always works on XPU. The fallback. | baseline |
| **vLLM-server, 1 trainer node** | `trl vllm-serve` on a head node; a single trainer node hits it over HTTP; on-policy weight-sync over XCCL. | works |
| **vLLM-server, 2+ trainer nodes** | Same, but the trainer's FSDP2 spans multiple nodes (the grad reduce-scatter is the hard part). | works (2026-07-06) |

All results below are the **vLLM-server** path (`--use_vllm --vllm_mode
server`), not `hf.generate`.

## Current status (2026-07-06)

| Config | Status | Evidence |
|---|---|---|
| **1 node** (server + trainer on `127.0.0.1`) | works | job 12468780 -- 5/5 steps, real weight-sync |
| **Cross-node generation** (server node A, **1** trainer node B) | works | job 12469976 -- 10/10 steps, cross-node weight-sync, accuracy reward moving |
| **Multi-trainer-node** (trainer spans 2+ nodes) | works | job 12470083 (3N = 1 server + 2 trainer, 24 ranks) -- 8/8 steps, loss -0.028, accuracy_reward -> 0.375, real cross-node FSDP grad reduce-scatter |

The multi-trainer-node blocker (open through 2026-07-01) was root-caused and
fixed 2026-07-06 -- **two ordinary bugs, not the "desync" earlier concluded:**
(1) a transport-default regression (`--no-oneccl-tcp-kvs` left
`CCL_ATL_TRANSPORT` at oneCCL's `mpi` default, which SIGSEGVs forming the
cross-world weight-sync PG in `atl_mpi::create_comm_id`); and (2) the AVG->SUM
FSDP patch never engaging (rebound the wrong `from`-import name). Full evidence
+ the all-rank `reduce_scatter_tensor` stack that refutes the desync theory:
[`2026-07-06_multinode-grpo-root-cause.md`](2026-07-06_multinode-grpo-root-cause.md).

## How to run it

**Scripts:**
- 1N smoke: [`rl/scripts/grpo/qwen3_vllm_server_smoke.sh`](../../../../rl/scripts/grpo/qwen3_vllm_server_smoke.sh)
- cross-node (1 trainer node): [`rl/scripts/grpo/aurora2b_sft_arithmetic_vllm_xnode.sh`](../../../../rl/scripts/grpo/aurora2b_sft_arithmetic_vllm_xnode.sh)
- multi-trainer-node (2+): [`rl/scripts/grpo/grpo_3n_multinode_validate.sh`](../../../../rl/scripts/grpo/grpo_3n_multinode_validate.sh)
- venv build: [`rl/scripts/build_rl_vllm_venv.sh`](../../../../rl/scripts/build_rl_vllm_venv.sh)

**What makes the cross-node path work (all landed):**
1. **Unified `venvs/rl-vllm/` for BOTH server and trainer.** The older
   `vllm_serve_xpu.sh` `venvs/vllm-test` + `PYTHONPATH=.venv` bridge (removed
   2026-07-17) broke vLLM-XPU platform detection (`RuntimeError: Device string
   must not be empty`).
2. **`--fsdp` for TRL 1.6 / transformers >= 5.11:** string parsing was dropped;
   `--fsdp full_shard` now parses to bare `True`, handled in `train_grpo.py`
   `_bootstrap_fsdp_env` (`0558eb592`: `fsdp is True` -> `full_shard`).
3. **Server launches as a plain local subshell** (NOT `mpiexec`-wrapped), with
   PMIx/CXI env scrubbed inside its subshell; vLLM stack checks run from a `.py`
   file with an `if __name__ == "__main__":` guard (vLLM's multiprocessing
   EngineCore re-imports the parent).
4. **Weight-sync uses TRL's own `StatelessProcessGroup`** on a dedicated
   host:port (`trl/scripts/vllm_serve.py:111`, XPU-aware) -- independent of the
   trainer's oneCCL transport.

**Extra requirements for multi-trainer-node (2+):**
5. **`ofi`/TCP-KVS transport ON** -- do NOT pass `--no-oneccl-tcp-kvs`. Leaving
   `CCL_ATL_TRANSPORT` unset selects oneCCL's `mpi` default, which SIGSEGVs
   forming the trainer-rank0<->server group across two mpiexec worlds.
6. **AVG->SUM FSDP patch** -- `patch_fsdp2_force_sum_reduction_for_xpu()`
   (auto-applied by `apply_all_xpu_patches`) converts the FSDP2 grad
   reduce-scatter from `ReduceOp.AVG` (no oneCCL scheduler-path kernel) to
   `SUM`+divide.

## Stack

| Component | Pin | Notes |
|---|---|---|
| Python | **3.12.12** | The only Python where `torchmonarch` (cp310-cp313) AND `triton-xpu==3.7.1` (cp312-cp314) both ship native wheels. |
| `torch` | `2.12.0+xpu` | From PyTorch XPU wheel index. `vllm-xpu-kernels` only links against 2.12. |
| `triton-xpu` | `3.7.1` | From PyTorch XPU index (NOT vanilla `triton` from PyPI -- missing Intel symbols). |
| `vllm` | `0.22.1` | `--no-deps` install to prevent vanilla triton via `xgrammar`. |
| `vllm-xpu-kernels` | `0.1.9.1` | `cp38-abi3` wheel from `vllm-project/vllm-xpu-kernels` GitHub release. |
| `trl` | `1.6.0` | First TRL with `vllm_mode="server"` + per-arg server URL. |
| `transformers` | `5.11.0` | |
| `accelerate` | `1.14.0` | |
| `torchmonarch` | `0.5.0` | Not used for GRPO; kept for a possible Monarch+TorchStore revisit. |
| `mpi4py` | `4.1.2` | Required by `ezpz.distributed`. |
| `omegaconf`, `hydra-core` | latest | Required by ezpz trainer modules. |
| `wandb`, `tensorboard`, `tyro`, `spmd-types`, `torchdata`, `renderers @ git+PrimeIntellect-ai` | latest | Upstream `torchtitan.experiments.rl` transitive deps. |

**Removed from the venv (CRITICAL):** `impi-rt`, `oneccl`, `oneccl-devel` --
torch's XPU wheel pulls them, but they install in-venv `libccl.so` / `libmpi*.so`
that shadow the system `/opt/aurora/.../oneapi/ccl` stack (in-venv oneCCL doesn't
know Sunspot's USM allocator). After uninstall, `ldd .../libtorch_xpu.so | grep
ccl` correctly resolves to the system path. Reproducible via
[`rl/scripts/build_rl_vllm_venv.sh`](../../../../rl/scripts/build_rl_vllm_venv.sh)
(commit `b43acb8b2`).

## Operational details

- **Why 8 trainer ranks (not 11) in the 1N smoke.** The Qwen3 GRPO config sets
  `num_generations=4`; TRL requires `trainer_world_size * per_device_batch_size`
  divisible by `num_generations`. With bsz=1, `trainer_world_size` must be a
  multiple of 4; 12 tiles - 1 server tile = 11 -> round down to 8.
- **Tile partitioning.** `ZE_AFFINITY_MASK` pins the server subshell to tile 0;
  the trainer launcher exports `ZE_AFFINITY_MASK=1..8` so its ranks see tiles
  1-8 (re-indexed `xpu:0..7` per rank).
- **Loopback proxy bypass.** `http_proxy` is global (for HF/W&B); the script sets
  `no_proxy=127.0.0.1,localhost` so self-loop curls (`/health/`, `/generate/`)
  bypass the proxy.
- **Perf caveat.** The trainer's intra-group XCCL currently runs on TCP-KVS, not
  Slingshot CXI -- slower than the production-trainer path. Production scaling
  wants a per-group transport split (CXI intra-node + TCP-KVS only for the
  cross-process server group). vLLM-XPU TP>1 (multi-tile server) is unexercised.

## What this does NOT solve

1. ~~**The Monarch+oneCCL architecture mismatch** -- still blocked~~
   **RESOLVED 2026-07-19** (this line was written 2026-07-06 and went stale;
   corrected 2026-08-16). The Monarch path was verified working -- see
   [POST-TRAINING-2B](../agpt/2b/post-training.md), which records MONARCH as the
   working GRPO path. `xpu_overrides` alone was indeed not sufficient, which is
   what this bullet got right; the `spawn_procs` vs PMIx issue beneath it was
   subsequently solved. Historical context:
   [`history/upstream-rl-port-status.md`](history/upstream-rl-port-status.md).
2. **Production-grade XCCL performance** for the trainer's intra-mesh group
   (TCP fabric everywhere; CXI would be faster -- see perf caveat).
3. **vLLM-XPU TP>1** (multi-tile server) -- unexercised by this work.

## History

The full bring-up chronology (26-job debug chain, TCP-KVS fix, `xpu_overrides`
shim tour, 1N smoke metrics) and the superseded 2026-07-01 desync investigation
are archived under [`history/`](history/README.md):
- [`history/2026-06-13-bringup-and-2026-07-01-desync.md`](history/2026-06-13-bringup-and-2026-07-01-desync.md)
  -- how the port was won + the desync dead-end (refuted 2026-07-06).
- Plus the earlier vLLM-XPU investigation, wiring plan, and Monarch deep-dive
  (see the [history index](history/README.md)).
