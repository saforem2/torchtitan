# AuroraGPT-2B: replicated TT versus MDS ZeRO-0

## Status

Preparing a new controlled campaign. No production-scale parity result yet.

The original Aurora production reference uses `MDS ZeRO-0` on `256` nodes,
`3072` FLAT devices, sequence `8192`, microbatch `1`, accumulation `2`, and
`SophiaG`. Its median logged throughput is `4787.15 tokens/s/device` over
`0.5-3.5T` tokens. The corresponding `TT-256` chain logs `3144.69` with global
`FSDP`. These histories describe the gap; they do not isolate its cause.
See the [full-history audit](2026-10-29-native-checkpoint-closure.md#production-scale-reference-aurora-not-polaris).

## Controlled configuration

| Setting | Contract |
|---|---|
| Model | `1,986,054,144` parameters, native `MDS global_step138650` weights |
| Precision | FP32 master parameters; BF16 compute; FP32 gradient reductions |
| Attention | Fused SDPA/Flash; confirm dispatch from traces |
| Geometry | `12` FLAT devices/node; microbatch `1`; accumulation `2` |
| Global batch | `2 * 12 * nodes`; `6144` at `256` nodes |
| Optimizer | Fresh `SophiaG`; betas `(0.9, 0.95)`, `rho=0.01`, weight decay `0.1` |
| LR | Constant `2e-6`, zero warmup; identical across fresh benchmark arms |
| Hessian policy | Reproduce pinned MDS: no Hessian update; fresh buffers remain zero |
| Rematerialization | Disabled |
| Data | Identical Sunspot `books` list, split `95,5,0`, seed `42` |
| W&B | Online; group `agpt-replicated-20261031`; preserve all evidence |
| Timing | `32` updates; unprofiled `21-32`; profile `12` separately |

The historical production LR is `2.17e-5`; this fresh-optimizer benchmark uses
the previously validated lower LR. It is a throughput comparison from common
weights, not a continuation of either production optimizer state.

`TT` native DDP currently uses BF16 autocast over FP32 storage; this leaves
embedding, norm and residual activations in FP32. `FSDP` and `MDS` can have
different intermediate dtypes. Record the actual policy rather than claiming
identical arithmetic from the dtype flags. The earlier native-weight audit
established the checkpoint representation mapping; input equality must be
audited again for the new global batches.

## Gates and arms

1. Run a two-node native-DDP probe with seed `42` and strict deterministic
   algorithms. Require unsynchronized first-microbatch gradients, bitwise
   agreement across ranks after the final synchronization, and accumulation
   agreement with the combined-batch reference at `atol=1e-6`, `rtol=1e-5`.
   This FP32 probe checks synchronization and token normalization, not full
   model BF16 numerical equivalence.
2. Run `TT` native DDP eager and loss compilation versus fresh `MDS ZeRO-0`.
   Verify `32` actual optimizer updates, checkpoint output, all-rank memory,
   complete batch hashes, loss/gradient history, dtype and fused-attention evidence.
3. Add the global `FSDP` control on the same workload. Isolate the layout
   without changing optimizer, microbatch or data.
4. Scale passing paths through `8`, `32` and `256` nodes on Aurora. A working
   Aurora runtime must pass the same parser, import and compute gates first.

Report steady-state step time, compilation cost, peak memory, exposed
collective time, tokens/sec/device, common model TFLOP/s and MFU. Keep the
fixed FLOP convention from the native-checkpoint report. Loss agreement is a
separate measurement; no performance result establishes numerical equivalence.

## Implementation

`NativeDDP.set_requires_all_reduce()` now honors the engine's accumulation
hook before each forward/backward. Only the final backward synchronizes.
The loss already uses the global token count across accumulated microbatches;
the existing DP-degree compensation cancels DDP gradient averaging. There is
no additional division by the accumulation count.

`SophiaGOptimizersContainer.Config.update_hessian_before_step` defaults to
`True`. The benchmark explicitly sets it to `False`. The checked Aurora
production and Sunspot pinned MDS `sophia.py` files have identical SHA-256
`1a3ff32890dba21ce538f96af5d59785f671ad9eafc9d42d6cfd317532b6ab3e`;
neither source tree has a caller of `update_hessian()`. Existing production
checkpoint state may contain nonzero Hessian values; the fresh benchmark
does not infer that state from source inspection.

The `MDS` XPU profiler patch only adds `ProfilerActivity.XPU`. Pin and archive
its separate source commit. Require optimizer state evidence because MDS
can catch optimizer-step exceptions without making the process exit nonzero.

## Jobs and evidence

Pending preflight and submission. Job IDs, immutable source commits, runtime
paths, exact commands, terminal status, artifact hashes and results go here.
