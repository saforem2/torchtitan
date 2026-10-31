# AuroraGPT-2B: replicated TT versus MDS ZeRO-0

## Status

The `24`-rank FP32 accumulation probe passed. Full-model comparison is next.
No production-scale parity result yet.

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
| Precision | FP32 optimizer masters; BF16 compute; gradient accumulation and wire dtype isolated below |
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

The pinned `DeepSpeed 0.18.5` ZeRO-0/no-pipeline path defaults to BF16
gradient accumulation and BF16 communication. The Megatron flag
`accumulate_allreduce_grads_in_fp32=True` does not establish wire precision:
the DeepSpeed branch bypasses Megatron's LocalDDP wrapper. Runtime engine
properties and observed collective tensor dtypes must confirm this.

| Path | Local gradient accumulation | Collective payload | Purpose |
|---|---|---|---|
| `MDS production` | BF16, expected from source | BF16, expected from source | Reproduce the pinned baseline |
| `MDS fp32_wire` | BF16 | FP32 | Isolate communication precision |
| `TT native DDP` | FP32 | FP32 | Preserve the current TT precision policy |
| `TT native DDP BF16 wire` | FP32 | BF16 | Isolate communication precision with FP32 masters/accumulation |

The TT BF16 path uses PyTorch's built-in `bf16_compress_hook`. It quantizes
the collective buffer and copies the result back to FP32 gradients. It does
not establish numerical equivalence with FP32 communication or BF16 local
accumulation. A dedicated probe compares it with an explicit cast/divide/
all-reduce reference. The pinned DeepSpeed ZeRO-0 path rejects BF16 model
parameters with FP32 gradient accumulation; do not request that combination
through a JSON setting and assume it works.

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

| Job | Source | Purpose | Result |
|---|---|---|---|
| `12481844` | `656a174b6` | `24`-rank accumulation probe | `PBS F`, exit `1`; stale console-script interpreter, before MPI launch |
| `12481845` | `6dade13b6` | Corrected `24`-rank accumulation probe | `PBS F`, exit `0`; all rank checks passed |

`12481845`: all `24` rank files are present. The first microbatch was
unsynchronized; final gradients were bitwise identical across ranks.
Accumulated versus combined-batch maximum gradient error was
`7.450580596923828e-9`; global loss error was `0`. The check used the explicit
`atol=1e-6`, `rtol=1e-5` gradient criterion. It proves FP32 DDP accumulation
and token normalization for the probe, not full AGPT BF16 numerical parity.
Online [run](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/agpt-replicated-probe-12481845);
artifact `agpt-replicated-accumulation-12481845:v0`, `COMMITTED`, `24` files.

`12481844` passed parser/API and repository preflight. On its compute node,
`torch 2.14.0+xpu` enumerated `12` FLAT devices. The installed `ezpz` console
script retained the older `/venvs/xpu-torch214/bin/python3` shebang and failed
to import `click.core`. This is a launcher failure, with no gradient result.
The fix rebinds a job-local copy of the installed console script to the
selected interpreter, runs `ezpz launch --help`, and leaves the borrowed venv
immutable.

Campaign source: `codex/agpt-replicated-20261031`, based on `5be47f7c`.
Sunspot workdir:
`/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan-agpt-replicated-20261031`.
Runtime:
`/lus/tegu/projects/datascience/foremans/venvs/xpu-torch214-fixedlr-v2-20260927`.
`MDS` base `b5617bac`, XPU-profiler commit
`9f407ed9e8e30920e28272daf8cb7db36c211e21`.
Native checkpoint SHA-256:
`e37d9ca2c9cb396a3207edf6132815e7a6ad32c247ebe5043782c54a20a69e1c`.
