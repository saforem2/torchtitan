# AuroraGPT-2B: replicated TT versus MDS ZeRO-0

## Status

The continued investigation has successful full-model runs on `16` and `24`
ranks. The initial replicated-state OOMs are real, but the initial `MDS`
configuration selected a different optimizer wrapper from production.
`DeepSpeed 0.18.5` can reproduce the older `BF16_Optimizer` behavior through
an explicit stage-1 configuration. The report's initial failed-arm results
below are retained as chronology.

The [November 1 audit](2026-11-01-handoff-audit.md) supersedes the earlier
interpretations and contains aligned timing, `TFLOP/s`, `MFU`, input-hash
coverage, precision results and unresolved gates. At `24` ranks, the recovered
window means are `2.201 s/update` for `MDS ZeRO-1`, `2.704 s` for `TT HSDP`
with BF16 reduction, and `2.713 s` with FP32 reduction. These are performance
results; full-model numerical equivalence and production-scale parity remain
unproven.

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
| Historical `MDS`, DS `0.17.5` | FP32 | BF16 | `BF16_Optimizer`; masters/state sharded despite nominal stage 0 |
| Fresh `MDS`, DS `0.18.5`, stage 0 | BF16 | BF16 | Different replicated optimizer-state path; OOM in these arms |
| `MDS ds0175` compatibility | FP32 | BF16 | DS `0.18.5` stage-1 JSON selects `BF16_Optimizer`; not an identical old runtime |
| `MDS fp32_wire` | BF16 | FP32 | Isolate communication precision |
| `TT native DDP` | FP32 | FP32 | Preserve the current TT precision policy |
| `TT native DDP BF16 wire` | FP32 | BF16 | Isolate communication precision with FP32 masters/accumulation |

The TT BF16 path uses PyTorch's built-in `bf16_compress_hook`. It quantizes
the collective buffer and copies the result back to FP32 gradients. It does
not establish numerical equivalence with FP32 communication or BF16 local
accumulation. A dedicated probe compares it with an explicit cast/divide/
all-reduce reference. The pinned DeepSpeed ZeRO-0 path rejects BF16 model
parameters with FP32 gradient accumulation; do not request that combination
through a stage-0 JSON setting and assume it works. An explicit stage-1
JSON with FP32 accumulation selects `BF16_Optimizer`; see the audited
compatibility run `12481887`.

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
| `9013231` (Aurora) | `24985bc50` | `24`-rank accumulation probe | exit `0`; all rank checks passed, max grad error `7.45e-9` |
| `12481846` | `d8fa48e14` | `TT` DDP, loss compile | exit `143`; DCP load missing `module.`-prefixed keys |
| `12481847` | `d8fa48e14` | DDP BF16-wire probe | exit `0` |
| `12481848` | `d8fa48e14` | `MDS ZeRO-0` | failed before training; private DS template absent from clone |
| `12481849` | `d8fa48e14` | `MDS ZeRO-0` retry | walltime kill (`-29`); training raised on all ranks, then diagnostics blocked on device work, so no error printed |
| `12481850` | `e566ba1bd` | `TT` DDP, loss compile | exit `143`; `inplace_wgrad_accum=True` rejected under autocast |
| `12481851` | `545d06f10` | `TT` DDP, loss compile | exit `137`; updates `1-2`, OOM at update `3` |
| `12481853` | `7586def1e` | `TT` DDP, memory snapshots | OOM at update `3`, same as `12481851`; snapshots for updates `1-2` |
| `12481854` | `7586def1e` | `TT` DDP, `AC=full` | exit `0`; `32` updates, peak `51.57 GiB` |
| `12481855` | `7586def1e` | `MDS ZeRO-0` | exit `143`; `UR_RESULT_ERROR_OUT_OF_RESOURCES` in first forward on `28` ranks |

### Fixes

- `cf18ed45d`: point the checkpointer's model state at the unwrapped modules.
- `545d06f10`: `wrap_native_ddp` disables `inplace_wgrad_accum`; native DDP
  always runs under autocast, which `Linear` rejects for that path.
- `fa1d445e1`: the benchmark entrypoint prints the training traceback before
  diagnostics and skips Sophia witness reductions after a failure.
- `7586def1e`: `BENCH_AC` and `BENCH_MEMORY_SNAPSHOT` launcher knobs.

### TT native DDP memory (rank 0, `12481853`)

| Snapshot | Active | Reserved | FP32 params | DDP grad buckets | `SophiaG` state |
|---|---:|---:|---:|---:|---:|
| After update `1` | `29.64 GiB` | `44.74 GiB` | `7.45 GiB` | not yet live | `14.80 GiB` |
| After update `2` | `29.64 GiB` | `60.37 GiB` | `7.45 GiB` | `7.40 GiB` | `14.80 GiB` |

Persistent TT state is `29.6 GiB`. The original MDS estimate omitted
FP32 master gradients. Its DS `0.18.5` unfused path is approximately
`37 GiB` persistent, whereas the historical `BF16_Optimizer` shards
masters and optimizer state. Do not treat their nominal stage-0 labels as
the same storage layout. At the OOM, `55.02 GiB` was allocated and `5.35 GiB`
was reserved but unallocated. The failed request was `3.91 GiB`, which is
one BF16 logits tensor: `8192 x 256128 x 2` bytes. Activations at sequence
`8192` take roughly the remaining `25 GiB`.

The Aurora production `MDS` source runs record this same layout with
activation checkpointing off. The later investigation explains the fit difference through the selected
DeepSpeed wrapper; see the November 1 audit.

### ZeRO-0 bisect and matched 16-rank pair

| Job | Layout | Ranks | Result |
|---|---|---:|---|
| `12481857` | MDS ZeRO-0 SophiaG, `--checkpoint-activations` | `24` | exit `127` before any rank output; transient |
| `12481858` | same, retry | `24` | `Invalid activation recompute method`; needs a recompute method |
| `12481859` | MDS ZeRO-0 SophiaG, uniform recompute per layer | `24` | GPU page faults (`NotPresent`, banned) at update `1` |
| `12481860` | MDS ZeRO-1 SophiaG | `24` | `32` updates; `2,200.75 ms` median `21-32`; max allocated `40.2 GB` |
| `12481861` | MDS ZeRO-0 AdamW | `24` | `UR_RESULT_ERROR_OUT_OF_RESOURCES` at update `1` |
| `12481864` | TT global FSDP | `24` | `FSDP does not support uneven sharding on dim 1: [2, 11008, 2048]` |
| `12481865` | MDS ZeRO-1 SophiaG | `16` (`2 x 8`) | `32` updates; `2,252.65 ms` median `21-32`; max allocated `40.4 GB` |
| `12481866` | TT global FSDP, loss compile | `16` (`2 x 8`) | `32` updates; `2.82 s` mean `22-32` from 1 s timestamps; `26.94 GiB` |

ZeRO-0 fails at update `1` with both optimizers on this runtime; ZeRO-1 runs.
Every MDS run here used `torch 2.10.0a0+git449b176` and `DeepSpeed 0.18.5`.
The historical `frameworks 2025.2.0` runtime is unavailable in the inspected
locations. A later current-runtime compatibility configuration does fit
(`12481887`); the original failure is not a blanket ZeRO-0 limitation.

Before the later padding fallback, TT global FSDP could not initialize
AGPT-2B over `12` or `24` ranks: since
`cde3c9322` the stacked `w13` shards on its `11008` rows, and FSDP2 rejects
uneven `Shard(1)`. `11008 = 2^8 x 43`. This blocked `12`-way HSDP at initialization. The subsequent experiment
fix falls back to padded `Shard(0)`, and successful `24`-rank runs now exist.
The padding still carries a communication and memory cost.

Matched 16-rank result, no rematerialization: MDS ZeRO-1 `2.2525 s/update`
(`7,273.6 tokens/s/device`) versus TT global FSDP `2.8699 s/update` (`5,708.8`).
The earlier coarse estimate was `1.25x`; aligned high-resolution event
timing gives `1.274x` for this pair. The historical Aurora gap is `1.52x`.
Losses at update `32`: MDS `11.99`, TT `12.32`; this run does not test
numerical equivalence.

`12481854` (`AC=full`): updates `22-32` take `3.18 s` mean from 1 s log
timestamps, so about `5,150 tokens/s/device` (`16,384` tokens/device/update).
The logged `tps` is a running average and reads `2,777` median over `21-32`.
This is a two-node number with rematerialization; it is not comparable with
the production `4,787` or with any `MDS` arm.

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
