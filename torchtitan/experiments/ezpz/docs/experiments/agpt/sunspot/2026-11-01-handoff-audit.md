# AuroraGPT-2B MDS/TT: recovered comparison and handoff audit

Audit date: `2026-11-01`. Source conversation: Claude session
`255a7e56-1404-4d91-be47-317b42c0aa0c`, titled `MDS vs TT performance comparison`.
The audit reads its complete retained conversation, source revisions,
TensorBoard events, complete training logs, consumed-input hashes and rank
diagnostics. Raw private transcripts stay outside the repository.

## Result and scope

At `24` FLAT devices on two Sunspot nodes, `MDS ZeRO-1` takes
**2.2005 s/update**. `TT HSDP 12 x 2`, with resharding disabled and deferred
reduction, takes **2.7041 s** with BF16 reduction or **2.7134 s** with FP32
reduction: `22.9%` and `23.3%` longer than that MDS control. The historical
Aurora production reference used a different DeepSpeed wrapper/runtime and
`256` nodes. This two-node result does not measure production-scale parity.

At `16` devices, the combined FSDP communication changes reach `2.2799 s`,
versus `2.2525 s` for MDS ZeRO-1 and `2.3894 s` for the historical-wrapper
compatibility arm. These are single-run estimates with different arithmetic
and runtime implementations. Full-model loss/gradient equivalence remains
unproven.

## Aligned performance

Every row uses updates `21-32`, twelve observations, after a `32`-update run.
TT durations are adjacent high-resolution loss-event `wall_time` differences
anchored at update `20`. Adjacent differences of its cumulative
`time_metrics/end_to_end(s)` agree within `0.0004 s`. MDS uses the native
per-iteration elapsed field. Means, rather than a mixture of means and
medians, determine throughput. TT's interval includes host logging between
loss events; identical timer/device boundaries across frameworks are not
established. Startup, active trace capture at update `12`, final checkpoint
and shutdown are outside this window.

| Job | Ranks | Path | Mean s/update | tok/s/device | Model TFLOP/s/device | MFU % |
|---|---:|---|---:|---:|---:|---:|
| `12481865` | `16` | MDS ZeRO-1 | `2.2525` | `7273.6` | `81.37` | `27.29` |
| `12481866` | `16` | TT global FSDP, default | `2.8699` | `5708.8` | `63.86` | `21.42` |
| `12481886` | `16` | TT global FSDP, never/BF16/defer | `2.2799` | `7186.4` | `80.39` | `26.96` |
| `12481887` | `16` | MDS BF16_Optimizer compatibility | `2.3894` | `6857.1` | `76.71` | `25.72` |
| `12481860` | `24` | MDS ZeRO-1 | `2.2005` | `7445.6` | `83.29` | `27.93` |
| `12481883` | `24` | TT global FSDP, padded | `4.0638` | `4031.7` | `45.10` | `15.12` |
| `12481884` | `24` | TT HSDP 12 x 2, default | `3.1049` | `5276.9` | `59.03` | `19.80` |
| `12481902` | `24` | TT global FSDP, never/BF16/defer | `2.7860` | `5880.8` | `65.79` | `22.06` |
| `12481901` | `24` | TT HSDP 12 x 2, never/BF16/defer | `2.7041` | `6058.9` | `67.78` | `22.73` |
| `12481901` | `24` | TT HSDP 12 x 2, never/FP32/defer | `2.7134` | `6038.3` | `67.55` | `22.65` |

Machine-readable complete scalar series, timing samples and exact executed
source heads: [timing evidence](2026-11-01-timing-evidence.json).

One device means one selected `FLAT` tile/rank, not a two-tile package.
Every update processes `16,384` tokens/device. The common accounting is
`11,186,515,968 FLOPs/token` and `298.1888 TFLOP/s/device` peak normalization:
`TFLOP/s = tok/s * 11,186,515,968 / 1e12`,
`MFU = 100 * TFLOP/s / 298.1888`.
These are normalized model-work estimates, not hardware-counter FLOPs.
The [native-checkpoint report](2026-10-29-native-checkpoint-closure.md#tokenssec-model-tflops-and-mfu)
defines the parameter/attention formula and denominator. Compilation in these
TT rows covers the loss only. Neither full-model compilation nor a matched
production-scale run is represented.

The BF16 HSDP mean has within-run sample deviation `0.283 s`, versus
`0.004 s` for the FP32 arm; the exact JSON retains every sample. Similar
rounded times do not establish that FP32 reduction costs zero. Repeated
independent jobs are needed to resolve small differences.

## Corrections to the Claude conversation

| Earlier claim | Audited finding |
|---|---|
| ZeRO-0 cannot fit, and more nodes cannot change its memory. | The new DS `0.18.5` stage-0 path replicated its optimizer state. Historical DS `0.17.5` selected `BF16_Optimizer`, which shards masters/state despite nominal stage 0. Its per-rank state therefore does depend on DP size. Plain DDP remains replicated. |
| Production MDS locally accumulates BF16. | This describes fresh DS `0.18.5` default stage 0 and the early ZeRO-1 controls. Historical production used FP32 local accumulation and BF16 communication. |
| `MDS_ZERO0_PATH=ds0175` reruns the original runtime/config. | It uses DS `0.18.5`, a stage-1 JSON and FP32 accumulation to select `BF16_Optimizer`. Megatron's flag remains stage 0. Wrapper/type behavior is reproduced; literal runtime/config identity is not. |
| TT cannot run 12/24-way FSDP. | A later ezpz fallback permits padded `Shard(0)` when stacked MLP rows cannot shard evenly. Successful 24-rank runs supersede the startup failure. Padding is still costly. |
| Identical first loss proves identical weights. | One scalar cannot establish parameter equality. The old conversion audit is exact; a new live post-wrap parameter audit is still needed. |
| MDS's 64 optimizer calls mean 64 updates. | DS ZeRO-1 calls Sophia once per parameter group, twice/update. Every checked rank's state counter is `32`; the compatibility wrapper makes `32` calls. |
| Loss above `ln(vocab)` implies a bad checkpoint. | Uniform prediction has loss `ln(vocab)`; cross entropy has no such upper bound. Confidently wrong predictions can exceed it. |
| The gap is communication; BF16 matches MDS numerics. | The combined communication settings improved one measured run. They also change accumulation precision. BF16 wire alone does not establish equivalent gradients. |
| Two-node results close the Aurora production gap. | No matched 256-node result exists in this continuation. The historical throughput gap remains a separate measurement. |
| Scheduler exit 0 validates the overlap sweep. | Jobs `12481898` and `12481903` returned 0 despite individual timeouts, MPI failures or a segfault. Per-variant return codes are required. |
| All campaign evidence is online. | W&B had metrics but 48/52 runs had no logged artifact at audit start. Evidence archiving was not called by the launchers. Recovery uploads are tracked separately. |

## DeepSpeed wrapper and precision

Production W&B run [`1kha62ra`](https://wandb.ai/aurora_gpt/AuroraGPT/runs/1kha62ra)
records `DeepSpeed 0.17.5`, `torch 2.8.0a0`, BF16 parameters, nominal stage 0,
no activation checkpointing and accumulation `2`. The older engine defaults
select `BF16_Optimizer` with FP32 accumulated gradients and sharded master
weights/optimizer state. DS `0.18.5` defaults select `FP16_UnfusedOptimizer`
for the fresh stage-0 BF16 config. That replicated-state path exhausts memory
in the new two-node workload.

Compatibility job `12481887` completed all `32` updates with the runtime
witness `BF16_Optimizer`, BF16 model/wire and FP32 accumulation. It used DS
`0.18.5` internal stage 1. All `16` rank diagnostics have state step `32`
and no recorded optimizer exception. This resolves the initial memory puzzle
without claiming that an unchanged historical runtime was rerun.

| Path | Local accumulation | Wire | Master/optimizer storage |
|---|---|---|---|
| Historical MDS DS `0.17.5` | FP32 | BF16 | Sharded via `BF16_Optimizer` |
| New default MDS stage 0 DS `0.18.5` | BF16 | BF16 | Replicated; failed fit |
| New MDS ZeRO-1 control | BF16 | BF16 | Sharded |
| MDS compatibility `12481887` | FP32 | BF16 | Sharded via `BF16_Optimizer` |
| TT deferred FSDP FP32 reduction | FP32 | FP32 | FSDP-sharded states |
| TT deferred FSDP BF16 reduction | BF16 | BF16 | FSDP-sharded states |
| TT native DDP + ZeRO-1 prototype | FP32 | BF16 compressed | FP32 master/optimizer shard; BF16 all-gather option |

The TT native-DDP/ZeRO-1 prototype completed at about `2.455 s/update` in
its earlier 16-rank experiment; the FSDP control in that allocation was
about `2.364 s`. Those are coarse transcript-recovered timing estimates,
not rows in the aligned table above. Adding a matching communication pattern
did not by itself beat FSDP in that allocation. Its checkpoint round-trip and
full-model numerical behavior remain experimental.

## Exact input and optimizer evidence

Consumed CPU tokens are hashed before transfer; comparisons include raw,
shifted input and shifted label SHA-256, keyed by rank/microbatch/sample.
The following complete comparisons have no missing, extra or mismatched rows:

| MDS | TT | Matching rows |
|---|---|---:|
| `12481865`, ZeRO-1 | `12481866`, FSDP | `1,024 / 1,024` |
| `12481887`, compatibility | `12481886`, sync0lite | `1,024 / 1,024` |
| `12481887`, compatibility | `12481893`, DDP-ZeRO-1 b100 | `1,024 / 1,024` |
| `12481860`, ZeRO-1 | `12481902`, global FSDP lite | `1,536 / 1,536` |
| `12481860`, ZeRO-1 | `12481901`, HSDP FP32 lite | `1,536 / 1,536` |

The saved conversion audit covers all `1,986,054,144` parameters (`75` native
and DCP tensors through `111` HF tensors), exact BF16 mapping and exact FP32
promotion. It does not establish that every new wrapped model loaded those
parameters correctly. Equal first losses and saved-checkpoint existence do
not replace that missing live-load audit. Loss and gradient histories already
differ between frameworks, so numerical equivalence is not claimed.

## BF16 reduction: recovered 200-step experiment

Job `12481899` finished all five `200`-update arms: two FP32 repeats, two
BF16 repeats and an FP32 per-microbatch control. All five consumed exactly
the same `6,400` sample hashes. Strict deterministic attempts in `12481896`
failed at SDPA with a `4 GiB` allocation after about `58.11 GiB` was allocated.
A math-attention fallback is the likely mechanism, but dispatch was not
recorded in those failed training traces. No warn-only mode was used.

| Loss-history pair | Mean abs difference | RMSE | Max abs difference |
|---|---:|---:|---:|
| FP32 repeat | `0.08060` | `0.14330` | `0.68392` |
| BF16 repeat | `0.10166` | `0.18080` | `1.27598` |
| FP32-a / BF16-a | `0.07424` | `0.14800` | `1.38340` |
| FP32-b / BF16-b | `0.04219` | `0.07095` | `0.57650` |

The same-state gradient probe measured relative L2 difference `1.6075%`
for FP32 repeats and `1.7204%` for BF16 versus FP32, with cosines
`0.999912` and `0.999888`. These paths are nondeterministic. Same-precision
variation is comparable to the cross-dtype variation, so these runs do not
isolate a BF16-only loss effect. They do not establish long-term convergence
or numerical equivalence. Full numbers: [precision evidence](2026-11-01-bf16-evidence.json).

## Attention and communication

At the measured `B=1, S=8192, Hq=16, Hkv=4, D=128` shape, the Torch `2.14`
XPU flash backward takes about `24.7 ms` causal and `25.0 ms` noncausal;
MDS's Torch `2.10`/IPEX xetla path takes about `10.2` and `19.0 ms`.
The causal tile-skipping difference is supported at this shape. It is not
proof of a universal 2x kernel gap.

The opt-in block-causal prototype partitions attention into lower-triangle
block pairs and merges outputs/LSE. The final BSHD-merge commit is
`eea894d7a`. Its microbenchmark exists, but the earlier training A/B ran a
previous implementation. Reported output/dQ/dK/dV max-absolute errors versus
an FP32 math reference were approximately `0.0077/0.0103/0.026/0.044`.
This is approximate numerical agreement. Final-source training benefit and
full-model numerical impact require a new run; the option remains off by
default.

The original TT traces show exposed collective kernels with essentially no
visible compute overlap. A later copy-engine microbenchmark reports
`310.88 + 222.96 - 368.91 = 164.93 ms` of timing-derived overlap, but the
all-gather transfer is absent from its GPU kernel/memcpy trace. The matching
reduce-scatter trace shows only `0.003 ms` of overlap. This establishes a
promising microbenchmark timing result, not trace-visible training overlap.
Training follow-up `12481904` logs `allgatherv using copy engines not supported
with tmp buffer`; the setting is not a usable training optimization on this
path. Its final variant outcomes must be inspected after the job terminates.

## Continuation gates

- Recover and upload full existing evidence to each online W&B run, using its
  recorded source SHA rather than a later checkout HEAD. Preserve failed arms.
- Run a pinned 24-rank canary with explicit per-rank step timing. Compare
  even-shard HSDP degree `4` against padded degree `12`, then the final block-4
  implementation and eager/loss-compiled paths. Keep FP32 reduction as a control.
- Run a fresh 24-rank MDS `BF16_Optimizer` compatibility control before assigning
  a production-numerics ratio to the TT candidates.
- Audit initial live parameters and same-batch full-model numerics. Verify all
  optimizer updates, rank coverage, CPU hashes and output checkpoints.
- Scale only passing candidates through the Aurora compute/runtime gates.
  No 256-node parity result is claimed here.
