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

A fresh `24`-rank historical-wrapper compatibility control takes
`2.3610 s/update`. The separate HSDP `4 x 6` canary takes `2.9030 s` and
does not improve on the earlier `12 x 2` result. Its complete input stream
matches MDS, but losses differ. The intended same-allocation comparison was
cancelled after a PALS startup failure. The live-load canary then ended with
scheduler exit `265` before application startup; the replacement comparison
has not run. The Aurora fallback communication probe completed;
[Sunspot startup isolation and exact recovery status](2026-11-01-startup-incident.md)
records the failed-start reproductions and its scope.

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

## Fresh 24-rank pair: complete input match, unresolved loss gap

Jobs `12481905` and `12481906` both completed 32 updates with PBS exit 0. The
TT run used `HSDP 4 x 6`, no activation checkpointing, deferred FP32 gradient
reduction, no reshard after forward, and loss-only compilation. The MDS run used
the nominal ZeRO-0 compatibility path selecting DeepSpeed `BF16_Optimizer`.
Both used `24` ranks and global batch `48`.

| Run | Update-21-32 timer | Mean s/update | tok/s/device | Model TFLOP/s/device | MFU % | Samples |
|---|---|---:|---:|---:|---:|---:|
| `12481906` MDS | Native iteration timer | `2.360975` | `6939.5` | `77.63` | `26.03` | `12` |
| `12481905` TT | Loss-event intervals, anchored at update `20` | `2.902996` | `5643.8` | `63.13` | `21.17` | `12` |
| `12481905` TT | Synchronized per-step maximum across `24` ranks | `2.902788` | `5644.2` | `63.14` | `21.17` | `12` |

TT's adjacent cumulative `time_metrics/end_to_end(s)` differences average
`2.902987 s`; the largest per-step difference from the loss-event intervals is
`0.000153 s`. The explicit synchronized hook's largest step is `4.071005 s`.
The two TT methods corroborate one another. MDS and TT timer boundaries are not
proven equivalent, so their means are not a controlled timing ratio.

All `1,536` `(rank, microbatch)` records match across input, labels, raw token
rows, shapes, world size, and local sample index. Per-rank JSONL digests and all
32 loss values are in the [fresh run evidence](2026-11-01-fresh-905-906-evidence.json).
Losses do not agree: MDS moves from `17.663555` to `10.828611`; TT moves from
`17.746710` to `12.389167`. Across 32 updates, TT minus MDS starts at
`+0.083155`, ends at `+1.560556`, and has RMS `0.352058`.

The MDS log confirms a successful load from native `mds-load`, tag
`global_step138650`. The captured TT launcher sets `MDS_ANNEAL_BASE` to the
converted `agpt-matched-native-weights/dcp/step-0`; the pinned config binds that
path as a model-only initial load. Neither job captured a live post-wrap
parameter hash. The loss gap is unexplained; identical input hashes do not
prove identical loaded parameters or numerical agreement.

Process high-water allocated/reserved memory was `46.396/50.613 GiB` for MDS
and `35.112/40.039 GiB` for TT. These are maxima of the recorded allocator high-water marks across ranks,
not update-window memory. TT's maximum-rank outer loss-compilation timer was
`15.139 s`; nested compiler timer keys are not summed.

The BF16 HSDP mean has within-run sample deviation `0.283 s`, versus
`0.004 s` for the FP32 arm; the exact JSON retains every sample. Similar
rounded times do not establish that FP32 reduction costs zero. Repeated
independent jobs are needed to resolve small differences.

## Peak memory and compilation cost

All expected rank diagnostics are present. The table takes the maximum
recorded allocated and reserved high-water marks across ranks, captured
before final witness reductions. In TT, the counters run from the metrics
processor's initialization reset, before model initialization; the ezpz
training override does not reset them at each log. They are not process-birth
or update-`21-32` peaks. No reset was found in the pinned MDS source, but
external DeepSpeed/runtime resets have not been audited. Identical memory
measurement windows across frameworks are therefore not established.
The `GB`/`GiB` shorthand is replaced with bytes converted consistently to `GiB`.

| Run/variant | Rank coverage | Peak allocated GiB | Peak reserved GiB | Max-rank loss compile timer s |
|---|---:|---:|---:|---:|
| `12481860` | `24/24` | `39.923` | `42.561` | `none recorded` |
| `12481865` | `16/16` | `40.387` | `42.727` | `none recorded` |
| `12481866` | `16/16` | `21.271` | `26.938` | `17.688` |
| `12481883` | `24/24` | `24.349` | `29.504` | `14.077` |
| `12481884` | `24/24` | `25.238` | `29.953` | `14.244` |
| `12481887` | `16/16` | `46.856` | `50.762` | `none recorded` |
| `12481886-base` | `16/16` | `21.271` | `26.938` | `14.725` |
| `12481886-sync0` | `16/16` | `21.271` | `26.938` | `3.002` |
| `12481886-sync0lite` | `16/16` | `27.232` | `33.418` | `2.179` |
| `12481886-sync1lite` | `16/16` | `27.232` | `33.418` | `2.281` |
| `12481901-bf16` | `24/24` | `35.252` | `41.578` | `10.312` |
| `12481901-fp32` | `24/24` | `38.948` | `44.736` | `2.516` |
| `12481902-bf16` | `24/24` | `40.627` | `47.146` | `10.744` |
| `12481906` MDS compatibility | `24/24` | `46.396` | `50.613` | `none recorded` |
| `12481905` TT HSDP `4 x 6` | `24/24` | `35.112` | `40.039` | `15.139` |

Compilation uses only the outer `_compile.compile_inner` timer. Nested timer
keys are not added. These are compiler-accounted costs, not total cold-start
latency or an amortization estimate. Serial variants share a job's cache
root; these diagnostics alone do not establish the cause of timer variation.
MDS has no Torch compilation timer recorded. Full rank coverage, runtime
versions, completion flags and timer fields are preserved in the
[diagnostic evidence](2026-11-01-diagnostic-evidence.json).

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
| `12481906`, MDS compatibility | `12481905`, TT HSDP `4 x 6` | `1,536 / 1,536` |

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

## Exposed collective time

I independently reanalyzed the saved rank-0 traces for profiled update `12`
of the `16`-rank runs. The local gzip SHA-256 values match the original
Sunspot files. This uses unions of GPU event intervals before computing
intersection; nested CPU annotations are not added. Collective kernels are
identified by oneCCL names. Other GPU activity includes copies and does not
mean GEMMs alone. Transfers absent from the profiler cannot be measured by
this method. These are single-step timeline measurements, not the steady
window means above or all-rank critical-path estimates.

| Run | Device span ms | Other GPU activity ms | Collective kernels ms | Intersection ms | Exposed collective ms | Idle ms |
|---|---:|---:|---:|---:|---:|---:|
| `12481865` MDS ZeRO-1 | `2269.5` | `2056.6` | `121.5` | `15.7` | `105.7` | `107.2` |
| `12481866` TT default FSDP | `2940.4` | `1988.8` | `452.7` | `0.0` | `452.7` | `498.9` |
| `12481886` TT sync0lite FSDP | `2306.8` | `1992.9` | `177.4` | `0.0` | `177.4` | `136.5` |

The combined FSDP settings reduce exposed collective-kernel time from
`452.7` to `177.4 ms` in these traces, while non-CCL GPU activity is
`1,988.8` versus `1,992.9 ms`. Idle gaps also shrink from `498.9` to
`136.5 ms`; the step improvement must not be attributed to collective
kernel duration alone. [Trace hashes and interval definitions](2026-11-01-collective-evidence.json).

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
Training follow-up `12481904` ended with PBS exit `1`. Its `defbase` and
`litebase` controls completed with return `0`; `defce` and `litece` timed out
with return `124`, both logging `allgatherv using copy engines not supported
with tmp buffer`. This setting is not a usable training optimization on this
path. Complete timeout logs are retained losslessly with their original
sizes and hashes.

Jobs `12481905` and `12481906` completed 32 updates on two Sunspot nodes with
`24` ranks. Full batch-hash comparison passed. The loss histories diverge despite
the matching input stream; live post-wrap parameter hashes were not captured.
The timing methods and result limits are recorded above and in the fresh JSON.

## W&B evidence recovery

At audit start, `48/52` group runs had no logged artifact despite online
training metrics. The serialized recovery has committed evidence for `60`
runs, including the fresh `12481905` and `12481906` arms. API verification
covered all then-existing `59` artifacts; five seeded random downloads also
passed source-SHA, manifest/file-presence and checkpoint-hash checks. Later
artifacts require the same final coverage pass. Checkpoint payloads remain
on project storage; W&B includes their full byte-count/SHA manifests.

The first uploads resumed old training runs. That replaced some online
package and machine metadata with the CPU uploader's environment and
inflated displayed runtimes. Recovery restored `52` original files across
`50` runs and `43` preserved runtime/hidden-SDK summaries. Seven failed runs
lack a final original summary; initial probe `12481844` has no preserved
original machine metadata. Server heartbeat/end timestamps still include
archival activity. Use captured training timers and rank provenance for
performance, not the current W&B UI elapsed span.

Future uploads use one separately labeled archival run and the public
`Run.log_artifact()` association API. The canary left training state, config,
runtime, heartbeat and Git metadata unchanged. Large repeated timeout logs
are stored as lossless gzip with original and compressed hashes and verified
decompression; no log content is trimmed. Private agent transcripts are
excluded.

## Continuation gates

- Finish the current failed/partial-arm uploads and final analysis artifact,
  then repeat the API coverage check. Preserve each arm's actual source SHA
  and its separate application/scheduler status.
- Reconcile the `12481905`/`12481906` loss gap with live parameter audits and
  same-batch numerical checks. Do not infer its cause from endpoint loss.
- Compare the final block-4 implementation and eager/loss-compiled paths with
  repeated jobs and FP32 reduction controls.
- Scale only candidates that pass checkpoint-load, numerical and resource
  gates through the Aurora compute/runtime path. No 256-node parity result is
  claimed here.
