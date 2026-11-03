# AuroraGPT-2B: recovered Sunspot MDS/TT comparison

<!-- measured-results:start -->
## Measured results: job `12482063`

Updates `21-32`, twelve samples after 32-update runs. `R/S` means data-parallel replica/shard degree. One device below is one selected `FLAT` tile/rank.

| Path | Mean s/update | tok/s/tile | Model TFLOP/s/tile | MFU % | Peak allocated / reserved GiB | Outer compile timer max s |
|---|---:|---:|---:|---:|---:|---:|
| MDS before, BF16_Optimizer compatibility | 2.364592 | 6928.9 | 77.51 | 25.99 | 46.396 / 50.613 | none |
| TT HSDP R12/S2 eager | OOM after 1 update | -- | -- | -- | incomplete | none |
| TT HSDP R12/S2 loss compile | 2.595684 | 6312.0 | 70.61 | 23.68 | 40.662 / 47.855 | 14.273 |
| TT native DDP/ZeRO-1, 100 MiB buckets | 2.444031 | 6703.7 | 74.99 | 25.15 | 49.064 / 56.820 | 3.013 |
| MDS after, same control | 2.363333 | 6932.6 | 77.55 | 26.01 | 46.396 / 50.613 | none |

Native TT DDP/ZeRO-1 is `3.39%` slower than the mean of the bracketing MDS controls. The MDS before/after timing drift is `-0.053%`. This is one allocation, not a production-scale or independent-repeat equivalence claim.

All completed arms have `24/24` completed rank diagnostics and 32 finite optimizer steps. All `1,536` consumed input/label/raw-token hash records match across completed arms. The eager failure has `96` matching records and `1,440` missing records; its partial data is not a complete-run comparison.

TT uses adjacent full-precision global-loss event wall times, cross-checked against synchronized per-rank training-step records. MDS uses its native iteration timer. Those framework boundaries are not proven identical. TFLOPs use `11,186,515,968 FLOPs/token`; MFU uses the existing report normalization of `298.1888 TFLOP/s/FLAT tile`. These are useful-model-work estimates, not hardware counters.

Compilation is loss-region compilation only. The outer timer is `_compile.compile_inner`; nested timers are not summed. TT variants share a job cache, so the later DDP compiler timer is not an independent cold-start measurement. Memory is the maximum process-lifetime allocated/reserved high-water mark across all 24 ranks, not a steady-window average.

### Loss agreement

Full 32-update loss differences versus MDS-before: HSDP loss compile max absolute `0.951191`, RMS `0.249758`; native DDP max absolute `0.304562`, RMS `0.123824`. MDS before/after itself has max absolute loss difference `0.756085` and RMS `0.148977`. No numerical-equivalence tolerance is silently applied.

The native DDP recipe uses autocast with FP32 embedding, RMSNorm, and residual tensors. That differs from uniform BF16 model compute. The loaded-weight audits establish exact initial parameter values, not identical forward/backward arithmetic or optimizer updates. Numerical equivalence and representative convergence remain open.

### Collective visibility

| Path | Trace coverage | Rank-0 exposed collective-kernel ms | Rank-0 device-idle ms | TT exposed min / median / max ms |
|---|---:|---:|---:|---:|
| MDS before, BF16_Optimizer compatibility | 1/24 ranks | 89.416 | 118.398 | -- |
| TT HSDP R12/S2 loss compile | 24/24 ranks | 221.650 | 210.645 | 167.402 / 214.802 / 256.602 |
| TT native DDP/ZeRO-1, 100 MiB buckets | 24/24 ranks | 187.461 | 85.213 | 101.226 / 168.960 / 204.614 |
| MDS after, same control | 1/24 ranks | 88.223 | 125.029 | -- |

These are interval unions from captured update 12. Device activity includes kernels, copies, and memsets. Recognized collective names contain `oneCCL` or `ccl::`; exposure subtracts overlap with other device activity. Each rank is analyzed independently; ranks are not summed or merged into an unsynchronized global timeline. CPU collective annotations are not added to GPU time. This is not a causal stall estimate or a complete network-transfer measurement.

The captured rank-0 non-collective device-activity union is similar for MDS and HSDP. HSDP has more exposed collective activity and idle time. This supports testing communication granularity and scheduling before attributing the gap to attention kernels alone. Native DDP removes the per-layer parameter all-gather pattern but still issues more reduction annotations than MDS. Larger DDP buckets are the next isolated variable.

[Numerical series, all-rank summaries, and methods](2026-11-02-pair-12482063-evidence.json).

<!-- measured-results:end -->

## Recovery evidence

`12481979` started once on two `Sunspot` nodes on `2026-11-02` and finished
with scheduler exit `0`. Its full stdout records two 10-step debug arms,
`1-baseline rc=0`, `2-mixed rc=0`, and `VERDICT mixed_demo rc=0`.
This establishes application execution after the November 1 startup incident.
It does not establish the site's root cause or validate the 2B comparison.

`REQ-17196` received a support response at `17:55 UTC` asking whether PBS
issues remained. The response supplies no diagnosis or repair details.

## Pending application gate

The next gate is the live loaded-parameter audit before the first 2B update.
It checks all `111` adapted parameter keys (`1,986,054,144` elements) against
the pinned native/DCP audit, independently on all `24` ranks.
The pinned audit manifest and checksum are retained in the linked W&B evidence.

Source is pinned to `a4057f84ebce1a46a42409d20eb6aee2aa7a0cf6` in an isolated project-filesystem checkout.
The checkout is separate from the shared training and earlier campaign trees.
The runtime remains the pinned Torch `2.14.0+xpu` environment. Exact paths and runtime manifests are retained in W&B.
Exact-runtime config parsing and mandatory dependency/asset preflight passed.
The launcher is `scripts/benchmark_agpt_replicated_tt.sh`, using `ezpz launch`.

| Setting | Value |
|---|---|
| Nodes / ranks per node | `2 / 12`, `FLAT` |
| Model / optimizer | `AuroraGPT-2B / SophiaG` |
| Native DDP / optimizer sharding | `24` replicas / ZeRO-1 |
| Local accumulation / wire / weight all-gather | FP32 / BF16 / BF16 |
| Sequence / microbatch / accumulation / GBS | `8192 / 1 / 2 / 48` |
| Updates / seed / activation checkpointing | `32 / 42 / none` |
| Compilation | Loss region only |
| Gradient bucket | `100 MiB` |
| Profiling | Disabled for the live-parameter canary |
| W&B | Online, `aurora_gpt/torchtitan.ezpz.train` |

## Restart attempts

The first login preflight returned after a delay while a second invocation
was in progress. Both submitted: `12482057` and `12482058`. Their training
output paths and W&B run IDs contain the unique PBS job ID. Their initial
PBS stdout destination was mistakenly shared and its parent did not exist.
Both entered scheduler exit processing with `Exit_status=-3`, then retried.
The missing stdout parent is a verified submission defect; its role in the
scheduler exit is an unverified inference pending the corrected run.
No training output existed in the first snapshot.

The stdout parent was created, and corrected job `12482059` uses PBS's
job-specific default stdout path. Both original jobs subsequently reached
application execution on their second attempts; `12482059` started on its
first attempt. All three produced `24/24` live-parameter audit JSONs with
`status=passed` and `all_ranks_passed=true`. Every rank matched all `111`
expected parameter hashes before training. These files establish live
loaded-parameter identity for this native DDP path, not full-model numerical
equivalence across frameworks. All three then completed `32` updates with scheduler exit `0` and complete
`24/24` rank diagnostics. Every rank recorded `32` successful optimizer calls,
`32` ZeRO-1 all-gathers, and finite parameter, momentum, and Hessian state.
No jobs were cancelled.

| Job | Audit rank coverage | Initial live parameters | W&B |
|---|---:|---|---|
| `12482057` | `24/24` | All exact | [Run](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/agpt-replicated-ddp-bf16-losscompile-live-resume-12482057) |
| `12482058` | `24/24` | All exact | [Run](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/agpt-replicated-ddp-bf16-losscompile-live-resume-12482058) |
| `12482059` | `24/24` | All exact | [Run](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/agpt-replicated-ddp-bf16-losscompile-live-resume-io-12482059) |

The initial stdout-directory defect is distinct from the November 1 failure:
these jobs now have real `stime` and application output. The site repair
mechanism remains unverified.

## Next comparison

After the live audit and optimizer-step witnesses pass, rerun the controlled
MDS/TT pair. Preserve complete input hashes, loss histories, rank diagnostics,
steady-state timing, compilation timers, peak memory, and raw profiler traces
in online W&B. Derive TFLOPs, MFU, and tokens/sec/device using the existing
report's common FLOP and hardware normalization. All arms must finish before
reporting a comparative result.

- [Recovered comparison and numerical limits](2026-11-01-handoff-audit.md)
- [Same-allocation comparison protocol](2026-11-01-paired-performance.md)
- [Original startup incident](2026-11-01-startup-incident.md)

## Native DDP canary results

These are loading/runtime canaries, not a new MDS/TT performance comparison.
Every rank has `32` successful synchronized step records. The timing window
is updates `21-32`, using the synchronized training-step timer rather than
cumulative throughput printed during warmup.

| Job | Rank-0 mean s/update | Largest per-rank mean s/update | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|
| `12482057` | `2.541401` | `2.542011` | `49.0643` | `56.8203` |
| `12482058` | `2.540682` | `2.540782` | `49.0643` | `56.8203` |
| `12482059` | `2.541382` | `2.541563` | `49.0643` | `56.8203` |

All `1,536` input/label/raw-token hash records match across all three runs.
The initial logged loss rounds to `17.75424` in each run, but update-32
losses are `11.82221`, `11.37010`, and `11.40153`. Identical live parameters
and complete input hashes therefore do not establish repeatable loss
trajectories with this configuration (`debug.deterministic=false`). The
cause of the later differences remains unresolved.

Raw terminal canary evidence, fetch manifests, and checksums are retained in W&B.
The online W&B runs retain the linked campaign-evidence artifacts.

## HSDP audit and MDS launcher gate

`12482062` runs the separate HSDP audit with `dp_replicate=12`,
`dp_shard=2`, no reshard after forward, FP32 reduction, and deferred gradient
reduction. Its materialization audit can affect lifetime memory peaks;
those peaks will not be used as clean performance measurements. The job
finished with exit `0`, `24/24` exact live-parameter audits, `24/24` completed
rank diagnostics, `32` successful SophiaG calls on every rank, and finite
parameters, momentum, and Hessian state.

MDS login preflight found that its borrowed helper environment's Click
`8.3.1` installation is missing `click/core.py`. The same Click version was
installed with no dependencies into the isolated campaign overlay
`.cache/mds-cli-overlay`. Neither borrowed runtime was modified. The complete replacement package and checksum are retained in W&B.
The old nested-separator command failed the exact installed Click CLI dry-run:
its forwarded payload began at `--finetune`, dropping Python and the benchmark
entrypoint. Removing the inner separator passed the same dry-run with MPI
execution mocked. The benchmark entrypoint forwards training flags through
`parse_known_args`, so the inner separator is unnecessary.

The narrow fix and explicit overlay support are pinned in
`4831cd2d94ad1dca29f1956d5b40e2fee45bdc12`. TT code remains at the audited
base. Borrowed MDS source remains `9f407ed9e8e30920e28272daf8cb7db36c211e21`.
The formal run will bracket TT eager, TT loss compilation, and native DDP
ZeRO-1 with MDS controls on the same two allocated nodes. Raw traces are
rank-local; they do not establish all-rank causal communication stalls.

## Same-allocation job `12482063`

Submitted through mandatory preflight after both all-rank canaries completed.
Source: [`4831cd2d9`](https://github.com/saforem2/torchtitan/commit/4831cd2d94ad1dca29f1956d5b40e2fee45bdc12).
Workdir: the clean isolated project-filesystem comparison checkout; its exact path is retained in W&B.
Parent evidence: `outputs/agpt-replicated-pair-12482063/run.log`.

| Phase | Path | Layout / precision | Compilation |
|---|---|---|---|
| Before | MDS historical-wrapper compatibility | `BF16_Optimizer`, FP32 local accumulation, BF16 wire | None |
| TT `hs12eager` | HSDP | `dp_replicate=12`, `dp_shard=2`, never reshard, FP32 deferred reduction | None |
| TT `hs12loss` | HSDP | Same as eager | Loss region |
| TT `ddpz1` | Native DDP / ZeRO-1 | FP32 local accumulation, BF16 wire and weight all-gather, `100 MiB` bucket | Loss region |
| After | MDS historical-wrapper compatibility | Same as before | None |

All phases use two nodes, `12` FLAT ranks/node, sequence `8192`, microbatch
`1`, accumulation `2`, GBS `48`, `32` updates, seed `42`, fresh SophiaG state,
and no activation checkpointing. W&B remains online. Profiler capture is
at update `12`; the steady window is `21-32`. The TT variants share a
job-local compiler cache, so later compile times are not independent cold
starts. Loading audits are disabled in this performance job after passing
separately; HSDP audit materialization cannot inflate its memory peaks.

Status at submission: awaiting application/phase completion. No new winner
is recorded. MDS uses Torch `2.10.0a0+git449b176` / DeepSpeed `0.18.5`; TT
uses Torch `2.14.0+xpu`. This is a framework-path comparison with explicit
runtime differences, not runtime identity or production-scale parity.

## First comparison terminal outcome

`12482063` finished with scheduler exit `1`. Both MDS controls, HSDP loss
compilation, and native DDP/ZeRO-1 completed `32` updates. Eager HSDP failed
in backward after its first completed update: XPU requested `7.81 GiB` with
`5.58 GiB` free on a `63.98 GiB` tile. The launcher returned `143` after
another rank's OOM, not after a user cancellation or a PBS pre-run failure.
Its partial output does not supply steady-state performance.

The eager W&B run is labelled `finished` despite the application OOM.
Application exit status, raw traceback, and evidence summary take precedence
when judging success. The failed run is retained with the completed arms.

A follow-up keeps sequence `8192`, microbatch `1`, accumulation `2`, GBS `48`,
`32` updates, SophiaG, and `24` FLAT ranks, and changes both eager and compiled
HSDP to `dp_shard=6`, `dp_replicate=4`. MDS controls again bracket the two TT
arms on the same allocated nodes. Both TT arms will audit loaded parameters;
materialization-audit memory peaks will be flagged as non-comparable. This
isolates sharding layout while keeping the workload fixed.

## Follow-up `12482081` terminal status

All four phases completed with `PAIR_COMPLETE rc=0` and scheduler exit `0`:
MDS before, HSDP R4/S6 eager, HSDP R4/S6 loss compilation, and MDS after.
Full source artifacts and all-rank loaded-parameter audits are being analyzed.
The six-shard eager configuration passed the memory-fit gate that failed at
two shards. Both layouts remain explicit variables in the comparison.

The next isolated test keeps the native DDP/ZeRO-1 recipe fixed and varies
`native_ddp_bucket_cap_mb` through `100`, `2048`, and `8192`, with the same
before/after MDS controls. It tests whether reduction granularity accounts
for the remaining measured gap; no result is assumed.
