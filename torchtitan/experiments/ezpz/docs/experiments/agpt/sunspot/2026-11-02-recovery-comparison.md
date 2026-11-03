# Sunspot recovery and MDS/TT continuation

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
The expected audit JSON SHA256 is
`b1584ff72d8b2fd4f53594cd1bba7b12efb0f6a102bda8407cd315a00fe6bad6`.

Source is pinned to `a4057f84ebce1a46a42409d20eb6aee2aa7a0cf6` in
`/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan-agpt-resume-20261102`.
The checkout is separate from the shared training and earlier campaign trees.
The runtime remains
`/lus/tegu/projects/datascience/foremans/venvs/xpu-torch214-fixedlr-v2-20260927`.
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
equivalence across frameworks. Training completion and optimizer witnesses
remain pending. No jobs were cancelled.

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
