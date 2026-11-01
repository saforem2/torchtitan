# AuroraGPT-2B: same-allocation MDS and TorchTitan comparison

## Execution

Sunspot job `12481907` runs `2` nodes with `12` FLAT ranks/node. Each arm
uses sequence `8192`, microbatch `1`, accumulation `2`, global batch `48`,
`32` updates, seed `42`, no activation checkpointing and fresh SophiaG
state. Every update processes `16,384` tokens/rank. Online
[W&B](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/groups/agpt-replicated-20261031)
is enabled for every training arm.

Source: TT [`1ef281d79021a96a7ca276b0989e4f9892412a3f`](https://github.com/saforem2/torchtitan/commit/1ef281d79021a96a7ca276b0989e4f9892412a3f),
MDS `9f407ed9e8e30920e28272daf8cb7db36c211e21`.
The clean pinned checkout is
`/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan-agpt-pair-20261101`.
The exact runtime parser accepted eager, loss-only and
`fused_binary_activation,loss` compilation. Mandatory submission preflight
passed with `ZE_FLAT_DEVICE_HIERARCHY=FLAT` after module loading.

The same two nodes execute the following sequence. A failed arm retains its
return code; the parent returns failure if any phase fails. TT arms have a
`600 s` timeout and MDS phases a `900 s` timeout. Scheduler success alone
is not application validation.

| Order | Arm | Layout | Reduction | Compilation | Attention |
|---:|---|---|---|---|---|
| `1` | MDS before | historical-wrapper compatibility | FP32 local/BF16 wire | none | IPEX xetla |
| `2` | `hs12regions` | HSDP `12 x 2` | FP32, deferred | SwiGLU local region + loss | Torch SDPA |
| `3` | `hs12loss` | HSDP `12 x 2` | FP32, deferred | loss | Torch SDPA |
| `4` | `hs12eager` | HSDP `12 x 2` | FP32, deferred | none | Torch SDPA |
| `5` | `hs12blk4` | HSDP `12 x 2` | FP32, deferred | loss | four-block causal candidate |
| `6` | `hs4blk4` | HSDP `4 x 6` | FP32, deferred | loss | four-block causal candidate |
| `7` | `ddpz1blk4` | native DDP + optimizer ZeRO-1 | FP32 local/BF16 wire and weight all-gather | loss | four-block causal candidate |
| `8` | MDS after | historical-wrapper compatibility | FP32 local/BF16 wire | none | IPEX xetla |

All HSDP arms use `reshard_after_forward=never`, deferred gradient reduction
and `CCL_OP_SYNC=1`. The region-compilation arm compiles only the supported
local SwiGLU activation and loss. No whole-model graph compilation is claimed.
The block-causal attention candidate has approximate BF16 microbenchmark
numerics; full training equivalence remains unproven.

MDS uses Torch `2.10.0a0+git449b176` and DS `0.18.5`; TT uses Torch
`2.14.0+xpu`. MDS's compatibility mode selects `BF16_Optimizer` through
internal stage `1` and FP32 accumulation. This reproduces the old DS wrapper's
types and sharding behavior, not the historical DS `0.17.5` runtime.
Optimizer Hessian updates are disabled in TT to match the pinned MDS path.
The shared native initialization is `global_step138650`, converted to TT DCP.
This does not by itself establish correct live post-wrap weights.

## Required evidence

Status at submission: **running; no winner recorded**.

Record all eight phase/variant return codes, rank completion and optimizer
step counts, consumed-input hashes, complete loss histories, peak allocated
and reserved memory, outer compilation timer and update-12 rank-0 traces.
Timing uses updates `21-32`, twelve intervals. Preserve TT's synchronized
per-rank step timers separately from loss-event intervals and MDS's direct
iteration timer. Their framework boundaries differ.

Use the report's common `11,186,515,968 FLOPs/token` and
`298.1888 TFLOP/s/FLAT tile` normalization. Useful model TFLOPs and MFU exclude
extra attention/block work and are not hardware-counter utilization.

The before/after MDS controls bound drift during this allocation. Serial TT
arms share a job-local compiler cache root; a later short compile timer is
not an independent cold-start estimate. Two-node results do not establish
Aurora production-scale parity.

Related: [recovered handoff audit](2026-11-01-handoff-audit.md).
