# AuroraGPT-2B: XPU throughput recovery on Sunspot

Follows the [November 2 recovery comparison](https://github.com/saforem2/torchtitan/blob/ezpz/torchtitan/experiments/ezpz/docs/experiments/agpt/sunspot/2026-11-02-recovery-comparison.md), which left the historical-throughput gate open.

## Results

Restoring per-block `torch.compile` raises TT native DDP/ZeRO-1 from `75.84` to
`79.42`-`80.20` model TFLOP/s/tile (`+5.7%` in the same allocation). At the
production-shaped batch (two sequences per rank per update) TT now runs
`2.3`-`2.6%` faster than the MDS SophiaG control in the same allocation
(`2.2852` vs `2.344`-`2.347 s/update` in `12482111`; `2.3079` vs `2.363 s` in
`12482126`).

MDS reproduces its historical number. Its `8dscnx65` configuration (GBS `384`,
accumulation `16`, AdamW, ZeRO-1) gives `89.76` today against `89.99` in March.
The lower November 2 MDS rate came from accumulation `2` and the
`BF16_Optimizer` compatibility path, not from a regression.

The historical TT anchor is `6.1%` below its July result on identical code and
runtime: `88.00` vs `93.75`. That gap belongs to the machine (node image,
driver, or node placement), not to either framework.

| Arm | Accumulation x microbatch | s/update | tok/s/tile | Model TFLOP/s/tile |
|---|---|---:|---:|---:|
| MDS control, SophiaG, `BF16_Optimizer` (`x4`) | `2 x 1` | `2.344`-`2.376` | `6895`-`6990` | `77.13`-`78.19` |
| MDS AdamW, ZeRO-1 (`x2`) | `2 x 1` | `2.203`-`2.214` | `7402`-`7436` | `82.80`-`83.18` |
| MDS SophiaG, `BF16_Optimizer` | `16 x 1` | `17.206` | `7618` | `85.22` |
| MDS AdamW, ZeRO-1 (historical `8dscnx65`) | `16 x 1` | `16.335` | `8024` | `89.76` |
| TT DDP/ZeRO-1, loss compile (November 2 path) | `2 x 1` | `2.4167` | `6780` | `75.84` |
| **TT DDP/ZeRO-1, per-block compile** (`x2`) | `2 x 1` | `2.2852`-`2.3079` | `7099`-`7170` | `79.42`-`80.20` |
| TT DDP/ZeRO-1, per-block compile, AdamW | `2 x 1` | `2.3031` | `7114` | `79.58` |
| TT DDP/ZeRO-1, per-block compile | `1 x 2` | OOM | -- | -- |
| TT HSDP R2/S12, per-block compile | `2 x 1` | `2.6234` | `6245` | `69.86` |
| TT HSDP R2/S12, per-block compile | `1 x 2` | `2.4914` | `6576` | `73.56` |
| TT HSDP R2/S12, per-block compile, AdamW | `1 x 2` | `2.3572` | `6951` | `77.75` |
| TT anchor `4a72fa1bf`, 1 node, global FSDP | `1 x 2` | `2.083` | `7866` | `88.00` |
| TT anchor `4a72fa1bf`, 1 node, global FSDP | `1 x 1` | `1.092` | `7505` | `83.95` |

All rows are `24` FLAT tiles on `2` nodes, sequence `8192`, `32` updates,
except the anchor (`12` tiles, `1` node, `20` updates). Rows marked `x2`/`x4`
are repeats in separate allocations.

## Workload and timing

| Setting | Value |
|---|---|
| Source | TT `586f1a542` (branch `codex/agpt-perf-recovery-20261102`); MDS `9f407ed9e` |
| TT runtime | Torch `2.14.0+xpu` |
| MDS runtime | Torch `2.10.0a0+git449b176`, DeepSpeed `0.18.5` |
| Tokens per update per tile | `16384` (`2 x 8192`) except the GAS-16 MDS rows (`131072`) |
| Timing window | Updates `21`-`32` (anchor: `9`-`20`) |
| FLOPs/token | `11,186,515,968`, the recovery report's normalization |

TT time is the mean over updates of the slowest rank's synchronized step
record (`rank-*.steps.jsonl`, `BENCH_STEP_TIMINGS=1`); the standard deviation
across updates is `1.7`-`3.9 ms` except HSDP `1 x 2` (`59 ms`). MDS time is its
native `elapsed_time_per_iteration_ms`. The anchor uses per-update
`time_metrics/end_to_end(s)` from W&B; the same calculation reproduces
`93.75` for `6tc71of4`. Log-line timestamps have `1 s` resolution (about `4%`
at these step times) and are not used.

## What changed in TT

Upstream replaced `apply_compile`, which compiled each `TransformerBlock` with
`fullgraph=True`, with named `@local_compile` regions. `compile.components=loss`
on the current tree therefore leaves the blocks eager; the historical
`components=[model, loss]` runs compiled them. `EZPZ_AGPT_COMPILE_BLOCKS`
(`agpt/parallelize.py`) restores the per-block compile after AC and before
FSDP/native DDP. It also turns off `inplace_wgrad_accum` inside compiled blocks:
a traced backward cannot add into `weight.grad` in place. The launcher selects
it with `BENCH_COMPILE_MODE=blocks-loss`.

Local batch `> 1` crashed on the current tree. The flat `[T]` token layout
(upstream `#4121`) makes RoPE read `cache[:T]`, which covers one sequence. The
blendcorpus loader now emits per-row positions when a microbatch holds more
than one sequence and document positions are off. RoPE then rotates every row
as it does at local batch `1`. SDPA builds no masks from positions, so
attention is unchanged.

Native DDP at `1 x 2` runs out of memory: `49.08 GiB` allocated plus a
`7.81 GiB` request on a `63.98 GiB` tile, the size of one FP32
`[8192, 256128]` logits tensor. HSDP at `1 x 2` fits (`55.20 GiB`).

## Reading the comparison

At equal tokens per update, SophiaG on both sides:

- TT DDP/ZeRO-1 with per-block compile is `2.3`-`2.6%` faster than MDS
  `BF16_Optimizer` in the same allocation.

AdamW on both sides:

- MDS gains `6.6%` throughput from SophiaG/`BF16_Optimizer` to AdamW/ZeRO-1. That arm uses
  DeepSpeed's runtime ZeRO-1 path. Its wrapper is not logged (the
  `DS_RUNTIME` witness runs only for SophiaG); based on the earlier login-node
  probe it is `DeepSpeedZeroOptimizer` with BF16 accumulation, so it changes
  numerics as well as the optimizer (**unverified**).
- TT gains nothing from AdamW (`2.3031` vs `2.3079 s`); its sharded optimizer
  step is already small.
- TT AdamW has `4.3%` lower throughput than MDS AdamW/ZeRO-1 (`2.3031` vs `2.203 s` in `12482126`).

Larger accumulation is worth `8`-`10%` to MDS (`77.5` -> `85.2` SophiaG,
`83.2` -> `89.8` AdamW). Aurora production runs two sequences per rank per
update (GBS `6144` at `3072` ranks), so the `2 x 1` rows are the ones that
apply to production. The historical `89.99` is a GBS-384 number.

Microbatch `2` is worth `4.8%` on the anchor (`83.95` -> `88.00`) and `5.3%` on
current HSDP (`69.86` -> `73.56`). Native DDP cannot use it without reducing
logits memory.

## Open

- Recover the `6.1%` machine gap. The anchor ran on one node; repeat it on
  other nodes, and compare against the July node image and driver.
- Fit native DDP at `1 x 2`, for example with chunked loss so FP32 logits are
  not materialized for both sequences at once.
- `mode="max-autotune"` (`EZPZ_AGPT_COMPILE_BLOCKS=max-autotune`) is wired but
  not measured; `xcs4xrqe` used it.
- Loss and gradient equivalence remain open, as in the
  [recovery report](https://github.com/saforem2/torchtitan/blob/ezpz/torchtitan/experiments/ezpz/docs/experiments/agpt/sunspot/2026-11-02-recovery-comparison.md).

## Jobs

| Job | Content | Exit |
|---|---|---|
| `12482111` | MDS controls; TT DDP base, `1 x 2`, per-block compile, per-block `1 x 2` | `1` (both `1 x 2` arms: RoPE shape error) |
| `12482112` | MDS control, GAS 16, AdamW/ZeRO-1, historical; TT HSDP per-block `1 x 2` | `1` (HSDP arm: `inplace_wgrad_accum` rejected under compile) |
| `12482114` | TT anchor `4a72fa1bf`, local batch `2` and `1` | `0` |
| `12482126` | MDS control and AdamW/ZeRO-1; TT DDP per-block compile, `1 x 2`, AdamW | `1` (both DDP `1 x 2` arms: OOM) |
| `12482127` | MDS control and AdamW/ZeRO-1; TT HSDP per-block compile `2 x 1`, `1 x 2`, AdamW | `0` |

Jobs `12482111`/`12482112` ran at `83fb8ad7b`; `12482126`/`12482127` ran at
`586f1a542`, which adds the positions, compiled-block WGRAD and AdamW changes.
The anchor ran from a clean `4a72fa1bf` worktree with the original RC2 conda
environment and `fw_ssp` venv (`benchmark_agpt_historical_anchor.sh`). Its
standard preflight imports torch without the job's module and conda setup, so
the runtime was checked by hand on the login node: torch
`2.13.0a0+gitcf30153`, ezpz `0.23.0`, `agpt_2b_real` with compile
`[model, loss]`.

Outputs are under
`/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan-agpt-replicated-20261031-perf/outputs/`
(TT) and `.../torchtitan-agpt-replicated-20261031/{logs,outputs}/mds-matrix-*`
(MDS). W&B runs are online in `aurora_gpt/torchtitan.ezpz.train`.
