# AuroraGPT-2B: XPU throughput recovery on Sunspot

Follows the [November 2 recovery comparison](https://github.com/saforem2/torchtitan/blob/ezpz/torchtitan/experiments/ezpz/docs/experiments/agpt/sunspot/2026-11-02-recovery-comparison.md), which left the historical-throughput gate open.

## Results

Restoring per-block `torch.compile` raises TT native DDP/ZeRO-1 from `75.84` to
`79.42`-`80.20` model TFLOP/s/tile (`+5.7%` in the same allocation). At the
production-shaped batch (two sequences per rank per update) TT now runs
`2.3`-`2.6%` faster than the MDS SophiaG control in the same allocation
(`2.2852` vs `2.344`-`2.347 s/update` in `12482111`; `2.3079` vs `2.363 s` in
`12482126`).

Opt-in block-causal attention (`EZPZ_XPU_ATTN_CAUSAL_BLOCKS=4`) brings the
same TT arm to `2.0788 s/update`, `88.17` model TFLOP/s/tile, `10.5%` faster
than its later same-job control (`12482301`). The op matches the default
kernel's accuracy on real training calls. Its training gradients differ from
the default path's by as much as any rounding-level change to the forward
output does; a longer loss comparison should come before it is turned on
([Same-batch gradients](#same-batch-gradients)).

MDS reproduces its historical number. Its `8dscnx65` configuration (GBS `384`,
accumulation `16`, AdamW, ZeRO-1) gives `89.76` today against `89.99` in March.
The lower November 2 MDS rate came from accumulation `2` and the
`BF16_Optimizer` compatibility path, not from a regression.

The historical TT anchor is `6.1%` below its July result on identical code and
runtime: `88.00` vs `93.75`. That rerun was the first launch on a cold node;
later launches on the same node give `92.6`-`95.1`. The slow first launch
costs TT `2`-`7%` and its cause is still open
([Anchor launch environment](#anchor-launch-environment)).

At the wider 2025 MDS shape (`z2rrt02w`: hidden `4096`, sequence `4096`,
accumulation `16`) TT DDP/ZeRO-1 is `26%` slower than MDS, and `14%` slower
with block-causal attention
([TT at the 2025 high-throughput shape](#tt-at-the-2025-high-throughput-shape)).

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
| TT anchor `4a72fa1bf`, 1 node, global FSDP, first launch | `1 x 2` | `2.083` | `7866` | `88.00` |
| TT anchor `4a72fa1bf`, 1 node, global FSDP, later launch (`x10`) | `1 x 2` | `1.927`-`1.979` | `8278`-`8501` | `92.63`-`95.10` |
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
| FLOPs/token | `11,186,515,968`, the recovery report's normalization; `1.3433e10` at the `z2rrt02w` shape, whose TFLOP/s are not comparable with the AGPT-2B rows |

TT time is the mean over updates of the slowest rank's synchronized step
record (`rank-*.steps.jsonl`, `BENCH_STEP_TIMINGS=1`); the standard deviation
across updates is `1.7`-`3.9 ms` except HSDP `1 x 2` (`59 ms`). MDS time is its
native `elapsed_time_per_iteration_ms`. The anchor uses per-update
`time_metrics/end_to_end(s)` from W&B; the same calculation reproduces
`93.75` for `6tc71of4`. Log-line timestamps have `1 s` resolution (about `4%`
at these step times) and are not used.

## Anchor launch environment

The July anchor ran from an interactive session; the November rerun ran under
PBS batch with `CCL_OP_SYNC=1`, `CCL_PROCESS_LAUNCHER=pmix` and
`OMP_NUM_THREADS=208`, none of which July set. None of these explains the gap.
What does is launch order: the first TT launch on a node that has not run this
workload in the last ~10 minutes is slow for its whole lifetime, and every
later launch in the same job matches July.

| Job | Node | First launch | Later launches |
|---|---|---:|---:|
| `12482147` | `x1921c2s6b0n0` (July's node) | `88.00` (batch env) | `93.49`, `93.36`, `93.24` (July env, no `CCL_OP_SYNC`, no `OMP_NUM_THREADS`) |
| `12482150` | `x1922c6s6b0n0` | `88.42` (July env) | `93.14` (batch env), `93.70` |
| `12482152` | `x1922c6s5b0n0` | `86.96`-`89.07` over 80 updates | -- |
| `12482153` | `x1922c6s4b0n0`, 3 min after `12482151` | `94.38` | `94.63` |
| `12482155` | `x1922c6s6b0n0`, 12 min after `12482150` | `89.80` | `93.53` |
| `12482158` | `x1921c0s3b0n0` | `90.57` | `95.10` |
| `12482243`-`12482246` | four fresh nodes | `90.27`-`92.18` | `93.62`-`94.28` |

Model TFLOP/s/tile, updates `9`-`20`, local batch `2`. A slow launch also
jitters: the per-update standard deviation is `20`-`170 ms` against `2`-`6 ms`
for a later launch, and its first update takes `40`-`47 s` against `17`-`19 s`.

Ruled out, each against a same-job control:

- Environment: each of the three variables removed on its own gives the same
  later-launch rate; the first launch is slow in the July environment too.
- Node, image and driver: July's own node shows it, and later launches on
  every node reach `92.6`-`95.1`. Kernel `53.66` (July) vs `53.78` (now).
- Hardware: bf16 GEMM runs at `139`-`170` TFLOP/s on every tile of a cold node.
- Page cache: reading the conda env and venv shared libraries first
  (`12482155`) leaves the first launch slow.
- CPU contention: a 15 s process sampler (`12482158`) shows no other busy
  process during the slow launch.
- XeLink: 90 s of 12-rank XCCL all-gather/reduce-scatter first (`12482238`)
  leaves the first launch slow (`87.01`).
- Expandable segments: one launch without `PYTORCH_XPU_ALLOC_CONF` reached
  `93.40` (`12482239`), but two replicates gave `91.19` and `90.27`.
- A 90 s GEMM prewarm (`12482151`) and a 55 GiB device allocation
  (`12482157`) leave the first launch at `90.8` and `90.6`.

MDS does not show it: its first matrix phase runs at the same rate as its last
(`2346.6` vs `2343.9 ms`, `12482111`). Production TT jobs are one launch on
nodes that are usually cold, so they likely pay this `2`-`7%`. The matrix TT
arms above never ran first in their job.

## MDS at the 2025 high-throughput shape

The AuroraGPT W&B project has many Aurora MDS runs above `100` TFLOP/s/tile,
none at the current AGPT-2B shape. The current shape (hidden `2048`, `12`
layers, vocab `256000`, sequence `8192`) peaks at `93.0` (`1j4xxxky`, Aurora,
October 2026). The runs above `100` are a wider, shallower model at sequence
`4096` with the Llama2 vocab, for example
[`z2rrt02w`](https://wandb.ai/aurora_gpt/AuroraGPT/runs/z2rrt02w) (July 2025,
1 Aurora node): hidden `4096`, `10` layers, `32` heads, `8` KV heads, FFN
`11008`, GBS `192`, ZeRO-1, AdamW, `102.9` TFLOP/s/tile at `7659` tok/s/tile.

`benchmark_mds_historical_shape.sh` reruns `z2rrt02w`'s argv on today's
Sunspot MDS stack (`9f407ed9e`, torch `2.10.0a0+git449b176`, DeepSpeed
`0.18.5`) from scratch on the Llama2-tokenized books data. Job `12482248`
gives `109.18` TFLOP/s/tile at `8128` tok/s/tile over iterations `21`-`32`,
`6.1%` above July 2025. MDS has not lost throughput; the `~90` on AGPT-2B is
the model shape.

## TT at the 2025 high-throughput shape

`mds-4096x10` (`agpt_mds_shape_benchmark_adamw`) is the `z2rrt02w` model in
TT: `2,034,323,456` parameters, the count W&B records for that run. Job
`12482280` ran TT and MDS on one node (`x1922c2s0b0n0`) with the same batch
(microbatch `1`, accumulation `16`, GBS `192`, sequence `4096`), AdamW, from
random init on the same books data. A discarded 8-update TT launch ran first,
so no measured TT arm is a cold first launch.

| Arm (phase order) | s/update | tok/s/tile | TFLOP/s/tile |
|---|---:|---:|---:|
| TT DDP/ZeRO-1, per-block compile (2) | `10.100` | `6489` | `87.17` |
| TT FSDP, per-block compile (3) | `15.122` | `4334` | `58.22` |
| MDS ZeRO-1 (4) | `7.977` | `8216` | `110.36` |
| TT DDP/ZeRO-1, per-block compile (5) | `10.042` | `6526` | `87.66` |

Updates `21`-`32`. TFLOP/s uses MDS's own count at this shape,
`1.3433e10` FLOPs/token (`110.36e12 / 8216`). TT DDP takes `26%` longer per
update than MDS. At AGPT-2B it was `2.3`-`2.6%` faster.

The rank-0 trace of update `12` (phase 5) shows TT DDP is compute-bound: the
GPU is busy `9.72` of `9.92 s`, and gradient reduction takes `0.19 s`.

| Kernel | Time per update | Rate |
|---|---:|---:|
| GEMM | `4.38 s` | `171` TFLOP/s (non-attention FLOPs) |
| Flash-attention backward | `2.32 s` | `38` TFLOP/s (dense count) |
| Flash-attention forward | `0.55 s` | `81` TFLOP/s (dense count) |
| Elementwise binary | `0.48 s` | -- |

Attention rates use MDS's dense (non-causal) attention FLOPs, `12 s h` per
token per layer; a causal count halves them. The backward/forward ratio is
the point: `4.3x` against about `2.5x` for a kernel that does the same work
per FLOP.

The attention backward is the outlier. The torch `2.14` XPU flash backward
does not skip fully masked key tiles under `is_causal`; IPEX xetla, which MDS
uses, does (`24.7` vs `10.2 ms` causal at the AGPT-2B shape,
[handoff audit](2026-11-01-handoff-audit.md#attention-and-communication)).

The opt-in block-causal path (`EZPZ_XPU_ATTN_CAUSAL_BLOCKS`) splits the
sequence into `n` chunks and computes only the lower-triangle block pairs.
Job `12482281` (`x1922c4s2b0n0`, a discarded warm-up launch first) ran it
between two controls:

| Arm (phase order) | s/update | tok/s/tile | TFLOP/s/tile |
|---|---:|---:|---:|
| Control (2) | `10.073` | `6506` | `87.39` |
| Block-causal, `4` chunks (3) | `9.111` | `7193` | `96.62` |
| Block-causal, `8` chunks (4) | `9.285` | `7058` | `94.81` |
| Control (5) | `9.777` | `6703` | `90.04` |

Four chunks is `7`-`10%` faster than the controls, whose spread is `3%`.
Update `12` traces: flash backward drops from `2.26` to `1.53 s` and the
extra merge work adds `0.16 s`; GEMM is unchanged (`4.57` vs `4.58 s`). Final
losses after `32` updates are `6.656`/`6.698` (controls) and `6.667`
(4 chunks), a smoke check, not an equivalence test. The two controls share a
seed and node and still differ by `0.042`, so the runs are not deterministic;
an equivalence test has to be stated against that spread.

TT with block-causal attention is still `14%` slower than MDS (`9.111` vs
`7.977 s`). That compares two jobs on two nodes; the TT controls agree across
jobs (`10.04`-`10.10` and `9.78`-`10.07 s`) and MDS agrees with `12482248`
within `1%`. Non-GEMM, non-attention kernels take `2.2 s` per update in TT;
that is the next place to look, with an MDS trace at this shape for
comparison.

FSDP spends `3.01 s` in all-gather and `2.43 s` in reduce-scatter per
update. This arm ran with the launcher default
`--parallelism.no-fsdp-defer-gradient-reduction`, so it reduce-scatters and
re-gathers on all `16` microbatches. Earlier work ran accumulation `2`, where
that costs little. `58.22` is not TT FSDP's rate at this shape;
`BENCH_FSDP_DEFER=1` is untested here.

## Block-causal attention at AGPT-2B

Job `12482301` (`x1922c1s2b0n0`, `x1922c1s3b0n0`) ran the best TT AGPT-2B arm
(DDP/ZeRO-1, per-block compile, SophiaG, `2 x 1`, `24` tiles) with and
without block-causal attention, after a discarded 8-update warm-up launch.

| Arm (phase order) | s/update | tok/s/tile | Model TFLOP/s/tile |
|---|---:|---:|---:|
| Control (2) | `2.4511` | `6684` | `74.78` |
| Block-causal, `4` chunks (3) | `2.0788` | `7882` | `88.17` |
| Block-causal, `8` chunks (4) | `2.0872` | `7850` | `87.81` |
| Control (6) | `2.2976` | `7131` | `79.77` |

Updates `21`-`32`, slowest rank per update. The second control matches
`12482126` (`2.3079 s`). The first control has `0.56 s` of idle GPU time in
its update-`12` trace and a `35 ms` step standard deviation, the pattern of the
slow launches in [Anchor launch environment](#anchor-launch-environment),
although it was not the job's first launch. Against the second control, four
chunks is `9.5%` less time per update.

| Arm | Flash backward | Flash forward | GEMM | Other | Collectives |
|---|---:|---:|---:|---:|---:|
| Control (6) | `0.613 s` | `0.148 s` | `0.904 s` | `0.451 s` | `0.123 s` |
| `4` chunks | `0.384 s` | `0.120 s` | `0.910 s` | `0.481 s` | `0.116 s` |
| `8` chunks | `0.355 s` | `0.120 s` | `0.911 s` | `0.511 s` | `0.125 s` |

Rank-0 kernel time in the profiled update. Eight chunks skips more masked
work but adds merge work; four chunks is as fast.

The MDS control phase did not run: the matrix needs `BENCH_EXPECTED_MDS_HEAD`
and the submission omitted it (exit `99` before launch). Against the earlier
MDS SophiaG controls (`2.344`-`2.376 s`, other allocations), four chunks is
`11`-`13%` faster.

Losses at update `1` agree: `17.741` (both controls), `17.742` (`4` chunks),
`17.733` (`8` chunks). Later updates are not deterministic: the two controls
differ by up to `0.62` (update `16`). Over updates `24`-`31` the 4-chunk arm is
above both controls on every update, by `0.03`-`0.14` against the closer one.
That is inside the control spread seen earlier in the run but one-sided.

### Same-batch gradients

`probe_block_causal_grads.py` (`BENCH_BLOCK_PROBE=1`) replays one update with
the same weights and microbatches under default and block-causal attention and
compares the all-reduced gradients on rank `0`. Job `12482320`, `12` tiles,
DDP/ZeRO-1 with per-block compile, `torch._dynamo.reset()` between variants:

| Update | Pair | Relative L2 | Cosine | Max abs diff |
|---:|---|---:|---:|---:|
| `1` | default repeat vs default | `1.80e-2` | `0.99986` | `3.0` |
| `1` | `4` chunks vs default | `0.938` | `0.458` | `164.5` |
| `1` | `8` chunks vs default | `1.80` | `0.655` | `391` |
| `24` | default repeat vs default | `8.36e-3` | `0.99998` | `0.0625` |
| `24` | `4` chunks vs default | `9.93e-2` | `0.99506` | `0.570` |
| `24` | `8` chunks vs default | `0.199` | `0.98040` | `1.22` |

The losses of the replayed updates agree (`1.47185` vs `1.47190` at update
`1`). At update `1` the block-causal gradient is `50`-`100x` further from the
default than the default is from itself.

The difference grows with backward depth. The LM-head gradient agrees
(relative L2 `6.9e-3`); the worst parameters are in layers `0` and `1` and the
embedding. At update `1` the gradients are also badly conditioned: the
embedding gradient norm is `1987` against `9.9` for the LM head. A small
attention-backward difference amplified through the early layers would look
like this.

The attention op alone is fine in eager mode. Job `12482321` measures it
against an FP32 math reference at the AGPT-2B shape (max abs error, torch
`2.14`):

| Case | out | dQ | dK | dV |
|---|---:|---:|---:|---:|
| Default flash | `0.0077` | `0.0103` | `0.0219` | `0.0303` |
| Block-causal, `4` chunks | `0.0077` | `0.0103` | `0.0262` | `0.0443` |
| Block-causal, `8` chunks | `0.0077` | `0.0103` | `0.0227` | `0.0332` |

Compile is not the cause. Job `12482325` runs the same op on one tile under
`torch.compile(fullgraph=True)`; relative L2 error against FP32 is the same
eager and compiled (`dK` `3.37e-3` block-causal, `2.93e-3` default). The eager
probe arm ran out of device resources at its first backward.

Amplification of a rounding-sized difference is not the cause either. Job
`12482328` puts the block-causal vs default difference at the op at `1.3e-3`
(`out`, `dQ`) to `3.7e-3` (`dK`, `dV`) relative L2, with random inputs. Job
`12482327` multiplies the default path's q/k/v gradients by
`1 + sigma * N(0, 1)` in training and compares against the unperturbed
default:

| Update `1` | Relative L2 | Cosine |
|---|---:|---:|
| Default repeat | `1.58e-2` | `0.99989` |
| `sigma = 1e-3` | `1.93e-2` | `0.99982` |
| `sigma = 4e-3` | `2.13e-2` | `0.99978` |
| Block-causal, `4` chunks | `0.928` | `0.468` |

Noise at least as large as the op-level block-causal difference barely moves
the training gradients. Random noise is not a systematic difference, but a
`45x` gap points at something the random-input check does not exercise:
the training call's dtypes, strides (q/k/v are views of one fused projection)
or attention patterns.

Those are cleared too. Job `12482329` saved the first and last attention calls
of training (bf16 under autocast, q/k/v strided views of the fused
projection) at updates `1` and `24`; job `12482330` replays them. On every
call block-causal's error against FP32 matches the default kernel's, and a
dense bf16 copy of the inputs gives identical numbers:

| Call | Default `dQ` / `dK` vs FP32 | `4` chunks `dQ` / `dK` vs FP32 | `4` chunks vs default `dQ` / `dK` |
|---|---:|---:|---:|
| Update `1`, layer `0` | `8.62e-3` / `3.52e-2` | `8.78e-3` / `3.56e-2` | `2.82e-3` / `4.97e-3` |
| Update `1`, layer `11` | `7.41e-3` / `2.57e-3` | `8.75e-3` / `2.76e-3` | `8.65e-3` / `3.28e-3` |
| Update `24`, layer `0` | `2.89e-2` / `0.248` | `2.89e-2` / `0.248` | `2.41e-3` / `3.93e-3` |
| Update `24`, layer `11` | `1.37e-2` / `2.25e-3` | `1.67e-2` / `2.86e-3` | `1.81e-2` / `3.45e-3` |

Relative L2. The default kernel itself is `25%` off FP32 in `dK` at layer `0`,
update `24`.

The same job ran the probe with blocks not compiled (loss compile only):

| Arm | Update | `4` chunks vs default | Default repeat |
|---|---:|---:|---:|
| Blocks not compiled (`12482329`) | `24` | `1.18e-2` | `4.32e-2` |
| Per-block compile (`12482327`) | `24` | `5.36e-2` | `6.75e-3` |
| Blocks not compiled | `1` | `2.53` | `1.86e-2` |
| Per-block compile | `1` | `0.928` | `1.58e-2` |

Job `12482333` reran both arms with `blk1`, one causal flash call through the
block-causal autograd function (same math as default):

| Arm | Update | Default repeat | `blk1` vs default | `4` chunks vs default |
|---|---:|---:|---:|---:|
| Per-block compile | `1` | `1.81e-2` | `1.25e-2` | `0.928` |
| Per-block compile | `24` | `3.17e-2` | `2.41e-2` | `0.411` |
| Blocks not compiled | `1` | `2.87e-2` | `2.44e-2` | `2.60` |
| Blocks not compiled | `24` | `1.40e-2` | `8.52e-3` | `7.78e-2` |

`blk1` is at the default path's own spread everywhere, so the autograd
wrapper and compile are cleared; the difference comes with chunking. The
uncompiled update-`24` result (`7.78e-2`) disagrees with `12482329`
(`1.18e-2`), so a single replay at update `24` is not stable evidence either
way. The chunked op is correct on the saved training calls, so the open
question is whether the training state is this sensitive to the chunked
forward output, which differs from default by `0.7`-`2e-3` relative.

It is. Job `12482340` (blocks not compiled) multiplies the default attention
output by `1 + sigma * N(0, 1)` in the forward pass, gradient passing through:

| Variant vs default | Update `1` | Update `24` |
|---|---:|---:|
| Default repeat | `1.50e-2` | `1.15e-2` |
| q/k/v gradient noise, `sigma = 4e-3` | `2.23e-2` | `1.19e-2` |
| Forward-output noise, `sigma = 1e-3` | `2.42` (cosine `-0.61`) | `8.04e-2` |
| Forward-output noise, `sigma = 4e-3` | `0.805` | `0.336` |
| Block-causal, `4` chunks | `2.56` | `0.130` |
| Block-causal, `8` chunks | `0.917` | `0.174` |

Relative L2. A forward perturbation at bf16 rounding size moves these
gradients as far as block-causal does; a backward perturbation does not. The
default path's repeat looks tight only because its forward is deterministic.

Conclusion: block-causal attention is correct. Its op-level error matches the
default kernel's on real training calls, the wrapper and compile are cleared,
and its effect on training gradients is that of a rounding-level forward
change, which these gradients are chaotic under. A same-batch gradient
comparison cannot separate it from the default path. A loss-curve A/B over a
few hundred updates (several seeds per arm) is the test that can, and should
come before it is turned on by default.

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

- Find the cause of the slow first TT launch on a cold node (`2`-`7%`). Its
  first update is `25 s` slower, so the next test is a profile of a cold first
  launch against a later launch on the same node.
- Close the remaining `14%` TT gap at the `z2rrt02w` shape
  ([TT at the 2025 high-throughput shape](#tt-at-the-2025-high-throughput-shape)):
  profile MDS at that shape and compare the `2.2 s` of other TT kernels.
- Run a loss-curve A/B of block-causal attention (`4` chunks) against the
  default path over a few hundred updates, several seeds per arm, before
  turning it on ([Same-batch gradients](#same-batch-gradients)). It is worth
  `9.5%` at AGPT-2B and `7`-`10%` at the `z2rrt02w` shape.
- Rerun TT FSDP at the `z2rrt02w` shape with deferred gradient reduction.
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
| `12482147`, `12482150`-`12482158`, `12482238`-`12482246` | TT anchor launch-order and cold-node tests ([Anchor launch environment](#anchor-launch-environment)) | `0` |
| `12482248` | MDS at the `z2rrt02w` shape | `0` |
| `12482280` | TT DDP/ZeRO-1 (`x2`), TT FSDP and MDS at the `z2rrt02w` shape, `ee9c5fd4a` | `0` |
| `12482281` | TT DDP/ZeRO-1 at the `z2rrt02w` shape: control (`x2`), block-causal `4` and `8` chunks, `ee9c5fd4a` | `0` |
| `12482301` | TT DDP/ZeRO-1 at AGPT-2B: control (`x2`), block-causal `4` and `8` chunks, `ee9c5fd4a` | `1` (MDS phase: `BENCH_EXPECTED_MDS_HEAD` unset) |
| `12482320` | Block-causal gradient probe, eager and per-block compile, `f214641b2` | `1` (eager arm: `UR_RESULT_ERROR_OUT_OF_RESOURCES`) |
| `12482321` | Attention microbenchmark against FP32: default, block-causal `4` and `8` chunks, `f214641b2` | `0` |
| `12482325`, `12482328` | One-tile block-causal check, eager vs compiled, against FP32 and default, `b60c37a2c`/`ad261add3` | `0` |
| `12482327` | Gradient probe with q/k/v gradient-noise calibration, per-block compile, `ad261add3` | `0` |
| `12482329`, `12482330` | Gradient probe, blocks not compiled, saving training attention calls; replay of those calls, `32a3595f3` | `0` |
| `12482333` | Gradient probe with `blk1`, per-block compile and blocks not compiled, `5e73faa7f` | `0` |
| `12482340` | Gradient probe with forward-output noise, blocks not compiled, `b5a221174` | `0` |

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
