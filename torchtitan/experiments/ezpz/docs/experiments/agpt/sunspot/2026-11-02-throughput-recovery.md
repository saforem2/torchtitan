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
than its later same-job control (`12482301`). In a 400-update loss A/B from
random init (three seeds) its loss stays within `0.013` of the default path
in every window, against a seed-to-seed spread of `0.018`-`0.034`
([Loss-curve A/B](#loss-curve-ab)).

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
Storing the ZeRO-1 model in BF16 (opt-in `BENCH_DDP_ZERO1_PARAMS=bfloat16`)
closes that to `0.9%`: `8.08 s/update`, `108.9` TFLOP/s/tile, half the peak
memory ([BF16 model storage](#bf16-model-storage)). At AGPT-2B over `400`
updates and three seeds its loss stays inside the seed spread; it is
`2.5`-`6.2%` faster there. It stays opt-in.

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
the next section compares them with an MDS trace.

### TT and MDS kernels at this shape

Job `12482583` (`x1922c4s2b0n0`, the node of `12482281`) ran MDS with
`MDS_PROFILE=1`, a rank-0 `torch.profiler` trace of update `12`, between TT
arms. MDS takes `8.008 s/update` (updates `21`-`32`); the traced update takes
`8.166 s`. `summarize_xpu_trace.py` buckets both traces by kernel name. The TT
column is the block-causal (`4` chunks) trace from `12482281`, whose update
`12` takes `9.216 s`. The block-causal trace from `12482583` is not used: its
update `12` took `10.23 s` and includes `0.81 s` in two `int64`
`oneccl_reduce_base` calls that no other trace has.

| Seconds per update (rank 0) | MDS | TT block-causal | TT - MDS |
|---|---:|---:|---:|
| GEMM (`gemm_kernel`, `1968` calls in both) | `4.443` | `4.581` | `+0.14` |
| Attention backward (with auxiliary kernels) | `1.099` | `1.635` | `+0.54` |
| Attention forward | `0.608` | `0.473` | `-0.14` |
| Casts, elementwise, copies, reductions, fills | `1.417` | `1.919` | `+0.50` |
| Norms | `0.162` | `0.205` | `+0.04` |
| Communication, optimizer | `0.104` | `0.139` | `+0.04` |
| Busy (union of kernel intervals) | `7.833` | `8.950` | `+1.12` |

The busy-time gap matches wall clock: `9.111` vs `8.008 s/update`, `+1.10 s`.
MDS runs the same GEMMs `3%` faster. Its IPEX xetla attention backward is
`0.54 s` faster than TT's block-causal path, and TT's CUTLASS forward is
`0.14 s` faster than xetla's.

The memory-traffic row is the other half of the gap. TT spends `1.00 s` in
dtype casts (`triton_poi_fused__to_copy*`, `CopyScalarFunc`) against `0.25 s`
for MDS, and `0.45 s` in FP32 elementwise adds that MDS does not have. Every
TT cast kernel's call count is a multiple of `160` (`10` layers x `16`
microbatches): `_to_copy_0` `640`, `_to_copy_1` `480`, `_to_copy_2` `160`. The
FP32 add has `1280` calls, `80` per microbatch. With
`BENCH_DDP_ZERO1=bfloat16` the model parameters hold BF16-rounded values in
FP32 storage, so autocast casts every weight on every microbatch, and DDP
accumulates FP32 gradients. MDS (DeepSpeed `bf16`) stores BF16 parameters, so
it has no per-microbatch weight cast. MDS
instead spends `0.35 s` in `CatArrayBatchedCopy`, which TT has no equivalent
of.

At AGPT-2B (accumulation `2`) casts are `0.22 s` of a `2.18 s` busy update
(`12482301` control trace), the same `10%` share as here, so the cast cost
does not by itself explain why TT is faster than MDS at AGPT-2B and slower
here.

### BF16 model storage

`BENCH_DDP_ZERO1_PARAMS=bfloat16` (`native_ddp_zero1_param_dtype`) stores the
replicated ZeRO-1 model in BF16, as MDS does; the FP32 master stays in each
rank's optimizer shard. Autocast then has no weight casts, the residual stream
is BF16, and DDP accumulates BF16 gradients across microbatches. It is off by
default. Job `12482585` (`x1922c4s2b0n0`) alternated it with FP32 storage,
both with block-causal attention (`4` chunks), after a discarded warm-up:

| Arm (phase order) | s/update | sd | tok/s/tile | TFLOP/s/tile | Peak memory |
|---|---:|---:|---:|---:|---:|
| FP32 storage (2) | `9.463` | `0.325` | `6926` | `93.03` | `41.83 GiB` |
| BF16 storage (3) | `8.082` | `0.005` | `8109` | `108.93` | `20.28 GiB` |
| FP32 storage (4) | `8.956` | `0.007` | `7318` | `98.30` | `41.83 GiB` |
| BF16 storage (5) | `8.087` | `0.006` | `8104` | `108.86` | `20.28 GiB` |

BF16 storage is `9.7%` faster than the steadier FP32 run and `0.9%` slower
than MDS (`8.008 s`, `12482583`, same node). The FP32 arms span `8.96`-`9.46
s` in this job and `9.11 s` in `12482281`. In the update-`12` trace, casts
drop from `1.00` to `0.48 s` and elementwise kernels from `0.80` to `0.51 s`;
GEMM is `4.45 s`. Losses after `32` updates are `6.637`/`6.636` (BF16) and
`6.642`/`6.668` (FP32), a smoke check.

The loss A/B repeats the [block-causal one](#loss-curve-ab) at AGPT-2B
(`agpt_2b_loss_ab`, `12` tiles, `400` updates, no block-causal attention),
FP32 vs BF16 storage, one job per seed (`12482597`-`12482599`); seed `2` runs
BF16 first. BF16 minus FP32 loss, mean over `12` ranks and the window:

| Updates | Seed 1 | Seed 2 | Seed 3 | Mean | FP32 seed spread |
|---|---:|---:|---:|---:|---:|
| 2-10 | `+0.034` | `+0.079` | `+0.041` | `+0.051` | `0.077` |
| 11-50 | `-0.053` | `-0.014` | `-0.056` | `-0.041` | `0.074` |
| 51-100 | `-0.017` | `+0.037` | `-0.072` | `-0.017` | `0.104` |
| 101-200 | `-0.004` | `-0.002` | `-0.029` | `-0.012` | `0.068` |
| 201-300 | `-0.012` | `-0.005` | `-0.007` | `-0.008` | `0.055` |
| 301-400 | `-0.014` | `+0.001` | `+0.006` | `-0.002` | `0.053` |

Update `1` losses agree to `1e-4`. The paired differences are larger than
block-causal's (at most `0.072` per window against `0.013`; up to `0.14` in a
`10`-update window) but stay inside the seed spread and shrink toward the end.
Over updates `391`-`400` BF16 is higher in all three seeds, by `+0.004`,
`+0.005` and `+0.008`, against a final seed spread of `0.044`. Three seeds
cannot tell that from noise. The FP32 arms reproduce the controls of
`12482374`-`12482376` (final windows `4.057`/`4.013`/`4.021` vs
`4.049`/`4.023`/`4.023`).

At accumulation `2` the speedup is smaller: `2.146`/`2.235`/`2.166` against
`2.289`/`2.294`/`2.284 s/update`, `2.5`-`6.2%`. The smallest gain is seed
`2`, where BF16 ran first in the job, and step time drifts by up to `5%`
within one arm, so this range is not a precise estimate.

BF16 storage stays opt-in. It is the larger change: gradients accumulate in
BF16 across microbatches and the residual stream is BF16. Before it is a
default it needs a run past `400` updates and a multi-node run, as
block-causal does.

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
update `24`, so that call shows only that both paths share the error; the
other three confirm block-causal against FP32.

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

Relative L2. The update-`1` column is saturated: `sigma = 1e-3` moves the
gradient further than `4e-3`, and the cosine goes negative, so magnitudes
there carry no information.

The robust result is the asymmetry. Forward-output noise at `1e-3` moves the
training gradients `7`-`100x` more than q/k/v gradient noise at `4e-3`. The
default path's repeat looks tight only because its forward is deterministic.
The block-causal vs default number at update `24` (blocks not compiled) is
not stable across jobs: `1.18e-2` (`12482329`), `7.78e-2` (`12482333`),
`0.130` (`12482340`). Job `12482343` (blocks not compiled) samples five
updates in one run:

| Update | Default repeat | `blk1` | q/k/v grad noise `4e-3` | Fwd noise `1e-3` | Fwd noise `4e-3` | `4` chunks | `8` chunks |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `6` | `3.46e-2` | `3.79e-2` | `2.60e-2` | `5.26` | `3.45` | `3.13` | `2.73` |
| `11` | `1.57e-2` | `1.20e-2` | `1.52e-2` | `1.02` | `0.826` | `1.91` | `1.33` |
| `16` | `6.62e-3` | `4.67e-3` | `9.60e-3` | `9.39e-2` | `0.368` | `0.169` | `0.227` |
| `21` | `1.32e-2` | `1.10e-2` | `1.58e-2` | `0.249` | `0.653` | `0.116` | `0.539` |
| `26` | `7.72e-3` | `1.21e-2` | `1.72e-2` | `0.399` | `0.695` | `0.114` | `0.447` |

Two groups, cleanly separated at every update: default repeat, `blk1` and
backward noise at `5e-3`-`4e-2`; forward noise and both block-causal variants
at `9e-2`-`5.3`. Block-causal sits inside the forward-noise range at every
sampled update.

What is established: block-causal's op-level accuracy matches the default
kernel's on four saved training calls (rank `0`, layers `0` and `11`, updates
`1` and `24`); the autograd wrapper and compile are cleared (`blk1`); and
across five sampled updates its gradient divergence falls in the same range
as random forward-output noise of `1e-3`-`4e-3`. A
same-batch gradient comparison therefore cannot separate block-causal from
the default path. A loss-curve A/B over a few hundred updates, several seeds
per arm, can, and should come before it is turned on by default.

### Loss-curve A/B

`agpt_2b_loss_ab` trains `2b-mds` from random init: AdamW `3e-4` (betas
`0.9`/`0.95`, weight decay `0.1`), `40` warmup updates then constant,
sequence `8192`, `12` tiles, DDP/ZeRO-1 with per-block compile, `400` updates
(`79M` tokens). `--debug.seed` sets the init and the blendcorpus order is
fixed, so the two arms of a seed share init and data. Jobs `12482374`-`12482376`
ran one seed each on its own node; seed `2` ran block-causal first.

Loss is the mean over all `12` ranks per update, then over the window.
Difference is block-causal (`4` chunks) minus default, per seed:

| Updates | Seed `1` | Seed `2` | Seed `3` | Mean | Default, SD across seeds |
|---|---:|---:|---:|---:|---:|
| `2`-`10` | `+0.0000` | `+0.0000` | `-0.0000` | `+0.0000` | `0.032` |
| `11`-`50` | `+0.0010` | `+0.0083` | `+0.0003` | `+0.0032` | `0.034` |
| `51`-`100` | `-0.0050` | `+0.0011` | `+0.0127` | `+0.0029` | `0.028` |
| `101`-`200` | `-0.0010` | `+0.0062` | `+0.0070` | `+0.0041` | `0.022` |
| `201`-`300` | `-0.0047` | `+0.0040` | `+0.0051` | `+0.0015` | `0.021` |
| `301`-`400` | `-0.0046` | `-0.0009` | `+0.0035` | `-0.0007` | `0.018` |

Final-window losses: `4.127`/`4.094`/`4.087` default, `4.122`/`4.093`/`4.090`
block-causal. The paired difference changes sign across seeds and is at most
`0.013` in any window, `5`-`10x` smaller than the spread from the init seed
alone. Over updates `301`-`400` the per-update difference has a standard
deviation of `0.010` in every seed. At `400` updates and three seeds the
two paths are indistinguishable; this does not test long-horizon or
production-scale training.

Updates `101`-`400`, slowest rank: block-causal `2.062`/`2.173`/`2.073 s`,
default `2.278`/`2.357`/`2.279 s`, `7.8`-`9.5%` less time per update in
every pair.

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
- Block-causal attention (`4` chunks) stays opt-in
  (`EZPZ_XPU_ATTN_CAUSAL_BLOCKS=4`), decided 2026-11-04. It is `7.8`-`9.5%`
  faster at AGPT-2B and matches the default path's loss over `400` updates
  and three seeds ([Loss-curve A/B](#loss-curve-ab)). Revisit after a
  multi-node run and a run past `400` updates.
- BF16 ZeRO-1 model storage (`BENCH_DDP_ZERO1_PARAMS=bfloat16`) stays opt-in:
  `0.9%` behind MDS at the `z2rrt02w` shape with block-causal attention; loss
  inside the seed spread at AGPT-2B over `400` updates, slightly higher in all
  three seeds over the last `10`. Needs a longer run and a multi-node run.
- The remaining `z2rrt02w` gap is attention backward: TT's block-causal path
  takes `0.54 s` more per update than IPEX xetla
  ([TT and MDS kernels](#tt-and-mds-kernels-at-this-shape)).
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
| `12482597`-`12482599` | AGPT-2B loss A/B, FP32 vs BF16 ZeRO-1 storage, seeds `1`-`3`, `400` updates | `0` |
| `12482585` | `z2rrt02w` shape, block-causal: FP32 vs BF16 ZeRO-1 model storage, alternated | `0` |
| `12482583` | `z2rrt02w` shape: TT warm-up, TT block-causal, MDS with `MDS_PROFILE=1`, TT control | `0` |
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
| `12482343` | Gradient probe at updates `6`-`26`, blocks not compiled, `5c4002ea1` | `0` |
| `12482374`-`12482376` | Loss-curve A/B, one seed each, default and block-causal `4` chunks, `400` updates, `b63a88d5a` | `0` |

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
