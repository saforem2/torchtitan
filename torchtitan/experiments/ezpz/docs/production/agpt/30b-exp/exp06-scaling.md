# exp06 -- 30B weak-scaling: 2N to 64N

> **2026-08-16, Sunspot, frameworks RC** (oneAPI 2026.1.0, torch
> `2.13.0a0+gitcf30153`). Job `12473198`, 64 nodes, 3h. Config: `agpt_30b`,
> TP=1, LBS=3, compiled, seq=4096, 20 steps, median of the last 10.

## What this establishes

**The 30B holds ~26% MFU from 2 to 64 nodes.** It loses **1.42% of MFU per
doubling**, against the 2B's **13.96%**. That 8x difference in decay rate is
the first direct evidence for the 30B-exp proposal's central claim.

| nodes | ranks | tps/GPU | MFU | memory | vs 2N |
|---:|---:|---:|---:|---|---:|
| 2 | 24 | 458 | 27.45% | 78.50% | 100% |
| 4 | 48 | 451 | 27.03% | 72.28% | 98.5% |
| 8 | 96 | 443 | 26.53% | 68.00% | 96.7% |
| 16 | 192 | 445 | 26.67% | 57.33% | 97.2% |
| 32 | 384 | 440 | 26.36% | 56.08% | 96.1% |
| 64 | 768 | 426 | 25.54% | 64.81% | 93.0% |

All six points come from one job, so they share a stack, a build and an
allocation. The ladder's own 2N (458 tps / 27.45%) sits ~1.6% under exp05's
466 / 27.89% for the same config -- a separate-job difference at the edge of
the <1% within-job noise floor, and the reason the ratios above are computed
against the ladder's own 2N rather than exp05's.

Smooth and shallow. No cliff anywhere in the measured range.

## Against the proposal's claim

The proposal argues per-GPU efficiency is the biggest available lever: the 2B
runs **8.79% MFU at 512N**, and recovering ~27% would be a "~3x
effective-compute multiplier."

| model | 2N | far point | retention | decay/doubling |
|---|---:|---:|---:|---:|
| 2B | 29.26% | 8.79% at 512N | 30.0% | **13.96%** |
| 30B | 27.45% | 25.54% at 64N | 93.0% | **1.42%** |

The mechanism the proposal names is the right one: the 2B collapses because
per-rank work is too small to hide communication, and a 14x larger model gives
each rank ~14x more arithmetic per unit of gradient traffic.

**What this does NOT show.** 64N is the largest point measured; 512N is three
more doublings out. A naive log-linear extrapolation lands near 24% MFU at
512N, but **the 2B's own collapse was not log-linear** -- it fell off a cliff
once per-rank work stopped covering communication, and an extrapolation
through a smooth region cannot predict where a cliff sits. The honest claim is
"no degradation through 64N, and a decay rate 8x better than the 2B", not a
512N number.

Sunspot cannot settle it. Confirming 512N needs Aurora.

## The batch-size ceiling is the real constraint

LBS is fixed at 3, so this is **weak scaling** -- global batch grows with node
count:

| nodes | ranks | GBS (sequences) | tokens/step |
|---:|---:|---:|---:|
| 2 | 24 | 72 | 0.3M |
| 16 | 192 | 576 | 2.4M |
| 32 | 384 | 1152 | 4.7M |
| 64 | 768 | 2304 | 9.4M |
| *512* | *6144* | *18,432* | *75M* |

That last row is extrapolated, and it is the problem. **A 75M-token step is
far past the batch size where more tokens still buy proportional learning.**
So "the 30B holds 26% MFU at 512N" would not by itself mean 512N is a good
place to train it -- past some GBS the extra ranks buy throughput that does
not convert into progress per token.

Two ways out, neither tested here:

- **Fewer nodes, same model.** If the 30B holds ~26% at 64-128N, the
  efficiency argument is already won at a node count with a sane GBS.
- **LBS=1 at high N.** Cuts GBS 3x at the cost of the batch-size lever that
  won +30% at 2N. Whether that trade nets out at 512N is unmeasured.

The MFU curve and the batch ceiling push in opposite directions, and the
proposal currently only argues the first.

## Method notes

- Node counts run **descending** (64 -> 2) so the largest allocation is used
  while the full reservation is held.
- 20 steps per point, median of the last 10, discarding warmup and compile.
- Run-to-run noise at this config is **under 1%** (measured in exp05: two
  independent LBS=2 runs gave 434 tps / 25.99% and 434 / 25.96%), so the
  2.35-point spread across the ladder is signal, not scatter.
- Memory is *lower* at 16-64N than at 2N (57-65% vs 78.5%) because FSDP shards
  parameters and optimizer state across more ranks. This means larger models
  or larger LBS become *more* affordable at scale, not less.

## Open

- **512N on Aurora** -- the only way to test the actual claim.
- **A GBS sanity bound.** Before proposing any 512N config, establish the
  largest global batch that still trains efficiently at this model size;
  that number, not MFU, may be what caps useful node count.
- **LBS=1 at 64N** -- measures the throughput cost of the GBS-reduction
  escape hatch above.
