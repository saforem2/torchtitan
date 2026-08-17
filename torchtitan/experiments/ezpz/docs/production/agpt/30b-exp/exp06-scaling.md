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


## Second ladder: the best config, and the price of a pinned global batch

> Job `12473204`, 64 nodes. `agpt_30b_llama3tok` (the 128k vocab, which exp05
> established as the better config), TP=1, compiled, seq=4096, 20 steps,
> median of the last 10. Two arms at every node count.

The first ladder measured gemma at LBS=3 and let global batch float with node
count. Both of those are wrong for a production decision: exp05 showed the
128k vocab at LBS=4 is the best config, and `global_batch_size` is a config
field, not a consequence of node count. This ladder fixes both.

| nodes | GBS | tps (LBS=4) | MFU | | GBS | tps (LBS=1) | MFU | gap |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 2 | 96 | 489 | **28.49%** | | -- | -- | -- | -- |
| 4 | 192 | 481 | 28.06% | | 48 | 362 | 21.09% | +6.97pp |
| 8 | 384 | 472 | 27.54% | | 96 | 350 | 20.41% | +7.13pp |
| 16 | 768 | 473 | 27.58% | | 192 | 355 | 20.71% | +6.87pp |
| 32 | 1536 | 468 | 27.27% | | 384 | 341 | 19.89% | +7.38pp |
| 64 | 3072 | 456 | 26.57% | | 768 | 322 | 18.77% | +7.80pp |

**LBS=4 beats the gemma LBS=3 ladder by ~1pp at every point** (+0.91, +0.91,
+1.03pp at 16/32/64N), confirming exp05's 2N result holds at scale.

**LBS=1 costs a stable 7-8pp of MFU** -- 29% of throughput at 64N -- and
decays more than twice as fast (2.87%/doubling against 1.35%), because there
is less per-rank work to hide the same collectives.

### Why LBS=1 is the number that matters

`GAS = GBS / (LBS * dp_degree)`, so a *pinned* global batch forces LBS down as
nodes grow. Holding GBS at the 2B's known-good **6,144**:

| nodes | max LBS at GBS 6,144 |
|---:|---:|
| 64 | 8 |
| **128** | **4** |
| 256 | 2 |
| 512 | **1** |

So the two arms above are not alternatives at 512N -- **LBS=1 is the only one
available there**, and LBS=4 is only available up to 128N.

### Extrapolated, with the constraint applied

| nodes | LBS=4 | its GBS | LBS=1 | its GBS |
|---:|---:|---:|---:|---:|
| 128 | 26.21% | **6,144** | 18.23% | 1,536 |
| 256 | 25.85% | 12,288 | 17.71% | 3,072 |
| 512 | 25.50% | 24,576 | **17.20%** | **6,144** |

Two rows are simultaneously feasible and sane: **128N at LBS=4 (GBS 6,144,
~26%)** and **512N at LBS=1 (GBS 6,144, ~17%)**. Everything else either blows
the batch budget or wastes the hardware.

**128N is ~1.5x better per GPU, on a quarter of the machine.**

### What this does to the proposal's claim

The proposal argues that recovering ~27% MFU from the 2B's 8.79% at 512N is a
"~3x effective-compute multiplier".

- **At 512N the 30B gets ~17%, not ~27%** -- because a sane global batch forces
  LBS=1 there. That is **~2x** the 2B, not 3x.
- **~27% is real, but it lives at 128N**, where the batch budget still allows
  LBS=4.

So the efficiency argument survives and the mechanism is confirmed, but it
argues for **a bigger model on fewer nodes**, not for the same node count.
Going wider than 128N with a fixed global batch gives back more MFU than the
extra ranks return.

**Caveat on the extrapolations:** they walk three doublings past the last
measured point using rates fitted over 4-64N. An earlier version of this
analysis used only the 32->64 doubling for LBS=1, got 5.63%/doubling, and
projected 15.8% at 512N -- the full-span fit gives 2.87% and 17.2%. Two-point
rates at the small end are unreliable because the curve is not monotonic
there (16N is the peak; 8N sits below it). Treat 17-18% as the bracket.

## The batch-size ceiling: how the first ladder got it wrong

The first ladder held LBS=3 and let GBS float, which made it look as though
global batch *necessarily* grows with node count -- 18,432 sequences / 75M
tokens per step at 512N, far past any useful batch size.

**That framing was wrong.** `global_batch_size` is a config field
(`configs.py:35`) and gradient accumulation is derived from it
(`trainer.py:425`). It only appears to track node count because the default
`-1` falls back to `LBS * dp_degree` (`trainer.py:410-414`) -- which is exactly
what the first ladder left it at.

The real constraint is the one measured above: pinning GBS does not blow up
the batch, it **forces LBS down**, and LBS is the lever that produced the MFU
in the first place. The 2B's own measurement is what sets the budget -- at
GBS 12,288 it lost 3-8pp per token against 6,144, with per-step parity
confirming the architecture was healthy.

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
