# 30B LR finder: AdamW / Mano / SophiaG at GBS=960

**Date:** 2026-08-23 | **Jobs:** 12473714 (adamw), 12473715 (mano), 12473716 (sophiag)
**Model:** agpt 30B (26.2B params), OLMo-2 tokenizer (vocab 100,352), seq 4096
**Data:** fineweb-edu-100BT (local parquet, 16 shards) | **Machine:** Sunspot, 16N x 3 concurrent

## Results

| arm | suggested LR | blow-up | min loss | LR at min | usable band top |
|---|---:|---:|---:|---:|---:|
| **AdamW** | **3.05e-05** | 3.05e-04 | 8.8381 | 3.594e-04 | 1.292e-03 |
| **Mano** | **5.61e-05** | 5.61e-04 | 8.9925 | 5.275e-04 | 2.448e-03 |
| **SophiaG** | **3.55e-05** | 3.55e-04 | 9.3441 | 4.642e-04 | 1.000e-03 |

"Suggested" is Smith 2015's blow-up/10. "Usable band top" is the largest LR
whose smoothed loss stays within 5% of that arm's minimum -- a width measure,
not a recommendation.

Sweep: 1e-6 -> 1e-1, 100 steps, `warmup_fraction=0.1`, `smooth_frac=0.1`.
All three produced a real blow-up, so every suggestion is measured rather than
an artifact of a sweep that ran out of range (a sweep that never diverges
yields no suggestion at all).

## Artifacts

Plots and raw data, per arm, under
`outputs/lrfind-30b-gbs960-<arm>/lr_finder/ezpz/ezpz.agpt/30b_olmo2tok/<arm>/`:

| arm | plot | csv |
|---|---|---|
| adamw | `.../adamw/lr_vs_loss.png` | `.../adamw/lr_finder_data.csv` |
| mano | `.../mano/lr_vs_loss.png` | `.../mano/lr_finder_data.csv` |
| sophiag | `.../sophiag/lr_vs_loss.png` | `.../sophiag/lr_finder_data.csv` |

Each CSV has 90 rows of `learning_rate,loss` plus `global_batch_size` and
`world_size` on every row. Those last two are worth checking rather than
trusting: all three arms record `global_batch_size=960, world_size=192`, which
is what makes the arms comparable to each other at all.

## Why these were measured per optimizer at this batch

The suggested LR moves by ORDERS OF MAGNITUDE with batch size for a single
optimizer. Mano, measured on this model family:

| batch | suggested LR |
|---|---:|
| 2B config | 4.79e-03 |
| 30B, GBS=480 | 1.29e-04 |
| **30B, GBS=960** | **5.61e-05** |
| 80B, GBS=6144 | ~3e-06 |

Inheriting a constant across batches would make an optimizer comparison
measure which optimizer got the luckier number.

**The inherited placeholder was on the divergence cliff.** Both the mano and
sophiag configs carried `lr=3.0e-4` from the 2B competition configs:

| | value | vs suggested | vs blow-up |
|---|---:|---:|---:|
| placeholder | 3.0e-4 | 8.5x too high (sophiag) | only 1.18x below |

Running SophiaG there would have read as "SophiaG is unstable at 30B" when the
truth is "SophiaG was run at 8.5x its usable LR" -- the same shape as the
2026-07-03 80B SophiaG NaN.

## What surprised me

At a FIXED batch the three optimizers land within **1.8x** of each other
(3.05e-05 to 5.61e-05). Compare that to the same optimizer across batch sizes
above: three orders of magnitude. **Batch size moves the usable LR far more
than optimizer choice does** -- so a batch change demands a re-measure, while
swapping optimizers at a known batch is a much smaller perturbation.

Mano also has the widest usable band (top 2.448e-03, ~2x AdamW's) and the
highest suggestion, consistent with the 30B/GBS=480 observation that it runs
calmer on grad_norm than AdamW at matched LR.

## Related

* Earlier 30B finder at GBS=480, mano only:
  `outputs/lrfind-30b-mano/...` (suggested 1.29e-04) -- different batch, NOT
  comparable to the table above.
* These LRs feed Phase 2 of the fixed-batch optimizer comparison:
  [`../../optimizer-comparison/README.md`](../../optimizer-comparison/README.md)
