# Mano LR finder at 30B (Sunspot, 2026-08-23)

**Job:** 12473671 | 8N / 96 ranks | agpt_30b_olmo2tok_mano | GBS=480, seq=4096
**Result:** suggested **1.29e-04**, blow-up **1.29e-03**, empirical min **1.468e-03**

## Verdict

Mano's LR does not transfer from the 2B. The 2B finder
(`outputs/lrfind-2b-100step/`) put mano's min-loss at **4.79e-03**; at 30B the
loss starts CLIMBING past 1.29e-03. Carrying the 2B number over would have run
**3.7x past the blow-up point**. This is the concrete case of the warning
already in CLAUDE.md, that mano's suggested LR is batch- and scale-dependent
and small-batch numbers do not transfer.

## The curve

90 points, 1e-6 -> 8.8e-2, EMA-smoothed, 10-step warmup at init_lr first.

| lr | smoothed loss |
|---|---|
| 1.000e-03 | 8.1749 |
| 1.136e-03 | 8.1579 |
| 1.292e-03 | 8.1324 |
| **1.468e-03** | **8.1295**  <- minimum |
| 1.668e-03 | 8.1395 |
| 1.896e-03 | 8.1497 |
| 2.154e-03 | 8.2044 |
| 2.783e-03 | 8.3048 |

**The basin is shallow**: 0.045 nats across a 1.5x LR range. That matters for
how much any single number in here should be trusted -- the difference between
3e-4 and 1.5e-3 is unlikely to dominate a run.

Smith's rule reports blow-up/10, which is deliberately conservative; the
empirical minimum sits ~11x above the suggestion.

## Where the running 64N job sits

Job 12473683 runs lr=3e-4:

- 2.3x ABOVE the conservative 1.29e-04 suggestion
- **0.23x the blow-up point** -- comfortably inside the stable region
- ~5x below the empirical minimum

That is a defensible, conservative setting. Not a misconfiguration, and given
the shallow basin, not obviously worth restarting 8h of 64 nodes to change.

## Two predictions I got wrong

Recording these because both were made from partial data and both were
confidently stated:

1. Reading the live sweep log mid-run, I estimated the suggestion would land
   near 1e-3 and said 3e-4 was "roughly 3x too low". The suggestion is
   1.29e-4; 3e-4 is 2.3x too HIGH relative to it. Eyeballing a minimum off a
   scrolling log is not the same as the derivative analysis the finder does.
2. Earlier, from the 2B data alone, I flagged 3e-4 as possibly "16x too low".
   The opposite: the 2B's preferred LR is past the 30B's blow-up.

The general lesson is the one this whole session keeps re-teaching: measure at
the target configuration. Both errors came from extrapolating -- once across
model scale, once across an unfinished curve.

## Method

    qsub 30b_mano_lrfind.pbs   # 8N, ~68 min

    --lr-finder.enable --lr-finder.init-lr=1e-6 --lr-finder.max-lr=1e-1
    --lr-finder.fraction=0.1 --lr-finder.warmup-fraction=0.1
    --lr-finder.smooth-frac=0.1

`warmup-fraction=0.1` holds 10 steps at init_lr before sweeping, so the early
points measure LR sensitivity rather than init transients.

Artifacts: `outputs/lrfind-30b-mano/lr_finder/ezpz/ezpz.agpt/30b_olmo2tok/mano/`
(`lr_finder_data.csv`, `.npz`, `lr_vs_loss.png`).
