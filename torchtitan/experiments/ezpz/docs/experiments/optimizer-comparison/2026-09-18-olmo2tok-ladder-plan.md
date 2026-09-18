# Optimizer comparison on the OLMo-2 ladder: plan

**Date:** 2026-09-18 | **Status:** LR finders SUBMITTED, training arms NOT submitted
**Sizes:** agpt `{5,10,30}b_olmo2tok` -- 4.64B / 9.48B / 26.2B, vocab 100,352, seq 4096
**Machine:** Aurora `next-eval`, 64N, GBS=6144 (dp 768, LBS=2, GAS=4)

## What is already answered, and by what

The fixed-batch comparison at **30B / GBS=960** is COMPLETE
([README](README.md)). Do not re-run it:

| arm | verdict |
|---|---|
| **Mano** | wins. Ahead of AdamW from ~1.0B tokens; **-0.143 nats at 10B**, **-0.075 at 23.59B** |
| **AdamW** | second, closing at ~0.0187 nats/1,000 steps; implied crossover ~35-40B |
| **SophiaG** | DISQUALIFIED -- diverged **4/4**, including a seed-pinned from-scratch replicate with ZERO precursor excursions |
| **Muon** | LR measured (5.68e-04 at GBS=960) but **never run as a training arm** |

So the open questions are narrower than "which optimizer":

1. **Does Mano's lead survive a decay phase?** The 30B arms were constant-LR
   **by design**, and the documented prior is exactly "Mano/Muon win short
   runs, AdamW wins in the cosine decay phase". The existing result is
   *consistent with* that prior, not a refutation. This is the single highest-
   value open question and the README names it as the obvious follow-up.
2. **Does the ranking hold at 4.64B and 9.48B?** Measured only at 26.2B.
3. **Where does Muon land as a training arm?** Won the 2B competition, has a
   30B LR, never trained at this scale.

## Precondition: LR per optimizer AT this batch

Nine sweeps submitted (3 sizes x adamw/mano/muon):

| size | adamw | mano | muon |
|---|---|---|---|
| 5b  | 8837334 | 8837373 | 8837374 |
| 10b | 8837335 | 8837375 | 8837376 |
| 30b | 8837336 | 8837377 | 8837378 |

This is not a formality. For Mano alone the suggested LR spans **4.79e-03**
(2B config) to **~3e-06** (80B at GBS=6144) -- three orders of magnitude for
one optimizer. The 30B campaign nearly ran SophiaG at an inherited 3.0e-4
placeholder that was 8.5x above its suggestion and only 1.18x below its
measured blow-up; that would have read as "SophiaG is unstable at 30B" for
entirely the wrong reason.

SophiaG is NOT swept here. Its curve is smooth through the low band, so a
sweep produces a number with no usable arm behind it -- the divergence is
state-dependent, not LR-driven. Including it would manufacture a
recommendation the 4/4 evidence contradicts.

## Proposed training arms

Gated on the finders. Budget per arm at GBS=6144 x 4096 = 25.2M tok/step:

| tokens | steps |
|---:|---:|
| 10B | 397 |
| 25B | 993 |
| 100B | 3,973 |

**Recommend 25B, not 10B, and not 100B.** At 10B the 30B campaign's gap was
still widening; the informative region was 10-24B where it narrowed
monotonically. 100B (~3,973 steps, est. 84h for the 30B at 64N) buys a fourth
size-point at the cost of the whole comparison fitting in one allocation
window -- and next-eval caps at 6h, so a 100B arm is a chained job, not a run.

**Stage 1 -- 5B ladder rung, all three optimizers, 25B tokens each.**
Cheapest size, so it answers "does the 30B ranking transfer down" for the
least allocation. 3 arms.

**Stage 2 -- decay phase at whichever size stage 1 favours.** This is the
question the 30B campaign explicitly did not answer. Two arms (Mano, AdamW)
with a cosine decay to 10% over the last 20% of steps, against the same two
arms constant-LR. Four arms total, and the only design here that can overturn
the existing conclusion rather than confirm it.

**Stage 3 -- 30B confirmation, only if stages 1-2 disagree with the GBS=960
result.** If they agree, the 30B answer already exists at a finer token
resolution than a fresh 25B arm would give.

## Non-negotiables for any arm

- `--grad-norm-abort=20.0` if SophiaG is ever run. It is **0.0 by default**.
  It fired on both documented divergences (8 and 13 steps before the peak) and
  stayed clean on adamw/mano. `nan_abort_consecutive` never fires here --
  every value in a SophiaG blow-up is finite.
- One seed per arm is what the 30B campaign had, and it is a stated weakness.
  If a stage-1 gap comes in under ~0.05 nats, that is inside the range a
  second seed could move and should not be reported as a ranking.
- W&B on every arm, per house rule.

## Open decision

Stage 1 is ~3 arms x 25B tokens at 4.64B params. Awaiting the finder LRs
before submitting anything; the arms are not launchable until each has its own
measured LR at GBS=6144.
