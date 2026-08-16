# 2026-08-16 -- the 20B ARC-Challenge decay is probably NOT the RoPE permute

> **Status: analysis complete, confirmatory job `8760122` RUNNING.**
> Written while the A/B converts, so the reasoning is on record independent of
> how the test comes out.

## Claim under test

The [RoPE investigation](../../../guides/known-bugs/rope-flavor-mismatch.md)
concluded that every HF export made from a cos_sin checkpoint was wrongly
permuted, and offered as its main evidence that **ARC-Challenge decays
monotonically while loss keeps improving**:

> 20B-512 0.3823 -> 0.2628; 2B-512 0.3456 -> 0.2381

That decay is real. **The attribution to the permute does not survive contact
with the boundary data.**

## The disproof: there is no discontinuity at the switch

MEASURED, from `outputs/evals/agpt-20b-v2-512n/step-*/results/results.json`.
The 20B-512 chain switched complex -> cos_sin at **step 4401**, so 4400 and
earlier are correctly converted and 4500 onward are allegedly corrupt:

| step | convention | ARC-C `acc` |
|---|---|---|
| 4100 | complex | 0.3345 |
| 4200 | complex | 0.3353 |
| 4300 | complex | 0.3430 |
| 4400 | complex | 0.3387 |
| **4500** | **cos_sin** | **0.3242** |
| 4600 | cos_sin | 0.3268 |
| 4700 | cos_sin | 0.3242 |
| 4800 | cos_sin | 0.3345 |
| 4900 | cos_sin | **0.3413** |
| 5000 | cos_sin | 0.3123 |
| 6000 | cos_sin | 0.2713 |
| 7000 | cos_sin | 0.2491 |
| 7600 | cos_sin | 0.2287 |

**Steps 4500-4900 are inside the complex controls' range.** Step 4900 scores
0.3413, *above* the 4400 control at 0.3387. The decline to 0.2287 unfolds
gradually over the ~3,000 steps that follow.

A wrong Q/K permute is a **step function**. It re-pairs every attention channel
with the wrong partner from the first converted tensor onward; it cannot leave
five consecutive checkpoints untouched and then start biting at step 5000. The
mechanism and the data disagree.

**The same pattern holds on 20B-256** (switch at step 3101):

| step | convention | ARC-C |
|---|---|---|
| 3000 | complex | 0.3268 |
| **4000** | **cos_sin** | **0.3285** |
| 5000 | cos_sin | 0.2765 |
| 8000 | cos_sin | 0.2543 |

Step 4000 is past the boundary and statistically identical to step 3000.

## What the decay is NOT

- **Not a length-normalization artifact.** MEASURED: `acc` and `acc_norm` decay
  together (0.3387 -> 0.2287 and 0.3823 -> 0.2628). If the permute were
  scrambling logits, the two would not track this cleanly.
- **Not a config change.** MEASURED, W&B `metadata.args` for the first and
  latest cos_sin-era runs (`tu1iseu1`, `c8zwrlqw`) are identical on every
  training knob: `sophiag`, `lr=2.28e-5`, `LBS=2`, `GBS=12288`, `seq-len=8192`.
- **Not a general capability collapse.** MEASURED: ARC-**Easy** holds at
  0.665-0.696 across the entire range and HellaSwag stays ~0.61 while ARC-C
  halves. The model keeps easy reasoning and loses hard reasoning.
- **Not loss divergence.** Train loss and the in-process validator loss both
  improve monotonically throughout.

## What it might be (UNTESTED)

Offered as hypotheses, not findings:

1. **Genuine capability regression** on hard multi-step reasoning while general
   LM quality improves. Would be the most important reading, and is consistent
   with ARC-Easy holding.
2. **Data-mix drift** -- something about the tokens seen after ~step 5000.
   Checkable against the blendcorpus shard order.
3. **Overfitting to the corpus** at fixed LR (constant 2.28e-5, no decay), where
   the model sharpens on web text at the expense of held-out reasoning.
4. **Eval-harness interaction** with a lengthening context or changed padding.
   Least likely given ARC-Easy is unaffected.

## The confirmatory test (job `8760122`)

Re-converts 20B-512 steps 5000 / 6000 / 7600 with `--model_flavor 20b_real`
from the **main repo** (the pinned v2 clones lack
`agpt/state_dict_adapter.py` and would permute regardless of the flag), then
re-runs `arc_challenge,hellaswag,arc_easy`. Only the flavor changes.

**Prediction, recorded before the job was submitted:**

- ARC-C returns to >= 0.3387 and stops decaying -> the permute WAS the cause,
  and this page is wrong.
- ARC-C reproduces the decay -> the permute is NOT the cause, the capability
  regression is real, and it needs its own investigation.

Given the boundary data above, **the second outcome is expected.**

Note the walltime is 1h for three 20B conversions at ~20-30 min each, so the
job may only complete step 5000. That single point is still decisive: 5000 is
the first clearly-decayed step (0.3123 vs a 0.3387 control).

## Blast radius, independent of the outcome

MEASURED counts of published eval results on the cos_sin side of each switch:

| chain | results | past the switch |
|---|---|---|
| `agpt-20b-v2-512n` | 72 | **36** |
| `agpt-20b-v2-256n` | 43 | **36** |
| `agpt-2b-v2-512n` | 36 | **8** |
| `agpt-2b-v2-256n` | 193 | 0 (never switched) |

Whether those numbers are *wrong* is exactly what `8760122` tests. The
flavor-mismatch mechanism is real and the eval scripts genuinely did hardcode
the wrong flavor -- the open question is whether it materially changed the
scores.

## Correction this page makes

The known-bugs page currently states the ARC-C decay as evidence FOR the
permute corruption. That inference is unsound for the reason given above, and I
repeated it before checking the boundary. The page should be amended once
`8760122` reports.

## Related

- [`rope-flavor-mismatch.md`](../../../guides/known-bugs/rope-flavor-mismatch.md)
  -- the mechanism, the per-step registry, and the shipped fail-loud mitigation.
  All of that stands; only the ARC-C attribution is in question.
- `scripts/eval/oneoff/reeval-20b-512-rope-ab.sh` -- the A/B job.
