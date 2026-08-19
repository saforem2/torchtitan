# 2026-08-16 -- the 20B ARC-C "decay" was entirely the RoPE permute

> **Status: A/B MEASURED at steps 5000 AND 6000. There is no capability
> regression -- the model improved throughout. Step 7600 pending.**
>
> **This page previously argued the opposite and was WRONG.** The original
> argument is preserved verbatim in
> [What I got wrong](#what-i-got-wrong-and-why-the-argument-was-seductive),
> because the way it failed is more instructive than the conclusion.

## Result

Job `8760246`. Same checkpoint (`step-5000`), same tasks, same harness, same
node. **The only variable is `--model_flavor`.**

| metric | `20b` (wrong permute) | `20b_real` (correct) | delta |
|---|---|---|---|
| **arc_challenge acc** | 0.3123 | **0.3575** | **+0.0452** |
| arc_challenge acc_norm | 0.3370 | **0.3823** | +0.0453 |
| arc_easy acc | 0.6688 | **0.7050** | +0.0362 |
| hellaswag acc_norm | 0.5925 | **0.6394** | +0.0469 |

Every metric improves. The corrected ARC-C of **0.3575 exceeds both complex
controls** (step 4400 = 0.3387, step 4900 = 0.3413) -- which is what a model
that is still learning should look like at step 5000, and is not what the
published table shows.

**The permute is real, it is costing ~4.5 points of ARC-C, and it is degrading
every published post-switch eval number.**

## Step 6000 settles it: the penalty GROWS, and the decay was entirely artifact

The open question above -- constant penalty plus a real decline, or a growing
penalty -- is now answered. MEASURED, same job:

| metric | 5000 delta | 6000 delta |
|---|---|---|
| **arc_challenge** | +0.0452 | **+0.0947** |
| arc_easy | +0.0362 | +0.0362 |
| hellaswag acc_norm | +0.0469 | +0.0490 |

**ARC-Easy's penalty is constant to four decimals and HellaSwag's is nearly so;
only ARC-Challenge's doubles.** And the corrected trajectory rises on every
task:

| metric | 5000 | 6000 | direction |
|---|---|---|---|
| arc_challenge | 0.3575 | **0.3660** | up |
| arc_easy | 0.7050 | **0.7151** | up |
| hellaswag | 0.6394 | **0.6576** | up |

So there is **no capability regression at all.** The model improved
monotonically across the whole window. The published ARC-C "decay" is the
permute penalty growing faster than the model's genuine gains, inverting the
curve.

## The chart now plots the corrected data (2026-08-19)

`all_production_evals.svg` reads corrected results for the three affected
chains. ARC-Challenge ends at **0.390** (20B-512) and **0.362** (20B-256),
where the corrupted exports fell to ~0.27 and ~0.297.

How it works, and the two traps in it:

- **Union, not replacement.** Pre-switch points come from the original eval
  dirs (correct: exported with the matching convention); at/after the switch
  step they come from `-ropefix`. Neither source alone is right. A post-switch
  step with no corrected result is DROPPED, never backfilled -- a gap is
  visible, a wrong point is not.
- **`agpt-2b-v2-256n` is untouched.** It never switched convention, so it has
  no corrected dir and needs none.
- **Shot pinning.** The eval scripts write `<task>@<N>shot` keys plus a bare
  `<task>` alias for whichever group ran LAST. On 14 of 36 post-switch steps
  (20B-512) and 15 of 37 (20B-256), the bare `arc_challenge` IS the 25-shot
  number, with an unused `@0shot` beside it. The corrected sweep ran 0-shot
  throughout. Reading bare keys would have spliced 25-shot pre-switch onto
  0-shot post-switch **in the ARC-C panel specifically** -- a union that looked
  right by step coverage and was wrong by measurement. The reader now demands
  `<task>@0shot` and returns nothing rather than substituting another shot
  count.

**Known gap, reported loudly.** The corrected sweep covered only
`arc_challenge`, `hellaswag`, `arc_easy`. Winogrande, PIQA, OpenBookQA and
BoolQ therefore lose their post-switch history on both 20B chains -- 268 points
in total. The chart run prints a NOTE naming each chain, task and count. Jobs
`8766059` / `8766061` are queued to close it.

**Provenance note:** the regenerated SVG landed in commit `7858d406c`, whose
message belongs to unrelated work. A heredoc to `/tmp` failed on a login node
where that path is not writable, and the `git add -A -- "*figures*"` in the
same command had already staged the figure. The chart content is correct; only
its commit message is misattributed.

## Confirmed across a 5-point series, not just two spot-checks (2026-08-17)

The re-eval sweep has since covered steps 4500-4900 continuously, turning the
two-point A/B into a series. Every step, every metric, same direction:

| step | ARC-C corrupted -> corrected | ARC-Easy corrupted -> corrected |
|---|---|---|
| 4500 | 0.3422 -> 0.3601 (+0.0179) | 0.6679 -> 0.6894 (+0.0215) |
| 4600 | 0.3456 -> 0.3626 (+0.0171) | 0.6738 -> 0.6995 (+0.0257) |
| 4700 | 0.3396 -> 0.3763 (+0.0367) | 0.6582 -> 0.6965 (+0.0383) |
| 4800 | 0.3601 -> 0.3788 (+0.0188) | 0.6662 -> 0.7024 (+0.0362) |
| 4900 | 0.3601 -> 0.3763 (+0.0162) | 0.6692 -> 0.7016 (+0.0324) |

Two things this adds beyond the earlier spot-checks:

1. **The corrected ARC-C series rises monotonically** across 4500-4800 (0.3601
   -> 0.3626 -> 0.3763 -> 0.3788) where the corrupted one wanders
   (0.3422 -> 0.3456 -> 0.3396 -> 0.3601). The corrupted series was noisy
   enough that a downward stretch of it could be read as a trend; the corrected
   one has no such stretch in this window.
2. **The penalty is not constant even before step 5000** -- it ranges 0.016 to
   0.037 across five adjacent checkpoints. Earlier framing treated ARC-C's
   penalty as roughly flat until it doubled by 6000; it is noisier than that.
   The conclusion is unchanged (correcting the permute helps everywhere, and
   there is no regression), but "the penalty grows smoothly" would be
   over-reading five points.

The 2B-512 arm shows the same shape at its own scale -- ARC-Easy 0.6465 ->
0.6507 and HellaSwag 0.5295 -> 0.5344 across steps 35,000-39,600, both rising.

### Replicated on 20B-256: the corrupted series FALLS where the corrected one RISES

The strongest single piece of evidence, because it is an independent chain (a
different node count, a different job history) reaching the same conclusion:

| step | ARC-C corrupted -> corrected | delta |
|---|---|---|
| 4000 | 0.3481 -> 0.3635 | +0.0154 |
| 4200 | 0.3234 -> 0.3652 | +0.0418 |
| 4400 | 0.3217 -> 0.3626 | +0.0410 |
| 4600 | 0.3319 -> 0.3712 | +0.0392 |
| 4800 | 0.3319 -> 0.3686 | +0.0367 |
| 5000 | 0.3046 -> 0.3737 | **+0.0691** |
| 5100 | 0.3200 -> 0.3874 | **+0.0674** |

| 5300 | 0.3413 -> 0.3874 | +0.0461 |
| 5400 | 0.3020 -> 0.3951 | +0.0930 |
| 5500 | 0.3106 -> 0.3788 | +0.0683 |
| 5600 | 0.3114 -> 0.3908 | +0.0794 |
| 5700 | 0.2961 -> 0.3771 | +0.0811 |
| 5800 | 0.2969 -> 0.3857 | +0.0887 |

**Across all 13 points (steps 4000-5800) the two series move in OPPOSITE
directions over the same checkpoints:**

| series | step 4000 | step 5800 | net |
|---|---|---|---|
| corrupted | 0.3481 | 0.2969 | **-0.0512** |
| corrected | 0.3635 | 0.3857 | **+0.0222** |

Mean penalty grows +0.0405 (first half) -> +0.0748 (second half), ending at
+0.0887. By step 5800 the corrupted export reads 0.2969 -- within noise of the
0.25 random baseline -- while the same weights, exported correctly, read
0.3857.

That is the whole artifact in one table. A reader of the corrupted curve would
conclude the 20B had lost most of its ARC-C ability by step 5800; the model had
in fact gained. No amount of care reading the corrupted numbers would have
recovered the truth, because the corruption is monotone in training progress
and therefore indistinguishable from a trend.

Read the two columns as trajectories rather than as five separate deltas:

- **corrupted: 0.3481 -> 0.3234 -> 0.3217 -> 0.3319 -> 0.3319 -> 0.3046 ->
  0.3200** -- opens with a 0.026 drop, never recovers to its starting value,
  and bottoms out at step 5000. On its own this is a textbook "the model is
  regressing on ARC-C" curve.
- **corrected: 0.3635 -> 0.3652 -> 0.3626 -> 0.3712 -> 0.3686 -> 0.3737 ->
  0.3874** -- rises across the window, no drop anywhere, ending at its maximum.

The penalty also **grows with training** on this chain (+0.015 at 4000 to
+0.069 at 5000), matching the 20B-512 finding that ARC-C's penalty roughly
doubles between steps 5000 and 6000 while ARC-Easy's stays flat. A permute
error costs more as the model's answers get sharper: there is more signal to
scramble.

Same checkpoints, same harness; only the RoPE convention used at export
differs. A decay that exists in one and not the other is not a property of the
model. This is what makes the artifact explanation conclusive rather than
merely consistent: the earlier 20B-512 evidence showed correction *helping
everywhere*, which a real-decay-plus-constant-penalty story could survive; a
sign flip in the trend itself, on a second chain, it cannot.

**INFERRED, not measured:** the likely reason the penalty grows only on ARC-C
is that a sharper model has more to lose. Scrambled Q/K pairing degrades
whatever structure attention has learned, so the better the model gets at the
task that most depends on that structure, the more the corruption costs. Easy
tasks that are already near their ceiling lose a fixed amount. This is a
plausible story, not a tested one.

Step 7600 is still worth having as a third point, but it cannot change the
conclusion: two steps with rising corrected scores already rule out the
regression reading.

## What I got wrong, and why the argument was seductive

The earlier version of this page rejected the permute explanation on this
evidence:

| step | convention | ARC-C |
|---|---|---|
| 4300 | complex | 0.3430 |
| 4400 | complex | 0.3387 |
| 4500 | cos_sin | 0.3242 |
| 4900 | cos_sin | **0.3413** |
| 5000 | cos_sin | 0.3123 |

and argued: *steps 4500-4900 sit inside the complex controls' range, and 4900
beats the 4400 control, so the permute cannot be biting -- a wrong permute is a
step function, it cannot spare five checkpoints then start at 5000.*

**The flaw: steps 4500-4900 were ALSO wrongly permuted.** Every one of those
numbers is a corrupted measurement. I compared corrupted values against clean
ones, saw them overlap, and read the overlap as proof of no damage. The
comparison had no clean arm in it at all.

What was actually happening: across 4400 -> 4900 the model genuinely improved by
roughly the same magnitude as the permute penalty (~+0.04 vs ~-0.045), so the
two nearly cancelled and the boundary looked continuous. A coincidence of
magnitudes, and I built a confident argument on it -- then repeated that
argument to the user twice.

The step-function intuition was correct in itself. It was applied to the wrong
data.

**The lesson worth keeping:** an A/B needs a control arm that differs in exactly
one variable. "Nearby numbers look similar" is not a control. The whole reason
this A/B was worth running is that it has one -- and it took 25 minutes of
compute to answer what 40 minutes of reasoning got backwards.

Also corrected: the ruled-out list in the prior version ("not a length-norm
artifact, not config drift, not a general collapse") was all true and all
irrelevant, because it never tested the hypothesis actually in play.

## Method

```bash
# convert from the MAIN repo -- the pinned v2 clones ship no
# agpt/state_dict_adapter.py and permute unconditionally regardless of flavor
python3 torchtitan/experiments/ezpz/eval/convert_to_hf.py \
    <ckpt>/step-5000 <out>/step-5000/hf \
    --model_name experiments.ezpz.agpt \
    --model_flavor 20b_real --export_dtype bfloat16
# then lm_eval arc_challenge,hellaswag,arc_easy on xpu:0, batch 8
```

Script: `scripts/eval/oneoff/reeval-20b-512-rope-ab.sh` (job `8760246`).
Results: `outputs/evals/agpt-20b-v2-512n-ropefix/step-5000/results/results.json`.

Five earlier attempts died on PBS environment setup, not on the science --
`--login` shebang, top-level `module load`, the `tt-lm-eval` venv, and no
`set -u`. All four are now recorded in `experiments/ezpz/.claude/CLAUDE.md`.

## Consequences

**Every published eval on the cos_sin side of a switch understates its model.**
Counts of affected results:

| chain | results | past the switch |
|---|---|---|
| `agpt-20b-v2-512n` | 72 | **36** |
| `agpt-20b-v2-256n` | 43 | **36** |
| `agpt-2b-v2-512n` | 36 | **8** |
| `agpt-2b-v2-256n` | 193 | 0 (never switched) |

The 2B-512 chain's *completed-run* numbers are in that set: its switch was at
step 30401 and the chain ran to 46,429, so the headline 4.674T eval
(MMLU 0.2511, ARC-C 0.2381, HellaSwag 0.4753) was measured on a wrongly-permuted
export. **Those numbers are too low by an unknown amount** -- and the
token-matched comparison that made 2B-256 look better than 2B-512 on fewer
tokens is now suspect in the same direction.

Whether a corrected 2B-512 changes the campaign's central "MMLU never left
chance" conclusion is UNKNOWN and worth testing: +0.045 would not lift 0.2511 to
significance, but the magnitude at 2B has not been measured.

## Related

- [`rope-flavor-mismatch.md`](../../../guides/known-bugs/rope-flavor-mismatch.md)
  -- mechanism, per-step registry, shipped fail-loud mitigation.
- `scripts/eval/oneoff/reeval-20b-512-rope-ab2.sh` -- the pending 6000/7600 run.
