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
