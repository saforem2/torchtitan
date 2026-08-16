# 2026-08-16 -- the 20B ARC-C decay IS (at least partly) the RoPE permute

> **Status: A/B MEASURED at step 5000. Steps 6000/7600 pending a queue slot.**
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

## What this does NOT settle

A flat -0.045 correction applied to the later steps gives:

| step | published | +0.045 | still below the 0.3387 control? |
|---|---|---|---|
| 5000 | 0.3123 | **0.3575 (MEASURED)** | no -- above it |
| 6000 | 0.2713 | 0.316 (inferred) | yes |
| 7600 | 0.2287 | 0.274 (inferred) | yes |

So **either** the permute penalty grows with training, **or** a genuine decline
sits underneath it. A single point cannot distinguish those. UNKNOWN until
steps 6000 and 7600 are re-evaluated -- `reeval-20b-512-rope-ab2.sh` is written
and waiting on a queue slot (currently at the 10-job cap).

Do not quote "the 20B chain regresses on ARC-C" until those land. Do not quote
the published post-switch numbers either.

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
