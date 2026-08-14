# 30B-exp experiment log

> **Last updated: 2026-08-14.**
>
> Tracking table for every experiment run against the
> [30B-exp proposal](README.md). One row per experiment, one file per
> experiment. **Status is honest**: an experiment that failed, was
> inconclusive, or contradicted the proposal is recorded as such -- a design
> doc whose own tests all conveniently agree with it is not evidence.

## Why these experiments exist

The proposal makes claims of three different kinds, and they need different
treatment:

- **Measured** (Section 1) -- already established by the 2B campaign. Not
  re-tested here.
- **Judgment** (Section 6) -- reasoning from measurements, not measurements.
  These are what the cheap experiments below attack.
- **Untested** -- the central data hypothesis. Needs a real run (exp03).

The tiering is set by *discriminating power*, not by cost. A toy model cannot
answer the MMLU question at all: MMLU has a capability floor around 1-3B
params, so a 20M or 500M model returns ~0.25 on **any** data and a null result
cannot separate "bad mix" from "model too small". Tier 0 experiments are cheap
*because* the questions they ask are scale-free, not because we are economising
on the important one.

## Tier 0 -- scale-free, no production allocation

| # | Experiment | Question | Cost | Status |
|---|---|---|---|---|
| [01](exp01-tokenizer-analysis.md) | Tokenizer analysis | Does gemma-7b's 256k vocab actually hurt us -- embedding cost, digit splitting, fertility, vocab utilisation? | CPU, minutes | RUNNING |
| [02](exp02-fp32-norms-ablation.md) | fp32 norms-only ablation | Is fp32 for *norm params only* sufficient, or does full fp32 master do real work? | debugmodel, ~2N | RUNNING |

## Tier 1 -- the gate

| # | Experiment | Question | Cost | Status |
|---|---|---|---|---|
| [03](exp03-1b-proxy-design.md) | 1B proxy design | Can a 1B on a candidate mix clear the MMLU floor, and is the 0.28 gate statistically defensible? | design only | RUNNING |

## Findings

Filled in as experiments land. Each entry states whether it **supports**,
**undermines**, or **does not settle** the corresponding proposal claim.

_(none yet -- experiments in flight)_

## Rules for this directory

1. **One file per experiment**, named `expNN-<slug>.md`, linked from the table
   above.
2. Every experiment file opens with a **## Verdict** section -- two to four
   sentences, stating plainly whether the evidence supports or undermines the
   claim it tested.
3. **Measured vs derived vs recalled** must be labelled. The proposal's
   Section 6 exists because that line was blurred once already.
4. A **negative or inconclusive result is a result** and gets the same
   treatment as a positive one. "Not settled, here is the blocker" is a valid
   verdict.
5. Job IDs and exact commands go in the **## Method** section so anything here
   can be re-run.
