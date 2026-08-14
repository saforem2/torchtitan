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
| [03](exp03-1b-proxy-design.md) | 1B proxy design | Can a 1B on a candidate mix clear the MMLU floor, and is the 0.28 gate statistically defensible? | design only | **DONE** -- gate redesigned |

## Findings

Filled in as experiments land. Each entry states whether it **supports**,
**undermines**, or **does not settle** the corresponding proposal claim.

### exp03 -- 1B proxy design (2026-08-14)

Four findings, three of which **undermine** the proposal as written. The
proposal has been amended in place; the originals are preserved here.

1. **UNDERMINES Section 5 ("1B proxy").** There is no 1B to build -- our
   production `agpt_2b` has **0.937B non-embedding params**, *below*
   Llama-3.2-1B's 0.973B. The 256,128-entry gemma vocab puts 525M in the
   embedding and 525M in the untied lm_head; **52.8% of the parameter budget
   does no depth-of-computation work.** Every smaller candidate costed
   (dim1536 L24 = 0.604B, dim2048 L16 = 0.721B) lands *below* the floor. The
   proxy is the production config, unchanged. *(Measured from
   `agpt/__init__.py:434`.)*

2. **UNDERMINES Section 5 ("hard gate 0.28").** Correct arithmetic, wrong
   shape. 0.28 is 8.2 binomial SE above chance at N=14,042 -- but only
   **1.45 pp above the highest near-chance MMLU we have ever recorded**
   (0.2655), and our across-config null is **2.4x overdispersed** (sd 0.00873,
   n=8), making 0.28 a 3.2 sigma gate under the empirical null. Structurally
   worse: a one-arm absolute threshold assumes the null is 0.25 and that
   harness/tokenizer/prompt-format contribute nothing -- all live hypotheses.
   Replaced with a **paired, control-referenced** rule (McNemar, +0.030 vs
   control, plus a same-mix reseed arm that measures the noise floor first).

3. **UNDERMINES Section 6's motivation, STRENGTHENS the design.** The "MMLU
   capability floor at 1-3B" premise appears **false**: Qwen2.5-0.5B scores
   47.5 while TinyLlama-1.1B at 3T tokens scores 25.3. Constraint is data, not
   scale. No published parameter threshold for MMLU emergence was located. A
   null at 0.937B is therefore *more* informative than the proposal assumed.
   *(Recalled, mixed harnesses -- not reproduced on ours.)*

4. **CORRECTS Section 1.3.** The 256N = 9.5% MFU row is a **torch 2.10**
   number; the current 2.13 stack measures **18.77%**. The table overstates
   the mid-scale collapse by ~2x. 512N ~9-11% is current; 64N is still the
   knee.

**Cost finding (supports feasibility):** FineWeb-Edu is *already on Aurora,
already gemma-tokenized, already in blendcorpus format* (1.551T tokens),
plus dolmino `flan` (17.1B) and `math` (11.7B). An arm is a weighted text
file, not a tokenization campaign. Four arms = **~771 node-hours = 0.14% of a
single 512N x 12h production job.** Counter-finding: the `_local` dataset
registrations in `datasets.py` point at `/lus/tegu/` (**Sunspot**), and
Aurora's `Nemotron-CC-Math-v1/4plus` is an **empty directory** -- a Nemotron
arm on Aurora would be planned against data that is not there.

**Provisional:** the 2.4x overdispersion estimate rests on n=8 spanning two
model sizes and five data treatments, so some spread may be real capability
difference. The reseed arm replaces it with a measurement; until then +0.030
is provisional.

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
