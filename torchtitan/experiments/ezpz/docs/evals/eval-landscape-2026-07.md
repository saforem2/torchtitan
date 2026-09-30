# AuroraGPT evaluation strategy: modern-suite review (2026-07)

> Last updated: 2026-08-31

> **Coverage audit (2026-09-30).** The modern backfill is no longer merely in
> progress, but its artifacts are not uniform enough for one combined curve.
> The reconciled 608-result corpus contains self-describing **MMLU 5-shot**
> ladders for 20B-256 (15 checkpoints, steps 6,900--8,300) and 20B-512
> (14 checkpoints, steps 6,550--7,600), plus five MDS base/tail/cooldown
> endpoints. The older 2B-v2 artifacts contain MMLU values at 8 checkpoints
> for 256N and 14 for 512N, but neither `n-shot` metadata nor an `@5shot` task
> key; they must not be silently mixed into a labeled 5-shot curve. GSM8K is
> sparse: two explicit 5-shot checkpoints per 20B lineage, eight MDS-tail
> checkpoints, three MDS-cooldown checkpoints, and only isolated or
> metadata-unknown 2B-v2 results. Therefore:
>
> - keep the seven-task commonsense charts as training thermometers;
> - publish MMLU separately, using only explicit 5-shot artifacts until the
>   older 2B provenance is recovered or those checkpoints are re-evaluated;
> - report GSM8K as an endpoint table, not a smooth training curve, until its
>   checkpoint coverage is dense and shot metadata is uniform.
>
> This is why MMLU/GSM8K are absent from the current overview SVGs: their
> omission is a hard-coded chart contract, not absence from the result corpus.

Decision-focused review of what we evaluate, what modern peers evaluate, and how
we compare. Grounded in a fact-checked research pass over primary sources (model
cards, papers, the HF Open LLM Leaderboard blog); numbers are marked
**[CONFIRMED]** (verified against a primary source) or **[internal]** (our own
reported AuroraGPT numbers).

## Bottom line

Our current 7-benchmark commonsense suite is **fine as a from-scratch training
thermometer but nearly useless for positioning against modern peers**. Every
model we'd compare against (SmolLM3-3B, Llama-3.2, OLMo-2) is differentiated on a
newer suite we don't run: **MMLU, GSM8K, MMLU-Pro, AGIEval**. Action taken: added
**MMLU (5-shot) + GSM8K (5-shot) + ARC-Challenge (25-shot)** to the eval driver
(`SHOTS_SPEC` mixed-few-shot support) and backfilled them across the checkpoint
ladders. Deferred IFEval/BBH/GPQA/MATH/HumanEval until instruct or many more
tokens.

## 1. Are the current evals meaningful?

**Partially -- right *kind* of eval for where we are, wrong *set* for the
comparisons we want.**

- **Right:** for a base, from-scratch model at ~12% of target tokens,
  loglikelihood multiple-choice commonsense tasks (HellaSwag, ARC, PIQA,
  Winogrande, OpenBookQA, BoolQ) show signal early -- no instruct-tuning, no chat
  template, no generation -- and climb smoothly from chance toward ceiling. Our
  20B ARC-Easy 0.27->0.65 is a legitimate, readable trajectory (it's what showed
  the fp32-master fix working). **[CONFIRMED this is the correct use of these
  tasks.]**
- **Wrong/limiting:** HuggingFace **retired all six OLL-v1 tasks in June 2024**
  for *saturation* ("models are now reaching baseline human performance on
  HellaSwag, MMLU, and ARC"). **[CONFIRMED]** They no longer separate strong
  models. We also run ARC-*Easy* while the field reports ARC-*Challenge*, and we
  had **no knowledge (MMLU) and no math/reasoning (GSM8K)** benchmark at all --
  the two most-cited base-model numbers.

**Verdict:** keep the commonsense suite as a training-progress dashboard; do not
present it as a competitive scorecard.

## 2. What we added (and what we deferred)

Exact `lm-evaluation-harness` (0.4.10) task names + shots. **Critical gotcha:**
few-shot is NOT hardcoded in the harness YAMLs for the classic MC tasks
(MMLU/ARC/HellaSwag default to `num_fewshot=0`); you MUST pass the shot count.
Only `gsm8k` pins its own (5). This is why the driver needed the `SHOTS_SPEC`
per-group change -- a single global `--num_fewshot` can't do mmlu-5 + arc_c-25.

### Tier A -- ADDED NOW (base-valid, informative at 2B/20B, 600B tok)
| Task | harness name | shots | Why |
|---|---|---|---|
| MMLU | `mmlu` | 5 | #1 base knowledge metric; every peer reports it |
| GSM8K | `gsm8k` | 5 | #1 base math/reasoning metric |
| ARC-Challenge | `arc_challenge` | 25 | the variant peers report (we ran ARC-Easy) |

### Tier B -- stretch (low expected, still standard)
`mmlu_pro` (5), `triviaqa`, `drop` (3), `agieval_en` (3-5).

### Tier C -- DEFER (instruct-tuned OR much more training)
`ifeval` (**base scores ~0 -- do NOT run on base**), `bbh`, `gpqa`, `math`,
`humaneval`/`mbpp`, `truthfulqa`. Avoid `*_instruct`, `gsm8k_cot_llama` (needs
chat template), `truthfulqa_gen` (needs a judge).

## 3. Peer comparison

**Big caveat first: we cannot fairly rank yet.** SmolLM3-3B trained on ~11T
tokens, Llama-3.2 on ~9T+; our 20B is at ~604B (**13%**). And peers report
benchmarks we're only now starting to run.

### 3a. Where benchmarks overlap
Only **HellaSwag** overlaps cleanly (ARC is a mismatch -- we ran ARC-Easy, peers
report ARC-Challenge). Peer numbers [CONFIRMED]:

| | AuroraGPT-20B @604B [internal] | SmolLM3-3B | OLMo-2-7B | OLMo-2-13B |
|---|---|---|---|---|
| HellaSwag | ~0.61 | 76.2 | 83.8 | 86.4 |

The gap is large but **expected** -- HellaSwag climbs steeply *late* in training;
0.61 at 12% of tokens is mid-trajectory, not final.

### 3b. Modern benchmarks (backfilled with coverage caveats) -- peer numbers [CONFIRMED]
| Benchmark | Llama-3.2-1B base | Llama-3.2-3B base | SmolLM3-3B base | OLMo-2-7B | OLMo-2-13B |
|---|---|---|---|---|---|
| MMLU (5-sh) | 32.2 | 58.0 | 44.1 (CF*) | 63.7 | 67.5 |
| MMLU-Pro | -- | -- | 19.6 CF / 32.7 MCF | 31.0 | 35.1 |
| GSM8K | (instruct 44.4) | (instruct 77.7) | 67.6 (5-sh) | 67.5 | 75.1 |
| ARC-Challenge | 32.8 | 69.1 | -- | 79.8 | 83.5 |
| AGIEval (En) | 23.3 | 39.2 | -- | 50.4 | 54.2 |

### Data-quality flags (so we don't cite bad numbers)
- **Llama-3.2 base card has NO published HellaSwag or GSM8K** -- any such figure
  is the *instruct* number mislabeled. [CONFIRMED]
- **SmolLM3 MMLU/ARC are "CF" (continuation-format) variants**, not standard
  `mmlu`/`arc_challenge` -- not directly comparable. [CONFIRMED]

## 4. The right peer: OLMo-2

OLMo-2 (7B/13B) is our **closest analog** -- trained on the *same olmo-mix data
family* we use, fully documented, evaluated via the OLMES suite. As we scale,
benchmark against OLMo-2 (with the token-count caveat explicit), not Llama/SmolLM.

## References (primary, verified)
- HF Open LLM Leaderboard v2 retirement blog (saturation rationale).
- Meta Llama-3.2 model card (base: MMLU, AGIEval, ARC-C, DROP; no base HellaSwag/GSM8K).
- HuggingFace SmolLM3-3B model card (CF-variant caveats).
- OLMo-2 paper "2 OLMo 2 Furious" (arXiv:2501.00656), Table 6; OLMES suite.
- EleutherAI lm-evaluation-harness task YAMLs (few-shot defaults).

## Implementation
- Driver: `scripts/eval/eval-{20b,2b}-v2.sh` -- `SHOTS_SPEC` mixed-few-shot loop.
- Backfill: `scripts/eval/oneoff/eval-backfill-{20b-512n,20b-256n,2b-512n}-modern.sh`.
- Aggregator: `eval/aggregate_evals.py` reads `exact_match` (GSM8K) + mmlu/gsm8k
  colors + baselines.

## Backfill runs

### 2026-08-03 -- modern-block tail backfill (`8729921` 256n, `8729922` 512n)
The commonsense-7 ladder is complete to each chain live tip (256n step-6800,
512n step-6500), but the modern block (MMLU-57 loglikelihood + gsm8k) was left
partial: the prior tail pass (`8714836` 256n) died Exit -29 -- walltime -- mid-MMLU
under the old 12h cap, and the 512n modern phase (`8714837`, Exit 0) only reached
step-6100. These two capacity-queue jobs run modern-only at **48h** walltime so the
slow ~56k-request/step MMLU pass cannot be walltime-killed again:

- `8729921` (256n): MMLU-5 + ARC-C-25 on steps 5900,6100,6400,6500,6600,6700,6800;
  gsm8k on 6800. REPO override -> relocated `agpt-20b-n256` clone.
- `8729922` (512n): MMLU-5 + ARC-C-25 on steps 6200,6300,6400,6500; gsm8k on 6500.

Content-aware skip-guard merges into the existing per-step `results.json`, so the
HF conversions cached by the commonsense pass are reused (no re-convert). Scripts:
`scripts/eval/oneoff/eval-backfill-20b-{256n,512n}-modern-tail.sh` (commit daeb510b7).
