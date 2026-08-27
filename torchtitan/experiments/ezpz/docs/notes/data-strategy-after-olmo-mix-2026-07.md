# Data Strategy After 4.67T olmo-mix-1124 Tokens

> Last updated: 2026-08-23

> **[2026-07-29 UPDATE -- section 1 prediction partially OVERTURNED by experiment.]**
> This memo predicted the primary lever is a "LR-decayed-to-zero (annealing)
> stage." The 2B MDS mid-training A/B
> ([sunspot/20260728-2b-mds-anneal-and-datamix.md](../records/experiments/agpt/sunspot/20260728-2b-mds-anneal-and-datamix.md))
> found the LR SCHEDULE is NOT the lever at 10B: constant-LR (flat) BEAT
> WSD-decay-to-0 on both the MDS and olmo bases (held-out FineMath NLL). What DOES
> move the needle is the DATA MIX: pure edu-web forgets math catastrophically
> (+0.308 nats), while a 75% math / 25% edu blend keeps ~all the math and captures
> ~all of edu's general gain. So the memo's "quality-upsampled mid-training" thesis
> is right that STAGE-2 DATA is high-ROI, but the "decay-to-zero" mechanism is not
> the source of the gain -- run stage-2 at constant LR and spend the effort on the
> mix (75/25 math/general validated; science-corpus blend is the next step).

Decision memo: where to get more pretraining data, and what to do once the
4.67T olmo-mix-1124 budget is consumed -- tailored to the AuroraGPT 2B/20B/80B
chains (Dolma-family mix, DOE science mission).

**Provenance.** Produced by a multi-agent research pass (18 agents: 4 parallel
corpus/strategy researchers -> adversarial verification of load-bearing
size/license/overlap claims -> synthesis) over primary sources (HF dataset
cards, papers, model reports), 2026-07-18. Numbers are marked **CONFIRMED**
(verified against a primary source) or **[approx]**. The verification pass
caught several real corrections (flagged inline): Nemotron-CC-v2 is ~5.9T not
6.6T; Zyda-2 is ~66% overlap not "near-total"; Nemotron-CC-v1's license was
misattributed; the DCLM "6.6x less compute" and "filtered top-25% repeats
safely" claims were overstated.

Related work already in flight:
- CPT stage-2 experiments: [`../production/cpt/`](../live/chains/cpt/README.md)
  (dolmino-style high-quality upsample at low constant LR -- this memo's
  primary recommendation).
- Synthetic-summary data POC:
  [`../experiments/synthetic/aurora/2026-07-11-summarize-olmo-mix-poc.md`](../records/experiments/synthetic/aurora/2026-07-11-summarize-olmo-mix-poc.md)
  (~9x compression via LLM summaries -- one of the blended synthetic styles
  below).

---

## 1. Bottom line

The highest-ROI move is not acquiring more raw web tokens -- it is a
**quality-upsampled, LR-decayed-to-zero mid-training (annealing) stage**,
sourced from (a) a model-based re-filter of the olmo-mix you already have,
(b) a small set of genuinely-additive **science/math corpora**, and (c) the
synthetic-summary POC used as *one of several* blended styles. This is exactly
the dolmino-style stage-2 experiment already running, and the literature says
it is worth roughly **+10 average downstream points at 2B-13B scale**, with
GSM8K/MATH gaining most (OLMo-2, MiniCPM, Llama-3 -- all CONFIRMED). Raw-token
acquisition matters far less for the 2B (which is ~115x past Chinchilla-optimal
token count, in the deployment-over-train regime) and most for the 80B (only
~3x Chinchilla, the most data-hungry). For unique tokens, almost every CC
corpus on the market is largely redundant with olmo-mix's DCLM backbone; the
real new-token value is a short list of **science-dense, low-overlap sources**
plus **LLM-synthetic** minting.

---

## 2. Where to get more UNIQUE tokens

### 2a. Adds genuinely new tokens (ranked for a science mission)

| Corpus | ~Tokens (new-value slice) | Domain | License | Overlap vs olmo-mix | Verdict |
|---|---|---|---|---|---|
| **Common Pile v0.1** (science/biomed/patent slices) | ~1-2T total [approx]; PubMed 36.6B, USPTO 157.4B, caselaw 19.7B, github_archive 11B | Openly-licensed papers, PubMed, patents, legal, code | Public-domain / open per-source (cleanest provenance) | **LOW** -- barely CC-derived | **ADD.** Cherry-pick non-overlap slices; skip its peS2o/arXiv/stackv2_edu (already in olmo-mix). Best license fit for DOE. |
| **Nemotron-CC-Math** | 133B (score>=3) / 52B (score>=4) | Scientific + math web (layout-aware CC extraction) | CC BY 4.0 [vendor-reported, arXiv:2508.15096] | MED (CC-sourced; distinct pipeline) | **ADD** -- top math/science lever; dedup vs olmo math + DCLM. |
| **Nemotron-CC-v2** | **~5.9T total** (corrected from 6.6T): ~3.36T base English CC + ~2.5T synthetic/QA derivatives; 8 new CC snapshots 2024-2025 | Web + synthetic QA/translated QA | **NVIDIA Data Agreement for Model Training** (corrected name); gated, model-training-only, 2025-08-18 | LOW-MED for the new snapshots (post-date DCLM); base CC overlaps | **ADD the synthetic/QA + 2024-25-snapshot slices**; dedup base CC. Vet license before any DOE release. Note: ~2.5T "synthetic" are LLM derivatives of the same CC base, not independent knowledge. |
| **FineWeb-Edu** | 1.3T (score>=3); 5.4T (score>=2) | Educational/knowledge web | ODC-By 1.0 (+ CC ToU) | MED (est.) -- edu filter differs from DCLM's fastText | **ADD** as targeted HQ upsample; dedup vs DCLM first. |
| **FineMath + MegaMath (synthetic split)** | FineMath 34B (3+) / 9.6B (4+); MegaMath synthetic ~64.5B | Math web + synthetic math QA | ODC-By | MED (CC/Stack-derived); synthetic split novel | **ADD** the 4+ and synthetic slices; dedup vs OpenWebMath/DCLM/StarCoder. |
| **Nemotron-CC v1 -- synthetic 1.9T** | 1.9T synthetic (of 6.3T; 4.4T real) | LLM-rephrased web | **Common Crawl ToU** (corrected; not "NVIDIA agreement") | Synthetic = LOW (additive); real 4.4T = doc-level overlap plausible, token-level redundancy **not proven** (corrected) | **ADD synthetic**; dedup real before use. |
| **HPLT v2 / FineWeb-2** (non-English) | HPLT 7.6T (Eng 2.86T), FineWeb-2 ~3T+ [approx] | Multilingual web; HPLT adds Internet-Archive WIDE-crawl pages CC never captured | HPLT **CC0-1.0**; FineWeb-2 ODC-By | LOW (non-English) | **ADD only if broadening to multilingual** -- lower priority for a currently-English mission. |
| **Cosmopedia + OpenCodeReasoning** | 25B + SFT-scale | Synthetic textbooks / code-reasoning | Apache-2.0 / CC BY 4.0 | LOW (synthetic) | **ADD to the anneal/CPT stage**, not the main mix. |
| **The Stack v2 (dedup)** | ~900B | Code, 658 langs | Per-file + Software Heritage/INRIA agreement | MED (superset of olmo's 83B StarCoder) | **ADD if growing code**; real ingestion friction (ships file IDs, not text). |

### 2b. Mostly-overlapping with olmo-mix -- SKIP as whole-corpus adds

| Corpus | ~Tokens | Why skip |
|---|---|---|
| **DCLM-baseline** | 4T | It **IS** olmo-mix (3.70T = ~93% already consumed). Only value: re-filter the 240T DCLM-Pool yourself. |
| **Zyda-2** | 5.07T | **~66% overlap** (the DCLM fraction), not "near-total" (corrected). Its distinct third is FineWeb-Edu -- source that directly. |
| **RedPajama-V2** | 30.4T deduped | Raw CC, same pool, HIGH overlap; you'd re-derive a DCLM-like filter. Skip for unique tokens. |
| **FineWeb (base)** | 15T (now ~18.5T) | Same CC pool, MED-HIGH overlap; only worth it post-dedup for its non-DCLM slice. |
| **TxT360** | ~5T (4.83T web) | Web bulk MED-HIGH overlap; **keep only the curated tail** (FreeLaw/USPTO/PG-19/StackExchange olmo lacks). |
| **CulturaX** | 6.3T | English HIGH overlap; non-English additive but FineWeb-2 is fresher. |
| **Already in olmo-mix (double-counts)** | -- | peS2o, arXiv, OpenWebMath, AlgebraicStack, StarCoder v1, and **Proof-Pile-2 in full**. Do NOT re-add. |

---

## 3. The strategy ladder (do in this order)

**(a) Is 4.67T even the right budget? -- Chinchilla vs over-train.**
Chinchilla-optimal is ~20 tokens/param. At full budget: 2B ~= 2,300 tok/param
(~115x optimal), 20B ~= 233 tok/param (~12x), 80B ~= 58 tok/param (~3x).
Implication: the **2B is deeply over-trained** (deployment-efficiency regime,
like Llama-3-8B on 15T) -- raw tokens are near-exhausted as a lever; the **80B
is only mildly over-trained** and, under data-constrained theory (excess params
decay faster, R_N*=5.3 < R_D*=15.4 -- CONFIRMED), genuinely benefits from more
*unique* tokens rather than more repeats. *Gain: reframes budget; Risk: none.*

**(b) Mid-training / annealing (the primary answer).** Short stage-2: decay LR
linearly to zero while heavily upsampling HQ + math/code/science data. Size
~10% of tokens is the sweet spot (MiniCPM; 2.5% too little), 50B (7B) to
100-300B (13B) in practice (OLMo-2). Put HQ data *in* the decay phase, not only
in post-hoc SFT. Soup 3-4 decay seeds. **Never** include benchmark train sets.
*Gain: +10.6/+10.3 avg pts (OLMo-2 7B/13B); GSM8K 24.1->67.5; MMLU gains modest
(~+3.9/+4.1); Llama-3 8B GSM8K +24%, MATH +6.4% -- all CONFIRMED. Risk: 405B saw
negligible gain, so payoff likely shrinks with scale; peak-LR/decay-fraction
sensitive.*

**(c) Domain upsampling for the science mission.** Use a **DoReMi proxy** (~8%
of main compute) or a **Llama-3-style 40B-token annealing probe** to *choose*
the stage-2 weights rather than guessing, then upweight math/code/scientific
domains. *Gain: DoReMi +6.5 avg pts and 2.6x faster-to-baseline on the Pile
(CONFIRMED; Pile-specific -- smaller on GLaM). Risk: weights tuned for balanced
perplexity may not match your science eval; keep a general-web floor.*

**(d) Controlled epoching (data-constrained scaling).** Repeating the fixed
corpus is **~free to ~4 epochs** (8.7B at 4 epochs = only 0.5% higher val loss)
and useful to ~16 epochs (R_D*=15.4), beyond which returns collapse (CONFIRMED,
Muennighoff 2305.16264). *Gain: cheap filler, especially for 20B/80B still
inside epoch 1. Risk: raises memorization/contamination; least attractive for
80B, which wants new tokens. Do not plan past ~4-8 pure-repeat epochs without
adding fresh/synthetic/filtered tokens.*

**(e) Re-filter + synthetic to feed the anneal.** Model-based quality
re-filtering makes a second pass worth more than a plain repeat: filter choice
alone spans **35% -> 44% MMLU** at 7B/280B (DCLM), and a classifier-ensemble HQ
subset gave **+5.6 MMLU / +3.1 avg over DCLM** at 8B/1T (Nemotron-CC).
*Correction: the "6.6x less compute" figure is on the 53-task average, and DCLM
only comes **close to** Llama-3-8B on MMLU (63.7 vs 66.2), it does not match it;
and there is no evidence that a classifier-filtered top-25% can be safely
repeated 2-4 epochs -- that was a splice of two unrelated results, so do not
rely on it.* Synthetic minting (WRAP, Nemotron-CC) adds fresh unique tokens:
WRAP ~3x speedup + >2% zero-shot QA at a strict **1:1 real:synthetic** ratio;
Nemotron-CC low-quality rephrasing +1.50 avg, and swapping 4-of-8 HQ epochs for
synthetic improved most benchmarks. *Corrections: synthetic does **not**
"sidestep the data wall entirely" (both recipes keep real data); the "~4x more
unique tokens" figure refers to their real extraction pipeline, **not** the
synthetic lever; and the "+5 MMLU" headline was vs a different model, not a
clean synthetic A/B. Keep synthetic in a **10-50% band, accumulate-never-
replace** -- accumulating real+synthetic provably bounds error (Gerstgrasser
2404.01413), while summary-only invites diversity collapse. Pure-synthetic
scaling also plateaus near ~300B tokens (SynthLLM), so size the synthetic
budget to that.*

---

## 4. AuroraGPT-specific recommendation (per chain)

**2B-256 (done at 4.674T -- the imminent case).** Raw-token acquisition is now
low-value; the deployment-over-train regime means the lever is quality. **Run
the stage-2 anneal now:** 50-100B tokens (~1-2% of budget; up to ~470B / 10% if
affordable), LR -> 0, on an upsampled mix from (a) a science-tuned re-filter of
olmo-mix's top slice, (b) the additive science set -- Nemotron-CC-Math-4+,
FineMath-4+, Common Pile PubMed/USPTO, Cosmopedia, (c) the synthetic summaries
as **one of >=3 styles, capped <50%**. Soup 3-4 seeds; decontaminate (10-gram +
difflib) against MMLU/ARC/HellaSwag/GSM8K. Expected ~+10 avg, GSM8K/MATH
biggest. (This generalizes the in-flight dolmino stage-2 + constant-LR fork
experiments.)

**20B (~12%, still inside epoch 1).** Keep training on the base mix. Before
end-of-budget, run a DoReMi proxy or a 40B annealing probe to *pick* the
stage-2 upsample weights; given the ~12x-Chinchilla headroom, consider
upweighting science domains during the remainder now. Anneal at end-of-budget
with the same recipe as 2B.

**80B (diverging, only ~3x Chinchilla = the most data-hungry).** **Fix
stability first** -- the NaN is a grad-path overflow, not the mix (see
[`../reference/guides/`](../reference/guides/) / the 80B fp32-residual investigation). This is the
chain with the most appetite for *genuinely new unique tokens*: prioritize
feeding it the low-overlap additive set (Nemotron-CC-v2 2024-25 snapshots +
Math, Common Pile science) over repeats. Expect a **smaller annealing payoff**
than 2B/20B (Llama-3's 405B saw negligible gains; 80B < 405B, so likely still
positive) -- validate on a small anneal before committing budget.

---

## 5. Caveats / open questions

- **Dedup is mandatory and will shrink every count.** All reported CC-corpus
  token counts are pre-cross-dedup. Overlap ratings here are curator estimates,
  not measured studies -- MinHash/Bloom every CC-derived add against the 3.70T
  DCLM backbone (and against each other, and code adds against the 83B
  StarCoder) before mixing.
- **License, for a DOE-redistributable model.** Common Pile is the cleanest
  (public-domain/open). The entire Nemotron family (CC-v2 web = NVIDIA Data
  Agreement for Model Training, gated/model-training-only; CC-v1 = Common Crawl
  ToU; synthetic slices may carry downstream Qwen/DeepSeek obligations) must be
  vetted before release. MathPile is non-commercial -- use MathPile_Commercial.
  The Stack v2 requires a Software Heritage/INRIA agreement and ships file IDs,
  not text.
- **Recent-science gap.** peS2o/S2ORC coverage stops ~2023-01-03; there is no
  public peS2o v3/v4. Fill 2023-2025 science via Common Pile pubmed/arxiv and
  Nemotron-CC-v2's 2024-2025 CC snapshots.
- **Confidence flags.** The math-corpus uplift numbers (Nemotron-CC-Math,
  FineMath, MegaMath) are vendor/arXiv-reported and were **not** independently
  re-verified in this pass. Nemotron-CC-v2's "~5.9T" and the license-name and
  token-scope corrections are verified. FineWeb-2 aggregate and Common Pile
  unique-token totals are [approx] (source cards publish no single aggregate).
- **Open question:** does the 80B annealing payoff hold at that scale? The only
  negligible-gain datapoint is 405B; a small 80B anneal probe should settle it
  before budget is committed.
