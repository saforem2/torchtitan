# AuroraGPT MMLU sits at chance because the models answer with a letter prior

> 2026-08-31. Sunspot. Affects every MMLU number the project has published.

## Bottom line

Every AuroraGPT MMLU score is at chance, and the cause is measured, not
guessed: **the models pick an answer letter largely independent of the
question.** Fitting the question-blind model

```
acc(subject) = sum_L p_L * keyfreq_L(subject)
```

recovers a fixed prior of roughly **p(A,B,C,D) = (0.37, 0.32, 0.20, 0.11)**
and collapses the per-subject residual variance to the binomial noise floor.
Per-subject accuracy correlates **+0.64 to +0.76** with how often that
subject's answer key is "A".

The sharpest statement of the result: **always answering "D" scores 0.2689,
which beats all three measured arms.** Our models are below the best
constant-letter baseline.

Constant-letter baselines, computed directly from the 14,042 `cais/mmlu`
test-split answer keys (`scripts/eval/mmlu_letter_baseline.py`):

| strategy | score |
|---|---:|
| always A | 0.2295 |
| always B | 0.2465 |
| always C | 0.2551 |
| **always D** | **0.2689** |

Note the spread: the answer key itself is not uniform, so "chance" for a
letter-guessing model is anywhere in 0.2295-0.2689 depending on which letter
it favours -- a wider band than the nominal 0.25 and wider than the
0.0036 standard error. Any MMLU number inside that band is uninterpretable
without knowing the model's letter distribution.

This is not "the model knows nothing." MMLU as we run it scores the *letters*
`["A","B","C","D"]`; ARC and HellaSwag score the *answer text*. Letter binding
is a learned format skill, and the same checkpoints that sit at chance on
MMLU score HellaSwag ~0.57 and ARC-Easy ~0.66. Those numbers are consistent,
not contradictory.

## The numbers

| model | tokens | MMLU (5-shot, MCF) |
|---|---:|---:|
| flagship 2B (`n512-gbs12288`) step 46,429 | 4.674T | 0.2579 |
| Megatron-DeepSpeed 2B (pre-torchtitan) | 7.064T | 0.2413 |
| 20B-256 | 0.39T | 0.2599 |
| 20B-512 | 0.71T | 0.2655 |
| mix arm `edu100` | ~600B + 10B | 0.2476 |
| mix arm `owm100` | ~600B + 10B | 0.2482 |
| mix arm `owm75` | ~600B + 10B | 0.2393 |
| **always answer "D"** | -- | **0.2689** |

Chance is nominally 0.25 (SE ~0.0036 at N=14,042), but the best
constant-letter strategy scores 0.2689 -- see the baseline table above. Twelve configurations
across a 20x token span, two model scales and five data treatments all land
between 0.2413 and 0.2655. **No ordering among them is meaningful.** The
most-trained model scores lowest.

The mix-arm figures are lm-eval's own sample-weighted `mmlu` group key. Use
that, not a hand-rolled macro: averaging every `mmlu_`-prefixed key
double-counts the four category rollups (`mmlu_stem`, `mmlu_other`,
`mmlu_social_sciences`, `mmlu_humanities`) alongside the 57 subjects, and
reads ~0.005 high.

## Why this is not an eval bug

Every plumbing hypothesis was checked and came back clean:

* **Few-shot was applied.** Job logs show `Overwriting default num_fewshot of
  mmlu_abstract_algebra from None to 5` for all 57 subjects on all three arms,
  and every subject's `dev` split has at least 5 rows.
* **Tokenizer matches.** All three HF exports share tokenizer.json md5
  `b60c6af2f42f20e636cdc604b5732d93`, GemmaTokenizer vocab 256000 against
  `config.json` vocab_size 256000. A real MMLU prompt round-trips byte-exact
  and the four scored continuations are single clean tokens
  (`_A`=586, `_B`=599, `_C`=585, `_D`=608). Not the Polaris failure mode.
* **No truncation.** `LIMIT` empty, all 57 subjects at full size, 14,042 total.
* **The harness is proven on this exact code path.** Job `8736838` ran
  Llama-3.2-1B = **0.3121** and Llama-3.1-8B = **0.6530** through the same
  `simple_evaluate(num_fewshot=5, device="xpu:0")`. A 1B model clears chance
  where our 2B at 4.674T does not.

## The trap in the per-subject spread

Per-subject accuracy is **over-dispersed** relative to an exact-chance null
(chi2/df = 1.74 / 2.17 / 2.24; variance about 0.25 is 1.7-2.0x the binomial
floor) and correlates **r = 0.70-0.77 across independent arms**. That pattern
normally means real signal, and reading it that way is the mistake this
document exists to prevent.

Conditioning on answer-key letter frequency alone removes it:

| arm | fitted p(A,B,C,D) | R2 | residual var / binomial floor |
|---|---|---:|---:|
| edu100 | (0.37, 0.32, 0.20, 0.11) | 0.32 | 1.14 (was 1.69) |
| owm100 | (0.37, 0.32, 0.21, 0.09) | 0.34 | 1.26 (was 1.90) |
| owm75  | (0.57, 0.34, 0.03, 0.06) | 0.55 | **0.88** (was 1.97) |

The "consistently strong" subjects (virology, human_aging, business_ethics)
are simply the A/B-heavy ones; the "weak" ones (professional_medicine, 0.45
of keys "D"; high_school_statistics, 0.47 "D") are D-heavy.

**Recorded as an error made during this investigation:** the fanning-out
spread on the 30B ladder -- subjects above chance going 24 -> 23 -> 31 of 57,
max 0.304 -> 0.372 -> 0.423 -- was read as "not the shape of a model at pure
chance." It is the shape of a letter prior interacting with per-subject key
distributions. Do not read per-subject structure as knowledge without fitting
the letter model first.

## What the literature says, and what it predicts

Two primary sources describe exactly this, and both point the same way.

**SmolLM2** (arXiv:2502.02737 Sec 4.3, Fig 6) reports above-chance MMLU in the
**cloze** formulation while the **multiple-choice (letter) formulation only
clears chance after 6T tokens**. Their own MCF number is 29.62 across 0-6T.

**OLMES** (arXiv:2406.08446) on OLMo-7B: *"Around 400B tokens, the model
starts gaining the ability on the MCF format... Before that point, there is
good signal from CF while MCF is random."* It recommends reporting
**max(CF, MCF)**.

Every AuroraGPT MMLU number is the MCF side, at scales below where either
paper saw MCF emerge.

## Two mechanisms, both actionable

**1. Format.** `mmlu` uses `doc_to_choice: ["A","B","C","D"]`;
`mmlu_continuation` uses `doc_to_choice: "{{choices}}"` and scores answer
text, directly comparable to our ARC/HellaSwag setup. Job `12474365` runs both
on 30B step-2000, 5-shot, same checkpoint and tokenizer, so `doc_to_choice`
is the only variable. A CF score clearly above chance where MCF sits on it
means the knowledge is present and the format is the barrier.

**2. Data composition.** Measured over the 1,438 files of
`data-lists/aurora/olmo-mix-1124.txt`:

| corpus | share |
|---|---:|
| dclm | **94.80%** |
| starcoder | 2.46% |
| pes2o | 1.47% |
| arxiv | 0.53% |
| open-web-math | 0.32% |
| algebraic-stack | 0.32% |
| wiki | 0.09% |

Faithfully OLMo-2 **stage 1 only**. No educational-filtered or textbook
corpus; academic content is 2.0% and is research papers, not pedagogy.
SmolLM2 stage 1 is **60% FineWeb-Edu / 40% DCLM**.

The FineWeb-Edu paper (arXiv:2406.17557) reports **33.6% MMLU at 38B tokens**
where the next-best dataset needs 300B, and its ablation has FineWeb-Edu
beating DCLM on MMLU (37.5 vs 35.5) while DCLM wins HellaSwag (62.3 vs 60.1)
-- our exact profile.

The closest precedent is our own twin. **OLMo-2 uses the same
`olmo-mix-1124`**, and its 1B gains **MMLU 26.9 -> 44.3 (+17.4) from Dolmino
mid-training alone** (arXiv:2501.00656 Table 9). At 7B/13B the same
intervention is worth only +3.9/+4.1, so at small scale mid-training is
nearly the entire MMLU signal.

### We did not skip Dolmino, and it did not help

An earlier draft of this document said we skip that stage. **That was wrong,
and it was wrong twice over.**

1. The torchtitan CPT sweep forked the plateaued 2B base onto dolmino at
   three blend ratios (`docs/production/cpt/README.md`, 2026-07). It
   DEGRADED benchmarks: HellaSwag -7.4pp, ARC-Easy -10.4pp, while train loss
   IMPROVED 0.31 nats. That doc attributes the damage to an LR re-warm shock
   (a converged base re-warmed to peak 2.28e-5) and calls the hypothesis
   open.
2. The pre-torchtitan Megatron-DeepSpeed 2B ran dolmino as its actual stage
   2, with no shock to blame:
   `DATA_FILE_LIST=ALCF/data-lists/aurora/dolmino-mix-1124-fused-file-list.txt`,
   `LR=2.17e-5`, `LR_DECAY_STYLE=constant`
   ([train_aGPT_2B_sophiag_stage2.sh](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/train_aGPT_2B_sophiag_stage2.sh)).
   The W&B report shows that model cycling three data lists across its
   lifetime -- `olmo-mix-1124`, `dolmino-mix-1124-fused`, and
   `stage1-33-stage2-33-stage3-34` -- at gbs 6144 out past 7T tokens.
   **It scores MMLU 0.2413, the lowest number in the table above.**

### Why: Dolmino is not the intervention the literature credits

Measured over the data-lists themselves, grouping weights by corpus:

| corpus | olmo-mix-1124 | dolmino-mix-1124 |
|---|---:|---:|
| dclm | 94.80% | **89.18%** |
| pes2o | 1.47% | 6.88% |
| flan | -- | 1.97% |
| math | 0.32% | 1.36% |
| wiki | 0.09% | 0.44% |
| stackexchange | -- | 0.16% |
| starcoder | 2.46% | -- |
| arxiv | 0.53% | -- |

(1,438 and 324 files respectively; olmo's weights sum to 2.717 and dolmino's
to 1.0, so both are normalized above.)

**The "high-quality mid-training mix" is still 89% DCLM.** It differs from
stage 1 by about 5.6 points of DCLM redistributed into peS2o, plus 2% FLAN
and 1.4% math. It is the same DCLM-dominated web distribution with a larger
academic slice -- not a different kind of data.

Contrast what the literature credits for MMLU: SmolLM2 stage 1 is **60%
FineWeb-Edu / 40% DCLM**, and FineWeb-Edu's own ablation reaches 33.6% MMLU
at 38B tokens via *classifier-filtered educational content*. That is a change
of kind. Dolmino is a change of degree.

This also explains the CPT loss/benchmark divergence without needing the LR
story: dolmino's lower entropy (more DCLM-filtered, more peS2o) yields lower
loss on its own distribution while moving the model away from the eval
distribution.

**Open question this does NOT settle.** OLMo-2 evaluates with OLMES, which
recommends reporting max(CF, MCF). If their +17.4 is a cloze or max number
while ours is letters-only, the two are not the same measurement and Dolmino
may have moved something we never scored. No CPT or MDS checkpoint has ever
been scored on MMLU in cloze format. That is eval-only work on checkpoints
that already exist.

**Cosmopedia is a red herring** -- 4% of SmolLM2 *stage 4* only, absent from
SmolLM3, no controlled ablation isolating it, and SmolLM v1 noted textbook
content helped everything *except* MMLU.

## How to report MMLU from here

1. Read lm-eval's `mmlu` group key, not a macro over `mmlu_*`.
2. State the format. An MCF number below ~1T tokens is close to
   uninformative on its own; report CF alongside it, or max(CF, MCF) per
   OLMES.
3. State the shot count. It now lands in `results.json` under `n-shot`
   (commit 5ab9e50f6); before that it was recoverable only from job stdout.
4. Do not compare against SmolLM3's published 44.1 -- that is **MMLU-CF**,
   Microsoft's *contamination-free* benchmark (arXiv:2412.15194), a different
   task. (Our earlier note expanded "CF" as continuation-format; the acronym
   was wrong, the "not comparable" conclusion was right.)
5. Compare against **OLMo-2**, which shares our data family, with the token
   count stated.

## Related

* [[dead-cli-flags-in-repo-root-pbs]] -- same family of failure: a launcher
  that runs and reports success while measuring the wrong thing.
* `docs/evals/eval-landscape-2026-07.md` -- task selection, the `SHOTS_SPEC`
  few-shot gotcha, confirmed peer numbers.
* Commits `60fe6bb24` (abort on silent 0-shot MMLU) and `5ab9e50f6` (persist
  `n-shot`; allow a task at two shot counts).

## Sources

* SmolLM2: https://arxiv.org/abs/2502.02737
* FineWeb / FineWeb-Edu: https://arxiv.org/abs/2406.17557
* OLMo-2: https://arxiv.org/abs/2501.00656
* OLMES: https://arxiv.org/abs/2406.08446
* MMLU-CF: https://arxiv.org/abs/2412.15194
