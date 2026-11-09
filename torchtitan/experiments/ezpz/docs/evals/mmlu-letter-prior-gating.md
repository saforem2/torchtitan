# MMLU letter-prior gating note: constant-D beats every agpt-2B model

> **Status 2026-11-09.** No new compute. This note is a short companion to the
> full audit in [`mmlu-letter-prior-at-chance.md`](mmlu-letter-prior-at-chance.md)
> (2026-08-31), restating its constant-letter finding and recording the
> implication for agpt-2B promotion gating. Evidence is job 12474379-era
> repro artifacts plus the 12-config sweep already on disk.

## Bottom line

**Constant-D (always answer "D") scores 0.2689, which beats all twelve
measured agpt-2B MMLU configurations (0.2413-0.2655).** This is not a model
bug — tokenizer, harness, few-shot, and truncation were checked and came back
clean (Llama-3.2-1B scores 0.3121 through the same path). It is a harness
prior: the `cais/mmlu` answer key is skewed, and every AuroraGPT base at
4.674T behaves as a letter prior rather than a question-conditional answerer.
**Do not gate agpt-2B promotion on MMLU.** Use ARC-Easy, HellaSwag, and
GSM8K-CoT (post-SFT) instead; report MMLU as `max(CF, MCF)` alongside
`mmlu_continuation` when available, with the constant-letter band stated.

## 1. What was measured

Constant-letter baselines computed directly from the 14,042 `cais/mmlu`
test-split answer keys (`scripts/eval/mmlu_letter_baseline.py`):

| strategy | score |
|---|---:|
| always A | 0.2295 |
| always B | 0.2465 |
| always C | 0.2551 |
| **always D** | **0.2689** |

The spread is not 0.25 +/- noise. The answer key itself is not uniform, so
"chance" for a letter-guessing model is a **band 0.2295-0.2689** — wider
than the nominal 0.25 and wider than the ~0.0036 standard error at N=14,042.
Any MMLU number inside that band is uninterpretable without the model's
letter distribution.

All agpt-2B-family MMLU scores land inside that band:

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

Twelve configurations across a ~20x token span, two model scales, and five
data treatments all between 0.2413 and 0.2655. **No ordering among them is
meaningful.** The most-trained model scores lowest.

Full table, verification from the `step-46429/results.json` artifact
(`mmlu` group key 0.25787 +/- 0.00369, leaf macro 0.2631), and the Llama-1B
control (0.3121) are in
[`mmlu-letter-prior-at-chance.md#the-numbers`](mmlu-letter-prior-at-chance.md#the-numbers).

## 2. Why this is a harness prior, not a model bug

Every plumbing hypothesis was checked and came back clean — full detail in
[`mmlu-letter-prior-at-chance.md#why-this-is-not-an-eval-bug`](mmlu-letter-prior-at-chance.md#why-this-is-not-an-eval-bug):

- **Few-shot applied.** Job logs show `Overwriting default num_fewshot of
  mmlu_abstract_algebra from None to 5` for all 57 subjects on all three
  arms, every subject's `dev` split has >=5 rows.
- **Tokenizer matches.** All HF exports share `tokenizer.json` md5
  `b60c6af2f42f20e636cdc604b5732d93`, GemmaTokenizer vocab 256000 = config
  vocab 256000. Four scored continuations are single clean tokens
  (`_A`=586, `_B`=599, `_C`=585, `_D`=608). Not the Polaris failure mode.
- **No truncation.** `LIMIT` empty, 57 subjects at full size, 14,042 total.
- **Harness proven on this code path.** Job 8736838: Llama-3.2-1B =
  **0.3121** and Llama-3.1-8B = **0.6530** through the same
  `simple_evaluate(num_fewshot=5, device="xpu:0")`. A 1B model clears
  chance where our 2B at 4.674T does not.
- **Letter binding is the format.** `mmlu` scores letters `["A","B","C","D"]`;
  ARC and HellaSwag score answer text. Same checkpoints at chance on MMLU
  score HellaSwag ~0.57 and ARC-Easy ~0.66 — consistent, not contradictory.

The letter-prior model `acc(subject) = sum_L p_L * keyfreq_L(subject)` was
fitted per arm:

| arm | fitted p(A,B,C,D) | R2 | residual var / binomial floor |
|---|---|---:|---:|
| edu100 | (0.37, 0.32, 0.20, 0.11) | 0.32 | 1.14 (was 1.69) |
| owm100 | (0.37, 0.32, 0.21, 0.09) | 0.34 | 1.26 (was 1.90) |
| owm75  | (0.57, 0.34, 0.03, 0.06) | 0.55 | **0.88** (was 1.97) |

Per-subject accuracy correlates +0.64 to +0.76 with key letter frequency. The
"strong" subjects (virology, human_aging, business_ethics) are A/B-heavy;
the "weak" ones (professional_medicine 0.45 "D", high_school_statistics
0.47 "D") are D-heavy. Conditioning on the letter prior collapses over-
dispersion to the binomial floor.

## 3. Why this blocks MMLU as a gating metric

**MMLU as we run it is MCF (multiple-choice letter) scoring.** Below
~1T tokens OLMES (arXiv:2406.08446) and SmolLM2 (arXiv:2502.02737) both
describe MCF sitting at chance while CF (cloze, scoring answer text)
already shows signal. SmolLM2 reports MCF only clears chance after 6T tokens
on the letter formulation while cloze is above chance throughout; their own
MCF number is 29.62 across 0-6T.

That predicts exactly the agpt-2B profile: knowledge present (HellaSwag,
ARC-Easy), letter binding absent. The same distinction explains the
Dolmino/CPT history:

- Dolmino mid-training at 89% DCLM changed MMLU minimally (+17.4 at 1B per
  OLMo-2 Table 9, but only +3.9/+4.1 at 7B/13B; our MDS dolmino arm scored
  0.2413, the *lowest* row).
- FineWeb-Edu filtering (HuggingFaceFW) is the distribution the literature
  credits for MMLU (37.5 vs 35.5 over DCLM in the FineWeb-Edu ablation, 33.6%
  at 38B vs 300B for the next-best dataset). See
  [`../production/sft/fineweb-edu-60-40-kind-shift.md`](../production/sft/fineweb-edu-60-40-kind-shift.md).

Job 12474382 ran both formats on 30B step-2000 (`--limit 10` per subject,
570 questions/arm):

| format | task | macro | above chance |
|---|---|---:|---:|
| MCF (letters) | `mmlu` | 0.2281 | 21/57 |
| CF (cloze) | `mmlu_continuation` | 0.2386 | 24/57 |
| CF - MCF | | +0.0105 | SE 0.0256 |

Not significant at this N, and at 3.93B tokens arguably too early for either
format. The null excludes only a *large* CF/MCF gap (double-digit, per the
literature) and cannot exclude a small one. The planned gate is the flagship
2B at 4.674T on Aurora — not this 30B early checkpoint.

## 4. How to report MMLU from here (and what NOT to gate on)

Adapted from
[`mmlu-letter-prior-at-chance.md#how-to-report-mmlu-from-here`](mmlu-letter-prior-at-chance.md#how-to-report-mmlu-from-here):

1. **Read lm-eval's `mmlu` group key**, not a hand-rolled macro over
   `mmlu_*` — the macro double-counts category rollups and reads ~0.005 high
   (verified 0.2631 vs 0.2579 on step-46429).
2. **State the format.** An MCF number below ~1T tokens is close to
   uninformative alone; report **CF alongside it**, or `max(CF, MCF)` per
   OLMES (arXiv:2406.08446). No CPT checkpoint has ever been scored on MMLU
   in either format — that is eval-only work on checkpoints that already
   exist.
3. **State the shot count.** It lands in `results.json` under `n-shot`
   since commit `5ab9e50f6`; older artifacts (including step-46429) have it
   unrecorded.
4. **Do not compare against SmolLM3's published 44.1** — that is MMLU-CF
   (contamination-free, arXiv:2412.15194), a different task with the same
   acronym. Compare against **OLMo-2**, which shares our data family, with
   token count stated.
5. **State the constant-letter band** (0.2295-0.2689) when reporting any
   new MMLU number, so a reader can see whether it clears chance after
   accounting for key skew.

### Gating rule for agpt-2B

**Do not gate promotion on MCF MMLU.** The 0.2413-0.2655 band is below or
within the constant-letter ceiling; movements inside it are letter-prior
noise, not knowledge signal.

Gate on:

- **ARC-Easy** and **HellaSwag** (commonsense, score answer text — the
  training thermometer that actually moves: ARC-Easy 0.27 -> 0.66 over
  4.674T, HellaSwag 0.29 -> 0.61).
- **GSM8K-CoT accuracy** post-SFT (B2 = 0.205 at format 0.985; every
  narrow-data extension lands 0.02-0.105 — see
  [`../production/POST-TRAINING-2B.md`](../production/POST-TRAINING-2B.md)
  and
  [`../production/sft/fineweb-edu-60-40-kind-shift.md`](../production/sft/fineweb-edu-60-40-kind-shift.md)).
- **MMLU `max(CF, MCF)`** as an *informational* metric once the CPT bases
  are scored in both formats — not as a gate until it demonstrably clears
  the constant-letter band.

When a new base does clear MMLU, the letter prior should be re-fitted
(`acc(subject) ~ sum_L p_L * keyfreq_L(subject)`) to confirm the model has
actually moved off the prior rather than shifted its `p_L`.

## Cross-refs

- Full audit (fitted priors, per-subject correlations, cloze experiment,
  reporting checklist):
  [`mmlu-letter-prior-at-chance.md`](mmlu-letter-prior-at-chance.md)
- Corpus composition lever (the pre-training fix):
  [`../production/sft/fineweb-edu-60-40-kind-shift.md`](../production/sft/fineweb-edu-60-40-kind-shift.md)
- Post-training ceiling (~0.2 GSM8K-CoT) and B2v2/B3/B4 degradation:
  [`../production/POST-TRAINING-2B.md`](../production/POST-TRAINING-2B.md)
- Evaluation strategy and peer comparison:
  [`eval-landscape-2026-07.md`](eval-landscape-2026-07.md)
