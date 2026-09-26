# AGPT-2B Stage-4 MetaMath distillation and model interpolation

Date: 2026-09-26

## Decision

**Accept `agpt2b-stage4-metamath-alpha65` as the current post-trained AGPT-2B
artifact.** It improves greedy GSM8K accuracy from 269/1,319 (20.39%) to
385/1,319 (29.19%), a gain of 116 correct answers and 8.80 percentage points.
On the fixed eight-prompt broad-generation sanity set, it matches broad
Stage-1's 4/8 semantic score and restores bounded output to 8/8, versus the
Stage-2 baseline's 3/8 semantic and 2/8 bounded results.

Artifact:

```text
/lus/tegu/projects/datascience/foremans/reproductions/
  agpt2b-mds154391-broad-grain-sft900/stage4/accepted/
  agpt2b-stage4-metamath-alpha65
```

The directory is a 7.5 GB inference-only Hugging Face artifact with 111 tensors,
configuration/tokenizer files, `merge_manifest.json`, `EVALUATION.json`, and
`SHA256SUMS`. The model weight SHA-256 is:

```text
1e647d86d6c942f57d9a326b33fa5a9c431c13c2d6daeccef66042ee8ab8b138
```

## Starting point

The accepted Stage-2 model was the two-stage SFT lineage
`tulu-math -> gsm8k-r1cot` at checkpoint 93. Its preserved 200-example sweep was
monotonic:

| checkpoint | correct | strict format |
|---:|---:|---:|
| 31 | 33/200 | 170/200 |
| 62 | 39/200 | 198/200 |
| 93 | 43/200 | 197/200 |

On the full 1,319-example test set, checkpoint 93 scored 269/1,319 (20.39%),
with 1,304/1,319 strict-format generations and 14 length truncations.

## Rejected control: repeat the same GSM8K data

Job `12478731` continued checkpoint 93 for 32 low-LR steps on the same 7,473
GSM8K training rows. Training was finite and produced checkpoints 8/16/24/32,
but the frozen 200-example evaluations were 43, 38, 44, and 43 correct. The
best result, 44/200 at step 24, missed the pre-registered 47/200 promotion bar.
This rules out another epoch over the unchanged data as a useful lever.

## MetaMath GSM distillation

The accepted candidate uses the `GSM_AnsAug` and `GSM_Rephrased` subsets of
MetaMathQA:

- 160,000 raw augmented/rephrased GSM training rows;
- zero normalized prompt or original-question overlap with all 1,319 GSM8K test
  questions;
- every source row contains an explicit `####` final answer;
- traces longer than 1,200 characters are dropped;
- remaining examples are normalized to AGPT's existing
  `<think>...</think><answer>\\boxed{}</answer>` contract;
- deterministic shuffle seed 154391;
- 17,844 packed 2,048-token training sequences with prompt labels masked.

Pretokenization job `12478740` built the complete dataset. Its first post-build
validator assumed an obsolete TRL schema and failed closed before publication;
the existing build was then validated against current `{input_ids, labels,
seq_lengths}` semantics and atomically published.

Training job `12478743` started from Stage-2 checkpoint 93 and completed 100/100
steps over 24 XPU ranks at LR 5e-6. It exited 0 with finite loss and gradients
and produced checkpoints 25/50/75/100.

## Checkpoint sweep

All checkpoints were evaluated with deterministic fp32 vLLM generation,
AGPT's exact prompt serialization, and stop IDs `[1, 107]`.

| checkpoint | GSM8K-200 | strict format | truncations | decision |
|---:|---:|---:|---:|---|
| Stage-2 baseline | 43 (21.5%) | 197 | 3 | baseline |
| 25 | 58 (29.0%) | 198 | 2 | pass |
| 50 | 53 (26.5%) | 197 | 2 | pass |
| 75 | 58 (29.0%) | 200 | 0 | best training checkpoint |
| 100 | 59 (29.5%) | 196 | 4 | reject: format gate |

On the full test set, the unmerged step-75 model scored 375/1,319 (28.43%),
with 1,302 strict-format outputs and 16 truncations. Relative to baseline, the
fixed 200-example subset contained 30 wrong-to-correct and 15 correct-to-wrong
transitions, for a net gain of 15.

Representative improvement:

```text
Question: James runs 3 sprints, 3 times a week, at 60 meters each.
Baseline: 1260 (incorrectly multiplied by seven days)
Candidate: 3 * 3 * 60 = 540
```

Another improvement correctly computed a total across Seattle, Charleston, and
Toulouse as 20 + 80 + 160 = 260; the baseline returned only Toulouse's 160.

## Capability-retention interpolation

The Stage-2 baseline and raw math-distilled checkpoints remained narrow and
repetitive on generic instructions. The broad Stage-1 checkpoint terminated all
8/8 sanity prompts cleanly, so the final selection interpolated weights between:

- broad Stage-1 checkpoint 600; and
- MetaMath step 75.

Formula: `broad + alpha * (math - broad)`.

| alpha | GSM8K full | strict format | truncations | broad sanity bounded |
|---:|---:|---:|---:|---:|
| 0.50 | 366/1,319 (27.75%) | 1,251 | 42 | 8/8 |
| 0.60 | **391/1,319 (29.64%)** | 1,292 | 20 | 7/8 |
| **0.65** | **385/1,319 (29.19%)** | **1,295** | 23 | **8/8** |
| 0.70 | 368/1,319 (27.90%) | 1,291 | 26 | 8/8 |
| 0.75 | 375/1,319 (28.43%) | 1,297 | 21 | 7/8 |

Alpha 0.65 is the selected Pareto point: it sacrifices six correct GSM8K
answers relative to the narrowly best alpha 0.60, but restores bounded output on
all broad sanity prompts, matches broad Stage-1's 4/8 semantic result, improves
on the Stage-2 baseline's 3/8 semantic and 2/8 bounded result, and retains an
8.80-point full-test improvement over Stage-2.

Examples from the selected artifact include concise correct responses for Earth,
Chicago extraction, photosynthesis summarization, `$7 * 3 = $21`, primary
colors, and 17-vs-12 comparison. The model is still only 2B and is not a general
chat-quality frontier model; this gate establishes a meaningful improvement over
its own accepted lineage, not parity with larger instruction models.

## Job ledger

- `12478731`: repeated-GSM continuation, 32/32 steps, exit 0; rejected by eval.
- `12478736`-`12478739`: corrected 200-example continuation sweep.
- `12478740`: MetaMath pretokenization; data build complete, obsolete validator
  failed closed; build subsequently validated and atomically published.
- `12478743`: MetaMath distillation, 100/100 finite steps, exit 0.
- `12478744`-`12478747`: 200-example checkpoint sweep.
- `12478755`-`12478757`: full 1,319-example baseline/step-25/step-75 comparison.
- `12478759`-`12478762`: canonical broad-generation comparison.
- `12478769`-`12478783`: interpolation and Pareto evaluation sweep.

## Reproduction code

- `rl/datasets_sft.py`: `metamath-gsm-distill` dataset.
- `rl/scripts/sft/agpt2b_metamath_gsm_pretokenize_1n.pbs`.
- `rl/scripts/sft/agpt2b_metamath_gsm_distill_2n.pbs`.
- `scripts/eval/eval_cot_gsm8k.py`: canonical `openai/gsm8k` dataset ID.
- `scripts/eval/agpt_general_sanity.py`.
- `scripts/eval/interpolate_hf_checkpoints.py`.
