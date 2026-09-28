# AGPT-2B Inkling synthetic-data distillation

Date: 2026-09-28

## Status and decision

**Stage-9 training is complete, but the model is not promoted.** The best
checkpoint, step 10, is coherent and bounded on 8/8 direct-generation prompts,
but scores 48/200 on the fixed GSM8K development slice versus 59/200 for the
accepted Stage-7 model. A final IFEval job (`12478986`) is queued. Stage-9 will
remain rejected unless that job shows a material instruction-following gain
large enough to justify the math regression.

The accepted production artifact remains Stage-7. See the prior
[Stage-4/Stage-7 lineage report](2026-09-26-agpt2b-stage4-metamath.md).

## Objective

Earlier MetaMath and OpenMath continuations either damaged GSM8K retention or
produced repetitive instruction behavior. This experiment tested a different
teacher and data regime: use ALCF Minerva's `inkling-bf16` endpoint to generate
short, structured responses for a source-balanced prompt manifest, verify the
outputs before training, and fine-tune from the accepted Stage-7 model rather
than from a damaged specialist.

The teacher is served by the
[ALCF Inference Service](https://docs.alcf.anl.gov/services/inference-endpoints/)
through its OpenAI-compatible Minerva chat-completions endpoint. The generation
code was introduced in
[`fa83e38827`](https://github.com/saforem2/torchtitan/commit/fa83e38827),
then hardened for cached MMLU decontamination, unique quota backfilling, and
bounded JSON output in
[`1ee992442d`](https://github.com/saforem2/torchtitan/commit/1ee992442d),
[`2ec54f7114`](https://github.com/saforem2/torchtitan/commit/2ec54f7114), and
[`f4d96579bf`](https://github.com/saforem2/torchtitan/commit/f4d96579bf).

## Prompt manifest

The deterministic seed was `154391`. The manifest contains exactly 5,000
unique prompts:

| category | rows | source |
|---|---:|---|
| GSM8K math | 1,000 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) train |
| broader math | 1,000 | [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) |
| instruction following | 500 | Tulu IFData |
| general instructions | 500 | Tulu No Robots |
| task instructions | 500 | Tulu FLAN |
| multilingual instructions | 500 | Tulu Aya |
| code | 500 | Tulu decontaminated CodeAlpaca |
| science | 500 | Tulu SciRIFF |

Tulu prompts come from
[allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
using its explicit source labels rather than heuristic classification.

Before generation, normalized prompt hashes were excluded against GSM8K test,
MATH-lighteval test, MATH-500, IFEval, and all 57 cached MMLU test subjects.
The held-out set contained 20,654 hashes. Cross-source duplicates were skipped
and deterministically backfilled so every category retained its exact quota.

Manifest SHA-256:

```text
d0e8310fc982afb0eabbd7a2531fce8ec8f7705f73b88062fdce5d80ca336de1
```

## Teacher-generation contract

Requests used `inkling-bf16`, temperature 0, `reasoning_effort="low"`, JSON
response mode, a 512-token output cap, concurrency 4, and up to five retries.
Each raw record retains the prompt ID/hash, source/category, request ID, model,
finish reason, usage, latency, attempt count, UTC timestamp, and unmodified
visible response. Tokens and credentials are never persisted.

The first 100-request canary showed why request-shape validation mattered:

| contract | accepted | length stops | total tokens | elapsed |
|---|---:|---:|---:|---:|
| 512 tokens, default reasoning | 51/100 | 40 | 48,621 | 115.6 s |
| 1,024 tokens, default reasoning | 64/100 | 25 | 63,577 | 169.9 s |
| 512 tokens, low reasoning + JSON mode | **85/100** | **5** | **30,238** | **64.0 s** |

Low reasoning plus JSON mode reduced hidden-reasoning exhaustion and improved
both yield and throughput. The full generation reused the accepted 100 calls
and generated the remaining 4,900 in 2,780 seconds. All calls completed; no API
retry was needed.

## Validation and accepted corpus

Validation rejects API errors, malformed or absent JSON fields, outputs over
2,000 characters, pathological repetition, and math answers that do not exactly
match the source gold after normalization.

| result | rows |
|---|---:|
| raw responses | 5,000 |
| accepted | **4,115 (82.3%)** |
| answer mismatch | 492 |
| invalid/truncated JSON | 254 |
| missing response | 130 |
| repetition | 8 |
| too long | 1 |

Acceptance by category was 96.4% code, 99.2% Aya, 99.6% FLAN, 91.8% IFData,
99.2% No Robots, 85.3% GSM8K, 45.9% OpenMath, and 74.4% science. The lower
OpenMath yield reflects exact-answer filtering rather than endpoint failures.

Accepted-corpus SHA-256:

```text
c16dccd578c385523735a0f641400b09c6c032fba1c34c4c204a2757536ac300
```

A stratified manual review found concise, on-task instruction responses,
executable-looking short code, bounded science answers, and correctly enveloped
math rationales. The accepted JSONL loader and launchers landed in
[`ced60b9d58`](https://github.com/saforem2/torchtitan/commit/ced60b9d58).

## Packaging and training

The 4,115 accepted examples packed into exactly 342 sequences of length at most
2,048, containing 282,496 supervised tokens. Dataset fingerprint:
`02dc0083d6841196`. The exact packed count was pinned in
[`afb85fe49d`](https://github.com/saforem2/torchtitan/commit/afb85fe49d).

Because an earlier broad filesystem cleanup removed legitimate files named
`core.py` from shared environments, this experiment used a separately named,
isolated Torch 2.14 XPU environment plus isolated TRL/Accelerate and vLLM
overlays. The protected/shared environments were not repaired in place. Runtime
and launcher corrections are recorded in
[`510daa7d8c`](https://github.com/saforem2/torchtitan/commit/510daa7d8c),
[`c0d8d4499b`](https://github.com/saforem2/torchtitan/commit/c0d8d4499b), and
[`8dfa789d3d`](https://github.com/saforem2/torchtitan/commit/8dfa789d3d).

A two-node one-step smoke (`12478967`) exited 0 with loss 1.823, gradient norm
33.59, token accuracy 0.6328, and a complete checkpoint. Training job
`12478970` then completed 40/40 finite steps on two Sunspot nodes in 6:11,
with final train loss 1.617 and complete checkpoints at steps 10/20/30/40.

## Checkpoint evaluation

All four checkpoints used deterministic fp32 generation, the exact AGPT prompt
serializer, and canonical stop IDs `[1, 107]`.

| checkpoint | GSM8K-200 | strict format | truncations |
|---:|---:|---:|---:|
| Stage-7 accepted | **59/200** | 194/200 | 6 |
| Stage-9 step 10 | 48/200 | 196/200 | 4 |
| Stage-9 step 20 | 44/200 | 197/200 | 3 |
| Stage-9 step 30 | 46/200 | 198/200 | 2 |
| Stage-9 step 40 | 43/200 | 199/200 | 1 |

Step 10 is the only checkpoint advanced. Its direct sanity outputs are concise
and semantically correct on 8/8 prompts, including Earth, Chicago, `$21`, `17`,
and `fish`; all terminate normally. The remaining IFEval job `12478986` is
queued as of this report's cutoff.

## Interpretation

Inkling is a viable synthetic-data teacher: the API was stable, structured
output became efficient after request tuning, and the resulting corpus is
balanced, auditable, concise, and mostly high quality. The student result is
less compelling. Forty steps progressively improved formatting and brevity but
eroded GSM8K correctness. This resembles earlier specialization damage, though
without the severe repetition seen in rejected OpenMath continuation.

The current evidence does **not** justify replacing Stage-7. If IFEval is not a
large improvement, reject Stage-9 and retain the dataset/generator as reusable
infrastructure. A future retry should use lower effective exposure--fewer
steps, lower LR, replay with the broad anchor, or interpolation--rather than
more continuation on the same 4,115 rows.

## Job ledger

- `12478962`, `12478964`, `12478966`: pretokenization bring-up; the final job
  produced the valid 342-sequence artifact, while earlier jobs failed safely on
  runtime/schema gates.
- `12478965`: compute-node oneAPI/Torch import canary, exit 0.
- `12478967`: one-step distributed training smoke, exit 0.
- `12478968`: rejected before training by an incorrect immutable-SHA guard.
- `12478970`: Stage-9 40-step training, exit 0.
- `12478971`-`12478978`: GSM8K/sanity sweep; GSM8K artifacts valid, first sanity
  submissions rejected before inference due to a variable-name mismatch.
- `12478985`: corrected step-10 sanity evaluation, 8/8 bounded, exit 0.
- `12478986`: step-10 IFEval, queued at report cutoff.
