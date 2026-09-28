# AuroraGPT-2B post-training campaign: SFT, distillation, synthetic data, and RL

Date: 2026-09-28

## Executive summary

This report is the current narrative for the AuroraGPT-2B post-training campaign. It connects the older SFT and GRPO work with the September checkpoint-selection, distillation, interpolation, and synthetic-data experiments.

**Current accepted model: Stage-7.** It combines the broad Stage-1 step-900 checkpoint with the Stage-4 MetaMath-GSM specialist at interpolation alpha 0.65. It is the best artifact validated across GSM8K, IFEval, MATH-500, MMLU, formatting, truncation, and direct-generation behavior. Stage-4 remains the rollback.

Stage-7 is not a large math breakthrough. GSM8K is statistically tied with Stage-4, and MMLU remains near chance. Its defensible gain is better instruction following and cleaner bounded behavior without measurable retention loss.

The major campaign lesson is consistent across SFT, STaR, GRPO, OpenMath, and Inkling: **small-model quality is governed more by checkpoint geometry, data structure, and exposure than by adding more nominally relevant tokens.** Continued narrow training repeatedly improved formatting while reducing correctness or broad behavior.

## Current model and rollback

| metric | Stage-4 rollback | Stage-7 accepted |
|---|---:|---:|
| GSM8K full | 385/1,319 (29.19%) | 386/1,319 (29.26%) |
| paired GSM8K wins | 45 | 46 |
| exact McNemar p-value | -- | 1.0 |
| GSM8K format-valid | 1,295 | 1,297 |
| GSM8K truncations | 23 | 20 |
| IFEval instruction-loose | 33.93% | **35.85%** |
| IFEval instruction-strict | 32.25% | **33.45%** |
| IFEval prompt-loose | 19.59% | **21.26%** |
| IFEval prompt-strict | 19.04% | **19.96%** |
| MATH-500 | 3.2% | 3.8% |
| MMLU letters | 25.30% | 25.10% |
| MMLU continuation | 35.50% | 35.51% |
| broad sanity | 8/8 bounded | 8/8 bounded, cleaner semantics |

The GSM8K difference is noise, not a claimed gain. Stage-7 was promoted because every IFEval axis improved, formatting and truncation improved, raw responses were cleaner, and MMLU/GSM8K were non-inferior within uncertainty.

Detailed lineage and checkpoint evidence: [Stage-4 MetaMath distillation and Stage-7 re-anchoring](2026-09-26-agpt2b-stage4-metamath.md).

## Campaign map

| stage | intervention | result | decision |
|---|---|---|---|
| Broad SFT | Tulu/math/UltraChat mixture, checkpoints 300/600/900 | Step 900 had cleaner reasoning and remained 8/8 bounded | retained as broad anchor |
| CoT SFT | GSM8K rationales in `<think>/<answer>` format | strong formatting; limited accuracy | retained as intermediate capability |
| GRPO | TRL and Monarch/TorchStore/vLLM paths | stack works; format/reward can improve, but held-out GSM8K did not materially improve | no RL artifact promoted |
| Stage-3 STaR | 4,294 self-generated, exact-correct GSM8K traces | 37/200 vs baseline 43/200 | rejected |
| Stage-4 MetaMath GSM | 160k concise, decontaminated augmented/rephrased GSM rows | strong GSM8K gain; broad behavior required interpolation | accepted specialist and rollback lineage |
| Stage-5 MetaMath MATH | 118,375 retained rows, 14,146 packed sequences | MATH-500 flat; full GSM8K fell to 21.53% | rejected |
| Stage-6 OpenMath | 150k augmented-MATH + 50k augmented-GSM, exact-answer verified | weak MATH gains, repetition, full GSM8K 25.78% | rejected |
| Stage-7 re-anchor | broad step 900 + Stage-4 math step 75, alpha 0.65 | better IFEval and behavior with flat GSM8K/MMLU | **accepted** |
| Stage-8 alpha sweep | step-900 anchor, alpha 0.45-0.75 | alpha 0.55/0.60 improved development GSM8K; alpha 0.55 retained MMLU | promising but not promoted over Stage-7 |
| Stage-9 Inkling | 4,115 verified synthetic examples, 40-step SFT | concise 8/8 sanity, but best GSM8K-200 only 48/200 | provisional rejection; IFEval queued |

## SFT and distillation findings

### Structured cold starts work

The earliest robust result was two-stage training: broad instruction SFT followed by a focused GSM8K CoT stage. This taught the explicit reasoning envelope and produced a stronger basis for later experiments than single-stage mixtures. The historical status and detailed ablations remain in [AuroraGPT-2B post-training status](../production/POST-TRAINING-2B.md).

### Self-distillation did not automatically improve reasoning

The teacher-free STaR experiment sampled 59,784 candidate traces, retained 4,294 exact-answer-correct and decontaminated examples, and completed finite SFT. Its final GSM8K result fell from 43/200 to 37/200. Correctness-filtered self-generated traces were therefore not sufficient to improve transfer. See [Stage-3 STaR](2026-09-25-mds154391-stage3-star.md).

### MetaMath improved math only when bounded and re-anchored

Concise MetaMath GSM augmentation produced the strongest specialist. Direct continuation improved GSM8K, but broad capability remained fragile. Weight interpolation recovered a usable Pareto point. Later re-anchoring against broad step 900 improved instruction following without sacrificing measured retention.

The re-anchor result is important: changing the broad anchor was more effective and cheaper than another SFT allocation. It also exposed that alpha 0.65 had originally been optimized for a different anchor; the later alpha sweep found development-set headroom at 0.55-0.60, though no replacement has passed the full promotion matrix.

### More math data repeatedly caused retention loss

MetaMath MATH and verified OpenMath were scientifically useful negative controls. Both pipelines were decontaminated and trained successfully, but their model checkpoints regressed broader capability:

- MetaMath MATH: no MATH-500 improvement and full GSM8K 21.53%.
- OpenMath: development MATH gains but severe repetition; full GSM8K 25.78%.

These failures were not hidden with interpolation. They established that continued narrow SFT on a damaged specialist is not a productive default.

## RL / GRPO findings

The campaign validated two real XPU RL stacks: historical TRL + vLLM-server and the current Monarch + TorchStore + vLLM path. Multi-host policy publication, generation, optimizer updates, checkpointing, and shutdown were validated on Sunspot with Gloo transport. See [Sunspot multi-host RL validation](2026-09-25-sunspot-multihost-rl-validation.md) and the [production RL hub](../production/rl/README.md).

RL produced valuable systems and optimization evidence:

- a 1,000-step arithmetic task improved reward from roughly 0.4 to 0.9;
- componentized reward shaping improved alphabet-sort reward by 168%;
- format adherence could be perfected without policy drift;
- convergence speed provided a useful probe of SFT checkpoint quality.

RL did **not** produce a promoted general-quality checkpoint. On GSM8K, gated GRPO moved accuracy from 20.5% to 21.5%, within noise. A weaker reward design increased reward while held-out accuracy fell, demonstrating reward hacking. The central lesson is that RL cannot rescue an underpowered cold start when most sample groups are entirely wrong and therefore provide no useful relative advantage.

## Inkling synthetic-data experiment

The Inkling experiment tested an external teacher rather than replaying existing distillation corpora. A deterministic 5,000-prompt manifest balanced math, instruction following, code, science, and multilingual prompts. It excluded normalized overlap with 20,654 held-out GSM8K, MATH, IFEval, and MMLU prompts.

ALCF Minerva served `inkling-bf16` through an OpenAI-compatible endpoint. Request tuning mattered: low reasoning effort plus JSON mode raised canary acceptance from 51% to 85% while reducing token usage. All 5,000 calls completed without retry. Validation retained 4,115 examples (82.3%); math rows required exact agreement with source gold answers.

The accepted corpus packed into 342 sequences with 282,496 supervised tokens. Stage-9 completed 40 finite steps and produced four checkpoints. Formatting and brevity improved monotonically, but GSM8K correctness declined:

| checkpoint | GSM8K-200 | format-valid | truncations |
|---:|---:|---:|---:|
| Stage-7 | **59** | 194 | 6 |
| Stage-9 step 10 | 48 | 196 | 4 |
| step 20 | 44 | 197 | 3 |
| step 30 | 46 | 198 | 2 |
| step 40 | 43 | 199 | 1 |

Step 10 remained 8/8 bounded and semantically correct on the direct sanity set. IFEval `12478986` is the final secondary gate. Full details, hashes, API settings, acceptance breakdowns, and job IDs are in [AGPT-2B Inkling synthetic-data distillation](2026-09-28-agpt2b-inkling-distillation.md).

## What the campaign established

1. **Formatting is easier than correctness.** Multiple interventions improved format-valid rates and truncation while leaving correctness flat or worse.
2. **Development slices are directional screens only.** OpenMath step 25 scored 59/200 but regressed on full GSM8K.
3. **Checkpoint selection and interpolation matter.** Re-anchoring existing checkpoints produced the best accepted model more cheaply than retraining.
4. **Exact-answer filtering is necessary but insufficient.** Verified teacher traces can still narrow the student or alter useful behavior.
5. **RL needs reachable tasks and dense reward variation.** A working RL system does not imply quality gains.
6. **Short, source-balanced teacher data is promising infrastructure.** Inkling generation was stable and auditable, but the student needs lower effective exposure or replay to avoid math regression.
7. **The current base is a hard constraint.** MMLU remains approximately chance; major general gains likely require a stronger base or substantially better curriculum, not another narrow continuation.

## Recommended next work

- Keep Stage-7 as accepted and Stage-4 as rollback.
- Close Stage-9 after IFEval; do not promote it on formatting alone.
- If reusing Inkling data, test lower exposure first: fewer than 10 steps, lower LR, broad-data replay, or interpolation with Stage-7.
- Require full held-out promotion gates after development screening.
- Treat a stronger base checkpoint as the highest-leverage route to substantial MMLU/general gains.

## Canonical detailed reports

- [Historical 2B SFT and GRPO status](../production/POST-TRAINING-2B.md)
- [Stage-3 teacher-free STaR](2026-09-25-mds154391-stage3-star.md)
- [Sunspot multi-host Monarch/TorchStore/vLLM validation](2026-09-25-sunspot-multihost-rl-validation.md)
- [Stage-4 MetaMath and Stage-7 lineage](2026-09-26-agpt2b-stage4-metamath.md)
- [Stage-9 Inkling distillation](2026-09-28-agpt2b-inkling-distillation.md)
- [Production RL hub](../production/rl/README.md)
- [CoT and GRPO experiment history](../production/rl/plans/cot.md)
