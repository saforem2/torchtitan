# Broad-Distill SFT Path: Deep Analysis

**Date:** 2026-11-16
**Author:** Thread B (Broad Distill)
**Status:** Analysis-only -- no training. All numbers cited from production docs.
**Branch:** `ezpz-cpt111k-eval-sft` @ `bcbca5bd90`
**Prerequisites read:** `POST-TRAINING-2B.md` (B2/B3/B4/B2v2), `sft/README.md`, `sft_distill_cot_4n.sh`, `_pretokenize_distill_cot_mix_1n.sh`, `datasets_sft.py`, `fineweb-edu-60-40-kind-shift.md`, `cpt/README.md`, journal entries through 2026-11-15.

---

## Summary

The broad-distill SFT path is the only open post-training lever that adds **new reasoning data** under a structure consistent with the only recipe that worked (B2's two-stage SFT). Every other SFT lever -- longer training on the same data, heavier in-distribution weighting, verbosity filtering, and single-stage broad mixes -- has been refuted with clean eval numbers. This report explains why they failed, what is different about the pre-tokenized distill mix now on disk, how to train on it, what cheap ablations gate the expensive runs, and an ordered queue of concrete experiments.

**Bottom line up front:** Do not treat the broad-distill mix as a license to re-attempt a single-stage SFT. The evidence says the two-stage *sequencing* is the accuracy lever, not the mix composition. The broad mix should be evaluated as a **stage-2 replacement** on the proven `checkpoint-900` base (and, once available, on the CPT-FWE base) -- not as a new cold-start from `gs138650`.

---

## Table of Contents

1. [Why every single-stage approach failed](#1-why-every-single-stage-approach-failed)
2. [What the broad-distill mix is and why it differs from failed narrow mixes](#2-what-the-broad-distill-mix-is-and-why-it-differs)
3. [Stage-2 SFT design: base, LR, epochs, GBS, one-stage vs two-stage](#3-stage-2-sft-design)
4. [Cheap ablations before committing 4N x 12h](#4-cheap-ablations)
5. [Ranked experiment queue](#5-ranked-experiment-queue)
6. [Appendix: evidence tables and cross-references](#appendix)

---

## 1. Why every single-stage approach failed

### 1.1 The evidence in one table

All eval numbers are from the same harness: 200-problem GSM8K-CoT, `temperature 0.0 max_tokens 768`, `frameworks/2026.1.0 vLLM 0.26 XPU fp32`, Sunspot `workq`. Cited from `POST-TRAINING-2B.md` B3/B4/B2v2 sections and `b4-finish-and-reweight/README.md`.

| Model | Base | Data lever | GSM8K-CoT acc | Format | Mean gen_len | n_unclosed /200 | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| **B2 = checkpoint-93** | `checkpoint-900` (tulu_math_uc_mix) | `gsm8k-r1cot` 7.4k ex, 3ep, 93 steps | **0.205** | 0.985 | 282 | ~0 | deliverable |
| B3 | `gs138650` | tulu 0.30 + OpenR1 0.25 + gsm8k-r1cot 0.15 + ultrachat 0.15 + OMI2 0.15, seq 8192, 1ep | **0.05** | 0.86 | 601 | 26 | long-CoT dilution |
| B4a | B3-3605 (`b3-3605-hf-diag`) | gsm8k-r1cot finish, 3ep, per-epoch ckpt | **0.02** | 0.26 | 1378 | 133 | verbose bias baked in |
| B4b | `gs138650` | gsm8k-r1cot 0.40 + OpenR1-filtered 0.15 + tulu 0.30 + ultrachat 0.15, seq 4096 | **0.065** | 0.86 | 611 | 27 | symptom fixed, accuracy stuck |
| B2v2 ep3 (6 steps) | `checkpoint-900-hf` | gsm8k-r1cot, 3->6ep (6 steps) | **0.09** | 0.86 | 416 | 22 | more narrow epochs hurt |
| B2v2 ep6 (12 steps) | `checkpoint-900-hf` | gsm8k-r1cot, 3->6ep (12 steps) | **0.105** | 0.91 | 378 | 15 | still -0.10 vs B2 |
| CPT-17000 SFT | CPT 17000 HF | narrow gsm8k-r1cot (3ep, LR 2e-5, GBS 384) | **0.015** | collapsed | -- | -- | format collapse on shifted manifold |

Additional anchor: `tulu_math_uc_mix_full` at full epoch (`checkpoint-8672`, 8672 steps, ~54B tokens, LR 2e-5 cosine) catastrophically forgot -- hellaswag 0.593 -> 0.273, arc_easy 0.694 -> 0.298 -- while `checkpoint-900` within the same run held IFEval 0.253 and base-LM intact. See `tulu_math_uc_mix_full/README.md`.

### 1.2 Format-collapse vs. reasoning: two distinct failure modes

The guardrail metrics (`mean_gen_len`, `n_unclosed`) introduced after the B3 investigation are essential because they **decouple two axes** that accuracy alone conflates. The B-series proves they are independently movable:

**Format-collapse (B4a phenotype):** The model loses the `<think>/<answer>\boxed{}` envelope. Generations ramble, never emit `</answer>`, and hit `max_tokens` with `finish_reason == "length"`. B4a is the extreme: format 0.985 (B2) -> 0.26, gen_len 282 -> 1378, unclosed 0 -> 133. The 3-probe B3 diagnosis confirmed these are coherent English with well-formed but *wrong* arithmetic -- the model did not forget language, it forgot how to *stop*.

**Reasoning failure (B4b phenotype):** The model holds the envelope (format 0.86, gen_len 611, unclosed 27 -- back to B3-level healthy) and still gets the answer wrong. Accuracy only moved 0.05 -> 0.065 despite the format rescue. This is the sharpest control in the series: fixing what the guardrails flagged did **not** fix what we cared about.

**What this tells us:**

1. **Style is a symptom, not the cause.** B3's verbose R1 style (gen_len 601 vs 282) and the unclosed-`</answer>` rate (26/200) were real, but they were downstream of a deeper capacity-allocation problem: the 2B's limited parameters, spread across five diverse corpora in one SFT, allocated insufficient capacity to short-arithmetic execution. Verbosity filtering (B4b's `OPENR1_MAX_THINK_CHARS=1200` clamp + reweighting gsm8k-r1cot to 0.40) removed the verbosity *expression* but not the underlying dilution -- because the dilution happened *during training*, not just at generation time.

2. **A verbose base cannot be rehabilitated by a short finishing stage.** B4a bolted B2's winning stage-2 recipe (gsm8k-r1cot, 3ep) onto the B3 checkpoint and made it worse (0.05 -> 0.02, unclosed 26 -> 133). The R1 verbosity bias was baked into the weights; a few dozen gradient steps on 7.4k narrow examples could not un-teach it -- they destabilized the envelope further. This rules out any "B3 is almost right, just needs a polish" narrative.

3. **More steps on narrow data is the same trap with a different knob.** B2v2 varied only duration (3ep/93 steps -> 6ep/6 steps and 6ep/12 steps at seq 2048, same mix, same LR) on the *correct* base (`checkpoint-900`) and still halved accuracy (0.205 -> 0.09/0.105) despite healthy training metrics (loss 0.8069, acc 0.8198, descending cleanly). The W&B runs `xakno7kg`/`ur6pqqhp` show no warning signal in training curves -- eval is the only detector. This is not an overfitting curve with a U-shape you can early-stop out of; it is a distribution-collapse where narrow data drives the model off the general-math manifold that `checkpoint-900` occupies, and more steps push it further off.

4. **The CPT-17000 probe generalizes the point.** SFT on a CPT-shifted base (FineWeb-Edu 60/40 at 17k steps) with the same narrow recipe scored 0.015 with format collapse -- worse than B3's 0.05 on the MDS base. A shifted pre-training manifold plus narrow SFT is strictly harder than SFT on the original `gs138650` manifold, at least at this CPT maturity. This does not refute CPT as a lever; it says the SFT recipe must be re-tuned to the new base, not blindly copied.

### 1.3 The structural hypothesis

The only surviving explanation that fits B2 vs B3/B4/B2v2 simultaneously:

> **Two-stage SFT is a sequential curriculum.** Stage 1 (`gs138650 -> tulu_math_uc_mix -> checkpoint-900`, 729 steps at GBS 6144, seq 1024) builds a general instruction-following + math-breadth manifold (IFEval 0.1645 -> 0.2440, base-LM intact). Stage 2 (`checkpoint-900 -> gsm8k-r1cot -> checkpoint-93`, 93 steps at GBS ~384, seq 2048) is a *format-and-execution* specialization that stays within that manifold. Separating them lets the model consolidate breadth before it learns the CoT envelope.
>
> **Single-stage SFT forces the model to learn breadth and envelope simultaneously.** At 2B, the gradient signal for envelope compliance (a strong, easy-to-learn token pattern) competes with the gradient signal for arithmetic correctness (a weak, hard-to-learn computation). The envelope wins -- the model learns to *look like* it is reasoning (long traces, correct tags) while allocating insufficient capacity to the underlying arithmetic circuit. Longer training or heavier in-distribution weighting does not fix this because the competition is structural.

Evidence for the structural view over any mix-tuning view:

- B4b moved gsm8k-r1cot to 0.40 (2.7x B3's 0.15) and filtered long traces -- still 0.065.
- B4b also halved seq_len (8192 -> 4096) -- still 0.065.
- B4a used a pure gsm8k-r1cot distribution -- still 0.02.
- B2v2 used no competing data at all -- still 0.09/0.105.
- No tested knob (weight, length filter, sequence length, epoch count, starting base) closed more than 0.015 of the 0.155 gap to B2.

**Implication for broad-distill:** Any broad-distill run that is framed as "one SFT from gs138650 on a broad reasoning mix" is a priori a B3-class experiment, regardless of which broad data it uses. The null hypothesis is that it will score ~0.05-0.07 and be indistinguishable from B3/B4b. The experiment must be designed to *beat* that null by operating within the two-stage structure.

---

## 2. What the broad-distill mix is and why it differs

### 2.1 What landed on disk

The pre-tokenized dataset at `/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192` was built by `_pretokenize_distill_cot_mix_1n.sh` (job `12483443`, NPROC=1, VENV `rl-monarch-torch214-locality-20261001`, bypassing `ezpz_setup_job` MPI wrapping). From `POST-TRAINING-2B.md` (2026-11-12 entry) and `journal.md`:

| Property | Value |
|---|---|
| Source | `_pretokenize_distill_cot_mix_1n.sh` (1N, `select=1`, `12h walltime`) |
| Job | `12483443` `Exit` -- cache hit `cba5c64` in 5.8s on final attempt after 6 segfaults (`NPROC 32->16->8->4->1`, fix `VENV` path) |
| Output | `/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192` |
| Size | 3.4 GB (`dataset_info.json` + 8 x `data-*.arrow`) |
| Raw rows | 91,481 (after envelope formatting + per-component decontamination, before packing) |
| Packed sequences | 36,934 @ `max_length=8192` (packing via TRL `SFTTrainer`, `assistant_only_loss`) |
| Mix spec | `distill_cot_mix` = 50% OpenThoughts-114k-decontam + 50% OpenR1-Math-220k-decontam (per `datasets_sft.py:_build_distill_cot_mix`) |
| Envelope | `<think>{trace}</think>\n<answer>\boxed{answer}</answer>` -- byte-identical to `gsm8k-r1cot` and `eval_cot_gsm8k.py` |
| Decontamination | GSM8K-test-decontaminated per-component (13-gram + normalized-exact, `decontam_traces.py`), baked into component builds, captured in materialized-mix recipe hash; `[decontam]` drop counts in `logs/pretokenize-distill-cot-mix-*/run.log` |
| Tokenizer base | `gs138650` (`global_step138650`, vocab 256000) |

**Note on the mix definition discrepancy:** The task prompt describes the mix as "50% OpenThoughts-114k-decontam + 50% gsm8k-r1cot" while `datasets_sft.py:_build_distill_cot_mix` registers it as `OpenThoughts-114k-decontam + OpenR1-Math-220k-decontam` (both frontier-distilled corpora, no gsm8k-r1cot). The `_pretokenize_distill_cot_mix_1n.sh` header describes it as `0.50 OpenThoughts-114k-decontam + 0.50 OpenR1-Math-220k-decontam` as well. The 91k-row artifact on disk matches the datasets_sft.py definition (both components are ~100k+ raw rows, so a 50/50 interleave at ~91k after filtering/decontam is consistent; a 50/50 with 7.4k gsm8k rows would be much smaller or heavily upsampled). The `sft_distill_cot_4n.sh` comment block still references `gsm8k-r1cot` in the "Alt spec" line, which is stale. For this report, the landed artifact is treated as the `datasets_sft.py` definition; the gsm8k-r1cot combination is available as an ad-hoc mix-spec via `get_sft_dataset("OpenThoughts-114k-decontam:0.5,gsm8k-r1cot:0.5")` if needed, but it is not what was pre-tokenized.

**If a gsm8k-r1cot blend is desired**, it must be staged as a separate pre-tokenize with that mix-spec; do not assume the landed 36.9k-packed artifact contains GSM8K problems. The experiment queue in section 5 covers both.

### 2.2 What makes this different from narrow mixes that failed

| Dimension | Narrow mixes (gsm8k-r1cot alone, B2v2) | B3/B4 broad mixes | **Broad-distill (this mix)** |
|---|---|---|---|
| **Diversity** | 7.4k GSM8K problems, one template, one difficulty band | 5 corpora but all math/instruct via TRL interleave | 2 frontier-distilled corpora spanning **math + science + code** reasoning (OpenThoughts) plus math-focused R1 traces (OpenR1) |
| **Reasoning depth** | Short CoT (mean trace ~hundreds of chars, B2 gen_len 282) | Mixed: short GSM8K + long R1 (median OpenR1 ~13k chars, filtered at 1200 for B4b, 16000 default now) | Frontier R1 traces kept long (default `DISTILL_MAX_THINK_CHARS=16000`, `OPENR1_MAX_THINK_CHARS=16000`) -- the point is to expose the 2B to *richer* reasoning, not just more of the same |
| **Envelope** | Correct by construction (GSM8K solutions wrapped) | B3: OpenR1 rows reformatted via `_openr1_format_row` (correctness-gated, `correctness_math_verify`); B4b: length-filtered | Same reformatters (`_openthoughts_format_row`, `_openr1_format_row`), code-generation rows without `\boxed{}` intentionally dropped (they would teach a malformed tail) |
| **Decontamination** | Trivially clean (training = eval domain, but no overlap beyond the 200 held-out vs 7.4k train split) | Not decontaminated (B3/B4 pre-date `decontam_traces.py`) -- but the eval delta was so large (0.05 vs 0.205) it cannot be explained by contamination | **GSM8K-test-decontaminated** (13-gram + normalized-exact), per-component, logged, hash-captured; a cache hit is guaranteed decontaminated |
| **Packing density** | 7.4k rows -> 93 steps at GBS 384, seq 2048 -- no packing pressure | 53M rows (full-mix) or ~93k (B3 guess) -> heavy packing, many logical problems per physical sequence | 91k rows -> 36.9k packed @ 8192 -- moderate packing (~2.5 rows/seq avg), so cross-problem contamination within a packed sequence is limited vs the 53M-row full-mix |
| **Token volume** | ~11k seqs x 3ep = ~33k packed-seq exposures at seq 1024-2048 | 1 epoch over variable-size mix at seq 8192/4096 | 36.9k packed @ 8192 (~302M tokens) at 1ep, ~906M at 3ep -- between B2's stage-1 (803M) and stage-2 (tiny), enough to matter without reaching the forgetting regime (see 2.4) |

The genuine novelty is the **breadth of reasoning types**. Prior mixes (including B3's 5-way) were breadth in *task format* (instruction, chat, math) but narrow in *reasoning provenance* -- all traces were either GSM8K solutions or OpenR1 math. OpenThoughts adds **science and code reasoning** distilled from DeepSeek-R1, tagged via `<|begin_of_thought|>` / `<|begin_of_solution|>` markers and re-wrapped. The hypothesis is that a 2B exposed to diverse reasoning *styles* (not just diverse topics) learns a more transferable CoT circuit, analogous to how `tulu_math_uc_mix`'s chat breadth helped stage-1 build a better manifold than pure math would have.

This is a **testable claim**, not an axiom. It could be wrong -- the B-series says mixing diverse CoT styles in one SFT dilutes short-arithmetic competence. The distill mix differs from B3/B4b in that its diversity is *within* the `<think>` trace (different reasoning structures) rather than *across* corpora with different envelopes, but whether that distinction matters for the dilution mechanism is unknown until measured. Hence the ablations in section 4.

### 2.3 Risks

#### 2.3.1 Distribution shift (the primary risk)

**What it is:** The 2B has been trained exclusively on `olmo-mix-1124` (94.8% DCLM) through pre-training, plus `tulu_math_uc_mix` for stage-1. Its tokenizer, positional embeddings, and weight statistics are all adapted to that distribution. OpenThoughts traces are **out-of-distribution** in vocabulary (science/code terms), trace structure (longer, more nested reasoning), and answer style (some problems are not GSM8K-like math at all). Training on them shifts the model's distribution, and the history of distribution shifts on this model is poor:

- CPT dolmino pilots (even 50/50 with olmo) degraded HellaSwag -7.4pp monotonically.
- `checkpoint-8672` (full OpenMathInstruct-2) forgot catastrophically at ~54B tokens / LR > 1e-5.
- CPT-17000 SFT probe scored 0.015 -- a shifted base plus narrow SFT is worse than an unshifted base plus narrow SFT.

**Mitigations already in place:**

- Decontamination prevents eval inflation (the most dangerous shift artifact).
- Pre-tokenization at `max_length 8192` keeps long traces intact rather than truncating mid-reasoning (truncation creates a different envelope at training vs eval).
- The `DISTILL_MAX_THINK_CHARS=16000` cap only drops pathological run-ons, not normal long traces -- unlike B4b's 1200-char clamp that filtered 100% of OpenR1 at median 13k chars (per the comment in `datasets_sft.py:474-476`, the old 1200 filtered all rows; the fix to 16000 was on the `sft_data.py 4b5bff5b6` lineage).

**Remaining risk:** The CPT-17000 result suggests that even a modest pre-training shift makes stage-2 SFT harder. If the distill mix shifts the SFT base similarly, the CoT eval could degrade even if the model genuinely learns broader reasoning -- because GSM8K-CoT measures *narrow* arithmetic execution, not broad reasoning. A broad reasoning gain could be invisible on GSM8K and require a different eval (MATH, science QA) to surface. The experiment queue should include a broad eval gate, not just GSM8K-CoT.

#### 2.3.2 Length (packing and attention)

**What it is:** 8192 is the longest sequence the TRL/XPU stack has been asked to train at on this model. B3 used 8192 and hit no OOM (micro-batch 1 @ 8192 = 8192 tok/device), but B4b fell back to 4096 for cost, and the `tulu_math_uc_mix_full` run explicitly avoided 8192 ("seq_len 2048 OOM'd the XPU tile -> reverted to 1024"). Attention is O(L^2) without flash-attention, so 8192 is 64x the per-token attention cost of 1024 at the same batch.

**Current sizing:** `sft_distill_cot_4n.sh` sets `per_device_train_batch_size 2, GAS 8, seq 8192` -> GBS 768 at 4N (48 GPUs), 1536 at 8N. Tokens/step at 4N: 48 * 2 * 8 * 8192 = ~6.3M; at 8N: ~12.6M. At 36.9k packed seqs, 1 epoch is ~48 steps at GBS 768 (4N) or ~24 steps at GBS 1536 (8N). At 3 epochs, that is 144 / 72 steps -- comparable to B2's stage-1 (729 steps @ 1024) in wall-time but much shorter in step count.

**Risks:**

- XPU memory: 8192 at batch 2 may OOM on the PVC tile if the model's KV cache + attention workspace exceeds tile HBM. B3 survived at batch 1 @ 8192; batch 2 is unproven for this mix. The fallback is batch 1 + GAS 16 (same GBS, half the per-device activation memory).
- Throughput: no throughput number has been recorded for TRL at 8192 on XPU. If tps drops far below the CPT rate (4625 tps at GBS 64, seq 8192 on `xpu-torch216-sync`), a 12h window may not cover 3 epochs. Estimate: 144 steps at even 30s/step = 72 minutes -- well within 12h, so walltime is not the risk; OOM is.
- Packing: at 8192, packing combines multiple short problems into one physical sequence with cross-problem attention (no block-causal masking in TRL). The model sees tokens from problem B while processing problem A. For the 7.4k-row GSM8K mix this is harmless (B2 proved it), but for the distill mix where problems are longer and more diverse, cross-problem attention could blend unrelated reasoning styles. No evidence this matters, but it is unmeasured.

#### 2.3.3 Quality (trace correctness and filtering)

**What it is:** OpenThoughts and OpenR1 traces are *distilled* from frontier models (DeepSeek-R1), not human-written. Their correctness is gated differently:

- OpenR1: `correctness_math_verify` + `is_reasoning_complete` flags per generation, with `_openr1_format_row` selecting a verified-correct generation and extracting `LAST \boxed{}`.
- OpenThoughts: no per-generation correctness flag; `_openthoughts_format_row` extracts the R1 trace + last `\boxed{}` but cannot verify correctness beyond the presence of a boxed answer. Code-generation rows (ending in code blocks, not `\boxed{}`) are dropped -- this is intentional but reduces the science/code breadth claim (pure-code problems are excluded).
- gsm8k-r1cot: correctness by construction (GSM8K ground truth).

**Risks:**

- OpenThoughts correctness is weaker: a trace can have a `\boxed{}` answer that is wrong (the R1 model hallucinated the final number but the regex still finds a box). Training on wrong answers teaches wrong arithmetic. The mitigation is that the traces are from a frontier model with high accuracy, but no correctness audit has been published for this mix.
- Length filtering at 16000 chars is permissive -- it keeps essentially all traces (median OpenR1 13k, so 16k keeps >80%). The B4a/B4b lesson was that *generated* verbosity matters, not *training* verbosity, but if the training traces themselves teach a verbose style, the model may still learn it. The `POST-TRAINING-2B.md` "More SFT tokens on narrow data actively hurts" finding suggests that the style-vs-accuracy tradeoff is not just about length.
- The dropped code-generation rows mean the "code reasoning" breadth is partially illusory -- only code-adjacent problems with a scalar answer survive (e.g., "what does this function return?"). Pure code synthesis traces are excluded because they do not fit `\boxed{}`. The mix's actual composition is therefore more math-heavy than the 50/50 OpenThoughts/OpenR1 label suggests.

#### 2.3.4 Decontamination completeness

**What it is:** The 13-gram + normalized-exact detector (`decontam_traces.py`) is conservative (13 tokens is long enough that false positives are negligible -- see `GSM8KDecontaminator` docstring), but it is not perfect. Reworded GSM8K test questions with <13 shared tokens and altered numbers would pass. The `fineweb-edu-60-40-kind-shift.md` notes that no corpus cures contamination at arbitrary scale and that re-scoring on both MMLU formats is recommended.

**Risk:** A non-trivial number of near-duplicate GSM8K test questions could survive decontamination and inflate the GSM8K-CoT eval, creating a spurious "gain" that is actually leakage. The `[decontam]` drop counts in the run log are the gate -- if they are near zero, the corpora may have little GSM8K overlap and contamination is not a concern; if they are large, the detector is working; if they are suspiciously small relative to the corpora sizes (114k + 220k), manual audit of surviving near-duplicates is warranted.

**Mitigation:** The materialized-mix recipe hash captures the decontamination step, so a cache hit is guaranteed decontaminated per that detector. Re-running decontamination with `n=8` (more sensitive) as an ablation would quantify the residual risk.

---

## 3. Stage-2 SFT design

### 3.1 Which base

Three candidates are discussed in `POST-TRAINING-2B.md` and `sft/README.md`:

| Base | What it is | Status | GSM8K-CoT relevance |
|---|---|---|---|
| **A. `checkpoint-900-hf`** via `global_step138650 -> tulu_math_uc_mix_full @ 900` | General-instruct deliverable, IFEval 0.253, base-LM intact, sharded `checkpoint-900/` on disk (96 `.distcp` shards) | **Ready** (re-consolidate to `-hf` if missing -- `consolidate_sft_ckpt.sh` via `accelerate merge-weights`) | Proven stage-1 for B2; the only base that has ever produced >0.10 CoT. Every comparison in the B-series uses this implicitly. |
| **B. CPT-FWE frontier** (`agpt-2b-cpt-fineweb-edu-n4-gbs64`, currently step 111.2k/190.7k, 58.3% budget) | FineWeb-Edu 60/40 kind-shift CPT from `gs138650/step-0`, Grain streaming, 100B budget | **Not yet deliverable** -- at 111k, no HF conversion since 74k (`74000/74200` 3.7G verified, Sunspot `~/torchtitan/outputs/hf*` empty). Raw CPT CoT 0/200 expected (no `chat_template`). Next convert at 112k gate. | The kind-shift hypothesis predicts this base will lift the ~0.2 ceiling -- but the CPT-17000 SFT probe (0.015) warns that a shifted base may need a different SFT recipe. Do not assume the B2 recipe transfers. |
| **C. `global_step138650` (MDS stage-3 pretrain)** | Raw pre-training checkpoint, no SFT, `bos 1 / eos 2`, `max_position_embeddings 131072`, `rope_theta 50000` | **Available** (`/home/foremans/global_step138650`, HF format) | The B3/B4 base. Every single-stage attempt from this base failed. Using it for broad-distill in one stage repeats the B3 shape. Only justified if the broad-distill hypothesis is "B3's mix was wrong, not its structure" -- which B4b refuted. |

**Recommendation: A is the primary base for the first experiments; B is the follow-on once it converts; C is for a controlled ablation only.**

Rationale:

- **A must be first** because it is the only base with a known-good GSM8K-CoT trajectory (B2). Any broad-distill result on A is directly comparable to B2's 0.205 and B2v2's 0.09/0.105. If broad-distill on A cannot beat 0.205, it has failed in the easiest setting; there is no reason to expect it to succeed on a harder base.
- **B should be gated on two conditions:** (1) HF conversion at 112k+ is verified (config + tokenizer + lm_head present, `model.safetensors` non-empty), and (2) at least one narrow SFT probe on B (the 0.015 datapoint at 17k, plus a fresh one at 112k+) has been measured to calibrate the base's format behavior. The CPT chain's `POST-TRAINING-2B.md` note says "park to CPT 50k then general SFT" -- general SFT on CPT-FWE is not yet done, so the CoT SFT on CPT-FWE would be operating on a base that has never seen *any* instruction tuning. That is a larger distribution gap than `checkpoint-900` ever was.
- **C is the null-hypothesis control:** One run from `gs138650` on `distill_cot_mix` at 8192, 1ep, LR 2e-5, to test whether the broad-distill mix *as a cold-start replacement* beats B3's 0.05. Expected result: ~0.05-0.07 (B3 class). If it scores higher, the mix itself is doing work; if not, structure is confirmed as the lever. This is ablation E5, not the main line.

**Base availability check (do before any run):**

```bash
# A: checkpoint-900-hf (preferred) or sharded fallback
ls -lh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900{,-hf}/ 2>&1 | head
# B: CPT-FWE frontier (check latest converted HF, not raw DCP)
ls -lh /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/hf* 2>&1; ls ~/torchtitan/outputs/hf* 2>&1
# C: raw pretrain
ls -lh ~/global_step138650/config.json ~/global_step138650/tokenizer.json 2>&1
# Distill mix (all three share this gate)
ls -lh /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json 2>&1
du -sh /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192 2>&1
grep -F "[decontam]" /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/logs/pretokenize-distill-cot-mix-*/run.log 2>&1 | tail -5
```

### 3.2 LR / epochs / GBS

All hparams below are anchored to the B2 recipe and the landed artifact's packing.

#### Learning rate

| Setting | Value | Rationale |
|---|---|---|
| **LR** | **2e-5 cosine** | Matches B2 stage-2 (LR 2e-5 cosine, 93 steps on `checkpoint-900`) and `sft_distill_cot_4n.sh` default. The `checkpoint-900` base was trained at LR 2e-5 -> ~0 over 8672 steps; re-warming to 2e-5 for stage-2 is the proven value. The `tulu_math_uc_mix_full` catastrophe (LR 2e-5 stayed >1e-5 through step 4350, forgetting) is a warning about *duration* at high LR, not about 2e-5 per se for short runs. |
| Warmup | TRL default (0 or small) | B2 stage-2 used no explicit warmup; the cosine schedule from peak is sufficient for <200 steps. Do not add warmup unless loss spikes at step 1. |
| Weight decay | TRL default (AdamW, 0.0 or 0.01) | Matches B2. Not a lever for this data volume. |
| **LR to vary in ablations** | **1e-5 cosine** | The `POST-TRAINING-2B.md` "cap full-mix SFT at O(1000) steps or drop LR substantially" rule, plus the RL note that "raise LR to 5e-6-1e-5 and group_size to 8-16" on B2 is now safe since format holds 1.000. For SFT, 1e-5 is the conservative alternative if 2e-5 shows format degradation at 3ep. Not the default -- test in E3. |

**Do not use:** LR 5e-5+ (untested at this scale, risk of immediate forgetting), constant LR without decay (the forgetting at `checkpoint-8672` was caused by LR staying high too long; cosine decay is load-bearing), or LR < 5e-6 (too slow to learn the envelope in <200 steps).

#### Epochs

| Setting | Value | Rationale |
|---|---|---|
| **Default** | **1 epoch** (36.9k packed @ GBS 768 -> ~48 steps at 4N) | The landed artifact's 1-epoch step count (~48 at 4N, ~24 at 8N) is in the same regime as B2 stage-2 (93 steps) and well below the `checkpoint-8672` forgetting regime (~4000 steps at LR > 1e-5). One epoch respects the "early stopping was load-bearing" finding from `tulu_math_uc_mix` (729 steps survived; 8672 did not). |
| **Extended** | **3 epochs** (144 steps at 4N, 72 at 8N) | Matches `sft_distill_cot_4n.sh` default `NUM_EPOCHS=3` and B2v2's 3ep baseline. Evaluate per-epoch (`save_strategy=epoch`) so the trajectory is visible. B2v2 shows 3ep on narrow data hurt (0.205 -> 0.105), but broad data may behave differently -- the per-epoch eval will tell. |
| **Do not exceed** | **3 epochs** without evidence | B2v2's 6ep also hurt (0.09), and `checkpoint-900`'s own forging above 1k steps is the warning. If 3ep helps, a follow-on can test 5-6ep; do not start there. |

**Save strategy:** `save_strategy=epoch` (one checkpoint per epoch) plus eval per checkpoint, mirroring B4a's per-epoch eval that correctly identified the overfitting point. For 1ep, save at end + every N steps (e.g., `save_steps=24` to get mid-epoch visibility).

#### GBS (global batch size)

| Scale | GBS | How | Notes |
|---|---|---|---|
| **4N (default)** | **768** | 48 GPUs x LBS 2 x GAS 8 | Matches `sft_distill_cot_4n.sh` header (`select=5` with 1 spare, `NHOSTS_TRAIN=4`, `PPN=12`). Tokens/step ~6.3M. Proven scale for `tulu_math_uc_mix_full` at 8N; 4N is within the safe 8N envelope (no 32N scale fault). |
| 8N (scale test) | 1536 | 96 GPUs x LBS 2 x GAS 8 | Via `qsub -l select=9 -v NHOSTS_TRAIN=8` override. Tokens/step ~12.6M. Tests whether larger-batch training (same LR) changes the format-vs-accuracy tradeoff. |
| 2N (cheap ablation) | 384 | 24 GPUs x LBS 2 x GAS 8 | Matches B2 stage-2's GBS 384 (batch 2 / GAS 8 / seq 2048 at 8N effective, but B2v2 used GBS 384 at 2N). Cheapest Sunspot allocation, ~1h walltime for 3ep. |

**GBS vs LR interaction:** LR is typically scaled with GBS (linear or sqrt). Here LR is held at 2e-5 across GBS values because the step count is small and the cosine schedule dominates. If 8N shows degraded accuracy vs 4N, it is a batch-size effect (fewer updates per epoch), not an LR effect.

**Sequence length:** Fixed at 8192 by the pre-tokenized artifact. Do not re-tokenize at 4096 or 2048 for the primary experiments -- the long traces would truncate mid-reasoning, breaking the envelope and creating a different training signal. A 4096 ablation is E4.

### 3.3 Should it be 2-stage (general -> CoT) or can broad-distill be single-stage?

This is the central design question. The answer is **two-stage**, with broad-distill as stage-2, for the following reasons:

**Evidence for two-stage:**

1. **B2 is the only recipe that worked.** Its two-stage structure (general math -> narrow GSM8K CoT) scored 0.205. Every single-stage variant (B3 0.05, B4b 0.065) failed, regardless of mix. The null hypothesis for any new single-stage SFT is failure.

2. **Broad-distill already mixes general+CoT internally.** OpenThoughts spans math + science + code; OpenR1 is math-focused. Their 50/50 interleave is itself a breadth+depth mix. Training on it in one stage from `gs138650` would be "general breadth (stage-1) + CoT depth (stage-2)" collapsed into one gradient stream -- the exact B3 shape that failed. The B3 mix was 45% instruction/chat + 55% math/CoT; the distill mix is ~50% broad reasoning + ~50% math reasoning. The proportions differ but the structural flaw is the same: one SFT must simultaneously learn to follow instructions and to execute arithmetic.

3. **The landed artifact's provenance is `gs138650` tokenization but the training base need not be.** The pre-tokenized dataset was built with `gs138650`'s tokenizer (vocab 256000, gemma-7b compatible) and is consumed via `--pretokenized_dataset` (input_ids, attention_mask, labels already materialized). The *base weights* it trains are independent -- `checkpoint-900` shares the same tokenizer, so the input_ids are valid for either base. There is no technical reason to limit SFT to the base that built the pre-tokenize.

4. **Stage-1 already did the general work.** `checkpoint-900` (IFEval 0.253, base-LM intact) is a stronger general manifold than anything a single-stage distill SFT could build from `gs138650` in ~50-150 steps. Re-doing general training from `gs138650` on distill traces would waste steps re-learning what stage-1 already consolidated.

**The single-stage counterargument and why it fails:**

One could argue: "The distill mix *is* the two-stage curriculum in one dataset -- OpenThoughts is the general stage, OpenR1 is the specific stage, interleaved at 50/50. Why not train on it directly?"

This conflates *data* breadth with *curriculum* sequencing. B3's 5-way mix also contained both general and specific data, interleaved at fixed weights, and it failed. The failure was not that the data lacked breadth -- it was that the model, in a single optimization trajectory, could not allocate capacity to both objectives without one starving the other. Interleaving at the batch level (TRL's `all_exhausted` interleave) does not create a curriculum; every batch contains both general and specific examples, so every gradient step optimizes both. A curriculum requires *temporal* separation: early steps on one distribution, late steps on another, so the model can consolidate the first before specializing on the second.

**Proposed stage-2 framing:**

```
gs138650 --[stage-1: tulu_math_uc_mix, 729 steps, seq 1024]--> checkpoint-900
       --[stage-2: distill_cot_mix, 48-144 steps, seq 8192, LR 2e-5]--> distill-cot checkpoint
                                          |
                                          +--[optional stage-3: gsm8k-r1cot polish, 6-12 steps]--> final CoT
```

Stage-2 trains on the broad-distill mix (rich reasoning breadth). An optional stage-3 -- a short gsm8k-r1cot polish (6-12 steps, same as B4a's shape but on the broad-distill base, not the B3 base) -- tests whether a narrow finishing stage on a *broad* base behaves like B2's stage-2 (helpful) or B4a's finish on a verbose base (harmful). B4a failed because the base was verbose-poisoned; a broad-distill base should not have that pathology, so the polish may work here even though it failed there.

**When single-stage would be justified:** Only if E5 (C = gs138650, single-stage distill, ~48 steps @ 8192) beats 0.205 or at least matches B2's 0.205 within noise. That would be evidence that the distill mix's *within-trace* reasoning diversity is qualitatively different from B3's *across-corpus* diversity and does not trigger the dilution mechanism. The prior for this is low (B3/B4b at 0.05-0.065), but it is worth one cheap test.

---

## 4. Cheap ablations before committing 4N x 12h

A full `sft_distill_cot_4n.sh` run is `select=5, walltime=12:00:00, workq`, 48 GPUs, ~48-144 steps, ~1-2h actual compute (the 12h is headroom for queuing + I/O). It is not expensive in compute, but it *is* expensive in **iteration latency** -- Sunspot `workq` queue times, manual HF consolidation, vLLM eval, and the need to re-queue CPT continuations in the same allocation. The ablations below are designed to reduce the number of full runs needed by answering structural questions at 1-2N scale or without any training.

### Ablation A0: Artifact audit (0 GPUs, 10 minutes, local or login node)

**Question:** Is the landed dataset what we think it is?

**Procedure:**

```bash
# On Sunspot login or via ssh
python3 - << 'PY'
from datasets import load_from_disk
ds = load_from_disk("/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192")
print(f"Rows: {len(ds):,}")
print(f"Columns: {ds.column_names}")
print(f"Features: {ds.features}")
# Length stats
lens = ds["seq_lengths"] if "seq_lengths" in ds.column_names else None
if lens:
    import numpy as np
    arr = np.array(lens)
    print(f"seq_lengths: min {arr.min()} max {arr.max()} mean {arr.mean():.0f} p50 {np.median(arr):.0f} p90 {np.percentile(arr,90):.0f}")
# Sample 3 rows decoded (need tokenizer)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/foremans/global_step138650")
for i in [0, len(ds)//2, len(ds)-1]:
    text = tok.decode(ds[i]["input_ids"][:512], skip_special_tokens=False)
    print(f"\n--- row {i} (first 512 tokens) ---\n{text[:2000]}")
PY
# Decontam gate
grep -F "[decontam]" /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/logs/pretokenize-distill-cot-mix-*/run.log
```

**Gates:**

- Rows ~91k, packed ~36.9k, columns include `input_ids` + `labels`/`attention_mask` (pre-tokenized contract: `train_sft.py` skips `skip_prepare_dataset` when `input_ids` present).
- `seq_lengths` distribution: if p50 >> 4096, the packing is dominated by long traces and 8192 is justified; if p50 ~ 1024, many packed sequences are half-empty and 4096 would be more efficient.
- Decoded samples: verify `<think>` / `</think>` / `<answer>` / `\boxed{}` envelope is present and well-formed; spot-check that OpenThoughts science/code problems look reasonable and not truncated.
- `[decontam]` counts: non-zero drops confirm decontamination ran; zero drops on 114k+220k corpora would be suspicious (expect at least dozens given GSM8K's public footprint).

**Cost:** 0 GPUs, login node Python. **Blocks all downstream experiments if failed** (wrong artifact, missing decontam, broken envelope).

### Ablation A1: Base-model probe -- format readiness (0 GPUs, 5 minutes, login node)

**Question:** How does `checkpoint-900-hf` currently behave on the CoT envelope? Is its GSM8K-CoT baseline still 0.205-proximal, or has weight drift / re-consolidation changed it?

**Procedure:** Run `eval_cot_gsm8k.py` (via `submit_cot_eval.sh` or direct `python -m` on 1 XPU tile) on `checkpoint-900-hf` itself -- not on any new SFT, just the base. This is the B2 stage-1 model's own CoT score (before stage-2). B2's stage-1 was never scored on GSM8K-CoT in the docs; only its stage-2 derivative (checkpoint-93) was. Measuring it tells you how much of B2's 0.205 was *already* in the base vs *added* by stage-2.

**Gates:**

- If `checkpoint-900` scores near 0.0 on GSM8K-CoT (expected -- it was never trained on the `<think>/<answer>\boxed{}` envelope, so format ~0.0), then stage-2 is doing all the work and the broad-distill SFT must teach both format and reasoning.
- If it scores partially (e.g., 0.05-0.10 with format 0.3-0.5), then stage-1 already learned some CoT-adjacent behavior and broad-distill has a warmer start.

**Cost:** 1 XPU tile, ~2 minutes of vLLM eval (200 problems). **Informs E1's expected delta.**

### Ablation A2: Mix composition -- does OpenThoughts add anything beyond OpenR1?

**Question:** Is the 50/50 OpenThoughts/OpenR1 interleave better than either component alone?

**Procedure (2N, GBS 384, 1ep, ~1h):** Two cheap SFT runs from `checkpoint-900`:

- A2a: `OpenThoughts-114k-decontam` only, 1ep, LR 2e-5, seq 8192, 2N (requires a one-off pre-tokenize of that single component at 8192, or a direct `get_sft_dataset` streaming run if pre-tokenize is skipped for this small scale).
- A2b: `OpenR1-Math-220k-decontam` only, same hparams.

The cheapest path: use the landed 36.9k-packed artifact's provenance to estimate per-component contribution, or run a **training-free proxy** -- embed the two corpora's questions with the base model's hidden states and measure embedding-space coverage (broader coverage predicts better generalization). But the direct 2N SFT is not expensive and is more reliable.

**Gates:**

- If A2a or A2b alone matches or beats the 50/50 mix on GSM8K-CoT, the interleave is unnecessary and a simpler mix (one corpus) is the deliverable.
- If both alone underperform the mix, the diversity hypothesis is supported.

**Cost:** 2 x 2N x ~1h. **Run after E1 if E1's result is ambiguous (e.g., 0.15 -- is it the mix or just one component?).**

### Ablation A3: Decontamination sensitivity (0 GPUs, 30 minutes, login node)

**Question:** How many GSM8K-test-near-duplicates survive the 13-gram filter?

**Procedure:**

```bash
python3 -m torchtitan.experiments.ezpz.rl.decontam_traces \
  --input /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192 \
  --n 8 \
  --report-only
```

Re-run the detector at `n=8` (more sensitive: 8-gram overlap instead of 13) on the landed artifact. Compare drop counts at n=13 (production) vs n=8 (sensitive). If n=8 drops significantly more, there is a tail of near-duplicate GSM8K test questions that the production filter missed.

**Gates:**

- Drop-rate delta < 0.5%: contamination risk is negligible; trust the eval.
- Delta 0.5-2%: worth noting in the report; consider re-staging the mix at n=8 for the final deliverable.
- Delta > 2%: the mix may be meaningfully contaminated; re-stage at n=8 before any "gain" claim.

**Cost:** 0 GPUs, CPU-only n-gram matching. **Run alongside A0; no training needed.**

### Ablation A4: Length truncation sensitivity (2N, GBS 384, 1ep, ~1h)

**Question:** Does training at 4096 (half the attention cost, truncating long traces mid-reasoning) hurt vs 8192?

**Procedure:** Stage a 4096 pre-tokenize of `distill_cot_mix` (one-off 1N job, `NPROC=4`, ~1-2h) and run SFT at `max_length 4096, LBS 4, GAS 4, GBS 384` (same GBS, half the seq_len, double the micro-batch to hold tokens/step). Compare to E1's 8192 result on GSM8K-CoT (format + accuracy + gen_len).

**Gates:**

- If 4096 matches 8192 within noise, the long traces' tails are not load-bearing and 4096 is the efficient operating point for future runs (64x less attention cost at scale).
- If 4096 degrades format (more unclosed) or accuracy, the long reasoning tails matter and 8192 is justified.

**Cost:** 1 x 1N pre-tokenize (~1-2h) + 1 x 2N SFT (~1h). **Only run if E1's gen_len suggests verbosity sensitivity (e.g., E1 gen_len > 500).**

---

## 5. Ranked experiment queue

Experiments are ordered by **information value per GPU-hour**, with early exits defined so that a negative result saves the downstream allocation. All evals are 200-problem GSM8K-CoT, `temperature 0.0 max_tokens 768`, fp32 vLLM, reporting `cot_accuracy / format_hit_rate / mean_gen_len / n_unclosed` vs B2's `0.205 / 0.985 / 282 / ~0`.

All runs use `rl-monarch-torch214-locality-20261001` VENV, `assistant_only_loss=True`, `packing=True`, `bf16`, `fsdp full_shard`, `dataset_num_proc 8` (the B2v2 fix from 32->8 after OOM in map subprocesses), `save_strategy epoch`, and `HF_DATASETS_OFFLINE=1` when reading the pre-tokenized artifact. Consolidation via `consolidate_sft_ckpt.sh` (`accelerate merge-weights`) then `submit_cot_eval.sh` / `eval_cot_gsm8k.py`.

### E1 -- Broad-distill stage-2 on checkpoint-900 (1ep, 4N) -- THE GATE

| Field | Value |
|---|---|
| **Priority** | **P0 -- run first** |
| **Base** | `checkpoint-900-hf` (re-consolidated from `checkpoint-900/` sharded if `-hf` missing; verify `config.json eos [1,107]`, `tokenizer.json`, `model.safetensors` present) |
| **Mix** | `distill_cot_mix` = `OpenThoughts-114k-decontam + OpenR1-Math-220k-decontam`, pre-tokenized `/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192` (36.9k packed @ 8192, 91k raw, 3.4G) |
| **Hparams** | LR 2e-5 cosine, 1 epoch (36.9k packed / GBS 768 @ 4N = ~48 steps), `per_device_train_batch_size 2`, `gradient_accumulation_steps 8`, `max_length 8192`, `bf16`, `fsdp full_shard`, `logging_steps 5`, `save_strategy epoch`, `dataset_num_proc 8` |
| **Scale** | 4N active (`select=5`, `NHOSTS_TRAIN=4`, `PPN=12`, `NGPUS_ACTIVE=48`, `GBS=768`), `workq`, `walltime 12:00:00` (actual ~1h, rest headroom) |
| **Script** | `sft_distill_cot_4n.sh` with `PRETOK_DIR=/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192`, `BASE_MODEL=<checkpoint-900-hf>`, `NUM_EPOCHS=1`, `NHOSTS_TRAIN=4` |
| **What it tests** | Does broad frontier reasoning distillation, as a stage-2 on the proven general base, beat or match B2's 0.205? This is the cleanest head-to-head: same base as B2, same LR, same structure, only the stage-2 *data* changes (7.4k narrow GSM8K -> 91k broad frontier traces). |
| **Gates** | **Go:** acc >= 0.15 (within 0.05 of B2) with format >= 0.80 and gen_len < 600 -- the mix is doing work; proceed to E2 (3ep). **Pivot:** acc 0.05-0.15 with good format -- broad reasoning did not transfer to GSM8K arithmetic; consider stage-3 polish (E2b) or evaluate on MATH/science before killing. **No-go:** acc < 0.05 or format collapse (<0.50) -- the mix dilutes like B3 did; stop and run A2 (component ablation) to isolate which half is the poison. |
| **Artifacts** | `outputs/sft/agpt2b-distill-cot-pretok-4n-ep1/` (checkpoints + `trainer_state.json` + `run.log`), W&B run, HF-consolidated `final/`, `eval_cot_gsm8k` JSONL + summary on `/lus/tegu` |

### E2 -- Same, 3 epochs (the B2 recipe's duration) -- 4N

| Field | Value |
|---|---|
| **Priority** | **P1 -- run if E1 is Go or Pivot** |
| **Base** | `checkpoint-900-hf` (same) |
| **Mix** | `distill_cot_mix` (same artifact) |
| **Hparams** | LR 2e-5 cosine, **3 epochs** (~144 steps @ 4N), otherwise identical to E1; `save_strategy epoch` so epoch-1/2/3 are individually eval-able |
| **Scale** | 4N, `GBS=768`, `workq`, `12h` (actual ~1.5h) |
| **What it tests** | Does more broad-distill training help or hurt? B2v2 says more narrow epochs hurt (0.205 -> 0.09/0.105); `checkpoint-8672` says more full-mix steps forget. Broad data may behave differently because it is diverse enough to avoid distribution collapse -- or it may follow the same curve. Per-epoch eval reveals the trajectory shape. |

| **Gates** | **Go:** epoch-3 acc > epoch-1 acc and >= 0.18 -- longer broad training helps; consider E3 (LR sweep) or E6 (CPT-FWE base). **Flat:** epoch-3 acc ~ epoch-1 acc (within 0.02) -- 1ep was sufficient; do not extend further. **No-go:** epoch-3 acc < epoch-1 acc or format degrades monotonically -- broad data also collapses with duration; cap future runs at 1ep. |
| **Artifacts** | `outputs/sft/agpt2b-distill-cot-pretok-4n-ep3/` (3 per-epoch checkpoints), W&B, per-epoch eval JSONLs |

### E2b -- Stage-3 narrow polish on the E1/E2 broad-distill checkpoint -- 2N

| Field | Value |
|---|---|
| **Priority** | **P1 -- run if E1 is Pivot (broad alone insufficient) or E2 is Flat** |
| **Base** | **E1 or E2's best-epoch HF-consolidated checkpoint** (not `checkpoint-900` -- the broad-distill model itself) |
| **Mix** | `gsm8k-r1cot` (7.4k ex, GSM8K train split, `<think>/<answer>\boxed{}` envelope; small enough to tokenize inline, no pre-tokenize needed) |
| **Hparams** | LR 2e-5 cosine (or 1e-5 if E1 showed format sensitivity), **3 epochs** (93 steps @ GBS 384 effective; but at 2N GBS 384 = batch 2 / GAS 8 / seq 2048, so use `max_length 2048` to match B2 stage-2's proven seq_len), `save_strategy epoch`, `dataset_num_proc 8` |
| **Scale** | 2N (`select=3` with spare, `NHOSTS_TRAIN=2`, `PPN=12`, `GBS=384`), `workq`, `01:00:00` (~30 min actual) |
| **Script** | `rl/scripts/sft/agpt2b_b4a_gsm8k_finish_2n.sh` pattern (B4a's shape) but on the broad-distill base |
| **What it tests** | Can a narrow GSM8K finishing stage on a *broad* base succeed where B4a failed on a *verbose* base? B4a's failure (0.05 -> 0.02, format collapse) was attributed to the B3 base's baked-in verbosity; a broad-distill base should not have that pathology, so the B2 stage-2 pattern (general -> narrow) may be recoverable here as a 3-stage curriculum (general -> broad reasoning -> narrow execution). |
| **Gates** | **Go:** acc > E1/E2 best and >= 0.15 with format >= 0.95 -- the 3-stage curriculum works; this becomes the deliverable candidate. **No-go:** acc flat or format collapses -- the narrow polish destabilizes even a broad base; the broad-distill model's manifold does not support narrow specialization (same B4a mechanism, different base). |
| **Artifacts** | `outputs/sft/agpt2b-broad-distill-gsm8k-polish-2n-ep3/` (3 per-epoch ckpts), per-epoch eval JSONLs |

### E3 -- LR ablation: 1e-5 cosine on the same setup -- 4N

| Field | Value |
|---|---|
| **Priority** | **P2 -- run if E1/E2 show format degradation or if E1 is No-go** |
| **Base** | `checkpoint-900-hf` (same) |
| **Mix** | `distill_cot_mix` (same) |
| **Hparams** | LR **1e-5 cosine**, 3 epochs (~144 steps @ 4N), otherwise identical to E2; `save_strategy epoch` |
| **Scale** | 4N, `GBS=768`, `workq`, `12h` |
| **What it tests** | Does a gentler LR preserve the stage-1 manifold better? The `tulu_math_uc_mix_full` post-mortem says "LR 2e-5 stayed >1e-5 through step 4350, forgetting" -- but that was ~4000 steps at high LR. At 48-144 steps, LR 2e-5 may still be safe; this tests whether it is. If E1 at 2e-5 shows format 0.6-0.8 with gen_len >500, LR is the suspect. |
| **Gates** | **Go:** acc >= E2 best or format >= 0.90 with gen_len < 400 -- lower LR helps; promote 1e-5 for future broad-distill. **No-go:** acc flat vs E2 -- LR was not the lever; the data or structure is. |
| **Artifacts** | `outputs/sft/agpt2b-distill-cot-pretok-4n-ep3-lr1e5/` |

### E4 -- Sequence-length ablation: 4096 vs 8192 -- 2N

| Field | Value |
|---|---|
| **Priority** | **P2 -- cheap, run alongside E3 or if E1 gen_len > 500** |
| **Base** | `checkpoint-900-hf` (same) |
| **Mix** | `distill_cot_mix` re-tokenized at `max_length 4096` (requires one-off 1N pre-tokenize: `_pretokenize_distill_cot_mix_1n.sh` with `max_length 4096` override, `NPROC=4`, ~1-2h, output `/tegu/datasets/datasets/agpt2b-distill-cot-mix-len4096`) |
| **Hparams** | LR 2e-5 cosine, 1 epoch, `per_device_train_batch_size 4`, `gradient_accumulation_steps 4`, `max_length 4096`, `GBS=384` at 2N (tokens/step matched to E1's 8192 @ GBS 768 by halving seq_len and doubling micro-batch), `bf16`, `fsdp full_shard` |
| **Scale** | 2N, `GBS=384`, `workq`, `01:00:00` |
| **What it tests** | Do the long reasoning tails (>4096 tokens) matter? B3's 8192 vs B4b's 4096 did not move accuracy (0.05 vs 0.065) but B4b also changed mix weights and filtering, so the seq_len effect was confounded. This isolates it: same mix, same base, only seq_len varies. If 4096 truncates mid-`<think>`, the envelope is broken and format should degrade. |
| **Gates** | **Go:** 4096 within 0.02 of E1 acc with format >= 0.85 -- long tails are not load-bearing; 4096 is the efficient default. **No-go:** 4096 format < 0.70 or acc -0.05 vs E1 -- long tails matter; keep 8192. |
| **Artifacts** | `/tegu/datasets/datasets/agpt2b-distill-cot-mix-len4096/` (new artifact), `outputs/sft/agpt2b-distill-cot-4096-2n-ep1/` |

### E5 -- Single-stage control from gs138650 (the null hypothesis) -- 4N

| Field | Value |
|---|---|
| **Priority** | **P3 -- one cheap control, not the main line** |
| **Base** | **`global_step138650` (raw MDS stage-3 pretrain, HF at `~/global_step138650`)** -- NOT `checkpoint-900` |
| **Mix** | `distill_cot_mix` (same 36.9k-packed @ 8192 artifact; tokenizer-compatible with gs138650) |
| **Hparams** | LR 2e-5 cosine, 1 epoch (~48 steps @ 4N), `per_device_train_batch_size 2`, `GAS 8`, `max_length 8192`, otherwise identical to E1 |
| **Scale** | 4N, `GBS=768`, `workq`, `12h` (actual ~1h) |
| **What it tests** | Is the broad-distill mix *as a cold-start replacement* better than B3's 5-way mix? Expected: ~0.05-0.07 (B3 class) because the structural flaw (one SFT must learn instruction-following + reasoning simultaneously) is unchanged. A higher score would be evidence that within-trace reasoning diversity is qualitatively different from across-corpus diversity. |
| **Gates** | **Surprise Go:** acc > 0.10 with format >= 0.80 -- the distill mix is doing something B3's mix could not; revisit the structural hypothesis. **Expected No-go:** acc 0.03-0.07 -- confirms structure is the lever; do not pursue further single-stage runs. |
| **Artifacts** | `outputs/sft/agpt2b-distill-cot-coldstart-4n-ep1/` |

### E6 -- CPT-FWE frontier base (follow-on, gated on CPT conversion) -- 4N

| Field | Value |
|---|---|
| **Priority** | **P4 -- gated on CPT 112k+ HF availability** |
| **Base** | **CPT-FWE 112k+ HF** (`step-112000` or nearest converted, from `outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64/step-112000` via `hf_batch_convert_cpt_fwe.sh`, flavor `2b-mds`) -- NOT raw DCP, NOT pre-112k. Gate: `config.json` + `tokenizer.json` + `model.safetensors` verified, not gibberish on a 5-problem GSM8K smoke. |
| **Mix** | `distill_cot_mix` (same artifact) |
| **Hparams** | LR **1e-5 cosine** (conservative -- CPT base is already distribution-shifted; 2e-5 re-warm may repeat the CPT dolmino LR-shock pattern), 1 epoch (~48 steps @ 4N), `max_length 8192`, otherwise as E1 |
| **Scale** | 4N, `GBS=768`, `workq`, `12h` |
| **Precondition** | CPT chain has converted at 112k+ AND a narrow SFT probe on that base (gsm8k-r1cot, 3ep, LR 2e-5, 2N, ~30 min) has been scored to calibrate the base's format behavior. Do not run blind -- the CPT-17000 probe (0.015, format collapse) says the shifted base is harder than `checkpoint-900`. |
| **What it tests** | Does the FineWeb-Edu kind-shift lift the ~0.2 ceiling when combined with broad reasoning distillation? This is the "bigger lever" from `fineweb-edu-60-40-kind-shift.md` + this report's SFT lever composed. SmolLM2's 60/40 at 38B reached 33.6% MMLU where DCLM needed 300B; if CPT-FWE moves the knowledge manifold, broad-distill SFT should convert it to reasoning accuracy. |
| **Gates** | **Go:** acc > E1 best (or > 0.15 if E1 was No-go) -- CPT+FWE + broad-distill composes; schedule full CPT-to-190k then re-train. **Pivot:** acc ~ E1 best within noise -- CPT did not lift the CoT ceiling (but may help MMLU -- score both). **No-go:** format collapse or acc < 0.05 -- the shifted base needs a different SFT recipe (general SFT first, then broad-distill as stage-2 on the CPT-SFT'd base -- i.e., mirror B2's two-stage on the new base). |
| **Artifacts** | `outputs/sft/agpt2b-cptfwe-distill-cot-4n-ep1-lr1e5/` |

### E7 -- Scale sensitivity: 2N vs 4N vs 8N (optional, low priority) -- 2N/8N

| Field | Value |
|---|---|
| **Priority** | **P5 -- only if E1's GBS sensitivity is questioned** |
| **Base** | `checkpoint-900-hf` |
| **Mix** | `distill_cot_mix` |
| **Hparams** | LR 2e-5 cosine, 1 epoch, same per-device batch/GAS adjusted to hold GBS per scale: 2N GBS 384 (LBS 2 / GAS 8), 8N GBS 1536 (LBS 2 / GAS 8, `select=9 NHOSTS_TRAIN=8`) |
| **Scale** | 2N + 8N, `workq` |
| **What it tests** | Does GBS (fewer updates per epoch at larger GBS, same LR) affect format-vs-accuracy? The 32N scale fault (GPU segfault at 144+ ranks) says XPU FSDP has a rank-count ceiling, but 2N/4N/8N (24/48/96 ranks) are all within the proven-safe 8N envelope. This is a throughput-vs-quality check, not a correctness gate. |
| **Gates** | Informational -- if 8N degrades vs 4N, future production should stay at 4N; if equivalent, 8N is the faster path. |
| **Artifacts** | `outputs/sft/agpt2b-distill-cot-pretok-{2,8}n-ep1/` |

---

### Queue summary and ordering

```
Time 0h ────────────────────────────────────────────────────── Time ~6h ── Time ~12h
A0 (0 GPU, artifact audit) ─┐
A1 (1 tile, base probe)     ─┤
A3 (0 GPU, decontam n=8)    ─┤  All three in parallel, <30 min total
                            v
E1 (4N, 1ep, ~1h) ──────────> eval (~2 min) ─┐
                                              ├─> branch:
E2 (4N, 3ep, ~1.5h)  ◄──────── Go/Pivot ──────┤───────────> E2b (2N polish, ~30 min)
E3 (4N, LR 1e-5)     ◄──────── format-sensitive┤
E4 (1N pretok + 2N SFT) ◄───── gen_len > 500 ─┤
E5 (4N, cold-start)  ◄──────── control ───────┤───────────> (informational)
                                              └───────────> E6 (CPT-FWE, gated on 112k+ HF)
```

**Minimum viable queue (if allocation is tight):** A0 + E1 + E2. That is 2 SFT runs (48 + 144 steps) + 4 evals, ~3h of compute, and it answers the binary question "does broad-distill on the proven base beat narrow-distill?" Everything else is conditional on E1's result.

**Full queue (if allocation is open):** A0 + A1 + A3 + E1 + E2 + E2b + E3 + E4 + E5 = ~8 SFT runs + 1 pre-tokenize + ~10 evals, ~10h of compute spread across Sunspot `workq` with 4N pack budgeting (`sft_distill_cot_4n.sh` budgets 16N active: this 4N + cpt_fineweb 4N + cpt_olmo2575 4N + rl_grpo 2N + test 2N).

---

## Appendix

### A. B-series raw numbers with job IDs

| Model | Jobs | W&B | Base | Mix | LR | Steps | Acc | Fmt | Len | Uncl |
|---|---|---|---|---|---|---:|---:|---:|---:|---|
| B2 = ckpt-93 | (stage-1: 12468404/409/437 chain; stage-2: gsm8k-r1cot 93 steps) | -- | gs138650 -> ckpt-900 | gsm8k-r1cot 7.4k, 3ep, seq 2048 | 2e-5 cos | 93 | **0.205** | 0.985 | 282 | ~0 |
| B3 | (b3_instruct_cot_mix_8n) | -- | gs138650 | tulu0.30+OAI0.25+gsm8k0.15+uc0.15+OMI2 0.15, s8192 | 2e-5 cos | ~3605? | 0.05 | 0.86 | 601 | 26 |
| B4a | 12471671 (ep1/2/3 = 12471676/77/78) | -- | B3-3605 | gsm8k-r1cot, 3ep | 2e-5 cos | 3ep | 0.02 | 0.26 | 1378 | 133 |
| B4b | 12471672 (eval 12471685) | -- | gs138650 | gsm8k0.40+OAI-filtered0.15+tulu0.30+uc0.15, s4096 | 2e-5 cos | 1ep | 0.065 | 0.86 | 611 | 27 |
| B2v2 ep3 | 12482423 (train) / 12482572 (eval) | xakno7kg | ckpt-900-hf | gsm8k-r1cot, 3->6ep, s2048 | 2e-5 cos | 6 | 0.09 | 0.86 | 416 | 22 |
| B2v2 ep6 | 12482485 (train) / 12482573 (eval) | ur6pqqhp | ckpt-900-hf | gsm8k-r1cot, 3->6ep, s2048 | 2e-5 cos | 12 | 0.105 | 0.91 | 378 | 15 |
| tulu_math_uc_mix_full @ 900 | 12470350..12470479 (chain) | vkwpxnqh etc. | gs138650 | tulu0.65+OMI2 0.15+uc0.20, s1024 | 2e-5 cos | 900 | (IFEval 0.253) | -- | -- | -- |
| tulu_math_uc_mix_full @ 8672 | same chain | -- | gs138650 | same, s1024 | 2e-5 cos | 8672 | (hellaswag 0.593->0.273) | -- | -- | -- |
| CPT-17000 SFT | (cpt34k-gsm8k-3ep) | -- | CPT 17000 HF | gsm8k-r1cot, 3ep, GBS384 | 2e-5 cos | 3ep | 0.015 | collapse | -- | -- |
| B2 stage-1 (tulu_math_uc_mix 729) | 12468404/409/437 chain | br7gopsj | gs138650 | tulu0.65+metamathqa0.15+uc0.20, s1024 | 2e-5 cos | 729 | (IFEval 0.1645->0.2440) | -- | -- | -- |

Eval: 200-problem GSM8K-CoT, `temperature 0.0 max_tokens 768`, `frameworks/2026.1.0 vLLM 0.26 XPU fp32`, Sunspot `workq`. Loss values: B2v2 0.8069 / acc 0.8198 (training, not eval). B4a/B4b gen_len/n_unclosed guardrails added in `eval_cot_gsm8k.py` (`mean_gen_len`, `n_unclosed`, `summarize()` helper; `finish_reason == "length"` = unclosed).

### B. Pre-tokenized artifact lineage

```
open-thoughts/OpenThoughts-114k (HF, ~3.4G download, 114k rows)
  -> _openthoughts_format_row (extract <|begin_of_thought|> trace + LAST \boxed{}, drop code-gen rows without \boxed{})
  -> _build_openthoughts_cot (map+filter)
  -> _decontaminate_against_gsm8k (13-gram + normalized-exact, GSM8KDecontaminator, logged [decontam])
  -> OpenThoughts-114k-decontam ─┐
                                 ├─ _build_distill_cot_mix (weights 0.5/0.5, seed 42,
open-r1/OpenR1-Math-220k (HF, cached, 220k rows)  │     _materialized_mix_load_or_build, all_exhausted interleave,
  -> _openr1_format_row (correctness_math_verify +  │     recipe hash captures decontam, cache under
     is_reasoning_complete + LAST \boxed{})        │     /tegu/datasets/datasets/ezpz_sft_mixes/cba5c64)
  -> _build_openr1_math_cot                       │
  -> _decontaminate_against_gsm8k                  │
  -> OpenR1-Math-220k-decontam ───────────────────┘
                                                    v
                                          distill_cot_mix (91,481 raw rows)
                                                    |
                              train_sft.py --pretokenize_to @ 8192 (TRL tokenize+pack, NPROC=1, VENV rl-monarch)
                                                    |
                              /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192
                                dataset_info.json + 8 x data-*.arrow (3.4G, 36,934 packed @ 8192)
                                                    |
                              sft_distill_cot_4n.sh --pretokenized_dataset (HF_DATASETS_OFFLINE=1,
                                load_from_disk, skip_prepare_dataset, FSDP full_shard)
```

The `DISTILL_MAX_THINK_CHARS=16000` and `OPENR1_MAX_THINK_CHARS=16000` caps are env-tunable and part of the recipe (component build), so changing them forces a fresh mix cache. The landed artifact used the defaults (16000). B4b's 1200-char clamp is not active here.

### C. Key file paths

| Purpose | Path |
|---|---|
| This report | `torchtitan/experiments/ezpz/docs/exploration/thread-b-broad-distill.md` |
| Production status | `torchtitan/experiments/ezpz/docs/production/POST-TRAINING-2B.md` |
| SFT index + deliverables | `torchtitan/experiments/ezpz/docs/production/sft/README.md` |
| B3 design (single-stage) | `torchtitan/experiments/ezpz/docs/production/sft/agpt/2b-mds/b3-instruct-cot-mix/design.md` |
| B4 design (finish+reweight) | `torchtitan/experiments/ezpz/docs/production/sft/agpt/2b-mds/b4-finish-and-reweight/design.md` |
| B4 results | `torchtitan/experiments/ezpz/docs/production/sft/agpt/2b-mds/b4-finish-and-reweight/README.md` |
| Kind-shift (60/40) | `torchtitan/experiments/ezpz/docs/production/sft/fineweb-edu-60-40-kind-shift.md` |
| CPT pilots | `torchtitan/experiments/ezpz/docs/production/cpt/README.md` |
| Dataset registry | `torchtitan/experiments/ezpz/rl/datasets_sft.py` (`distill_cot_mix`, `decontaminate_against_gsm8k`, `DISTILL_MAX_THINK_CHARS`) |
| Decontam | `torchtitan/experiments/ezpz/rl/decontam_traces.py` (`GSM8KDecontaminator`, 13-gram + exact) |
| SFT trainer | `torchtitan/experiments/ezpz/rl/train_sft.py` (TRL `SFTTrainer`, `assistant_only_loss`, `packing`) |
| Pre-tokenize (1N) | `torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh` |
| Training (4N) | `torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh` |
| B2v2 narrow SFT | `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh` |
| B4a finish | `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b4a_gsm8k_finish_2n.sh` |
| B4b reweight | `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b4b_reweight_8n.sh` |
| Eval (CoT) | `torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py` + `consolidate_and_eval_cot.sh` / `submit_cot_eval.sh` |
| Landed artifact | `/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192` (Sunspot, `datasets` group quota) |
| Mix cache | `/tegu/datasets/datasets/ezpz_sft_mixes/cba5c64` (materialized interleave, <5s load) |
| Base: ckpt-900 | `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-{900,900-hf}/` |
| Base: CPT-FWE | `outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64/` (DCP, 556 dirs at 111.2k; HF converts at 74k verified, 112k gate) |
| Base: gs138650 | `~/global_step138650` (HF, Sunspot) |

### D. Glossary

- **B2 (`checkpoint-93`):** Two-stage SFT `gs138650 -> tulu_math_uc_mix -> checkpoint-900 -> gsm8k-r1cot -> checkpoint-93`, 0.205 GSM8K-CoT, format 0.985. The only recipe that worked; the deliverable.
- **B3:** Single-stage SFT `gs138650 -> b3_instruct_cot_mix` (tulu + OpenR1 + gsm8k + ultrachat + OMI2, seq 8192), 0.05. Long-CoT dilution.
- **B4a:** gsm8k-r1cot finishing stage on B3 base, 0.02, format 0.26. Verbose bias baked in; finishing stage cannot un-teach it.
- **B4b:** Reweighted single-stage `gs138650 -> b4_reweight_mix` (gsm8k 0.40 + filtered OpenR1 0.15 + tulu 0.30 + ultrachat 0.15, seq 4096), 0.065, format 0.86. Symptom fixed, accuracy stuck.
- **B2v2:** Extended narrow stage-2 `checkpoint-900 -> gsm8k-r1cot` at 6/12 steps (3->6ep), 0.09/0.105. More narrow epochs hurt; training metrics healthy, eval halved.
- **checkpoint-900 (`tulu_math_uc_mix_full @ 900`):** Stage-1 general-instruct deliverable, IFEval 0.253, base-LM intact. Sharded on disk; `-hf` re-consolidatable.
- **checkpoint-8672:** Same run at full epoch (8672 steps), catastrophically forgot (hellaswag 0.593->0.273). Do NOT use.
- **CPT-FWE:** FineWeb-Edu 60/40 continued pre-training, 4N Sunspot, Grain streaming, 100B budget (190734 steps), currently 111.2k/190.7k (58.3%). The kind-shift lever.
- **distill_cot_mix:** `OpenThoughts-114k-decontam (50%) + OpenR1-Math-220k-decontam (50%)` reasoning-distillation cold-start mix, GSM8K-test-decontaminated, `<think>/<answer>\boxed{}` envelope, pre-tokenized to 36.9k packed @ 8192.
- **Format / guardrails:** `format_hit_rate` (fraction with valid `<think>/<answer>\boxed{}` envelope), `mean_gen_len` (chars), `n_unclosed` (generations with `finish_reason == "length"`, i.e., never emitted `</answer>`). Added after B3; accuracy alone hides envelope collapse.
- **GBS:** Global batch size = `NGPU * per_device_train_batch_size * gradient_accumulation_steps` (TP=1 for SFT). B2 stage-2: 384; full-mix stage-1: 6144; broad-distill: 768 @ 4N, 384 @ 2N, 1536 @ 8N.
- **Decontam:** GSM8K-test decontamination via `GSM8KDecontaminator` (13-gram + normalized-exact), `decontam_traces.py`, logged `[decontam]`, hash-captured in materialized-mix cache.
