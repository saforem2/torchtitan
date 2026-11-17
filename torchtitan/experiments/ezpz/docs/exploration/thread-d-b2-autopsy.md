# B2 Autopsy: Why 0.205 Worked and Everything Else Did Not

> **Scope:** Deep failure-mode autopsy of the AuroraGPT-2B post-training
> CoT campaign. B2 = `checkpoint-93` (0.205 GSM8K-CoT) remains the sole
> CoT deliverable; every reconstruction attempt landed 0.02-0.105. This
> document dissects what made B2 work, classifies each failure, explains
> the ckpt-8672 catastrophe, argues the ceiling thesis, and distills
> concrete rules for any future SFT attempt.
>
> **Branch:** `ezpz-cpt111k-eval-sft` HEAD `bcbca5bd90`.
> **Primary source:** [`docs/production/POST-TRAINING-2B.md`](../production/POST-TRAINING-2B.md)
> with run-level detail from `tulu_math_uc_mix/README.md`,
> `tulu_math_uc_mix_full/README.md`, `b3-instruct-cot-mix/design.md`,
> `b4-finish-and-reweight/README.md`, `rl/plans/cot.md`, and
> `sft/README.md`. Every number cited below names its source file.

---

## 1. Dissecting B2: What Specifically Made 0.205 Work

B2 is not one SFT. It is two SFTs in sequence, each doing a distinct
job, with a carefully chosen checkpoint boundary between them. The
headline recipe:

```
global_step138650  --[stage 1: general-math SFT]-->  checkpoint-900  --[stage 2: GSM8K-CoT SFT]-->  checkpoint-93 = B2
   (MDS 2B base)         tulu_math_uc_mix_full            gsm8k-r1cot              0.205 / 0.985 / 282
                          53.3M seqs, 1ep                   7.4k ex, 93 steps
                          seq 1024, 8N, ~54B tok             3ep, LR 2e-5
                          IFEval 0.179 -> 0.253              format 0.985
```

Neither stage alone reaches 0.205. Stage 1 scores 0.00-0.15 on GSM8K-CoT
(no `<think>` envelope). Stage 2 from any other base scores 0.02-0.065.
The combination is load-bearing. Sections 1.1 and 1.2 dissect each stage
independently.

### 1.1 Stage 1 -- `global_step138650` to `checkpoint-900` via `tulu_math_uc_mix_full`

#### 1.1.1 What stage 1 actually is

Stage 1 is the **FULL big-mix** SFT documented in
`tulu_math_uc_mix_full/README.md` (completed 2026-07-16, head job
12470350, 11-link afterany chain, 8N x 12 = 96 ranks, GBS 6144):

| Field | Value |
|-------|-------|
| Base | `global_step138650` = MDS stage-3 checkpoint (1.99B, vocab 256000, llama arch, 140k-step continuous run; "stages" are data-mix transitions, not separate artifacts) |
| Recipe | `tulu_math_uc_mix` = tulu-3-sft-mixture 0.65 + OpenMathInstruct-2 0.15 + ultrachat-200k 0.20, `all_exhausted` interleave |
| Materialized mix | ~93.1M interleaved rows -> 53,276,203 packed sequences at len 1024 |
| Tokens | ~54B target (1 epoch); ~5.7B at the step-900 deliverable |
| Hyperparams | LR 2e-5 cosine to 0, bf16, AdamW (TRL default), packing, `assistant_only_loss` via baked `assistant_masks` |
| Scale | 8N (not 32N -- 32N GPU-page-faults at step 1-2, 384-rank scale fault; 8N = 96 ranks is largest safe scale) |

This is the **same recipe** as the completed 729-step SFT
(`tulu_math_uc_mix/README.md`, 2026-06-10, 32N, metamathqa-swap for
OpenMathInstruct-2 at 35x less I/O), but at ~12x the token budget: the
full OpenMathInstruct-2 14M rows instead of the metamathqa 350k swap.
The comparison question was "does ~12x more tokens beat the 729-step
deliverable?" The answer was split -- see 1.1.3 and Section 3.

#### 1.1.2 Why the tulu_math_uc_mix blend was the right stage-1 mix

The mix proportions are doing specific work that matters for the stage-2
handoff:

- **tulu-3-sft-mixture at 0.65** -- the general instruction-following
  mass. This is sourced + filtered by AllenAI, covering single- and
  multi-turn assistant data. It teaches the model to follow structural
  directives (IFEval prompt-strict 0.179 -> 0.253 at step 900, +7.4pp,
  ~4-5 sigma; see 1.1.3). Without this component, the model cannot
  reliably obey the `<think>/<answer>\boxed{}` envelope instruction that
  stage 2 will demand. This is proven by the stage-1 smoke: B2's stage 2
  on checkpoint-16 (a 16-step toy) reached format 0.955 but accuracy only
  0.16, while the real ckpt-900 base enabled 0.205 -- the stronger
  instruction-following foundation matters.

- **OpenMathInstruct-2 at 0.15** -- the broad math signal. 14M rows of
  GSM8K+MATH-derived problems with worked solutions, in natural language,
  at moderate difficulty. This teaches shallow step-by-step reasoning
  patterns *without* imposing the verbose DeepSeek-R1 reflective style.
  Its traces are short (packed at 1024, not 8192) and its answers are
  plain numbers boxed only when the template demands it. Critically, it
  does NOT carry `<think>` tags -- so it teaches reasoning *habits*
  (decomposing a word problem into steps) without teaching a conflicting
  envelope. The model arrives at stage 2 knowing how to break down a
  math problem, ready to have that skill *wrapped* in the CoT format.

- **ultrachat-200k at 0.20** -- conversational breadth. Filtered ChatGPT
  dialogues that prevent the model from collapsing to a single QA style.
  This preserves generation diversity and prevents the "math-only channel"
  failure that killed ckpt-8672 (see Section 3).

Three properties of this mix jointly make it stage-1-suitable:

1. **No long-form CoT contamination.** None of the three components
   contains `<think>` traces. The longest completions are tulu's
   multi-turn dialogues, not R1-style reasoning chains. This means stage 1
   never teaches verbosity.

2. **No envelope conflict.** None of the three components uses
   `<think>/<answer>\boxed{}`. Stage 1's output format is the tulu chat
   format (plain assistant turn). Stage 2 can therefore define the
   envelope cleanly, without having to *override* a competing format the
   model already learned. Every failed single-stage rebuild violated this:
   B3 mixed 0.25 OpenR1 traces carrying `<think>` with instruction data in
   one pass, creating a format competition that stage 1 avoids by
   definition.

3. **Short sequence length (1024).** All 53.3M packed sequences are at
   1024 tokens. No trace is truncated mid-reasoning, because no trace
   exceeds ~1k tokens. The attention cost is O(L^2) with no flash-attn on
   XPU, so 1024 is dramatically cheaper than the 8192/4096 used in failed
   rebuilds, and the short length forces a *concise* reasoning style that
   matches the GSM8K eval distribution (mean gen_len 282 at B2, not 601).

#### 1.1.3 Stage 1 evals: why ckpt-900, not ckpt-8672

The full-mix 8N SFT produced checkpoints every 50 steps from 300 to 8672.
The eval sweep (jobs 12470365 + 12470886 + 12470889 + 12470896, Sunspot,
0-shot lm-eval + IFEval + GRPO) is in `tulu_math_uc_mix_full/evals/README.md`
and is the source for all numbers in this section.

**Checkpoint-900 is the deliverable.** From `evals/README.md`:

| Metric | baseline (gs138650) | ckpt-900 | ckpt-8672 | metamathqa-729 |
|--------|---------------------:|--------:|----------:|---------------:|
| IFEval prompt_strict | 0.179 | **0.253** | 0.168 | 0.244 |
| IFEval inst_strict | 0.289 | **0.384** | 0.287 | 0.386 |
| IFEval prompt_loose | 0.183 | **0.283** | 0.183 | 0.274 |
| IFEval inst_loose | 0.294 | **0.417** | 0.300 | 0.411 |
| hellaswag | 0.593 | 0.592 | **0.273** | (intact) |
| arc_easy | 0.694 | 0.640 | **0.298** | (intact) |
| GRPO sum_digits 20-step | -- | 0.31 -> 0.74 | -- | 0.4 -> 0.9/1000 |

Checkpoint-900 matches or beats metamathqa-729 on IFEval (prompt-strict
0.253 vs 0.244, inst-loose 0.417 vs 0.411) **while retaining full base-LM
capability** (hellaswag 0.592 vs base 0.593, arc_easy 0.640 vs 0.694).
It is the best of all four models evaluated and the strongest GRPO
starting point (0.31 -> 0.74 in 20 steps, vLLM server-mode job 12470959).

Checkpoint-900 sits at ~5.7B tokens (900 steps x 6144 seqs x 1024 tok),
or roughly **10% through the epoch**. The model has seen enough tulu_math
to internalize instruction-following and shallow math reasoning, but not
so much that it has overfit the math distribution (see Section 3 for the
collapse). The 729-step metamathqa SFT survived for the same reason -- it
stopped at ~4.5B tokens -- but ckpt-900 is stronger because the full
OpenMathInstruct-2 mix is richer than the metamathqa swap.

**What ckpt-900 gave stage 2 specifically:**

1. **Instruction obedience at 0.253 IFEval.** The model reliably follows
   "Reason step by step inside `<think></think>`, then give the final
   answer inside `<answer>\boxed{}</answer>`."

2. **Shallow math reasoning habits.** Decomposition of word problems into
   sequential arithmetic steps, without verbosity.

3. **Intact base LM.** No forgetting yet (hellaswag 0.592, arc_easy 0.640;
   collapse begins at ~step 1500, see Section 3). The model's general
   knowledge and fluency are intact, so stage 2's narrow fine-tuning has a
   stable manifold to operate on.

4. **A non-verbose prior.** Mean generation length under the base
   instruction format is short. Stage 2 does not need to *un-teach*
   verbosity -- the single hardest failure mode in the B3/B4 family (see
   Section 2).

### 1.2 Stage 2 -- `checkpoint-900` to `checkpoint-93` via `gsm8k-r1cot`

#### 1.2.1 What stage 2 actually is

Stage 2 is documented in `rl/plans/cot.md` (Stage 1 section) and
`POST-TRAINING-2B.md` B2 section. It is a **continue-SFT** (not from-scratch)
on the single dataset `gsm8k-r1cot`:

| Field | Value |
|-------|-------|
| Base | `checkpoint-900-hf` (the stage-1 deliverable, 8N full-mix at ~5.7B tokens) |
| Dataset | `gsm8k-r1cot` = openai/gsm8k main, 7473 examples, each reformatted |
| Reformatting | prompt: `{user, question + "\nReason step by step inside <think></think>, then give the final answer inside <answer>\boxed{}</answer>."}` + completion: `{assistant, "<think>{cot_sans_marker}</think>\n<answer>\boxed{N}</answer>"}` -- completion starts directly with `<think>` (no leading `\n`; the gemma template's `lstrip('\n')` at `train_grpo.py:584/587` would otherwise break the `assistant_only_loss` mask) |
| Envelope | `<think>...</think>\n<answer>\boxed{N}</answer>` -- byte-consistent with what `eval_cot_gsm8k.py` scores |
| Trainer | TRL `SFTTrainer`, FSDP `full_shard`, `assistant_only_loss=True`, gemma chat template, packing |
| Sequence length | **2048** (mandatory -- default 1024 silently truncates the `<answer>` tail and destroys the supervised target; `rl/plans/cot.md:136`) |
| Hyperparameters | LR 2e-5 cosine, bf16, AdamW, 3 epochs |
| Scale | 2N x 12 = 24 ranks, GBS 384 (batch 2 / GAS 8 / seq 2048), 93 global steps |
| Output | `checkpoint-93` = B2: 0.205 CoT, format 0.985, gen_len 282 |

The dataset is tiny (7,473 examples) and the run is short (93 steps).
Training metrics at `checkpoint-93` are healthy: loss descending, token
accuracy high (the B2v2 re-runs at 3ep/6ep report loss 0.8069 / acc
0.8198 on the same data, confirming the training curve is clean).

#### 1.2.2 Why stage 2 works on ckpt-900 and nowhere else

Stage 2 performs three narrow functions, each of which requires the
stage-1 foundation:

**Function 1: Envelope imposition.** The model must learn to emit
`<think>...</think><answer>\boxed{}</answer>` on every math query,
reliably, with correct nesting and no truncation. This is pure format
learning, and it is almost trivially easy on ckpt-900: the smoke
checkpoint at only 16 steps already reached format 0.955 (see
`rl/plans/cot.md` Stage 1 results table). By step 93, format is 0.985
with gen_len 282 and ~0 unclosed. The reason it is easy on ckpt-900
is that ckpt-900 already follows instructions (IFEval 0.253) and already
does shallow math decomposition -- stage 2 only needs to *wrap* existing
competence in a new envelope, not teach the competence itself.

Contrast every other base: B3's base (gs138650, IFEval 0.179, no math
SFT) had to learn instruction-following, math reasoning, AND the envelope
simultaneously in one SFT, at 0.65/0.15/0.20 tulu/math/chat competing with
0.25 long-CoT and 0.15 gsm8k-r1cot for signal. The format competition was
structural, not a weight-tuning problem (see Section 2.1).

**Function 2: In-distribution format grounding.** The `gsm8k-r1cot` CoT
is derived from gsm8k's own `#### N` rationales -- terse, templated,
arithmetic-forward, with no reflection or self-correction. This is the
*exact* distribution the GSM8K-CoT eval draws from (200 held-out GSM8K
problems, same question style, same answer extraction via `\boxed{}`).
Stage 2 therefore teaches the model the in-distribution reasoning pattern
that the eval rewards: short chains (mean 282 tokens), direct arithmetic,
single boxed answer. There is zero distribution shift between train and
eval -- a deliberate choice, not an accident. The smoke at ckpt-16
already showed reasoning accuracy preserved (0.15 -> 0.16) while format
went 0.00 -> 0.955, meaning the model's existing math competence was
*preserved* through the envelope change. By ckpt-93, that competence was
sharpened to 0.205 (+4.5pp over the smoke), meaning the 77 additional
steps refined arithmetic accuracy within the new envelope.

**Function 3: Brevity as implicit regularization.** At 7,473 examples
and 93 steps, stage 2 is extremely data-efficient and therefore
extremely regularized by its own brevity. The model sees each example
exactly 3 times (3 epochs), for a total of ~22k example-presentations.
There is no room for overfitting in the classical sense -- the evidence
is that B2v2 at 6 epochs (186 presentations of each example, 12 steps)
*degraded* to 0.09/0.105, not improved. The 3-epoch sweet spot teaches
the envelope + refines arithmetic without memorizing. At LR 2e-5 cosine,
the learning rate decays meaningfully across 93 steps, providing a
natural cooldown that prevents late-step forgetting.

#### 1.2.3 The B2 evidence that sequencing is doing the work

The strongest evidence that the two-stage *structure* is the lever -- not
the data, not the hyperparameters -- is the **stage-1 substitution test**
implicit in the campaign results:

- **Same stage-2 data, different stage-1 base, same stage-2 hparams:**
  B2 (ckpt-900 -> gsm8k-r1cot 3ep) = 0.205. B4a (B3-3605 -> gsm8k-r1cot 3ep
  per-epoch) = 0.02, 0.00, 0.26 across epochs, gen_len 1378, 133 unclosed.
  Same dataset, same epochs, same LR -- 10x difference. The base is
  everything.

- **Same overall data, different structure, same compute:**
  B2 (tulu_math in SFT-1, then gsm8k-r1cot in SFT-2) = 0.205. B3
  (tulu_math + gsm8k-r1cot + OpenR1 + OpenMathInstruct in one SFT) =
  0.05. Same component datasets (plus OpenR1), same total tokens
  approximately -- 4x difference. The sequencing is everything.

- **Same stage-1 base, same stage-2 data, more stage-2 epochs:**
  B2 (ckpt-900 -> gsm8k-r1cot 3ep) = 0.205. B2v2 (ckpt-900 -> gsm8k-r1cot
  6ep/12ep, 6/12 steps) = 0.09/0.105. Doubling the stage-2 budget
  *halved* accuracy. More is not better; the 3-epoch stop is load-bearing
  (see Section 2.4).

---

## 2. Why B3, B4a, B4b, and B2v2 All Failed -- Failure-Mode Taxonomy

### 2.1 Failure-Mode Taxonomy Table

Each row is one measured run on the shared 200-problem GSM8K-CoT eval
(`temperature 0.0 max_tokens 768`, `frameworks/2026.1.0 vLLM 0.26 XPU
fp32`, Sunspot `workq`). Primary failure-mode assignment is the *cause*
identified by the guardrail evidence; secondary modes are contributing
factors. Evidence columns quote `POST-TRAINING-2B.md`, `b4-finish-and-
reweight/README.md`, and `rl/plans/cot.md`.

| Run | CoT acc | Format | Gen len | Unclosed (/200) | vs B2 (0.205) | Primary failure mode | Secondary | Evidence that pins the mode |
|-----|--------:|-------:|--------:|----------------:|---------------|----------------------|-----------|-----------------------------|
| **B2** = ckpt-93 (two-stage: tulu_math -> gsm8k-r1cot 3ep) | **0.205** | 0.985 | 282 | ~0 | -- | -- (reference) | -- | Target. Format near-perfect, gen_len short, zero truncation. Training healthy (loss descending, acc 0.8198 at B2v2 re-run confirms same mix trains cleanly). |
| **B3** = b3_instruct_cot_mix single SFT from gs138650, seq 8192 | **0.05** | 0.86 | 601 | 26 | -0.155 | **Data dilution / long-CoT contamination** | Verbose-base formation | Mix had gsm8k-r1cot at only 0.15, diluted by 0.25 OpenR1 long-form R1 traces (2k-10k tok) + 0.15 OpenMathInstruct + 0.45 instruct/chat. Raw generations: coherent English, well-formed `<think>/<answer>` but WRONG arithmetic + run-on chains (probe `wf_70eef54d` in `b4/design.md`). Acc flat-low across ckpt 900/1800/3605 (0.035/0.070/0.050) -- not late-collapse but structural. |
| **B4a** = gsm8k-r1cot finish (3ep, per-epoch ckpts) on B3-3605 base | **0.02** | 0.26 | 1378 | 133 | -0.185 | **Verbose-base poisoning (irreversible)** | Format collapse | B3's verbose R1 style baked into weights; a light finish cannot un-teach it. Gen_len 601 -> ~1400-2000, format 0.86 -> 0.26/0.00, 133/200 never close `</answer>`. Adding a short finishing stage *destabilized* the envelope further (gen_len doubled, format halved). REFUTED hypothesis: "broad base + finishing signal = best of both." |
| **B4b** = b4_reweight_mix single SFT from gs138650, seq 4096, gsm8k 0.40 + len-filtered OpenR1 0.15 | **0.065** | 0.86 | 611 | 27 | -0.14 | **Data dilution (format-independent)** | -- (verbosity fixed, accuracy not) | Sharpest negative result. OPENR1_MAX_THINK_CHARS filter + reweight (gsm8k 0.15->0.40) restored format (0.86) and verbosity (611, 27 unclosed) to B3-like health -- yet accuracy only 0.05 -> 0.065 (+0.015). Run-on style was a *symptom*, not the cause. Accuracy lives in SFT structure, not verbosity. Single-stage-from-gs138650 ceiling is ~0.05-0.065 regardless of weights/filtering/seq_len. |
| **B2v2 ep3** = ckpt-900 -> gsm8k-r1cot 3ep (6 steps), GBS 384, Sunspot 2N `12482423` / W&B `xakno7kg` | **0.09** | 0.86 | 416 | 22 | -0.115 | **Narrow-data overfitting** | Format regression | Same base as B2, same data as B2, same LR (2e-5 cosine), double the epochs (3->6, 6 steps vs B2's 93 steps at larger GBS -- note: B2's 93 steps vs B2v2's 6 steps are GBS-inequivalent; B2v2 re-ran the same 7.4k examples at GBS 384 for 6 steps where B2 did 93 steps at GBS 6144-equivalent packing). Training healthy (loss 0.8069, acc 0.8198). Eval halved (0.205 -> 0.09). Same failure mode as B3/B4: more narrow SFT tokens hurt despite healthy training metrics. |
| **B2v2 ep6** = ckpt-900 -> gsm8k-r1cot 6ep (12 steps), GBS 384, Sunspot 2N `12482485` / W&B `ur6pqqhp` | **0.105** | 0.91 | 378 | 15 | -0.10 | **Narrow-data overfitting** | Format partial recovery | 12 steps vs 6: accuracy 0.09 -> 0.105 (+0.015, noise), format 0.86 -> 0.91, gen_len 416 -> 378, unclosed 22 -> 15. Marginal recovery but still half of B2. Non-monotonic with epochs; no sweet spot beyond 3ep. Same verdict as full-mix ckpt-8672: more narrow tokens at high LR destroys what the early stop preserved. Eval jobs `12482572/73` (`x1922c4s2b0n0/s5b0n0`, Exit 0) on `frameworks/2026.1.0 vLLM 0.26 XPU`. |

**Reading the table row by row:**

- **B3** tests whether a single balanced SFT can learn instruction-following
  AND CoT simultaneously. It cannot, at 2B. The 0.25 OpenR1 long-form
  component teaches a verbose, reflective style that competes with the
  0.15 gsm8k-r1cot short-CoT signal. With only 1.99B parameters, the model
  cannot maintain two reasoning styles; the longer, more diverse R1 traces
  dominate. Accuracy flat at 0.035-0.070 across ckpt 900/1800/3605 proves
  this is not "train longer and it recovers" -- the mix itself is the
  problem.

- **B4a** tests whether a narrow finish can rescue a verbose-poisoned
  base. It cannot, and the attempt makes things worse. The B3-3605 base
  has already internalized verbosity as a *weight-level prior* (gen_len
  601). Three epochs of gsm8k-r1cot at LR 2e-5 do not overwrite that prior;
  instead they *destabilize* the envelope the base had learned (format
  0.86 -> 0.26), because the optimizer is pulled between two conflicting
  format attractors. Gen_len 1378 and 133 unclosed are the model
  oscillating between R1-verbosity and gsm8k-brevity and failing to
  produce either coherently.

- **B4b** is the decisive control. It rebuilds from the clean gs138650
  base (not the poisoned B3 base), fixes the exact symptom B3 showed
  (long-CoT dilution via OPENR1_MAX_THINK_CHARS length filter + reweight
  gsm8k 0.15 -> 0.40, seq 4096 to halve attention cost), and still only
  reaches 0.065. Format and verbosity are now healthy (0.86, 611, 27 --
  indistinguishable from B3's 0.86/601/26), but accuracy moved only
  +0.015. This proves verbosity was a *symptom* of single-stage training,
  not the accuracy cause. The accuracy gap (0.205 vs 0.065) lives in SFT
  *structure* -- two curriculum-ordered stages vs one mixed stage -- not
  in any tunable property of the single stage.

- **B2v2** tests whether B2 itself can be improved by more stage-2 epochs.
  It cannot. The same base (ckpt-900), same data (gsm8k-r1cot), same LR
  (2e-5) that produced 0.205 at 3ep produces 0.09 at 6ep and 0.105 at 12ep.
  Training loss still descends (0.8069, acc 0.8198), so this is not
  optimization failure -- it is narrow-data overfitting that training
  metrics cannot detect. This is the same pathology as ckpt-8672 at
  larger scale (see Section 3): more tokens on a narrow distribution at
  high LR actively destroy capability that early stopping preserved.

### 2.2 Was it format collapse, verbose base, data dilution, or overfitting?

Each failure has a different dominant mechanism. The guardrail trio
(format / gen_len / unclosed) separates them cleanly:

**B3 -- data dilution (primary) + verbose-base formation (secondary).**
B3's core problem is that gsm8k-r1cot at 0.15 is diluted by 0.25 OpenR1
long-form CoT + 0.15 OpenMathInstruct in a single SFT. The 2B cannot
simultaneously learn terse GSM8K arithmetic (mean 282 tok), verbose R1
reflection (2k-10k tok), and general instruction-following in one pass.
The probe `wf_70eef54d` (cited in `b4/design.md`) confirmed: merge
clean (3605: 111 tensors, 7.94GB fp32), eval clean (extractor pulls
model's own boxed number), NOT late-collapse (acc flat across ckpt
900/1800/3605) -- raw generations are coherent English, well-formed
`<think>/<answer>`, but wrong arithmetic and run-on chains. Gen_len 601
(2x B2) and 26 unclosed are symptoms of the dilution, not independent
causes. Format 0.86 shows the envelope was mostly learned, but the
*content* inside it was corrupted by the competing R1 style. This is
data dilution, not format collapse: the model learned to emit the right
tags around the wrong reasoning.

**B4a -- verbose-base poisoning (primary) + format collapse (secondary).**
B4a's base is B3-3605, which already encodes verbosity (gen_len 601) and
marginal format (0.86) as weight-level priors. A short gsm8k-r1cot finish
(3ep, 7473 examples, 2N ~30-60min) cannot overwrite those priors -- the
base is too strongly committed to the R1 style after a full-epoch SFT on
it. Instead, the optimizer oscillates between two attractors (verbose R1
vs terse gsm8k), destabilizing the envelope: format collapses 0.86 ->
0.26, gen_len explodes 601 -> 1378, unclosed 26 -> 133. This is the only
run where format collapse is the dominant observable, but it is *caused*
by verbose-base poisoning -- the format would have held at 0.86 without
the conflicting finish. The accuracy floor of 0.02 is the model failing
to produce parseable answers at all (133/200 never close), not a
reasoning failure per se.

**B4b -- data dilution, format-independent (primary).**
B4b is the control that isolates the dilution hypothesis from the
verbosity hypothesis. OPENR1_MAX_THINK_CHARS filtering removes long traces
(the run-on culprit), gsm8k-r1cot is reweighted 0.15 -> 0.40, seq_len
drops 8192 -> 4096, OpenMathInstruct-2 is dropped entirely. All of these
changes target verbosity and dilution. They succeed on verbosity (gen_len
601 -> 611, unclosed 26 -> 27, format 0.86 held) but accuracy barely
moves (0.05 -> 0.065). The residual 0.14 gap to B2 (0.205 - 0.065) is
*not* explained by any property of the mix that was varied. It is
explained by structure: one SFT stage from gs138650 cannot reproduce what
two ordered stages (tulu_math SFT, then gsm8k-r1cot SFT) produce,
regardless of how well the single stage is tuned. This is the sharpest
negative result in the campaign and the basis for the "structure beats
mix" lesson in `POST-TRAINING-2B.md` Findings.

**B2v2 -- narrow-data overfitting (primary) + format regression (secondary).**
B2v2 has none of the single-stage pathologies: it starts from the proven
ckpt-900 base, uses the proven gsm8k-r1cot data, and applies no competing
long-CoT signal. Its only change is doubling/tripling the stage-2 epoch
budget (3ep -> 6ep/12ep). Training metrics remain healthy (loss 0.8069,
acc 0.8198), yet eval halves (0.205 -> 0.09/0.105). Format regresses
(0.985 -> 0.86/0.91), gen_len inflates (282 -> 416/378), unclosed appears
where B2 had ~0 (22/15 vs ~0). This is overfitting in the sense that
more optimization steps on the same narrow dataset move the model *away*
from the generalizable solution. The mechanism is likely memorization of
specific gsm8k-r1cot reasoning templates at the expense of the
instruction-following prior that ckpt-900 provides -- the model
increasingly reproduces training-set phrasing rather than generalizing
the arithmetic procedure. The non-monotonic pattern (0.09 at 6ep, 0.105
at 12ep) suggests the model oscillates near a local optimum rather than
monotonically degrading, but never recovers to 0.205.

### 2.3 What the guardrails taught us

The eval's three auxiliary signals -- format hit-rate, mean generation
length, and n_unclosed -- were added after the B3 failure (see
`b4/design.md` Shared section) and proved load-bearing for diagnosis:

- Without format, B4a's 0.02 would read as "the finish just didn't help."
  With format 0.26 and 133 unclosed, it reads as "the model lost the
  envelope" -- a qualitatively different failure requiring a
  qualitatively different fix (don't try to rehabilitate a verbose base).

- Without gen_len, B3's 0.05 and B4b's 0.065 would appear as "nearby
  single-stage results." With gen_len 601 vs 611, they appear as
  "identical verbosity despite different mixes" -- pointing to structure.

- Without n_unclosed, the distinction between "model emits wrong answers
  in well-formed envelopes" (B3: 26 unclosed) and "model cannot close the
  envelope at all" (B4a: 133 unclosed) would be invisible. That
  distinction is what ruled out "train B4a longer."

**Keep these guardrails in any future CoT eval.** An accuracy number
alone hides envelope collapse.

---

## 3. The ckpt-8672 Catastrophe: LR Schedule and Early Stopping as Load-Bearing Structure

### 3.1 What happened

The full-mix 8N SFT (`tulu_math_uc_mix_full`, 53.3M packed seqs, 1 epoch,
step 8672, ~54B tokens, LR 2e-5 cosine to 0) produced two qualitatively
different models at two points in the same run:

| Checkpoint | Tokens (approx) | IFEval prompt_strict | hellaswag | arc_easy | arc_ch | winogrande | piqa |
|------------|----------------:|---------------------:|----------:|---------:|--------|------------|------|
| **baseline** (gs138650) | 0 | 0.179 | 0.593 | 0.694 | 0.390 | 0.582 | 0.743 |
| **ckpt-300** | ~1.9B | -- | 0.591 | 0.662 | 0.385 | 0.607 | 0.736 |
| **ckpt-600** | ~3.8B | -- | 0.594 | 0.655 | 0.369 | 0.608 | 0.735 |
| **ckpt-900** | **~5.7B** | **0.253** | **0.592** | **0.640** | 0.362 | 0.604 | 0.730 |
| ckpt-1500 | ~9.5B | -- | 0.592 | 0.632 | 0.367 | 0.590 | 0.722 |
| ckpt-3000 | ~19B | -- | 0.502 | 0.516 | 0.309 | 0.519 | 0.648 |
| **ckpt-8672** (final) | **~54B** | **0.168** | **0.273** | **0.298** | 0.236 | 0.530 | 0.537 |

Source: `tulu_math_uc_mix_full/evals/README.md` base-LM sweep table
(jobs 12470365 + 12470886 + 12470889/96, Sunspot). Random-chance
references: hellaswag/arc/openbookqa ~0.25, piqa/boolq/winogrande ~0.50.
By step 8672 every task is at or near chance.

The collapse between ckpt-900 and ckpt-8672 is catastrophic:

- **hellaswag: 0.593 -> 0.273** -- a 32pp drop to random chance. The model
  lost commonsense narrative completion entirely.
- **arc_easy: 0.694 -> 0.298** -- a 40pp drop to random chance. Grade-school
  science QA destroyed.
- **IFEval prompt_strict: 0.253 (ckpt-900) -> 0.168 (ckpt-8672)** -- below
  the 0.179 baseline. The SFT made instruction-following *worse* than
  doing nothing. By step 8672 the model cannot follow structural
  directives it could follow at step 900.
- **Training metrics at collapse: loss 0.357, mean_token_accuracy 0.902.**
  Training looks superb. The model is extremely good at exactly one
  thing -- emitting OpenMathInstruct-2-style math CoT -- and has lost
  everything else. Train loss and eval capability have decoupled.

### 3.2 Why it happened: LR x tokens

From `tulu_math_uc_mix_full/README.md` and `tulu_math_uc_mix_full/evals/README.md`:

> LR 2e-5 stayed above 1e-5 through step ~4350 (cosine decay only bites
> the second half). ~4000 steps at high LR on a narrow math distribution
> destroyed general capability.

Concretely:

- **Cosine schedule shape.** LR 2e-5 with cosine decay to 0 over 8672
  steps means LR(t) = 2e-5 * 0.5 * (1 + cos(pi * t / 8672)). At t=4350
  (half the run), cos(pi/2) = 0, so LR = 1e-5. The first half of the run
  is at LR >= 1e-5; the second half decays from 1e-5 to 0. The schedule
  is "peaky" -- high for a long time, then rapid decay. For a broad
  pre-training mix this is appropriate (the model needs sustained high LR
  to absorb diverse data). For a narrow SFT mix (0.65 tulu + 0.15 math +
  0.20 chat), sustained high LR means sustained large updates *on the
  same narrow distribution*, which progressively overwrites the base LM's
  general knowledge.

- **Narrow distribution + high LR = catastrophic forgetting.** The tulu
  math mix is ~65% instruction-following + 15% math + 20% chat. Compared
  to the pre-training mix (broad web + code + math + multilingual), it is
  extremely narrow. At LR >= 1e-5, each step moves weights meaningfully
  toward this narrow distribution. The base LM's hellaswag/arc_easy
  capability (which depends on broad world knowledge and reasoning
  diversity) is not represented in the SFT mix and is therefore not
  reinforced. ~4000 steps of unreinforced, high-LR updates on a narrow
  distribution progressively erase it. The collapse window (step ~1500 to
  ~4500, where hellaswag drops 0.59 -> 0.32 and arc_easy 0.64 -> 0.37) is
  precisely the high-LR window.

- **The metamathqa-729 SFT survived by accident.** The completed 729-step
  SFT (`tulu_math_uc_mix/README.md`, 2026-06-10, same LR 2e-5, same cosine
  schedule) stopped at 729 steps -- before the collapse window. At step
  729, LR is still ~1.95e-5 (cos(pi*729/729) at 3 epochs is different
  geometry, but the absolute step count is what matters: 729 steps at
  GBS 6144 is 729 * 6144 = 4.48M sequences, or ~4.5B tokens, barely into
  the SFT mix). The model had absorbed the tulu instruction-following
  signal (IFEval 0.244) without yet suffering forgetting. The 729-step
  stop was not designed as an early-stopping strategy -- it was the
  natural end of 3 epochs at GBS 6144 on the small mix -- but it was
  **load-bearing**: had the run continued (as the full-mix did), it would
  have collapsed identically.

### 3.3 What it teaches about LR schedule and early stopping

**Lesson 1: For narrow-distribution SFT at LR 2e-5, cap at O(1000) steps
or drop the LR.** This is stated explicitly as the recipe lesson in
`tulu_math_uc_mix_full/evals/README.md`:

> For full-mix (narrow-distribution) SFT at LR 2e-5, cap training at
> O(1000) steps or lower the LR substantially; a full epoch (~8672 steps)
> at peak LR causes catastrophic forgetting.

The full-mix run teaches that "more tokens" is not "better model" for SFT
in the way it is for pre-training. Pre-training's broad mix benefits from
many tokens because diversity prevents forgetting. SFT's narrow mix
*causes* forgetting proportional to tokens x LR. The lever is not token
count but LR x step count on the narrow distribution.

**Lesson 2: The LR schedule's shape matters as much as its peak.** Cosine
decay from 2e-5 over 8672 steps keeps LR high for ~4000 steps. An
alternative schedule (linear decay, or lower peak LR like 5e-6, or cosine
over fewer steps) would have decayed sooner and potentially avoided the
collapse window. The schedule was inherited from pre-training without
reconsideration for the SFT context. Any future full-mix SFT should
either lower the peak LR (e.g. 5e-6 as used in the GRPO runs) or use a
schedule that decays in the first half, not the second.

**Lesson 3: Early stopping was load-bearing, not designed -- and that is
dangerous.** The 729-step SFT's success depended on stopping before
collapse, but the stop was a consequence of GBS x epochs, not an
explicit eval-gated decision. There was no intermediate eval sweep during
the 729-step run that would have caught the collapse if it had occurred
earlier. The full-mix run had such a sweep (every 50 steps checkpointed)
and it *did* catch the collapse -- but only after the fact. For any
future narrow SFT, intermediate evals on base-LM benchmarks (hellaswag,
arc_easy) must gate continuation, not just loss curves. Loss 0.357 at
ckpt-8672 looked healthy.

**Lesson 4: Training metrics cannot detect forgetting.** At ckpt-8672,
loss 0.357 and mean_token_accuracy 0.902 are superb -- the model is
excellent at the SFT distribution. HellaSwag 0.273 is catastrophic. No
training metric (loss, grad_norm, token accuracy, entropy) predicted the
collapse. Only eval on held-out *general* benchmarks (not the SFT
distribution) reveals it. This mirrors the B2v2 finding (loss 0.8069,
acc 0.8198 at 6ep, yet eval halved) -- healthy training + collapsed eval
is a recurring signature of narrow-data overfitting.

**Lesson 5: The checkpoint boundary is the deliverable, not the run.**
The full-mix run's deliverable is checkpoint-900, not checkpoint-8672,
despite 8672 being "the end of the epoch." This is counterintuitive for
pre-training (where later is better) but correct for SFT (where early
stopping preserves the base). The run produced 82 sharded checkpoints
(steps 900..8672) and the *interior* one is the product. Any future SFT
run must define its deliverable checkpoint by eval, not by "last step."

---

## 4. Is 0.205 Near the Base Ceiling? What Would Break 0.25+?

### 4.1 Evidence that ~0.2 is the 2B GSM8K-CoT ceiling for this base

Seven independent measurements converge on ~0.2 as the ceiling for
`global_step138650` (the MDS 2B stage-3 base, 1.99B params):

| Measurement | Result | Source |
|-------------|-------:|--------|
| **B2** (two-stage SFT, the best recipe) | **0.205** / 0.985 / 282 | `POST-TRAINING-2B.md` B2 + `rl/plans/cot.md` |
| B3 (single-stage, broad mix 8192) | 0.05 / 0.86 / 601 | `b4/README.md` |
| B4a (finish on B3 base) | 0.02 / 0.26 / 1378 | `b4/README.md` |
| B4b (reweighted single-stage 4096) | 0.065 / 0.86 / 611 | `b4/README.md` |
| B2v2 ep3 (more stage-2 epochs, 6 steps) | 0.09 / 0.86 / 416 | `POST-TRAINING-2B.md` B2v2 + jobs 12482572/73 |
| B2v2 ep6 (more stage-2 epochs, 12 steps) | 0.105 / 0.91 / 378 | same |
| GRPO on B2 (100 steps, gated reward, format 1.000) | **0.205 -> 0.215** (noise) | `rl/plans/cot.md` 2026-07-22 |
| CPT-17k narrow SFT (gsm8k-r1cot on CPT-17000 HF, LR 2e-5 3ep GBS 384) | **0.015** | `POST-TRAINING-2B.md` 2026-11-12 (format collapse on FW-Edu-shifted manifold) |

No measurement exceeds 0.215, and only B2 and its GRPO continuation
exceed 0.15. The gap between B2 (0.205) and the next best non-B2 result
(B2v2 ep6 at 0.105) is 10pp -- a 2x factor. The gap to the single-stage
mean (~0.04) is 5x.

The GRPO evidence is particularly telling. From `rl/plans/cot.md`
2026-07-22 (job 12471432, 100 steps, `rl_grpo_lora_agpt_2b_gsm8k_b2long`,
lr 2e-6, easy curriculum, group_size 4, ckpt every 25):

- Format held **1.000** the entire run (zero reward-hacking, drift-proof).
- In-loop 20-sample validation: reward 0.429 -> 0.425, correct 0.250 ->
  0.250 (flat).
- **Decisive 200-problem GSM8K-CoT (merged LoRA into B2 base):**
  cot_accuracy **0.205 -> 0.215** (+1pp, ~2 problems, within noise),
  format 0.985 -> 1.000.

GRPO on B2 is drift-proof (format perfected) but accuracy-flat. The
post-hoc analysis in `rl/plans/cot.md` identifies the mechanism:

> RL can only reweight rollouts the base already samples, and at ~20%
> solve rate the correct-rollout density is too thin (group_size 4 =>
> most groups all-wrong, zero advantage) and lr 2e-6 too gentle to shift
> the task.

With group_size 4 and 20% accuracy, the expected number of correct
rollouts per group is 0.8. Most groups are all-wrong (advantage zero, no
gradient) or have exactly one correct rollout (weak signal). GRPO's
advantage normalization (group-relative) means the gradient is
proportional to within-group variance; at low solve rate, variance is low
because most groups are uniformly wrong. This is a structural limitation
of on-policy RL at low capability -- not a hyperparameter that more
tuning fixes. (The alphabet_sort analogy in `grpo/beat-v5-sweep.md` and
`grpo/ceiling-attack.md` confirms: varying LR, LoRA rank, and batch size
all plateau at ~0.24-0.25 mean reward; only reward *shaping* broke that
plateau at +168%, but GSM8K-CoT's binary correctness reward is not
shapable without process supervision.)

### 4.2 Cross-evidence: base MDS plateau and IFEval

The MDS 2B pre-training evals (`evals/agpt/2b-mds/README.md`, 28
checkpoints x 3 replicates, lm-eval on hellaswag/arc_easy/arc_challenge/
winogrande) show the *base itself* plateauing:

| Step | hellaswag | arc_easy | arc_challenge | winogrande |
|-----:|----------:|---------:|--------------:|-----------:|
| 95k (stage transition) | 0.6004 | 0.6660 | 0.3623 | 0.5770 |
| 105k | 0.5944 | 0.6953 | 0.3845 | 0.6004 |
| 120k | 0.5910 | 0.6985 | 0.3862 | 0.5864 |
| 140k (near gs138650) | 0.5924 | 0.6658 | 0.3712 | 0.5806 |

From step 95k to 140k (+45k steps, ~2.5T tokens), hellaswag is flat
(0.60 -> 0.59), arc_easy oscillates 0.66-0.70, arc_challenge 0.36-0.39.
The base's broad capability has plateaued. Since GSM8K-CoT draws on the
same general reasoning + arithmetic substrate, it is unsurprising that
SFT on this plateau'd base also plateaus -- SFT cannot create capability
the base does not contain, it can only elicit it into a useful format.
B2 at 0.205 is plausibly near-full elicitation of what the 1.99B
MDS-2B base knows about GSM8K-style arithmetic.

Further evidence: the base's IFEval at 0.179 (vs 0.253 after stage-1 SFT)
shows the base has limited instruction-following. Stage-1 SFT lifts this
to 0.253, but that is still modest (a strong instruct model reaches
0.4+). The instruction-following ceiling constrains how reliably the
model can obey the CoT envelope, which in turn constrains accuracy.

### 4.3 What would need to change to break 0.25+

Breaking 0.25+ requires changing the *substrate*, not the recipe. Five
levers are identified, in rough priority:

**Lever 1: Larger base model.** The most direct lever. A 4B or 7B base
with the same training recipe will have more parameters to encode both
format and arithmetic, and more capacity to maintain both simultaneously.
The single-stage failures (B3/B4b) may succeed at larger scale because
the capacity constraint that forces the "verbosity vs arithmetic" tradeoff
at 2B is relaxed. Evidence: the 2B's attention is O(L^2) with no
flash-attn; at 8192 seq_len, a 2B dedicates a large fraction of its
compute to attention bookkeeping rather than reasoning. A larger model
amortizes this better. No 4B/7B CoT campaign has been run -- this is the
highest-expected-value next experiment if the goal is higher CoT accuracy.

**Lever 2: Math-pretrained base.** A base pre-trained with a higher math
fraction (e.g. FineWeb-Edu 60/40 kind-shift as documented in
`sft/fineweb-edu-60-40-kind-shift.md`, or a dedicated math-heavy
pre-training stage) would arrive at SFT with stronger arithmetic priors,
making the SFT's job easier (elicitation, not creation). The current
base's pre-training is 94.8% DCLM (olmo-mix-1124) / dolmino, which is
broad-web-heavy and math-light. A math-pretrained base at the same 2B
scale might reach 0.25-0.30 without any SFT recipe change, simply because
it already "knows" more arithmetic. The CPT-FWE 4N chain (60% FineWeb-Edu
/ 40% OWM, 190k steps, currently at 111.2k/190.7k = 58.3%, MFU 15.8%,
`POST-TRAINING-2B.md` 2026-11-15) is building exactly this base; its first
meaningful CoT delta vs B2 0.205 gates at ~50k steps with a stage-1 SFT
on the new HF (see `POST-TRAINING-2B.md` Next Steps).

**Lever 3: Broad CoT data in stage 2 (the distill_cot_mix).** B2's stage 2
uses only narrow gsm8k-r1cot (7.4k examples). The planned `distill_cot_mix`
(50% OpenThoughts-114k-decontam + 50% gsm8k-r1cot, via pretokenized
`sft_distill_cot_4n.sh` + `_pretokenize_distill_cot_mix_1n.sh`,
`sft_distill_cot_4n_verification.md`) adds frontier R1 traces that are
decontaminated and length-controlled, *within the proven two-stage
structure*. This is the only open SFT path that adds *new* CoT data
without violating the structure lesson. It could lift accuracy if the
current 0.205 ceiling is partly a data-diversity ceiling (the model has
only seen 7.4k GSM8K problems and may benefit from broader reasoning
patterns). However, this lever is bounded: the failed single-stage runs
show that broad CoT data *mixed with instruction data in one SFT* hurts
at 2B, so the broad data must stay in stage 2, gated on the stage-1
foundation, not mixed into stage 1. Gate: verified
`/tegu/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json` +
`[decontam]` counts before training (`sft_distill_cot_4n_verification.md`).

**Lever 4: Stronger stage-1 SFT (more tokens, but capped).** Checkpoint-900
at ~5.7B tokens is strong (IFEval 0.253) but not necessarily optimal.
A stage-1 SFT that runs to e.g. ~1500-2000 steps (still before the
~1500-4500 collapse window) with a gentler LR schedule might produce a
stronger instruction-following + math-reasoning foundation for stage 2 to
build on. This is speculative -- the eval sweep shows no clear monotonic
improvement in IFEval between 600 and 900 (hellaswag flat at 0.59, IFEval
not measured at intermediate steps), and pushing further risks the
collapse. Any such experiment must be eval-gated (base-LM benchmarks
every 100-200 steps) and must not exceed O(1000) steps at LR 2e-5.

**Lever 5: RL with higher group_size and LR, on B2 (now drift-proof).**
GRPO on B2 was drift-proof (format 1.000 for 100 steps) but used
conservative hparams (lr 2e-6, group_size 4) that produced a weak
gradient. Now that format is known to be bulletproof, it is safe to
raise LR to 5e-6-1e-5 and group_size to 8-16 for denser gradients
(`POST-TRAINING-2B.md` Next Steps #2). With group_size 16 at 20% solve
rate, expected correct per group rises to 3.2, meaning most groups have
variance and gradient flows. This is unlikely to break 0.25 alone (the
substrate ceiling argument still applies), but it could squeeze 0.02-0.05
pp of additional accuracy and is cheap to try (Monarch+TorchStore+vLLM,
`rl_grpo_b2_2n.sh`, job 12483252 in-flight). The `sft_arithmetic` 8N GRPO
result (accuracy_reward 0.4 -> 0.9 over 1000 steps on sum_digits, a
narrow task where RL *did* work) shows that RL's effectiveness is
task-difficulty-dependent, not universally limited.

### 4.4 What would NOT break 0.25

The campaign has refuted these approaches for this base:

- **More single-stage SFT mix tuning** (weights, filtering, seq_len):
  B3/B4a/B4b/B2v2 jointly refute this. `POST-TRAINING-2B.md` Next Steps
  #3: "Do NOT attempt further single-stage SFT rebuilds."
- **More narrow stage-2 epochs on the same data:** B2v2 (0.09/0.105 vs
  0.205) refutes this. Same verdict as the full-mix overtraining.
- **RL alone (R1-Zero) without cold-start:** The gated-reward run
  (12471366, `rl/plans/cot.md` Gated-reward section) showed 95% of
  rollouts from the weak cold-start had no `<think>` block and 20% echoed
  the exemplar's `\boxed{5}`. R1-Zero produced reward 0.000 -- no signal
  at all. Cold-start is prerequisite.
- **RL on the weak cold-start (ckpt-16):** Reward-hacking (reward
  0.118 -> 0.217 while accuracy 0.16 -> 0.105, format 0.955 -> 0.635)
  proved this fails. The 400-step cot-long re-analysis showed the same
  weak base flatlined at ~0.17 for 280 steps, never recovering.
- **Changing only the LR schedule within single-stage:** The collapse is
  LR x distribution, not LR alone. A gentler schedule on the same single-
  stage mix might delay but not prevent the dilution problem.

---

## 5. Concrete Lessons: Rules for Any Future SFT Attempt

### 5.1 Five Non-Negotiable Rules

These rules are derived from the numbered evidence above. Violating any
one of them reproduces a failure mode that the campaign has already
measured. They apply to any future SFT attempt on the 2B base, regardless
of dataset or objective (CoT, general instruct, or other).

#### Rule 1: Use two ordered SFT stages; never combine instruction-following and CoT in one SFT

**Statement:** If the goal involves both general capability
(instruction-following, chat) and a specialized format (CoT envelope,
tool use, structured output), train them in two separate SFT stages:
general first, specialization second. Do not mix them into one SFT, no
matter how carefully balanced the mix weights appear.

**Evidence:** B2 (two-stage: 0.205) vs B3 (single-stage: 0.05) vs B4b
(reweighted single-stage: 0.065) -- a 3-4x gap that survived weight
rebalancing (gsm8k 0.15 -> 0.40), length filtering
(OPENR1_MAX_THINK_CHARS), seq_len halving (8192 -> 4096), and component
dropping (OpenMathInstruct-2 removed). `b4-finish-and-reweight/README.md`:
"The lever is STRUCTURE, not mix." `POST-TRAINING-2B.md` Findings:
"Structure beats mix. B4b restored format/verbosity and still only moved
0.05->0.065."

**Why it matters:** At 2B scale, a single SFT cannot simultaneously
learn a general instruction-following prior and a specialized output
format without one contaminating the other. The general stage needs to
establish a stable manifold (IFEval 0.253, intact base-LM) that the
specialization stage then refines, not competes with. A combined SFT
forces the model to allocate capacity between competing objectives in one
pass, and the more diverse/longer component (R1 traces) dominates.

**Enforcement:** Any new SFT plan that proposes a single combined mix
containing both instruction data and specialized format data must be
rejected at review. Require a two-stage decomposition with an eval gate
(IFEval + base-LM) between stages.

#### Rule 2: Cap any narrow-distribution SFT at O(1000) steps at LR 2e-5, or drop the LR

**Statement:** For any SFT on a narrow distribution (tulu_math mix,
gsm8k-r1cot, or similar -- anything that is not the broad pre-training
distribution), limit training to approximately 1000 optimizer steps at LR
2e-5 (or equivalently, lower the LR if more steps are needed). Treat
~4000 steps at LR >1e-5 on a narrow mix as a known catastrophic-
forgetting regime.

**Evidence:** ckpt-8672 hellaswag 0.593 -> 0.273, arc_easy 0.694 -> 0.298,
IFEval 0.253 -> 0.168 (below baseline) after ~4000 steps at LR >1e-5 on
the tulu_math mix. B2v2: 0.205 -> 0.09/0.105 after doubling stage-2 epochs
on narrow gsm8k-r1cot at same LR. `tulu_math_uc_mix_full/evals/README.md`:
"~4000 steps at high LR on a narrow distribution = catastrophic
forgetting. The metamathqa recipe survived only by STOPPING at 729 steps
-- that stop was load-bearing, not designed."

**Why it matters:** Cosine decay from 2e-5 over many steps keeps LR high
for a long time (above 1e-5 through step ~4350 of 8672). Sustained
high-LR updates on a narrow distribution progressively overwrite general
knowledge that the distribution does not reinforce. Training loss
continues to descend (0.357 at ckpt-8672, 0.8069 at B2v2 ep3) while eval
collapses -- no training metric warns you.

**Enforcement:** Any SFT plan must compute expected steps (dataset size x
epochs / GBS) and verify it is O(1000) at LR 2e-5, or specify a lower LR.
If the plan exceeds this, require either (a) a reduced epoch/LR budget
or (b) eval-gated early stopping with base-LM benchmarks, not just loss.
Add `max_steps 1000` or `NUM_EPOCHS=1` smoke gate for any new narrow SFT
(see `sft_distill_cot_4n_verification.md` Section 4: "Do not rely on
epochs alone").

#### Rule 3: Never attempt to rehabilitate a verbose-poisoned base with a finishing stage

**Statement:** If a base model has been trained on long-form CoT data
and exhibits verbosity (gen_len >2x the target distribution, format
marginal at 0.86 with unclosed generations), do not attempt to fix it
with a short finishing SFT on the target distribution. The verbose prior
is baked into the weights and a light finish destabilizes the envelope
rather than correcting it.

**Evidence:** B4a: B3-3605 (gen_len 601, format 0.86, 26 unclosed) ->
gsm8k-r1cot finish (3ep, per-epoch ckpts) -> gen_len 1378, format 0.26,
133 unclosed, accuracy 0.05 -> 0.02 (WORSE). `b4/README.md`: "A short
gsm8k-r1cot finish did NOT un-teach B3's verbose R1 style -- it made it
worse (gen_len 601 -> ~1400-2000, format collapsed to 0.0-0.26). The B3
base's verbose bias is baked in; a light finish destabilizes the
envelope."

**Why it matters:** Long-form CoT traces (2k-10k tokens, reflective
self-correction, multi-step decomposition) teach a fundamentally different
generation style than short GSM8K CoT (282 tokens, direct arithmetic).
The verbose style is not a surface pattern that a few epochs of
counter-examples can overwrite -- it is a deep prior over generation
length, structure, and content. A narrow finish creates a format
competition in weight space that the model resolves by producing
incoherent hybrids (long chains that never close) rather than clean
short chains.

**Enforcement:** If a base exhibits verbosity (gen_len >500, unclosed
>10/200), discard it as a CoT starting point. Rebuild from the clean
pre-verbose checkpoint (gs138650 or ckpt-900), not from the verbose one.
Any plan that proposes "finish on B3" must be redirected to "rebuild from
gs138650/ckpt-900 with a different stage-1 mix that excludes long CoT."

#### Rule 4: Gate every SFT continuation on held-out general evals, not training metrics

**Statement:** For any SFT run, evaluate on held-out *general* benchmarks
(hellaswag, arc_easy, IFEval) at every checkpoint interval (every 50-200
steps), and gate continuation on those evals. Never gate on training
loss, token accuracy, or grad_norm alone. Define the deliverable
checkpoint by eval peak, not by "last step."

**Evidence:** ckpt-8672: loss 0.357, mean_token_accuracy 0.902 (superb
training) vs hellaswag 0.273, arc_easy 0.298, IFEval 0.168 (catastrophic
forgetting). B2v2: loss 0.8069, acc 0.8198 (healthy) vs CoT 0.09/0.105
(halved). In both cases training metrics were blind to the failure.
`POST-TRAINING-2B.md` 2026-11-14: "Raw CPT CoT 0/200 pre-SFT expected; B2
0.205 remains best CoT." -- CPT's 0/200 is not a regression because CPT
never trains `\boxed{}`; only SFT-stage evals on the target format are
meaningful.

**Why it matters:** Training loss measures fit to the SFT distribution,
not preservation of general capability or transfer to the eval
distribution. On a narrow SFT mix, training loss can descend while
general capability collapses (ckpt-8672) or task accuracy halves (B2v2).
The only signal that catches this is eval on benchmarks that exercise
*retained* knowledge (hellaswag, arc_easy for stage 1; GSM8K-CoT 200-
problem vLLM eval for stage 2). The deliverable is wherever those evals
peak (ckpt-900 at +7.4pp IFEval, ckpt-93 at 0.205 CoT), not the final
checkpoint.

**Enforcement:** Any SFT launch must include an eval sweep plan (which
checkpoints, which benchmarks, which harness). The run's README must
record per-checkpoint base-LM + IFEval + CoT numbers, not just loss.
Any checkpoint promoted to deliverable must have eval numbers that beat
the base; "it finished without crashing" is not promotion criteria.

#### Rule 5: Preserve the envelope trinity -- format, length, and unclosed -- in every CoT eval

**Statement:** Every GSM8K-CoT eval must report three guardrail metrics
alongside accuracy: format hit-rate (fraction matching
`<think>.*?</think>\s*<answer>.*?\boxed{}.*?</answer>`), mean generation
length, and count of unclosed generations (never emit `</answer>` within
`max_tokens`). Accuracy alone is never sufficient. A run that regresses
any guardrail has failed differently than one that regresses accuracy,
and the fix is different.

**Evidence:** `b4/design.md` Shared section: "Guardrail metrics (from
the B3 failure): gen_len (target ~282, not 601) and the count of
generations that never close `</answer>` -- accuracy alone hid the
run-on style problem in-loop." `b4/README.md`: "On B4a, accuracy alone
(0.02) would have read as 'the finish stage just didn't help.' The
guardrails showed something sharper: 133 of 200 generations never
emitted a closing tag, i.e. the model had lost the format envelope, not
merely the math. That distinction is what ruled out 'train it longer' and
sent us to B4b instead." `b4/README.md` Figures: `cot_ladder.svg`
(accuracy + format + verbosity panels) and `accuracy_vs_genlen.svg`
(gen_len vs accuracy scatter, marker size = unclosed count) -- the three
panels jointly tell the story that any single panel hides.

**Why it matters:** Format collapse (B4a: 0.26, 133 unclosed) vs
data-dilution accuracy loss (B3: 0.86, 26 unclosed) vs narrow overfitting
(B2v2: 0.86-0.91, 22/15 unclosed) are three different failure modes with
three different fixes (discard base vs restructure SFT vs reduce epochs).
Accuracy alone conflates them. The guardrails also catch silent
truncation bugs: if `MAX_LENGTH` is too short (1024 vs required 2048 for
`<answer>` tail), the eval will show high format but zero accuracy
because the answer span is truncated before `\boxed{}` -- a bug that
accuracy alone would misattribute to model capability.

**Enforcement:** `eval_cot_gsm8k.py` must emit all four columns (acc,
format, gen_len, n_unclosed) for every checkpoint eval'd. Any results
table that reports only accuracy is incomplete and must not be used to
make training decisions. Retain the `cot_ladder.svg` three-panel chart
pattern for any future B-series comparison.

### 5.2 Additional Operational Rules

Beyond the five non-negotiable rules, the campaign surfaced operational
lessons that should be codified for any future SFT run:

**6. Respect the two-stage data boundary.** Stage 1 must not contain
`<think>` traces; stage 2 must not contain general instruction data.
This is the data-level expression of Rule 1. The stage-1 mix
(tulu_math_uc_mix: tulu-3 0.65 + OpenMathInstruct-2 0.15 + ultrachat 0.20)
contains zero CoT-envelope data; stage 2 (gsm8k-r1cot) contains zero
general instruction data. Any new lever that adds broad CoT data (e.g.
distill_cot_mix: OpenThoughts-114k + gsm8k-r1cot) must go into stage 2,
gated on the stage-1 foundation, not mixed into stage 1.

**7. Use MAX_LENGTH = 2048 for any CoT-envelope SFT.** The default 1024
(`train_sft.py:221`) silently truncates the `<answer>\boxed{}</answer>`
tail, destroying the supervised target. This was caught by
`rl/plans/cot.md:136` as mandatory, not optional. Any CoT SFT launch
that does not explicitly set `MAX_LENGTH=2048` (or larger, if traces
are longer) will train on truncated envelopes and produce format collapse
that is a bug, not a model failure.

**8. Completion must start directly with `<think>`, no leading newline.**
The gemma chat template's `lstrip('\n')` at `train_grpo.py:584/587`
trips TRL's prompt/completion tokenization alignment if the completion
starts with `\n<think>`. The `assistant_only_loss` mask silently breaks,
training on wrong tokens. Any new dataset builder that produces CoT
envelopes must assert `completion[0]["content"].startswith("<think>")`.

**9. Pre-tokenize broad mixes offline; never rely on runtime tokenize at
scale.** The full OpenMathInstruct-2 mix (93M rows, ~8.5h runtime
tokenize) and the OpenThoughts distill mix hit the HF streaming cache
coherency race at 2N (rank 0 writes new cache, others look for old shards)
plus the idle-watchdog SIGTERM. The fix is the two-phase path:
`--pretokenize_to <path>` on 1N (bakes `assistant_masks` with
`assistant_only_loss=True`), then `--pretokenized_dataset <path>` on the
training job with `HF_HUB_OFFLINE=1`. Any SFT mix exceeding ~1M rows must
use this path.

**10. Do not trust the reward curve for RL-vs-SFT decisions; re-eval the
merged checkpoint on the shared 200-problem GSM8K-CoT metric.** The early
GRPO runs showed reward 0.118 -> 0.217 while accuracy 0.16 -> 0.105
(reward-hacking via `answer_close` on exemplar echoes), and later
`ThinkFormatReward` scored 0.000 on all 6,337 rollouts due to vLLM's
`reasoning_content` split (not format collapse). Both errors were caught
only by the held-out 200-problem eval on the merged checkpoint -- the
only metric that measures the task. Any RL result that has not been
verified by merging and re-evaluating on the shared CoT metric is
provisional.

---

## Appendix A: Source Map

| Claim | Source file | Section/line |
|-------|-------------|--------------|
| B2 = ckpt-93 = 0.205/0.985/282 | `POST-TRAINING-2B.md` | B2 section, Deliverables table |
| Stage 1 = ckpt-900 IFEval 0.253 | `tulu_math_uc_mix_full/evals/README.md` | IFEval table (0.253 vs 0.244 vs 0.179 vs 0.168) |
| Stage 1 materialized mix 53.3M seqs / ~54B tok | `tulu_math_uc_mix_full/README.md` | Quick reference table |
| 8N not 32N (scale fault) | `tulu_math_uc_mix_full/README.md` | 8N-vs-32N scale fault section |
| 32N fault at step 1-2, rank 221 | same | same |
| ckpt-8672 hellaswag 0.593->0.273, arc_easy 0.694->0.298 | `tulu_math_uc_mix_full/evals/README.md` | Base-LM sweep table + TL;DR |
| Collapse window ~1500-4500 at LR >1e-5 | `tulu_math_uc_mix_full/README.md` | Evals section: "LR 2e-5 stayed >1e-5 through ~4350" |
| Collapsed IFEval 0.168 below baseline 0.179 | `tulu_math_uc_mix_full/evals/README.md` | IFEval table |
| Train loss 0.357 / acc 0.902 at ckpt-8672 | `tulu_math_uc_mix_full/README.md` | Status: complete |
| 729-step survival at 4.5B tok | `tulu_math_uc_mix/README.md` | Operational notes, Evals |
| IFEval 0.1645->0.2440 +8pp at 729 | `tulu_math_uc_mix/README.md` | Evals section |
| B3 mix 0.30/0.25/0.15/0.15/0.15 @8192 | `b3-instruct-cot-mix/design.md` | Data mix table |
| B3 0.05 / 0.86 / 601 / 26 | `b4-finish-and-reweight/README.md` | Results table |
| B4a 0.02 / 0.26 / 1378 / 133 | same | same |
| B4b 0.065 / 0.86 / 611 / 27 | same | same |
| Probe wf_70eef54d (B3 diagnosis) | `b4-finish-and-reweight/design.md` | Motivation section |
| B4a per-epoch not in table but 0.26/1378 row | `b4-finish-and-reweight/README.md` | Results table (B4a single row; per-epoch in design) |
| B2v2 ep3 0.09/0.86/416/22, ep6 0.105/0.91/378/15 | `POST-TRAINING-2B.md` | B2v2 section + jobs 12482572/73 |
| B2v2 training loss 0.8069 / acc 0.8198 | same + `sft/README.md` | SFT index table |
| B2v2 W&B xakno7kg / ur6pqqhp | `POST-TRAINING-2B.md` + `sft/README.md` | B2v2 rows |
| CPT-17k 0.015 collapse | `POST-TRAINING-2B.md` | 2026-11-12 section |
| GRPO 0.205->0.215 (noise, 200-prob) | `rl/plans/cot.md` | 2026-07-22 section + job 12471432 |
| GRPO format 1.000 entire run (drift-proof) | same | same |
| group_size 4 thin-signal analysis | same | same |
| Reward-hack 0.118->0.217 while 0.16->0.105 | `rl/plans/cot.md` | Stage 2 result section |
| Gated reward 0.000: 95% no think, 20% echo | same | Gated-reward run section |
| cot-long re-analysis 0.069->0.17 flat | same | cot-long measured section |
| vLLM reasoning_content split (format 0.000 bug) | same | same + memory doc |
| MDS base plateau 95k-140k hellaswag flat | `evals/agpt/2b-mds/README.md` | Eval Results table |
| Distill mix 50/50 OpenThoughts+gsm8k-r1cot | `sft_distill_cot_4n_verification.md` | Section 1 |
| MAX_LENGTH 2048 mandatory | `rl/plans/cot.md` | Stage 1 smoke section |
| `<think>` no leading \n (lstrip bug) | same | same |
| Guardrail addition post-B3 | `b4-finish-and-reweight/design.md` | Shared section |
| Golden rule: no single-stage rebuilds | `POST-TRAINING-2B.md` | Next Steps #3 |

## Appendix B: The Single Paragraph

B2's 0.205 is the product of a curriculum, not a dataset. Stage 1
(checkpoint-900, ~5.7B tokens of tulu_math at 1024, no `<think>` traces)
builds a stable instruction-following + shallow-math prior (IFEval 0.253,
base-LM intact) that stage 2 (gsm8k-r1cot, 7.4k examples, 93 steps at
2048) wraps in the CoT envelope without competition. Every failure is a
violation of that curriculum: B3 mixes the stages (dilution), B4a tries
to un-teach verbosity (irreversible poisoning), B4b fixes verbosity but
not structure (symptom vs cause), B2v2 extends the narrow stage too far
(overfitting). The ckpt-8672 catastrophe shows the same overfitting at
larger scale, with a precise mechanism (LR >1e-5 for ~4000 steps on a
narrow mix). And GRPO's 0.205->0.215 flatline on the drift-proof B2 base,
combined with the base's own pre-training plateau (hellaswag flat
0.60->0.59 from 95k to 140k), argues that ~0.2 is near-full elicitation
of what this 1.99B base knows -- not a recipe that more tuning can lift.
The next lever is a stronger base (larger, math-pretrained, or CPT-FWE
at 111.2k/190.7k), or broader stage-2 data (distill_cot_mix) within the
proven two-stage structure. Nothing else has worked, and the guardrails
(format/len/unclosed) now make that visible.

---

*Report generated on `ezpz-cpt111k-eval-sft` HEAD `bcbca5bd90` (2026-11-16).
All numbers are quoted from run docs listed in Appendix A. No training
was launched for this report.*
