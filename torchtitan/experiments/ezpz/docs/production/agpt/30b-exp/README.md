# agpt 30B-exp -- a proposed next flagship

> **Last updated: 2026-08-14.**
>
> **Status: PROPOSAL. Nothing here has been submitted or run.** This is a
> design document, written the day the 2B-512 canonical chain completed its
> full 4.674T olmo-mix-1124 budget. It argues for what the *next* flagship
> should be, given what that campaign actually measured.
>
> Read the [Evidence](#1-evidence-what-this-campaign-actually-measured)
> section first. Every claim there is a measurement from our own runs with a
> pointer to the doc it came from. Section 6 lists what is judgment rather
> than measurement, and Section 7 lists what would falsify the plan.

## 0. The one-paragraph version

Train **one ~30B dense model on ~10T tokens of a rebuilt mix**, not another
2B/20B/80B ladder on olmo-mix. The 2B campaign showed the corpus cannot
produce MMLU at any token count we can afford, that the last ~11% of the
budget bought nothing measurable, and that a 2B at 512 nodes runs at ~10%
MFU. Those are three separate multipliers being left on the floor: data
composition, budget allocation, and hardware efficiency. Fixing the third
alone is worth ~3x compute. Gate every mix on a 1B proxy *before* spending
production tokens -- the real failure of this campaign was not the mix, it
was taking 4.674T tokens to discover the mix was wrong.

---

## 1. Evidence: what this campaign actually measured

### 1.1 MMLU never left chance -- and it is the DATA

| model | tokens | MMLU | RoPE-corrected? |
|---|---|---|---|
| 20B-256 | 0.39T | 0.2599 | complex-era, clean |
| 20B-512 | 0.71T | 0.2655 | complex-era, clean |
| 2B-512 | 3.99T | 0.2473 | **wrong permute** |
| 2B-256 **complete** | 4.674T | 0.2437 | never switched, clean |
| **2B-512 complete** | **4.674T** | ~~0.2511~~ **0.2579** | **MEASURED corrected** |
| 2B MDS dolmino | 7.064T | 0.2413 | different codebase, clean |
| MDS stage3-mix | 7.771T | 0.2463 | clean |
| MDS stage3 math/code | 7.771T | 0.2591 | clean |

> **RoPE correction (2026-08-16).** Only the two 2B-512 rows were affected --
> that chain switched convention at step 30,401 and its evals were converted
> with the complex flavor. The corrected endpoint is **0.2579**, up 0.007.
> Every other row is either from a complex-era checkpoint (converted
> correctly), a chain that never switched, or a different codebase.
> **The conclusion is unchanged and now rests on a corrected number**: 0.2579
> is still inside noise of the 0.25 four-way floor. The 3.99T row is still
> uncorrected (job `8760307` covers 41,000-46,429 only).
>
> One earlier claim retracted: I wrote that the corrected endpoint "remains
> below `MDS stage3 math/code` at 0.2591". Corrected MMLU across the four
> steps spans **0.2579-0.2625**, straddling that value. Everything here is
> ~0.008-0.013 from the 0.25 floor against a binomial SE of ~0.0037 at
> N=14,042, so **no ordering among these configurations is meaningful** --
> the honest statement is that they all sit at chance and cannot be ranked.

~12 configurations, a 20x token span, two model scales, five data
treatments. No upward trend -- the most-trained model scores among the
lowest. Meanwhile the same checkpoints improve monotonically on everything
else (HellaSwag 0.405 -> 0.561 across the completed 2B run).

**The harness is clean.** Job 8736838 ran three cached public models through
the identical `simple_evaluate(num_fewshot=5, device="xpu:0")` path:
**Llama-3.2-1B = 0.3121** (published ~0.32) and **Llama-3.1-8B = 0.6530**
(published ~0.66). A 1B model clears chance on this exact code path; our 2B
at 4.674T does not.

Eliminated in order: tokens (a COMPLETED 4.674T run finished at chance),
scale (7.06T with our best-ever ARC-C still 0.2413), tokenizer (gemma-7b
assets against gemma-tokenized data), harness (above). What remains is the
mix: `olmo-mix-1124` carries little multiple-choice academic content.

### 1.2 The last 500B tokens bought nothing

> [!CAUTION]
> **The numbers in this table were measured on WRONGLY-PERMUTED exports.**
> The 2B-512 chain switched RoPE convention at step 30,401, and every eval
> after that was converted with `--model_flavor 2b` (complex) against cos_sin
> weights. MEASURED correction at the endpoint (job `8760307`, same
> checkpoint, correct `2b_real` flavor):
>
> **CORRECTED table, all four steps** (job `8760307`, `--model_flavor 2b_real`,
> same checkpoints and harness). Published values in parentheses:
>
> | step | tokens | MMLU | ARC-C | HellaSwag |
> |---|---|---|---|---|
> | 41,000 | 4.13T | **0.2617** (0.2501) | **0.2892** (0.2474) | **0.5344** (0.4785) |
> | 43,000 | 4.33T | **0.2625** (0.2516) | **0.2961** (0.2389) | **0.5376** (0.4775) |
> | 45,000 | 4.53T | **0.2592** (0.2498) | **0.3012** (0.2398) | **0.5386** (0.4764) |
> | **46,429** | **4.674T** | **0.2579** (0.2511) | **0.2978** (0.2381) | **0.5384** (0.4753) |
>
> Mean correction: MMLU **+0.009**, ARC-C **+0.059**, HellaSwag **+0.060**.
>
> **The section's finding is confirmed on corrected data across the whole
> window**, not merely preserved by shared bias: HellaSwag moves 0.004 across
> ~550B tokens, ARC-C 0.012 (and non-monotonically), MMLU drifts *down*
> 0.2617 -> 0.2579. The last 500B tokens really did buy nothing.
>
> **The section's conclusion survives** -- MMLU is still at chance (0.2579 is
> inside noise of 0.25), and the flatness across the final 11% of the budget
> is a within-table comparison where every row carries the same bias. But the
> absolute capability numbers are ~6 points too low on the tasks the model
> actually learned. Full account:
> [`20260816-arc-c-decay-vs-rope-permute.md`](../../../experiments/agpt/aurora/20260816-arc-c-decay-vs-rope-permute.md).
> Corrected rows for 41,000 / 43,000 / 45,000 pending in job `8760307`.

Final eval of the completed chain (job 8754664, 2026-08-14):

| step | tokens | mmlu | arc_c | hellaswag | arc_e | wino | piqa | obqa | boolq | gsm8k |
|---|---|---|---|---|---|---|---|---|---|---|
| 41,000 | 4.13T | 0.2501 | 0.2474 | 0.4785 | 0.6048 | 0.5359 | 0.7018 | 0.3160 | 0.5560 | 0.0000 |
| 43,000 | 4.33T | 0.2516 | 0.2389 | 0.4775 | 0.5993 | 0.5193 | 0.7024 | 0.3240 | 0.5630 | 0.0000 |
| 45,000 | 4.53T | 0.2498 | 0.2398 | 0.4764 | 0.5985 | 0.5288 | 0.6980 | 0.3260 | 0.5489 | 0.0000 |
| **46,429** | **4.674T** | **0.2511** | **0.2381** | **0.4753** | **0.6006** | **0.5233** | **0.6997** | **0.3220** | **0.5538** | **0.0000** |

Every metric flat within noise across the final ~11% of the budget. This is
what a run that has saturated its data mix looks like. It is not a defect --
but it means ~500B tokens of 512-node time produced no measurable capability.

### 1.3 Throughput collapses with scale -- the biggest single multiplier

2B, torch 2.13, compiled:

| nodes | TPS/GPU | MFU |
|---|---|---|
| 2 | 7,142 | 27.6% |
| 4 | 7,068 | 27.3% |
| 16 | 6,995 | 27.0% |
| 64 | 6,702 | 25.9% |
| 256 | 2,500 | 9.5% (torch 2.10; **18.77% on current 2.13** -- see below) |
| **512 (production)** | **~1,478** | **~9-11%** |

> **Correction ([exp03](exp03-1b-proxy-design.md)).** The 256N row is a
> **torch 2.10** measurement. The current torch-2.13 stack measures 256N at
> **18.77% MFU**, so this table overstates the mid-scale collapse by ~2x. The
> 512N ~9-11% figure is current and real, and 64N is still the efficiency
> knee -- the argument below survives, but on a 512N-vs-64N gap, not a
> 256N cliff.

Live from the completed chain's last steps: `tps: 1,478 tflops: 26.22
mfu: 8.79%`. We ran a 2B model at dp=6144 -- per-rank work small enough
that communication dominates. **A 30B at the same node count has ~15x the
per-rank compute, which is the entire argument for going bigger rather than
wider.** Recovering 27% MFU is a ~3x effective-compute multiplier, larger
than any data change proposed below.

> **Measured, partially ([exp06](exp06-scaling.md), job `12473198`).** The 30B
> holds **25.54% MFU at 64N** and decays **1.75% per doubling** against the
> 2B's **13.96%** -- an 8x better rate, supporting the mechanism above.
>
> Two honest caveats this section should carry:
>
> 1. **64N is not 512N.** Three doublings separate them, and the 2B's collapse
>    was a cliff, not a slope -- extrapolating through a smooth region cannot
>    locate a cliff. Confirming 512N needs Aurora.
> 2. **At 64N the 2B is not collapsed either** (25.9% in the table above vs
>    the 30B's 25.54%). So the measured range does not yet separate the two
>    models; it establishes that the 30B does not degrade *earlier*, which is
>    necessary but not sufficient for the 512N claim.
>
> exp06 also raises a constraint this section omits: holding LBS=3 to 512N
> implies **GBS 18,432 sequences / 75M tokens per step**, far past the batch
> size that still converts tokens into learning. The MFU argument and the
> batch ceiling push in opposite directions.

### 1.4 Catastrophic forgetting is real and fast

MDS stage-3 ladder, both arms branching from ~140,300 and running to
154,391:

| | stage3-mix (33/33/34) | stage3 (pure math/code) |
|---|---|---|
| hellaswag | flat ~0.587 | **0.564 -> 0.431** |
| arc_easy | 0.687 -> **0.722** | 0.6216 |
| arc_challenge@25 | 0.4164 | 0.3703 |
| gsm8k | 0.0167 | 0.0303 |

Pure math/code lost ~8 HellaSwag points in the first 2,000 steps then bled
monotonically, to buy gsm8k that is still ~0. The balanced mix stayed flat
and *kept gaining* ARC-Easy. **Never drop below ~50% general in any phase.**

### 1.5 Schedule is not the lever; mix is

The 2026-07-28 anneal A/B (pre-registered, frozen holdouts disjoint from
every training arm, ~1600 steps / ~10B tokens per arm): constant-LR beat
WSD-decay-to-zero on *both* bases. Data mix mattered enormously over the
same span -- pure edu-web forgot math catastrophically; **75% math / 25%
edu** kept essentially all the math while capturing essentially all of edu's
general-domain gain. Three later LR cooldowns (52k/72k/92k steps) showed no
capability jump, corroborating at 5-9x the token count.

---

## 2. Data: the whole ballgame

Target **~10T tokens**, rebuilt. Rough shape:

| source | ~tokens | why |
|---|---|---|
| DCLM-pool re-filtered with an edu-quality classifier | 2-3T | We own the pool. FineWeb-Edu showed the filter is the lever; we used the wrong one |
| **Synthetic rephrasing** (WRAP / Cosmopedia style) | 0.5-1T | The clearest small-model MMLU lever in the literature. A POC already exists in-repo |
| Nemotron-CC-Math-4+, FineMath-4+ | ~150B | Already on disk |
| peS2o + PubMed + USPTO + arXiv full text | ~250B | DOE science mission, low CC overlap, clean provenance |
| Stack v2 permissive, quality-filtered | ~500B | Code correlates with reasoning transfer |
| Textbooks (OpenStax / LibreTexts / MIT OCW) | ~50B | Highest knowledge density per token available |
| FLAN-style instruction data folded into the final phase | 2-5% | Mid-training instruction exposure |

Curriculum, constrained by 1.4: **never below ~50% general in any phase.**
Phase A (0-60%) broad web-heavy; Phase B (60-90%) upweight science/code/math
while holding general >= 50%; Phase C (90-100%) high-quality + instruction,
WSD decay.

**Decontamination is mandatory and non-negotiable** -- 10-gram + difflib
against MMLU/ARC/HellaSwag/GSM8K/MATH before anything enters the mix. All
corpus token counts above are pre-cross-dedup curator estimates, not
measurements; MinHash/Bloom every CC-derived add against the existing 3.70T
DCLM backbone and expect the totals to shrink.

## 3. Tokenizer

> **Substantially revised by [exp01](exp01-tokenizer-analysis.md).** Two of the
> four bullets below were **wrong** and have been struck; the embedding cost was
> understated by 2x. The conclusion (shrink the vocab) survives, but for
> different reasons than I originally gave.

Drop gemma-7b's 256,128-entry vocab. At 2B it is **1,049M params (52.8% of the
model)** -- production agpt has weight tying *off*, so the 256128x2048 matrix is
instantiated **twice**, as embedding and as untied `lm_head`. (My original
"525M / ~26%" counted one matrix; `config_registry.py:163` already said ~53%.)
At 30B (dim 6144) the same vocab is only 10.5%, so this is a small-model
argument, and the reasons that survive are cost, code, and concentration:

- **~64k custom BPE** trained on the actual mix. Justified by measurement:
  **99% of all token mass sits in the top 62,108 IDs, and 129 IDs cover 50%.**
- **Code fertility.** gemma is within 2-4% of Llama-3.1 on every prose/science
  domain measured (it *beats* Llama on peS2o at 0.983x) but costs **+17% on
  starcoder and +26% on Python**. Qwen3 matches Llama on code at 151k vocab,
  so this is gemma's merges, not vocab size.
- Keep 128-alignment (the 256128-vs-256000 gap is padding, worth preserving).

- ~~**Single-digit tokenization.**~~ **FALSE as applied to gemma.** exp01
  measured 9,999/9,999 integers splitting one-token-per-digit with **zero
  exceptions**. The sharpest probe: prepending a `0` leaves gemma's
  segmentation untouched (`2024` -> `2 0 2 4` becomes `0 2 0 2 4`), while
  **Llama-3.1 re-segments the whole number** (`202`,`4` -> `020`,`24`). Gemma
  is perfectly compositional; the pathology I described is Llama's, not ours.
  **A custom tokenizer cannot buy this, and digit handling cannot explain
  GSM8K = 0.0000.** That failure needs a different explanation.
- ~~**Explicit handling for LaTeX, units, indentation.**~~ **Not supported.**
  gemma encodes a 24-space run as a single token and beats Llama-3.1 on
  `\begin{equation}`, `kPa`, and `\nabla^2`.

**Internal contradiction exp01 caught:** the original section asked for
single-digit tokenization *and* fewer wasted tokens. The +16% GSM8K fertility
*is the price of* single digits -- any tokenizer preserving that property pays
it. The two bullets were in tension.

**Do this first, before any re-tokenization: turn on weight tying.** It
recovers **525M params at 2B for free** -- no re-tokenization of the
4.674T-token corpus, no loss of checkpoint comparability. `agpt_2b_tied`
already exists.

## 4. Model + training

**One ~30B dense model**, not a ladder.

- 20B already trains stably on this stack; 80B does not (see the dp>186
  grad-path NaN wall). 30B is the largest size with a credible stability
  story here.
- Per-rank work at 512-1024 nodes is ~15x a 2B's -- the direct fix for 1.3.
- ~10T tokens at 30B is ~330 tok/param (~17x Chinchilla) -- inference-optimal
  over train-optimal, correct for a model meant to be deployed.
- **MoE (~120B total / ~12B active) is strictly better FLOPs-per-loss**, and
  is the right long-term answer -- but EP is currently broken on this stack
  (a2a-under-selective-AC SIGABRT at EP=2). Stretch goal, not the primary bet.

Training config:

- **fp32 master weights + bf16 compute.** See Section 6 for why this is
  broader than the measurement strictly requires.
- **QK-norm + z-loss** rather than fp32-residual patches. The 80B NaN is a
  grad-path overflow at dp>186; score-bounding addresses the cause, and a
  qk_norm TP-sharding bug is already fixed (`41663e59e`).
- **HSDP** (`dp_shard=NGPU_PER_HOST`, `dp_replicate=NHOSTS`) -- measured
  +2.7% MFU over pure FSDP, and the win should compound at 30B.
- **WSD schedule.** Constant-LR trunk + short decay, so an anneal can branch
  from *any* point without recompute. This suits a queue where jobs die
  constantly far better than a single global cosine.
- Sequence 4k for the trunk, extend to 32k in the final ~5%.

## 5. Process: the actual failure of this campaign

The mix being wrong was survivable. **Taking 4.674T tokens to find out was
not.** Every finding in Section 1 was reachable on a 1B proxy in ~a day.

Proposed gate, before any mix touches a production chain:

1. **1B proxy, ~50B tokens** on the candidate mix (~3h at 64N).
2. Decontaminated eval on the full modern ladder, shot-namespaced.
3. **Hard gate: MMLU must clear 0.28** (above the 4-way floor) on the proxy.
   If it does not, the mix does not go to production. Full stop.
4. Only then commit production tokens.

> **Two revisions from [exp03](exp03-1b-proxy-design.md), which specified this
> gate and checked it against the repo:**
>
> 1. **There is no "1B proxy" to build -- we are already running it.** Our
>    production 2B has **0.937B non-embedding parameters**, *below*
>    Llama-3.2-1B's 0.973B: the 256,128-entry gemma vocab puts 525M in the
>    embedding and 525M more in the untied lm_head, so 53% of the parameter
>    budget does no depth-of-computation work. Every smaller config lands
>    below the floor. The proxy is `agpt_2b` unchanged, which also keeps the
>    arms comparable to the 12 existing datapoints.
> 2. **The absolute 0.28 gate is the wrong *shape*.** A one-arm absolute
>    threshold assumes the null is exactly 0.25 and that harness, tokenizer,
>    and prompt format contribute nothing -- all three of which are live
>    hypotheses here. A gate cannot assume away what it exists to test. 0.28
>    is 8.2 binomial SE above chance but only **1.45 pp above the highest
>    near-chance value we have ever recorded** (0.2655), and our
>    across-config null is **2.4x overdispersed**. exp03 keeps 0.28 as a
>    secondary conjunct and makes the primary rule **control-referenced and
>    paired**: McNemar on discordant pairs, +0.030 arm-vs-control, with a
>    same-mix reseed arm measuring the noise floor before the candidate is
>    unblinded.

Also worth institutionalizing from this campaign: eval the tail *as it is
produced*, not at the end (coverage on the completed chain stopped at 39,600
and the final 6,800 steps were unevaluated until the day it finished); and
keep `arc_challenge@Nshot` shot-namespaced -- a bare key silently overwrote
0-shot with 25-shot and produced a retracted "ARC-C decline" finding.

## 6. What here is JUDGMENT, not measurement

Stated separately so nobody cites it as evidence.

- ~~**fp32 master for ALL parameters** is a risk-asymmetry judgment;
  norms-only fp32 would have fixed the observed bug.~~ **RETRACTED
  2026-08-14 -- this is now MEASURED, and the "norms-only would have
  sufficed" half was wrong.** [exp02](exp02-fp32-norms-ablation.md) ran the
  three-arm ablation (bf16 master / norms-only fp32 / full fp32) on the
  debugmodel: norms-only fp32 does unfreeze all 13 RMSNorm weights, but it
  leaves `tok_embeddings.weight` frozen at `frac_changed` **0.000345 vs
  1.000000** under full fp32, because agpt initializes the embedding at
  `std=1.0` (`_EMBEDDING_INIT`) -- the *same* scale as `RMSNorm.weight`, so
  the same ~7.8e-3 ULP. **38.36% of all parameter elements never move under
  norms-only, against 0.000% under full fp32.** The earlier reasoning ("every
  other parameter sits near 0.005") holds for *linear* layers only; the
  embedding is neither a linear layer nor at that scale. Full fp32 master
  stays, now on evidence rather than caution. The recurrence prediction in
  the old bullet was correct and under-stated: the vulnerability had
  *already* recurred in the shipped 2B/20B/80B configs, on the embedding, and
  went unnoticed because the v1 post-mortem only checked the norms.
- **~30B and ~10T** are round numbers chosen from scaling-law reasoning and
  the stability envelope, not from a fitted scaling study on this stack.
- **The corpus token counts** in Section 2 are curator estimates
  pre-cross-dedup, not measured after dedup against our existing data.
- **The synthetic-rephrasing MMLU claim** rests on external literature
  (WRAP/Cosmopedia/phi), not on our own runs. Our synthetic POC produced a
  pipeline, not an eval result.
- **MoE FLOPs-per-loss superiority** is from published work, not measured
  here.
- Comparisons to other ~1-3B models clearing MMLU at ~2-11T tokens are
  recalled from published results and should be re-verified before being
  cited in a proposal or paper.
- **The "MMLU capability floor at 1-3B params" premise appears to be false.**
  [exp03](exp03-1b-proxy-design.md) found Qwen2.5-**0.5B** at 47.5 against
  TinyLlama-1.1B at 25.3 *with 3T tokens* -- which says the binding constraint
  is data, not scale, and that no published parameter threshold for MMLU
  emergence could be located. This **weakens the stated motivation** for the
  proxy gate while strengthening the gate's design: a null result at 0.937B
  non-embedding params is more informative than assumed, not less. (Those
  literature rows are themselves recalled, with differing harnesses and shot
  counts.)

## 7. What would falsify this

- **1B proxy on the new mix still lands at chance MMLU** -> the problem is
  not the mix, and the tokenizer/architecture/eval-protocol hypotheses move
  to the front.
- **30B at 512N does not recover MFU well above ~11%** -> the bottleneck is
  the stack (CCL, dataloader, checkpointing), not per-rank work, and going
  bigger buys nothing.
- **Dedup shrinks the additive corpora below ~6T** -> ~10T at 30B is not
  reachable without repeats, and the size/token split needs rebalancing.
- **EP gets fixed** -> the MoE option dominates dense and this plan should be
  revisited.

## Experiments

Claims in Sections 2-4 are being tested rather than asserted. See
**[EXPERIMENTS.md](EXPERIMENTS.md)** for the live tracking table -- what has
been run, what it found, and which proposal claims it supports or undermines.

Note on scope: a toy-scale model **cannot** settle the central MMLU question.
MMLU has a capability floor around 1-3B params, so a 20M or 500M model returns
~0.25 on *any* data and a null result cannot distinguish "the mix is bad" from
"the model is too small". Tier 0 experiments are cheap because their questions
are scale-free (tokenizer statistics, precision ablation), not because the
important question is being economised on -- that one needs the 1B proxy.

## Related

- [`2b/n512/README.md`](../2b/n512/README.md) -- the completed chain
- [`../../../notes/data-strategy-after-olmo-mix-2026-07.md`](../../../notes/data-strategy-after-olmo-mix-2026-07.md)
  -- corpus survey this builds on
- [`../../../guides/training-dtype-bf16-norm-freeze.md`](../../../guides/training-dtype-bf16-norm-freeze.md)
  -- the fp32/bf16 investigation Section 6 refers to
- [`../../../experiments/agpt/sunspot/20260728-2b-mds-anneal-and-datamix.md`](../../../experiments/agpt/sunspot/20260728-2b-mds-anneal-and-datamix.md)
  -- the anneal vs data-mix A/B
- [`../../POST-TRAINING-2B.md`](../POST-TRAINING-2B.md) -- post-training
  counterpart: accuracy lives in SFT structure, not RL
