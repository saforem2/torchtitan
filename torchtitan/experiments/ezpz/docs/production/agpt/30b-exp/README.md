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

| model | tokens | MMLU |
|---|---|---|
| 20B-256 | 0.39T | 0.2599 |
| 20B-512 | 0.71T | 0.2655 |
| 2B-512 | 3.99T | 0.2473 |
| 2B-256 **complete** | 4.674T | 0.2437 |
| **2B-512 complete** | **4.674T** | **0.2511** |
| 2B MDS dolmino | 7.064T | 0.2413 |
| MDS stage3-mix | 7.771T | 0.2463 |
| MDS stage3 math/code | 7.771T | 0.2591 |

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
| 256 | 2,500 | 9.5% |
| **512 (production)** | **~1,478** | **~9-11%** |

Live from the completed chain's last steps: `tps: 1,478 tflops: 26.22
mfu: 8.79%`. We ran a 2B model at dp=6144 -- per-rank work small enough
that communication dominates. **A 30B at the same node count has ~15x the
per-rank compute, which is the entire argument for going bigger rather than
wider.** Recovering 27% MFU is a ~3x effective-compute multiplier, larger
than any data change proposed below.

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

Drop gemma-7b's 256,128-entry vocab. At 2B it is **525M params (~26% of the
model) in embeddings**; at 30B the same vocab is a much smaller fraction, so
this matters most if we keep training small models -- but the other reasons
stand at any size:

- **~64-100k custom BPE** trained on the actual mix, not on someone else's.
- **Single-digit tokenization.** We score exactly 0.0000 on GSM8K at every
  checkpoint ever evaluated. Digit handling is a known contributor.
- Explicit handling for LaTeX, units, and code indentation -- our corpus is
  science-heavy and the general-purpose tokenizer wastes tokens on all three.
- Keep 128-alignment (the 256128-vs-256000 gap is padding, worth preserving).

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

1. **1B proxy, ~50B tokens** on the candidate mix (~1 day at 64N).
2. Decontaminated eval on the full modern ladder, shot-namespaced.
3. **Hard gate: MMLU must clear 0.28** (above the 4-way floor) on the proxy.
   If it does not, the mix does not go to production. Full stop.
4. Only then commit production tokens.

Also worth institutionalizing from this campaign: eval the tail *as it is
produced*, not at the end (coverage on the completed chain stopped at 39,600
and the final 6,800 steps were unevaluated until the day it finished); and
keep `arc_challenge@Nshot` shot-namespaced -- a bare key silently overwrote
0-shot with 25-shot and produced a retracted "ARC-C decline" finding.

## 6. What here is JUDGMENT, not measurement

Stated separately so nobody cites it as evidence.

- **fp32 master for ALL parameters.** What was *measured* is narrower: bf16
  master freezes `RMSNorm.weight` because those params sit at 1.0, where the
  bf16 ULP is ~7.8e-3 against ~1.6e-5 updates. Every other parameter sits
  near 0.005 where the ULP is ~3.8e-5, and the investigation explicitly
  confirmed they update fine ("Why other parameters update fine"). So
  **norms-only fp32 would have fixed the observed bug.** I still recommend
  full fp32 master because: the compute dtype is bf16 either way (the policy
  is `param_dtype=bf16` with fp32 master, so there is no throughput cost);
  the memory saving is ~10 GB at 20B, negligible against 88% peak; the
  failure is silent and cost a full restart; and the vulnerability is
  scale-dependent, so it recurs for any future parameter initialized near
  1.0 -- QK-norm gains being exactly that. That is a risk-asymmetry
  argument, not a measurement.
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
