# exp03 -- the 1B proxy that gates every future mix

> **Last updated: 2026-08-23.**
>
> **Status: DESIGN. Nothing submitted. No jobs touched.** This specifies the
> Section-5 process gate from [`README.md`](README.md) precisely enough to
> submit, and revises two of its numbers after checking them against the repo
> and the cluster.
>
> **Two headline revisions to the proposal:**
>
> 1. **Do not build a new "1B" config.** Our production `agpt 2B` is already a
>    1B-class model by the only measure that matters -- it has **0.937B
>    non-embedding parameters**, within 4% of Llama-3.2-1B's 0.973B. The
>    "2B" label counts a 525M-param embedding table twice. Building a smaller
>    config would take us *below* the floor, not to it. See
>    [Why this size](#why-this-size).
> 2. **The 0.28 gate is right at the edge of defensible, and it is the wrong
>    *shape* of gate.** 0.28 is 8.2 binomial SE above chance -- but only
>    **1.45 pp above the highest near-chance MMLU we have ever recorded**
>    (0.2655), and our across-config null is **2.4x overdispersed** relative
>    to binomial. Replace the absolute one-arm threshold with a
>    **control-referenced, paired** rule. See [Decision rule](#decision-rule).
>
> A third finding changes the cost picture entirely: **FineWeb-Edu is already
> on Aurora, already gemma-tokenized, already in blendcorpus format** -- 1.551T
> tokens at `/flare/AuroraGPT/datasets/fineweb-edu-v1.4.0/data-fused-tok`.
> The candidate arm needs a text file of weights, not a tokenization campaign.

---

## Design summary

Four arms, one model size, one token budget, one node count. All arms are
identical except the `--dataloader.dataset-path` data list.

| Arm | Data mix | Model | Tokens | Nodes | Walltime | Node-hours |
|---|---|---|---|---|---|---|
| **A (CONTROL)** | `olmo-mix-1124` -- the known-bad baseline | agpt 2b (0.937B non-emb) | 50B | 64 | ~3.0 h | ~190 |
| **B (CANDIDATE)** | 70% FineWeb-Edu / 10% dolmino-flan / 10% dolmino-math / 10% peS2o | same | 50B | 64 | ~3.0 h | ~190 |
| **C (ISOLATOR)** | 100% FineWeb-Edu | same | 50B | 64 | ~3.0 h | ~190 |
| **D (SEED)** | repeat of A, `--debug.seed` changed | same | 50B | 64 | ~3.0 h | ~190 |
| | | | | | **total** | **~760** |

Plus eval: 4 arms x ~4 checkpoints x ~40 min on 1 node in `capacity` = **~11
node-hours**. **Grand total ~771 node-hours (~0.14% of one 512N x 12h prod job's
6,144 node-hours).**

Fixed across all arms (anything not listed is the production 2B default):

```
--module ezpz.agpt --config agpt_2b_real
--training.seq-len 8192  --training.local-batch-size 2      # GBS = 64*12*2 = 1536 seq = 12.58M tok/step
--training.steps 3974                                        # 3974 * 12.58M = 50.0B tokens
--training.dtype float32                                     # fp32 master (the v2 fix)
--data-parallel-shard-degree 12 --data-parallel-replicate-degree 64   # HSDP, +2.7% MFU
--compile.enable                                             # safe at 64N; only 512N OOMs
--lr-scheduler.warmup-steps 200 --lr-scheduler.decay-ratio 0.0 --lr-scheduler.min-lr-factor 1.0
--checkpoint.interval 500 --checkpoint.no-enable=false       # keep-latest-k stays 0
--debug.seed 42
```

Constant LR (`decay_ratio=0.0`) is deliberate: the 2026-07-28 anneal A/B found
constant-LR beat WSD-decay-to-zero on both bases, and a flat trunk lets any arm
be extended or branched without recompute.

**Why arm D exists.** Three arms measure mix; the fourth measures *us*. Without a
same-mix, different-seed replicate we cannot separate "the candidate mix moved
MMLU" from "two 50B runs of the same model differ by that much anyway." Arm D
is the only thing that turns the 2.4x overdispersion in
[Decision rule](#decision-rule) from an assumption into a measurement, and at
190 node-hours it is the cheapest insurance in the design. **Do not drop it.**

**Why arm C exists.** If B beats A, C says *why*. B is FineWeb-Edu plus explicit
multiple-choice/instruction formatting (FLAN); C is the same edu-web filter with
no MC exposure. B > C > A means the edu filter is the lever and MC formatting
adds on top. B > A but C ~ A means the lever is *format exposure*, not knowledge
density -- a much cheaper thing to buy, and it changes the 30B data plan.

**Node count: 64, not 512.** From the measured Aurora torch-2.13 sweep
([`docs/scaling/agpt-2b.md`](../../../reference/scaling/agpt-2b.md)), 64N runs at
**22.8-24.6% MFU** (6,083-6,553 TPS/GPU) versus 18.5-18.8% at 128-256N and
~7.5-11% at 512N. 64N also fits `debug-scaling` (max 256 nodes, 1 h) for smoke
and `prod`/`small` for the real thing, and its ~3 h walltime is short enough to
land inside one queue slot without a continuation chain. Node-hours are
essentially flat from 4N to 64N (158-190) and rise past it, so 64N is the
efficiency knee and the wall-clock sweet spot simultaneously.

### Note on the proposal's throughput table

Section 1.3 of [`README.md`](README.md) lists 256N at 9.5% MFU. That figure is
from the **torch 2.10** sweep. The current torch-2.13 stack measures 256N at
**18.77%**. The 512N ~9-11% production figure is real and current. This does not
change the recommendation -- 64N is still the knee -- but the "throughput
collapses at 256" framing overstates the current stack's problem by ~2x, and
the 30B proposal leans on that number.

---

## Why this size

### The proxy is not "a 1B" -- it is the model we already have

`agpt_configs["2B"]` (`agpt/__init__.py:434`) is `dim=2048, n_layers=12,
n_heads=16, n_kv_heads=4, hidden_dim=11008, vocab=256128`, untied embeddings.
Computed from those dims:

| | total | non-embedding | embedding+lm_head |
|---|---|---|---|
| **agpt 2B (ours)** | **1.987B** | **0.937B** | 1.049B (**52.8%**) |
| Llama-3.2-1B (tied, vocab 128256) | 1.236B | 0.973B | 0.263B (21.3%) |
| TinyLlama-1.1B (vocab 32000) | 1.100B | 0.969B | 0.131B (11.9%) |
| SmolLM2-1.7B (vocab 49152) | 1.812B | 1.611B | 0.201B (11.1%) |

Our "2B" has **less non-embedding capacity than Llama-3.2-1B**, the very model
the proposal cites as proof that 1B can clear chance on our harness (0.3121).
The gemma 256,128-entry vocab puts 525M params in the embedding and another
525M in the untied lm_head; 53% of the parameter budget does no
depth-of-computation work.

**Consequence: there is no new config to add, and adding one would be a
mistake.** Every smaller candidate makes the null *less* interpretable:

| candidate | dim | L | non-emb | emb % |
|---|---|---|---|---|
| dim2048 L12 ffn5632 | 2048 | 12 | 0.541B | 66.0% |
| dim1536 L24 | 1536 | 24 | 0.604B | 56.6% |
| dim2048 L16 ffn5632 | 2048 | 16 | 0.721B | 59.3% |
| **agpt 2B (use this)** | **2048** | **12** | **0.937B** | **52.8%** |

Reusing the production config buys three more things a fresh config cannot:
the arms are directly comparable to the 12 existing near-chance datapoints;
the eval path (`convert_to_hf.py --model_flavor 2b_real`, gemma-7b assets) works
unmodified; and the measured 64N throughput above is *this exact config*, so the
walltime estimate is measurement, not extrapolation.

### Is 50B tokens enough for MMLU to move -- and is 1B above the floor?

Here the literature is more encouraging than the proposal assumes, and it argues
**against** a hard capability floor.

> **Recalled-and-verified-this-session, not our own measurements.** The
> following are published third-party results retrieved from primary sources
> during this design pass. Harnesses and shot counts differ between rows and are
> **not** mutually comparable; treat them as order-of-magnitude evidence about
> data-vs-scale, not as a calibrated scale.

| model | params | tokens | MMLU | source |
|---|---|---|---|---|
| OPT-1.3B | 1.3B | 300B | 24.9 | via TinyLlama paper (arXiv 2401.02385v2) |
| Pythia-1.4B | 1.4B | 300B | 25.4 | via TinyLlama paper |
| **TinyLlama-1.1B** | **1.1B** | **3T** | **25.3** | arXiv 2401.02385v2, 5-shot |
| **Phi-1.5** | **1.3B** | **150B** | **37.6** | arXiv 2309.05463, **2-shot, in-house harness** |
| MiniCPM-1.2B | 1.2B | 1.1T | 49.6 | arXiv 2404.06395 |
| Llama-3.2-1B | 1.23B | up to 9T | 32.2 | Meta model card, 5-shot |
| Qwen2.5-0.5B | **0.5B** | 18T | **47.5** | arXiv 2412.15115, 5-shot |
| FineWeb-Edu ablation | 1.71B | **38B** | **33.6** | arXiv 2406.17557v2 |

Two rows carry the argument:

- **TinyLlama-1.1B at 3T tokens scores 25.3 -- chance.** Same parameter count as
  Llama-3.2-1B, one-third the tokens, and it never leaves the floor. Generic web
  data at 3T does not produce MMLU. **This is our result, reproduced
  independently.** It is the single best evidence that our finding is real and
  is about data.
- **Qwen2.5-0.5B scores 47.5.** Half a billion parameters, 22 points above every
  generic-data 1.4B model. If a hard capability floor at 1-3B existed, this row
  could not exist.

The proposal's premise -- "MMLU has a capability floor around 1-3B params, so
below ~1B a null is uninformative" -- is **not supported** by what I could
verify. The floor is better described as a **data floor**: models below ~1B on
*generic web* stay at chance, and models at 0.5-1.7B on *curated* data clear it
comfortably. This is good news for the design, because it means our 0.937B
non-embedding proxy sits above the relevant threshold and a null result at 50B
on a good mix is genuinely informative.

**Token budget: 50B is defensible, and the FineWeb-Edu row is why.** The
FineWeb-Edu ablation reports **33.6% MMLU at 38B tokens** on a 1.71B model --
below chance-plus-noise is not where they landed. Our proxy has ~55% of that
model's non-embedding capacity, so 50B tokens is the right order and buys a
~30% margin over their 38B. The caveats are honest ones: their model has a
compact 32k-class vocab (more of its params compute), and their harness may not
be ours.

I recommend **50B as the primary budget with a pre-committed 100B extension**
(one more 3 h/64N slot per arm, +760 node-hours) *only* if arm B lands in the
[Decision rule](#decision-rule) "ambiguous" band. Do not extend a clean null and
do not extend a clean pass.

**Checkpoint at 500-step intervals and eval the trajectory, not just the
endpoint** (steps ~1000/2000/3000/3974 ~= 12.6B/25B/38B/50B). If MMLU is going
to move, it should move *monotonically* across that ladder. A single endpoint
number cannot distinguish a real trend from a lucky draw; four points can, and
this directly fixes the "eval the tail as it is produced" process failure called
out in Section 5 of the proposal.

---

## Decision rule

**This is the part of the design most worth arguing about, so here is the
arithmetic in full.**

### The sample size and the naive answer

MMLU `test` split = **14,042 questions** (verified against the `cais/mmlu`
dataset card; the other splits are auxiliary_train 99,842 / validation 1,531 /
dev 285). At p = 0.25:

```
SE_binomial = sqrt(0.25 * 0.75 / 14042) = 0.00365  = 0.365 pp
```

So 0.28 sits **(0.28 - 0.25) / 0.00365 = 8.2 SE** above chance, one-sided
p ~ 1e-16. **On binomial grounds the 0.28 gate is not merely distinguishable
from 0.25 -- it is wildly conservative.** A 95% one-sided threshold would be
0.2560.

### Why the binomial number is the wrong number

Our own near-chance observations are **overdispersed** relative to binomial.
Taking the 8 independent configurations from Section 1.1 of the proposal
(excluding the 4 correlated tail checkpoints of one run):

```
values: 0.2599 0.2655 0.2473 0.2437 0.2511 0.2413 0.2463 0.2591
mean = 0.2518   sd = 0.00873   = 2.39x the binomial SE
```

The within-run tail checkpoints, by contrast, have sd = 0.00084 (0.23x
binomial) -- so the extra variance is **between configurations**, not between
evaluations. That is the signature of real systematic effects near chance:
answer-option bias, tokenizer-boundary effects on the answer letters, and
prompt-format interaction. A near-chance model is not flipping a fair 4-sided
coin 14,042 times; it has a mild, config-dependent preference among A/B/C/D that
shifts the whole score together.

Using the empirical null (mu = 0.2518, sd = 0.00873):

| threshold | binomial null (mu 0.25, sd 0.00365) | **empirical null (mu 0.2518, sd 0.00873)** |
|---|---|---|
| 95% one-sided | 0.2560 | **0.2661** |
| 97.5% | 0.2572 | **0.2689** |
| 99.5% | 0.2594 | **0.2743** |
| 99.87% (3 sd) | 0.2610 | **0.2780** |

**Verdict on 0.28: statistically defensible, but by a thinner margin than it
looks, and for a different reason than stated.** It is 3.2 sd above the
empirical null (one-sided p ~ 0.0006) -- so it is *not* indefensible, and I am
not recommending it be loosened. But:

- Its real margin is **0.28 - 0.2655 = 1.45 pp above the highest near-chance
  value we have ever recorded**, which is only ~4 binomial SE. It is a
  ~3-sigma gate, not the ~8-sigma gate the binomial arithmetic advertises.
- It is estimated from **n = 8**, so the null sd itself has ~25% relative
  uncertainty. A 3-sigma threshold built on an 8-sample sd estimate is not a
  3-sigma threshold in practice.
- **The deeper problem is structural, not numerical: it is a one-arm gate.** An
  absolute threshold silently assumes the null is 0.25 and that our harness,
  tokenizer, and prompt format contribute nothing. Every one of those
  assumptions is a live hypothesis in this very investigation (the proposal's
  own Section 7 lists tokenizer and eval-protocol as the fallback hypotheses).
  A gate cannot assume away the things it is supposed to test.

### The rule I recommend instead

Run the control. Compare arms. Pre-register all of this **before** any job is
submitted.

**Primary (paired, arm B vs arm A).** Both arms answer the *same* 14,042
questions through the *same* harness. Save per-question correctness with
lm-eval's `--log_samples` and run **McNemar's test on the discordant pairs**.
This removes per-question difficulty variance entirely and is strictly more
powerful than comparing two marginal accuracies.

**Secondary (unpaired, for reporting).**

```
SE_diff(two arms near 0.25) = sqrt(2 * 0.25*0.75 / 14042) = 0.00526
MDE @ 80% power, alpha=0.05 one-sided  = 1.31 pp
MDE @ 80% power, 2.4x overdispersion   = 3.1 pp
```

**The gate:**

| outcome | condition | meaning |
|---|---|---|
| **PASS** | `MMLU(B) - MMLU(A) >= +0.030` **AND** McNemar p < 0.01 **AND** the 4-checkpoint trajectory is monotone non-decreasing **AND** `MMLU(B) >= 0.28` absolute | The mix is the lever. Proceed to production tokens. |
| **AMBIGUOUS** | gap in `[+0.010, +0.030)`, or gap >= 0.030 but non-monotone | Extend B and A to 100B (pre-committed, +380 node-hours). Do not proceed on this evidence. |
| **FAIL** | `MMLU(B) - MMLU(A) < +0.010` | The mix is **not** the lever at this scale. Tokenizer / architecture / eval-protocol hypotheses move to the front, exactly as Section 7 of the proposal says. |

The `+0.030` primary threshold is chosen to sit just above the 3.1 pp
overdispersion-inflated MDE, so a PASS is powered at >=80% even if the
arm-to-arm noise is as bad as our worst historical spread. The absolute
`>= 0.28` is retained as a **secondary** conjunct, not the primary criterion --
it guards against the degenerate case where B beats A only because A drew low.

**Arm D calibrates the whole rule.** `|MMLU(D) - MMLU(A)|` is a direct
measurement of same-mix run-to-run noise. If that gap is itself >= 0.010, the
+0.030 threshold is too loose and must be raised to `3 x |D - A|` before B is
unblinded. Compute this **first, and before looking at B.**

**One more pre-registration item.** Report `acc` and `acc_norm` and fix which
one decides, in advance. Report the 4-way answer-distribution entropy of each
arm's predictions; a model that "passes" by learning to prefer option C is a
false positive that the accuracy number alone will not reveal.

---

## Data arms

### What actually exists (verified on Aurora, not assumed)

The proposal's Section 2 lists corpora to acquire. Two of them are **already on
Aurora, already tokenized with the same gemma-7b tokenizer as olmo-mix, already
in the Megatron `MMIDIDX` blendcorpus format** the trainer reads. Verified by
inspecting the `.idx` headers and the tokenization script:

| corpus | path (`/lus/flare/projects/AuroraGPT/datasets/`) | tokens | tokenizer |
|---|---|---|---|
| olmo-mix-1124 | `olmo-mix-1124/data_fused_gemma_eod/` | **3.999T** | gemma-7b |
| **FineWeb-Edu v1.4.0** | `fineweb-edu-v1.4.0/data-fused-tok/` | **1.551T** | **gemma-7b** (`tk_gemma.sh`: `TOKENIZER=.../gemma-7b/`) |
| dolmino-mix-1124 | `dolmino-mix-1124/data-fused-tok/` | 0.853T | gemma-7b |

Token counts derived from `du -sb` / 4 (the `.idx` headers report `dtype_code=4`
= int32, 4 bytes/token). 989 `.bin` files across 110 CC snapshots for
FineWeb-Edu.

dolmino breaks down into slices that matter here:

| slice | tokens | why it matters |
|---|---|---|
| `dclm` | 760.7B | generic web -- **not** used (it is the olmo-mix backbone) |
| **`flan`** | **17.1B** | **instruction/multiple-choice format exposure.** The single most MMLU-shaped thing on disk |
| **`math`** | **11.7B** | dolmino's curated math |
| `pes2o` | 58.7B | open-access science papers, DOE-mission-relevant |
| `wiki` | 3.8B | knowledge-dense |
| `stackexchange` | 1.4B | Q&A format |

**This changes the cost of the experiment by an order of magnitude.** No
tokenization campaign, no HF streaming (and so no repeat of the
open-web-math 429-storm at 384 ranks), no gated-dataset access negotiation. An
arm is a **weighted text file** in
`torchtitan/experiments/ezpz/data-lists/aurora/` -- the same three-column
`weight  path  tag` format as `olmo-mix-1124.txt` -- passed via
`--dataloader.dataset-path`. That directory is writable.

Note the corollary: the corpora referenced by `fineweb_edu_local` /
`cosmopedia_science_local` / `nemotron_cc_math_4plus_local` in
`datasets.py:430-448` all point at **`/lus/tegu/`**, which is **Sunspot, not
Aurora**. Those registrations are unusable for an Aurora run, and
`Nemotron-CC-Math-v1/4plus` on Aurora is an **empty directory** (4.0K, zero
`.bin` files). Do not plan an arm around Nemotron on Aurora without staging it
first.

### The arms

**Arm A -- CONTROL: `olmo-mix-1124`.** Existing list, unmodified. This is the
known-bad baseline and the whole experiment is referenced to it. Running it at
50B/64N rather than reusing a production checkpoint is deliberate: the control
must share the proxy's model size, token count, batch size, LR schedule, seed,
and eval date. A comparison against the 4.674T production number would confound
mix with five other things.

**Arm B -- CANDIDATE: the maximum-contrast mix.**

| source | weight | tokens @50B | available | epochs |
|---|---|---|---|---|
| FineWeb-Edu | 0.70 | 35.0B | 1,551B | 0.02 |
| dolmino `flan` | 0.10 | 5.0B | 17.1B | 0.29 |
| dolmino `math` | 0.10 | 5.0B | 11.7B | 0.43 |
| dolmino `pes2o` | 0.10 | 5.0B | 58.7B | 0.09 |

Every slice stays **well inside one epoch** -- no repeat-induced memorization,
which also keeps the contamination story clean.

The design logic: FineWeb-Edu is the classifier-filtered edu web whose own
ablation reports 33.6 MMLU at 38B tokens, and it is exactly the "we used the
wrong filter" lever from Section 2 of the proposal. FLAN at 10% supplies the
multiple-choice/instruction *format* that olmo-mix demonstrably lacks -- the
hypothesis under test is literally "the corpus lacks multiple-choice academic
content," and FLAN is the direct antidote. Math and peS2o hold the DOE science
mission and keep the mix from being a pure-web monoculture. General content
stays at 70%, comfortably above the >=50% floor that Section 1.4's
catastrophic-forgetting result mandates.

**Arm C -- ISOLATOR: 100% FineWeb-Edu.** Single source, no MC format. Separates
"knowledge-dense edu filter" from "explicit MC/instruction exposure."

**Arm D -- SEED REPLICATE: `olmo-mix-1124`, different seed.** Identical to A
except `--debug.seed`. Measures the noise floor. See
[Decision rule](#decision-rule).

### Arms deliberately excluded

Every arm is a full run, so the ones left out need justifying. **Nemotron-CC-Math**
(empty on Aurora, needs staging + gated access). **Cosmopedia** (Sunspot-only;
its synthetic-textbook value is real but it is the *second* question, after
"does any better mix move MMLU at all"). **Tokenizer variants** (a different
vocab is a different experiment and cannot share arm A as a control).
**Curriculum/phase ordering** (only meaningful once a mix has cleared the gate).

---

## Contamination control

A false positive here is the worst possible outcome: it would send ~10T tokens
of 30B-scale compute after a mirage. FineWeb-Edu and dolmino-FLAN are both
*higher* contamination risk than olmo-mix -- FLAN is built from academic task
pools that overlap MMLU's sources, and FineWeb-Edu is filtered *for* educational
content, which is what MMLU questions are made of. **Treat FLAN as the primary
suspect.**

Four layers, in order. Layers 1-2 are mandatory before submission.

**Layer 1 -- decontaminate at the source, on the raw text.** The raw pre-tokenized
text is still on disk (`fineweb-edu-v1.4.0/data/` and `data-fused/`, 4.1 TB each),
so this is feasible rather than aspirational -- the `.bin` files are opaque, the
text is not.

We already own the detector. `torchtitan/experiments/ezpz/rl/decontam_traces.py`
implements exactly the two-signal scheme the proposal asks for: normalized-exact
match plus **13-gram** overlap, pure-Python/CPU, importable and CLI-runnable,
with a `--self-test`. It is currently GSM8K-specific
(`GSM8KDecontaminator`, `load_gsm8k_test_questions`). Generalize it to a
`BenchmarkDecontaminator` taking any question list, and build the reference set
from **MMLU test (14,042) + ARC-C + ARC-E + HellaSwag + GSM8K + OBQA + BoolQ +
PIQA + Winogrande** -- every task on our eval ladder, not just MMLU, so a
"contamination-driven MMLU pass" cannot hide behind clean sibling metrics.

Match against the **question stem AND each answer option** concatenated, not the
stem alone: FLAN's failure mode is reproducing the full MC item, and a stem-only
match would miss a reworded stem with verbatim options. Use 13-gram (keep the
existing, validated n) rather than the proposal's 10-gram -- 10 raises the
false-positive drop rate on generic academic prose for no real gain in recall.

Emit a `DecontamReport` per source: documents scanned, documents dropped, drop
rate, and the top-20 matched n-grams. **Commit that report next to the data
list.** If FLAN's drop rate is >1%, stop and inspect before running anything --
that would mean the slice is substantially MMLU-derived and the arm needs
rebuilding.

**Layer 2 -- re-tokenize only what survives.** Drop-then-retokenize the affected
slices with the same `tk_gemma.sh` path so the arm's `.bin` files provably
contain no flagged document. Cheap for FLAN/math/wiki/stackexchange (~34B tokens
total). For FineWeb-Edu's 1.551T, scan the full corpus but **retokenize only the
snapshots that had hits**, and record which ones.

*Honest limitation:* the 50B draw samples ~2% of FineWeb-Edu, so a full-corpus
scan is ~50x more work than the run strictly needs. If wall-clock forces a
shortcut, scan the specific shards the dataloader will touch and **say so in the
report** -- do not silently scan a subset and claim the corpus is clean.

**Layer 3 -- verify after the fact, on the model.** Decontamination that is
never checked is a hope. On every arm's final checkpoint, run a
**canary/memorization probe**: take 500 MMLU test items, feed the question stem,
and measure per-token NLL of the *correct answer text* versus a
length/domain-matched control set of never-published items. A contaminated model
shows anomalously low NLL on the real items. This catches leaks the n-gram
filter missed (paraphrases, translations, reformatted items) and it catches
leaks that entered through **olmo-mix** -- arm A is not automatically clean, it
is merely the incumbent.

Also compare **MMLU-subject profiles** across arms. Genuine capability lifts
broad clusters (STEM, humanities, social science) together; contamination
spikes isolated subjects. A B-arm pass driven by three subjects is a red flag
regardless of the aggregate.

**Layer 4 -- hold one benchmark completely out.** Do not decontaminate against
**MMLU-Pro**, and do not look at it until the gate has been called. If B passes
MMLU but shows nothing on the untouched held-out benchmark, that asymmetry is
evidence of leakage rather than capability. This is the only layer that can
detect contamination introduced by the decontamination process itself (e.g. a
reference list built from a corrupted MMLU copy).

---

## Open questions / what I could not determine

**Verified in this pass** (so the rest can be weighed against it): all config
dims and the derived parameter counts (from `agpt/__init__.py`); the
`num_flops_per_token` formula and the Max-1550 peak-FLOPs constant (from
`torchtitan/models/utils.py:450` and `tools/utils.py:166`, reproducing the
documented 27.55% MFU at 4N to within 0.01pp, which confirms Aurora runs
standard-EU 448-CU mode); the 64N throughput numbers (from
`docs/scaling/agpt-2b.md`); FineWeb-Edu's presence, gemma tokenizer, format, and
1.551T size (by inspecting `.idx` headers and `tk_gemma.sh` on Aurora); the
dolmino slice sizes; the MMLU test-split size of 14,042; and the existence of
the 13-gram decontaminator.

**Recalled/third-party, NOT our measurements:** every row of the published-MMLU
table in [Why this size](#why-this-size). These were retrieved from primary
sources during this design pass, but harnesses and shot counts differ between
rows (Phi-1.5 is 2-shot on an in-house harness; OLMo-2 and SmolLM2 use cloze
format; Qwen and Llama are 5-shot). **They are not mutually comparable and none
of them were reproduced on our harness.** Re-verify before citing any of them in
a proposal or paper.

**Open questions:**

1. **The proposal's "MMLU capability floor at 1-3B" premise appears false.**
   Qwen2.5-0.5B at 47.5 and TinyLlama-1.1B at 25.3 with 3T tokens together say
   the binding constraint is data, not scale. I did not find any published
   statement of a parameter threshold for MMLU emergence. The DCLM paper reports
   MMLU only at 7B (not at its 1B-1x/1B-5x scales), which is *consistent* with a
   small-scale floor but does not assert one; the FineWeb paper implies the
   opposite, having kept MMLU specifically because it scored above random at
   1.71B. This weakens the stated motivation for the proxy while
   *strengthening* the design -- a null at 0.937B non-embedding params is more
   informative than the proposal assumed.

2. **The 2.4x overdispersion estimate rests on n = 8**, and those 8 are not a
   clean sample -- they span two model sizes and five data treatments, so some
   of the spread may be genuine (tiny) capability differences rather than noise.
   Arm D exists to replace this estimate with a real measurement. Until it runs,
   the +0.030 threshold is provisional.

3. **I could not verify what fraction of FineWeb-Edu the 50B draw actually
   touches**, because blendcorpus's shard-sampling order depends on the weights
   and seed. This determines whether Layer-2 decontamination needs the full 1.551T
   corpus or a few snapshots. Resolve by instrumenting the dataloader for one
   short run and logging touched shards before committing to a scan strategy.

4. **The `2b_real` (cos_sin RoPE) vs `2b` (complex RoPE) choice.** Production
   2B/20B default to `_real` for compile-lowerability, and these arms train from
   scratch so either is self-consistent. I recommend `2b_real` to match the
   production submit scripts and the measured throughput. Flagging it because
   the RoPE convention is load-bearing at conversion time and has silently
   corrupted a fork before.

5. **Queue availability at 64N is unmodeled.** `debug-scaling` caps at 1 h
   (enough for smoke, not the 3 h run), so the real arms go to `prod`/`small`
   alongside the production chain. Four 3 h jobs may still wait days for slots.
   The design is cheap in node-hours but not necessarily fast in wall-clock.

6. **A 2N smoke of the arm-B data list is mandatory before the 64N submissions**
   (per the standing smoke-before-prod rule) -- specifically to confirm the
   hand-built weighted data list parses, that all `.bin`/`.idx` pairs resolve,
   and that the blendcorpus index builds without the Lustre cold-cache race.
   Not costed above; ~1 node-hour.

7. **Not addressed here:** whether a *tokenizer* change would move MMLU
   independently of the mix. Our 256,128-entry gemma vocab spends 53% of the
   2B's parameters on embeddings, and Section 3 of the proposal wants to replace
   it. If all four arms come back at chance, that is the next experiment -- and
   it cannot reuse arm A as its control.

## Related

- [`README.md`](README.md) -- the 30B proposal this gates (Section 5 process gate, Section 7 falsification)
- [`../../../notes/data-strategy-after-olmo-mix-2026-07.md`](../../../notes/data-strategy-after-olmo-mix-2026-07.md) -- corpus survey
- [`../2b/n512/README.md`](../../../production/agpt/2b/n512/README.md) -- the completed 4.674T chain that motivates this
- [`../../../scaling/agpt-2b.md`](../../../reference/scaling/agpt-2b.md) -- the measured throughput table used for sizing
- [`../../../evals/agpt/2b/README.md`](../../evals/agpt/2b/README.md) -- eval history the null distribution comes from
