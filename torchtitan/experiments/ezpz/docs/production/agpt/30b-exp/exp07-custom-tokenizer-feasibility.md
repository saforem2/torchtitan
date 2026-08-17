# exp07 -- tokenizer bake-off: custom 64k does NOT pay; OLMo-2 wins and is confirmed on hardware

> **2026-08-16.** Corpus study on Aurora against the real `olmo-mix-1124`
> trees, plus a measured fertility comparison on Sunspot (job `12473205`).
> Follows
> [exp01](exp01-tokenizer-analysis.md) (which measured vocab utilisation) and
> [exp05](exp05-2n-performance.md) (which measured the 128k alternative).

## The two questions

1. Is it worth training our own tokenizer on `olmo-mix-1124`?
2. What do we do about the corpus already being tokenized on disk?

**Short answers: NO on the first, and the second is moot as a result.**

The feasibility argument below held up on every term except the one that
matters most, and that term was then **measured and came out negative**: a
candidate 64k trained on our own corpus is **2.5% WORSE than gemma and 5.9%
worse than Llama-3** in tokens per byte. See
[the measurement](#measured-the-custom-tokenizer-loses-job-12473205), which
supersedes the recommendation this page originally carried.

**Use the vendored Llama-3 128k.** It measured best on fertility here, exp05
already measured it at +1.12pp MFU on the 30B, and it needs no new artifact.

## The corpus, measured

`/lus/flare/projects/AuroraGPT/datasets/olmo-mix-1124/` holds both forms:

| tree | contents | size |
|---|---|---:|
| `data/` | original `.json.gz` source | **6.9 TB** |
| `data_gemma_eod/` | Megatron `.bin` / `.idx`, gemma-7b | 15.9 TB |
| `data_fused_gemma_eod/` | flattened view of the same | -- |

Per-domain, from the `.bin` sizes (int32, so bytes/4 = tokens):

| domain | .bin | tokens |
|---|---:|---:|
| dclm | 15.058 TB | 3,764.5B |
| starcoder | 0.391 TB | 97.8B |
| pes2o | 0.234 TB | 58.5B |
| arxiv | 0.085 TB | 21.2B |
| open-web-math | 0.051 TB | 12.8B |
| algebraic-stack | 0.050 TB | 12.5B |
| wiki | 0.015 TB | 3.8B |
| **total (unique)** | **15.88 TB** | **3.97T** |

`starcoder_fused` is excluded as a re-layout of `starcoder`, not new data:
103 files against 863, but the same document names and identical byte sizes
(`ada-0000_text_document.bin` is 125,643,376 bytes in both).

**dclm is 94.8% of the corpus.** Any cost estimate here is essentially a
statement about dclm.

## Question 2 first: retokenizing is cheaper than it sounds

**The raw text was never discarded.** `data/arxiv/train/arxiv-train-0000.json.gz`
and friends are still there -- 6.9 TB of source. Retokenizing means re-running
a tokenizer over text we have, not reconstructing text from token ids.

**And the output would be half the size.** Read from the Megatron index header
(`data_gemma_eod/arxiv/train/arxiv-train-0000_text_document.idx`, magic
`MMIDIDX`, dtype code **4 = int32**): the corpus stores **4 bytes per token**,
because gemma's 256,128 ids do not fit in 16 bits.

| | vocab | bytes/token | corpus |
|---|---:|---:|---:|
| gemma (today) | 256,128 | 4 (int32) | 15.9 TB |
| Llama-3 128k | 128,256 | 4 (int32) | 15.9 TB |
| **custom <=65,535** | **~64k** | **2 (uint16)** | **~7.9 TB** |

A 64k vocab is the *only* option that crosses the uint16 boundary -- 128k does
not. So retokenizing to 64k **frees ~8 TB of Lustre** and halves the bytes read
per training step, which matters more at 512N than at 2N.

## Question 1: what a custom tokenizer buys

In descending order of value:

**1. Fertility -- the one that matters.** A tokenizer trained on our own mix
has merges tuned to dclm / starcoder / peS2o text. Fewer tokens for the same
content means the same knowledge for less compute:

| fertility gain | 3.97T becomes |
|---:|---:|
| 5% | 3.77T |
| 10% | 3.57T |
| 15% | 3.38T |

> **MEASURED AND REFUTED.** This whole table was built on "5-15% is the usual
> range for in-domain versus general-purpose tokenizers." That assumption is
> wrong for this corpus -- the measured number is **-2.5%** (i.e. the custom
> tokenizer needs MORE tokens, not fewer). The section below has the data; the
> table above is retained only to show what the argument rested on.

**2. Embedding size.** At dim=6144 with untied embeddings:

| vocab | embedding | share of model |
|---|---:|---:|
| gemma 256,128 | 3.15B | 11.2% |
| Llama-3 128,256 | 1.58B | 5.9% |
| **custom ~64k** | **0.79B** | **3.0%** |

Roughly double the saving the 128k delivers, and exp05 showed *why* that
matters: freed HBM converts into batch size, which is the dominant throughput
lever on this model (LBS 3 -> 4 was worth +1.12pp MFU).

**3. Storage and I/O.** The uint16 halving above.

**4. It is the right size for our data.** exp01 measured **99% of token mass in
the top 62,108 ids**, and 129 ids covering half. A ~64k vocab is fitted to the
measured distribution rather than guessed -- this is the proposal's
best-supported claim and it survives scrutiny.

## Hard constraint: the existing gemma data is untouchable

**Nothing in this line of work writes to `olmo-mix-1124/` at all.** The
`data_gemma_eod/` trees back the completed 4.674T 2B flagship and every
checkpoint derived from it; `data/` is the only surviving copy of the raw
source. Both are read-only inputs.

A retokenized corpus is a **new sibling tree** (e.g. `data_custom64k_eod/`)
built beside them, never a replacement or an in-place conversion. The fertility
study below opens `data/**.json.gz` for reading only and writes exclusively to
a scratch directory outside the dataset root.

## The honest costs

- **dclm dominates.** 6.6 of 6.9 TB raw. Retokenization is embarrassingly
  parallel but it is a real job, not a background task.
- **It invalidates every gemma-trained checkpoint**, including the completed
  4.674T 2B flagship. Strictly a fresh-flagship decision, never a continuation.
- **A new tokenizer is its own artifact** to train, validate, version and keep
  reproducible -- and every eval harness has to agree on it.
- **The fertility gain is unmeasured.** It is also the single largest term in
  the argument.

## The decision rule this page set itself (and then failed)

Before measuring, the rule was:

- **>= 8-10% better fertility:** compelling, retokenize.
- **~5%:** marginal, prefer the vendored 128k.
- **< 5%:** do not train one.

The measured result is **-2.5%** -- not merely under the threshold but on the
wrong side of zero. Recording the rule in advance is what makes that a clean
answer rather than an argument.


## Measured: the custom tokenizer loses (job `12473205`)

Candidate byte-level BPEs trained on a **705 MB stratified sample** of
`olmo-mix-1124` (7 domains, equal bytes each), measured on **176 MB of
held-out text from disjoint source shards**. Tokens per MB, lower is better:

| domain | custom 64k | gemma 256k | Llama-3 128k | winner |
|---|---:|---:|---:|---|
| algebraic-stack | 263,564 | 268,075 | **256,752** | Llama-3 |
| arxiv | **266,802** | 282,694 | 267,074 | custom |
| **dclm** (94.8% of corpus) | 238,298 | 232,014 | **224,782** | Llama-3 |
| open-web-math | 260,205 | 265,404 | **253,468** | Llama-3 |
| pes2o | **213,881** | 221,057 | 220,296 | custom |
| starcoder | 297,499 | 296,945 | **241,110** | Llama-3 |
| wiki | 235,275 | 233,309 | **226,916** | Llama-3 |
| **corpus-weighted** | **239,695** | 233,944 | **225,539** | **Llama-3** |

| comparison | result |
|---|---:|
| custom 64k vs gemma | **+2.5% (WORSE)** |
| custom 64k vs Llama-3 | **+5.9% (WORSE)** |
| Llama-3 vs gemma | **-3.6% (better)** |

A 32k candidate is worse still (**+7.8%** vs gemma), so this is a direction,
not a vocab-size accident.

**The custom tokenizer beats gemma on 4 of 7 domains and still loses**, because
it loses on dclm -- and dclm is 94.8% of the corpus. This is exactly why the
report weights by real token share: the unweighted table looks like a close
contest, and the weighted one is not close. A stratified sample flatters
whatever wins on the small specialist domains.

### Why the assumption was wrong

Two reasons, both of which should have been obvious in advance:

1. **705 MB is far too little.** Production tokenizers are fit on hundreds of
   GB. The candidate is undertrained rather than badly designed.
2. **dclm is generic web text, which is precisely what Llama-3's tokenizer was
   fit on**, at vastly greater scale. There is no in-domain advantage to
   capture when the dominant domain IS the generic domain. The "train on your
   own data" argument only pays when your data is unusual, and 94.8% of ours
   is not.

**This is a lower bound.** A 64k trained on the full 6.9 TB could beat this
candidate. But it must close 5.9% and then beat Llama-3 on top, against a
threshold that was set at 8-10% -- a far larger ask than the argument
assumed, for a benefit (0.79B vs 1.58B embedding, uint16 storage) that would
be paid for with more tokens on every step forever.

### Verdict

**Do not train a custom tokenizer. Use the vendored Llama-3 128k.**

It wins fertility outright here, exp05 measured it at +1.12pp MFU on the 30B
(its freed HBM buys LBS=4), and it requires no new artifact, no tokenizer
validation, and no retokenization gamble. The uint16 corpus halving is
genuinely lost by this choice -- 128k does not fit in 16 bits -- and that is
the real price of the decision.


## Nine tokenizers compared (job `12473206`)

Same 176 MB held-out sample, no training -- pure measurement. Corpus-weighted
tokens per MB, lower is better, sorted best first:

| tokenizer | vocab | tok/MB | vs gemma | embedding @dim=6144 | fits uint16 |
|---|---:|---:|---:|---:|:---:|
| SmolLM3 | 128,256 | 225,534 | -3.6% | 1.58B | no |
| **Llama-3.1** | 128,256 | 225,539 | -3.6% | 1.58B | no |
| **OLMo-2** | **100,278** | **225,749** | **-3.5%** | **1.23B** | no |
| DeepSeek-V3 | 128,815 | 226,651 | -3.1% | 1.58B | no |
| Qwen3 | 151,669 | 230,702 | -1.4% | 1.86B | no |
| gemma-7b (incumbent) | 256,000 | 233,944 | -- | 3.15B | no |
| custom 64k (ours) | 64,000 | 239,695 | +2.5% | 0.79B | **yes** |
| Mistral v0.3 | 32,768 | 262,190 | +12.1% | 0.40B | **yes** |
| Llama-2 | 32,000 | 269,327 | +15.1% | 0.39B | **yes** |

### Three findings

**1. SmolLM3 is the Llama-3 tokenizer.** Identical to five significant digits
on all seven domains (225,534 vs 225,539). Not an independent datapoint --
worth noting so nobody counts it as corroboration.

**2. OLMo-2 settles the in-domain question.** Its tokenizer was fit on
dolma/olmo-mix at production scale -- *our corpus, done properly*, which is
exactly what my 705 MB candidate could not be. It lands at **225,749 tok/MB,
0.09% behind Llama-3.1** -- a tie -- **with a 22% smaller vocab**.

So training on our own data buys **no fertility advantage over a good general
tokenizer**. It buys a *smaller vocab at the same fertility*. That is a real
but different benefit from the one the proposal claimed, and it confirms the
diagnosis in the section above: my candidate lost on training scale, not on
the idea, and the idea's actual payoff is vocab size rather than tokens.

**3. Fertility saturates above ~100k; the knee is 64k-100k.**

| step | fertility cost |
|---|---:|
| 128k -> 100k | **+0.09%** |
| 100k -> 64k | **+6.2%** |
| 64k -> 32k | +12.4% |

Below ~100k the trade turns sharply bad. This is the quantitative reason the
proposal's "~64k" target is the wrong number: it sits just past the knee. The
`exp01` mass-concentration argument (99% of mass in the top 62,108 ids) does
correctly identify that *most ids are rare*, but rare ids are cheap to keep and
expensive to drop -- fertility depends on the tail, not the head.

### Revised recommendation

**OLMo-2 (100,278) is the best-supported choice**, narrowly ahead of Llama-3
on the merits:

- fertility tied with the best measured (+0.09%, far inside noise)
- **0.34B fewer embedding params** than Llama-3 at dim=6144 (1.23B vs 1.58B),
  which by exp05's mechanism is more HBM for batch
- trained on our own corpus family, so its merges match our text distribution

**Llama-3.1 remains the safe choice**: already vendored, already measured on
the 30B end-to-end at +1.12pp MFU (exp05), no new artifact. The gap to OLMo-2
is 0.35B params -- worth a 2N A/B before switching, not worth assuming.

Neither fits uint16, so the corpus stays int32 either way. **The only vocabs
that halve the corpus cost 6-15% in fertility**, which is a bad trade at
3.97T tokens.


## Confirmed on hardware: OLMo-2 frees 9pp of HBM at identical speed (job `12473207`)

Fertility said OLMo-2 should match Llama-3 on tokens while costing 0.34B fewer
embedding params. Both halves were tested on the 30B at 2N (TP=1, compiled,
seq=4096, all four arms in ONE job so there is no cross-allocation drift):

| tokenizer | LBS | tps | MFU | memory |
|---|---:|---:|---:|---|
| Llama-3 128k | 4 | 496 | 28.94% | 48.62 GiB (75.98%) |
| **OLMo-2 100k** | 4 | **499** | **28.94%** | **43.58 GiB (68.11%)** |
| Llama-3 128k | 5 | 507 | 29.58% | 85.84% |
| **OLMo-2 100k** | 5 | **512** | **29.67%** | **49.16 GiB (76.83%)** |

**MFU is identical to the digit at LBS=4** (28.94% both) -- exactly what a
fertility tie predicts, and a useful confirmation that the tokens-per-MB
measurement transfers to real training. The difference is entirely memory:
**7.9 points of HBM at LBS=4, 9.0 points at LBS=5.**

That is larger than the raw parameter delta (0.34B params = 0.69 GB at bf16)
because fp32 master weights plus optimizer state multiply the embedding saving
several times over.

### Why this decides it

exp05 rejected LBS=5 under Llama-3: 85.84% leaves nothing for checkpoint and
eval allocations. Under OLMo-2 the same batch runs at **76.83%**, which is
comfortable -- so **LBS=5 is production-viable with OLMo-2 and was not with
Llama-3**, worth +0.73pp MFU over the LBS=4 pick (29.67% vs 28.94%).

At 29.67% the 30B now exceeds the 2B's small-N 29.26%, on a model 14x larger.

**Recommendation: OLMo-2 (100,352) at LBS=5.** See the batch-size section
below for how that was measured.

### How far to push batch: LBS=5, measured properly

The obvious follow-on was "spend the headroom on a bigger batch". Answering it
took two attempts, because the first was built on a bad noise estimate.

The full ladder (job `12473208`, one job) shows the lever flattening at the top:

| LBS | tps | MFU | memory |
|---:|---:|---:|---:|
| 5 | 498 | 28.85% | 76.83% |
| 6 | 504 | 29.23% | 87.69% |
| 7 | 507 | 29.38% | **94.88%** |

LBS 5 -> 7 buys +1.8% tps for +18 points of HBM, and 94.88% leaves nothing for
checkpoint saves, eval allocations, or fragmentation over a long chain.
**LBS=6 and 7 are out on headroom.**

I then extrapolated that flattening *downward* and predicted 4 vs 5 would be
indistinguishable too, which would have made LBS=4 the pick on headroom.
**That was wrong.** Job `12473210` measured the pair interleaved
(4,5,4,5,4,5), three repeats each, in one job:

| arm | rep 1 | rep 2 | rep 3 | mean | spread |
|---|---:|---:|---:|---:|---:|
| LBS=4 | 499 / 28.92% | 499 / 28.94% | 500 / 28.99% | **499.3 / 28.95%** | 0.2% |
| LBS=5 | 512 / 29.68% | 512 / 29.65% | 510 / 29.55% | **511.3 / 29.63%** | 0.4% |

Within-job repeatability is **0.2-0.4%** (sd 0.47 tps for LBS=4), so
LBS=5's **+2.4% tps / +0.68pp MFU** is ~27x the spread. The flattening is
real but starts *after* LBS=5:

| step | gain | memory cost | verdict |
|---|---:|---:|---|
| 4 -> 5 | **+2.4% tps, +0.68pp** | +8.7pp | **worth it** |
| 5 -> 6 | +1.2% | +10.9pp | marginal |
| 6 -> 7 | +0.6% | +7.2pp | no |

**Production pick: LBS=5 at 76.83%.** A real gain with comfortable headroom --
and precisely the configuration OLMo-2's smaller vocab makes reachable, since
Llama-3 needs 85.84% for the same batch.

**Method note.** The first version of this claim (+0.73pp) came from comparing
two *different* jobs, and the controlled answer is +0.68pp -- so the number was
right but the evidence for it was not, and a later cross-job run (498 tps)
made it look refuted. Interleaving both arms inside one job turned an
undecidable comparison into a 27-sigma one. **At this scale, cross-job spread
(~3%) exceeds most effects worth measuring; run competing arms in the same
job, alternating, or the allocation becomes the variable.**

## Relation to the other tokenizer findings

- [exp01](exp01-tokenizer-analysis.md) refuted the proposal's digit-splitting
  and LaTeX arguments but **supported** the vocab-oversizing claim on measured
  mass concentration. This page rests on that surviving half.
- [exp05](exp05-2n-performance.md) measured the 128k alternative end to end:
  a tie in MFU at matched batch, but +1.12pp best-vs-best because its freed
  memory buys LBS=4. A 64k would free about twice as much again.
