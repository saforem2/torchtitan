# exp07 -- would a custom ~64k tokenizer pay, and what does retokenizing cost?

> **2026-08-16.** Desk study on Aurora against the real `olmo-mix-1124` trees.
> No training runs. Follows
> [exp01](exp01-tokenizer-analysis.md) (which measured vocab utilisation) and
> [exp05](exp05-2n-performance.md) (which measured the 128k alternative).

## The two questions

1. Is it worth training our own tokenizer on `olmo-mix-1124`?
2. What do we do about the corpus already being tokenized on disk?

**Short answers: probably yes, and the second is much less of an obstacle than
it sounds -- the raw text is still on disk, and a <=65,535 vocab would HALVE
the tokenized corpus rather than cost extra storage.**

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

**This is an assumption, not a measurement** -- 5-15% is the usual range for
in-domain versus general-purpose tokenizers, but we have not measured what
*our* corpus would give. See "cheap next step" below.

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

## Cheap next step, before anything expensive

Train a candidate 64k BPE on a stratified sample and **measure** fertility
against gemma-256k and Llama-3-128k on held-out text from each domain. CPU
only, hours not days, no cluster allocation. That converts the one unmeasured
term into a number, and the decision follows from it:

- **>= 8-10% better fertility:** compelling. Compute saving alone justifies the
  retokenization job, before counting the 2.36B freed params and 8 TB.
- **~5%:** marginal. The embedding and storage wins still stand, but they are
  available at lower risk from the already-vendored 128k.
- **< 5%:** do not train one. Use Llama-3 128k, which exp05 already measured at
  +1.12pp MFU over gemma and needs no new artifact.

## Relation to the other tokenizer findings

- [exp01](exp01-tokenizer-analysis.md) refuted the proposal's digit-splitting
  and LaTeX arguments but **supported** the vocab-oversizing claim on measured
  mass concentration. This page rests on that surviving half.
- [exp05](exp05-2n-performance.md) measured the 128k alternative end to end:
  a tie in MFU at matched batch, but +1.12pp best-vs-best because its freed
  memory buys LBS=4. A 64k would free about twice as much again.
