# Synthetic-summary data generation (POC)

Generate high-quality **synthetic mid-training data** by summarizing our existing
`olmo-mix-1124` tokens. The corpus is stored as gemma-tokenized Megatron/blendcorpus
indexed datasets (`.bin`/`.idx`), so the pipeline is a three-step round-trip:

```
olmo-mix .bin/.idx  --(1 detok)-->  source text (JSONL)
source text         --(2 summarize)-->  synthetic summaries (JSONL)
summaries           --(3 retok)-->  synthetic .bin/.idx  (drop-in training shard)
```

The synthetic summaries are dense, faithful restatements of the source facts. The
hypothesis: mid-training on higher-information-density text improves benchmark
performance per token vs. re-seeing raw web text (the same lever that made the
dolmino stage-2 mix help -- see [`../docs/live/cpt/`](../docs/live/chains/cpt/README.md)).

This is a **proof-of-concept pilot** (a ~2k-doc wiki slice), not a production
data-gen system. It validates the round-trip is lossless and the summaries are
usable before committing compute to a full generation run.

## Components

| Step | Script | Env | Where |
|------|--------|-----|-------|
| 1. detok | `detok_to_text.py` | `.venv` (transformers + blendcorpus) | login node (CPU) |
| 2. summarize | `summarize_text.py` + `submit_summarize.sh` | `.venv` (transformers) | compute node (1 XPU tile) |
| 3. retok | `retok_to_bin.py` | `.venv` (transformers + blendcorpus) | login node (CPU) |

### 1. Detokenize (`detok_to_text.py`)
Reads a shard prefix via `blendcorpus.data.indexed_dataset.MMapIndexedDataset`,
gemma-decodes each document, writes JSONL `{"id","n_tok","text"}`. The `--verify`
mode is the **correctness gate**: it checks the recovered text is a tokenizer
**fixed-point** (`decode -> encode -> decode` is stable). Exact orig-id match is
NOT the gate -- SentencePiece decode is many-to-one and the corpus carries a
leading BOS + trailing EOD, so id-equality is expectedly low; text faithfulness
is what matters.

### 2. Summarize (`summarize_text.py`)
Batched HF `generate` with an instruction-tuned model (default
`Llama-3.1-8B-exvocab`) on a single XPU tile. Prompt is tuned for faithfulness
(no new facts, no opinions). Writes JSONL `{"id","n_tok_src","summary"}`.
Uses plain transformers (not vLLM) -- there is no XPU vLLM venv in this clone and
building one touches the torch stack (forbidden by the ezpz env rules); for a
pilot, batched HF generate is adequate. Some on-disk instruct checkpoints ship a
`chat_template` that ignores `add_generation_prompt`; the script detects this and
appends the ChatML assistant cue manually.

### 3. Retokenize (`retok_to_bin.py`)
Gemma-tokenizes each summary, appends EOD (`eos_token_id`), writes an int32
`MMapIndexedDataset` shard (`.bin`/`.idx`) via `MMapIndexedDatasetBuilder` --
same format as the source olmo-mix shards, so it is a drop-in training input.
Reopens the shard and decodes doc0 as a readback sanity check.

## Report

Pilot results + eval gate: [`../docs/records/experiments/synthetic/aurora/`](../docs/records/experiments/synthetic/aurora/).
