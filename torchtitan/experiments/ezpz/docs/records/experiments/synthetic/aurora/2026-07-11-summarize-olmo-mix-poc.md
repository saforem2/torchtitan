# Synthetic-summary data generation POC (olmo-mix-1124)

> Last updated: 2026-07-14

**Goal.** Generate high-quality synthetic mid-training data by summarizing our
existing `olmo-mix-1124` tokens, and validate the full round-trip is lossless and
the summaries usable, before committing compute to a large generation run.

**Motivation.** The dolmino stage-2 mix helped by raising information density of
mid-training tokens (see [`../../../production/cpt/`](../../../../production/cpt/README.md)).
Dense, faithful summaries of the pretraining corpus are another way to raise
information-per-token. This POC tests the machinery + a first quality read.

## Pipeline

The corpus is stored as gemma-tokenized Megatron/blendcorpus indexed datasets
(`.bin`/`.idx`), so generation is a three-step round-trip
([code](../../../../../synthetic/README.md)):

1. **detok** (`detok_to_text.py`): `.bin` -> source text (JSONL), login node.
2. **summarize** (`summarize_text.py` + `submit_summarize.sh`): text -> summary
   (JSONL), 1 XPU tile via HF `generate` with `Llama-3.1-8B-exvocab`.
3. **retok** (`retok_to_bin.py`): summary -> synthetic `.bin`/`.idx`, login node.

Pilot source slice: first 2000 docs of
`/flare/AuroraGPT/datasets/olmo-mix-1124/data_fused_gemma_eod/wiki/fused_0001_of_0002_text_document`
(3.22M docs total); 1787 kept after the <64-token filter.

## Results

### Step 1 -- detok (CONFIRMED)
Correctness gate is **text fixed-point** (`decode -> encode -> decode` stable),
not exact orig-id match (SentencePiece decode is many-to-one; the corpus carries
a leading BOS + trailing EOD, so id-equality is expectedly low and irrelevant).

- 300-doc wiki verify: **265/265 text fixed-point (100.0%)**; core-span id match
  35/265 (13.2%, as expected); 35 skipped (<64 tok).
- Produced `outputs/synthetic/wiki_slice.jsonl`: 1787 docs (213 skipped).

### Step 3 -- retok (CONFIRMED)
Ran end-to-end on the raw-text slice as a machinery test (`--field text`):

- **1787 docs / 2,057,269 tokens** (avg 1151/doc) -> `wiki_rawtext_text_document.{bin,idx}`.
- Readback: 1787 docs; **doc0 len 553 == source doc0 len 553**; ends with EOD;
  decodes to the correct source text. Round-trip (detok -> retok) is lossless.
- Gotcha fixed: `MMapIndexedDatasetBuilder.add_item` wants a torch tensor (calls
  `.numpy()` internally), not a numpy array.

### Step 2 -- summarize (IN PROGRESS)
- Smoke (`LIMIT=4`, `Llama-3.1-8B-exvocab`, 1 tile): job `8664871` (debug-scaling).
  - **Root cause of the "no XPU available" failures (jobs 8664598, 8664725): a
    bad `ZE_AFFINITY_MASK`, NOT the env.** An env probe (`8664810`) showed
    `torch.xpu.is_available()` True in every combo (bare / frameworks / oneapi /
    `ezpz_setup_job` / `ezpz_setup_env`) -- 12 flat tiles. The wrapper had pinned
    `ZE_AFFINITY_MASK="0.0"`, but under `ZE_FLAT_DEVICE_HIERARCHY=FLAT` tiles are
    flat integers (0..11); the composite `device.subdevice` form `"0.0"` selects
    NOTHING -> 0 visible devices. Fixed: don't set the mask (let `.to("xpu")`
    use `xpu:0`); if pinning is ever needed, use a bare int. See
    [`memory/project_ze_affinity_mask_flat_format`].
  - Model `chat_template` ignores `add_generation_prompt` (ChatML); the script
    detects this and appends the assistant cue manually.
- **Smoke `8664871` succeeded (rc=0)**: 4 docs summarized in 17s (incl. model
  load; ~0.23 docs/s cold). Quality is **mixed** -- 2/4 clean:
  - docs 0, 3: excellent, faithful, self-contained summaries.
  - doc 2: degenerate -- echoed the system prompt (10021-tok source truncated at
    6000 chars).
  - doc 4: degenerate -- meta-commentary ("The summary is a bit too long...")
    on a short 118-tok source.
  - `Llama-3.1-8B-exvocab` is an extended-vocab continued-pretrain, a weak
    instruction-follower.
- **Model comparison decided it: `DeepHermes-3-Llama-3-3B` wins 4/4 vs 2/4**
  (job `8665047`, rc=0, 0.31 docs/s). Every summary is a faithful, dense
  restatement -- including doc 2 (correctly summarized the alkenes content
  despite the 10021->6000-char truncation) and the short doc 4 that broke
  exvocab. It is also smaller (3B) and faster. **DeepHermes-3 is the pilot
  model.** QC keeps 4/4.
- **QC filter** (`qc_summaries.py`): rejects prompt-echo / meta-commentary /
  too-short between summarize and retok. exvocab smoke: kept 2 good (0,3),
  rejected doc 2 (prompt_echo) + doc 4 (too_short). DeepHermes smoke: kept 4/4.
- **Full pilot COMPLETE: job `8665137`** (DeepHermes-3, batch 16, 1787 docs) --
  rc=0, **1787/1787 summarized in 1274s = 1.40 docs/s** (batch 16 is ~4.5x the
  cold-smoke rate on 1 tile).
  - **QC at scale: 1784/1787 kept (99.8%)** -- 2 too-short, 1 meta-commentary.
    DeepHermes-3 is a clean summarizer; the QC filter is a light safety net, not
    load-bearing.

### Step 3 -- retok pilot (CONFIRMED)
- 1784 QC'd summaries -> `outputs/synthetic/wiki_synth_text_document.{bin,idx}`,
  a drop-in blendcorpus training shard. Readback: **1784 docs / 229,104 tokens**
  (avg 128 tok/doc), doc0 ends with EOD, decodes to clean summary text.

**Compression finding (the crux for the eval gate).** The 1784 source docs held
2,057,269 tokens (raw-text retok); their summaries are **229,104 tokens -- ~9x
compression.** So synthetic mid-training trades token *count* for information
*density*. The eval gate must therefore compare at matched conditions carefully:
"same tokens" and "same docs" are different budgets here.

## POC verdict

The full round-trip works and is lossless where it must be:
detok (265/265 text fixed-point) -> summarize (DeepHermes-3, 1787/1787, faithful)
-> QC (1784/1787 kept, 99.8%) -> retok (drop-in .bin, readback clean). DeepHermes-3
clearly beat Llama-3.1-8B-exvocab (4/4 vs 2/4 on the smoke). Throughput on 1 XPU
tile: 1.40 docs/s at batch 16 (a full-corpus run would need many nodes + ideally
vLLM; see below).

## Eval gate (planned)

Fork a short 2B mid-training run (from the completed stage-1 base, gentle
constant LR -- same recipe as the CPT stage-2 experiments) and compare:

- **arm A (synthetic):** the `wiki_synth_text_document` shard (dense summaries).
- **arm B (raw-text control):** the `wiki_rawtext_text_document` shard (same 1784
  source docs, detok'd verbatim -- already built as the retok machinery test).

Compare benchmark deltas (ARC-Easy / HellaSwag) via the standard eval pipeline.
Because of the ~9x compression, run BOTH budgets: (1) equal *token* count (arm B
sees ~9x fewer docs) and (2) equal *doc/epoch* count (arm A sees ~9x fewer
tokens). Synthetic wins if it lifts benchmarks per *token* over raw re-exposure.
NOTE: the pilot slice (229K tokens) is far too small to move 2B benchmarks --
the eval gate needs a much larger synthetic corpus first (see scaling).

## Scaling (for a real run, not this pilot)

- Throughput here is **1.40 docs/s on ONE XPU tile** (HF `generate`, batch 16).
  A full olmo-mix pass (billions of docs) needs many nodes and, ideally, **vLLM**
  (no XPU vLLM venv exists in this clone yet -- building one touches the torch
  stack, so it was out of scope for the pilot). `submit_summarize.sh` shards by
  input JSONL, so a real run fans out N nodes over N input slices.
- Generating summaries over the entire 4.67T-token corpus would cost more than
  pretraining; a real synthetic-data effort should target a curated subset
  (high-value domains) rather than the whole corpus.

## Artifacts

- Code: [`torchtitan/experiments/ezpz/synthetic/`](../../../../../synthetic/README.md)
- Outputs: `outputs/synthetic/` (JSONL slices + `.bin`/`.idx` shards; not committed)
- Logs: `logs/synthetic-summarize-<jobid>/run.log`
