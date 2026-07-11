#!/usr/bin/env python3
"""Retokenize synthetic summaries into a blendcorpus indexed dataset (.bin/.idx).

Step 3 (final) of the "summarize olmo-mix" synthetic-data POC. Reads the
summaries JSONL (from summarize_text.py, {"id", "summary", ...}), gemma-tokenizes
each summary, appends the end-of-document token, and writes a Megatron/blendcorpus
MMapIndexedDataset shard that the training dataloader can consume directly (same
format as the source olmo-mix shards).

Convention (matches blendcorpus preprocessing): one document per summary, tokens
= gemma-encode(summary) + [EOD], dtype int32, EOD = tokenizer.eos_token_id.

Usage (frameworks .venv, has transformers + blendcorpus):
  python -m torchtitan.experiments.ezpz.synthetic.retok_to_bin \\
    --in outputs/synthetic/wiki_summaries.jsonl \\
    --out-prefix outputs/synthetic/wiki_synth_text_document \\
    --tokenizer assets/hf/gemma-7b

Writes <out-prefix>.bin and <out-prefix>.idx. Point a training run at it with
  --dataloader.dataset blendcorpus --dataloader.dataset-path <a .txt listing the prefix>
(the same way the olmo-mix data lists reference their shard prefixes).
"""
from __future__ import annotations

import argparse
import json
import os


def _read_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="summaries JSONL")
    ap.add_argument(
        "--out-prefix",
        required=True,
        help="output prefix WITHOUT .bin/.idx (convention: *_text_document)",
    )
    ap.add_argument("--tokenizer", default="assets/hf/gemma-7b")
    ap.add_argument(
        "--field",
        default="summary",
        help="JSONL field to tokenize (summary; use 'text' to retok raw text)",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=32,
        help="skip summaries shorter than this (degenerate/empty generations)",
    )
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from blendcorpus.data.indexed_dataset import MMapIndexedDatasetBuilder

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eod = tok.eos_token_id
    assert eod is not None, "tokenizer has no eos_token_id for EOD"

    # Source olmo-mix shards are int32; match that so the shard is a drop-in.
    dtype = np.int32
    bin_path = args.out_prefix + ".bin"
    idx_path = args.out_prefix + ".idx"
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    builder = MMapIndexedDatasetBuilder(bin_path, dtype=dtype)
    n_docs = n_tok = skipped = 0
    for row in _read_jsonl(args.inp):
        s = (row.get(args.field) or "").strip()
        if len(s) < args.min_chars:
            skipped += 1
            continue
        ids = tok.encode(s, add_special_tokens=False)
        ids.append(eod)  # end-of-document marker (Megatron convention)
        # MMapIndexedDatasetBuilder.add_item calls tensor.numpy() internally,
        # so it wants a torch tensor (int32 to match the source shards), not a
        # numpy array.
        builder.add_item(torch.tensor(ids, dtype=torch.int32))
        builder.end_document()
        n_docs += 1
        n_tok += len(ids)
    builder.finalize(idx_path)

    avg = (n_tok / n_docs) if n_docs else 0
    print(
        f"=== wrote {n_docs} docs / {n_tok:,} tokens (avg {avg:.0f}/doc) to "
        f"{bin_path} + {idx_path}; {skipped} skipped (<{args.min_chars} chars) ==="
    )

    # Round-trip sanity: reopen and confirm doc count + a decoded sample.
    from blendcorpus.data.indexed_dataset import MMapIndexedDataset

    ds = MMapIndexedDataset(args.out_prefix)
    assert len(ds) == n_docs, f"readback docs {len(ds)} != written {n_docs}"
    sample = [int(x) for x in ds[0]]
    print(f"readback OK: {len(ds)} docs; doc0 len={len(sample)}, ends with EOD={sample[-1]==eod}")
    print(f"doc0 decoded[:200]={tok.decode(sample, skip_special_tokens=True)[:200]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
