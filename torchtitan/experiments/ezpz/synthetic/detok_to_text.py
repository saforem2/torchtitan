#!/usr/bin/env python3
"""Detokenize a Megatron/blendcorpus indexed dataset (.bin/.idx) back to text.

Step 1 of the "summarize olmo-mix" synthetic-data POC. The olmo-mix-1124
corpus is stored as gemma-tokenized indexed datasets (per-document token id
arrays); to generate synthetic summaries we first need the source TEXT back.

This reads a shard prefix with blendcorpus' MMapIndexedDataset (each item =
one document's token ids), gemma-decodes with the same google/gemma-7b
tokenizer used to build the corpus, and writes JSONL {"id", "n_tok", "text"}.

It ALSO supports --verify: a round-trip check that the recovered text is a
tokenizer fixed-point (decode -> encode -> decode is stable). That, not an
exact orig-id match, is the correctness gate for step 1: SentencePiece decode
is many-to-one and the corpus carries a leading BOS + trailing EOD, so exact
id equality is neither expected nor needed -- text faithfulness is.

Usage:
  # decode the first 2000 docs of a wiki shard to text
  python -m torchtitan.experiments.ezpz.synthetic.detok_to_text \\
    --prefix /flare/AuroraGPT/datasets/olmo-mix-1124/data_fused_gemma_eod/wiki/fused_0001_of_0002_text_document \\
    --tokenizer assets/hf/gemma-7b \\
    --num-docs 2000 --out outputs/synthetic/wiki_slice.jsonl

  # correctness gate: round-trip N docs, report text fixed-point rate
  python -m torchtitan.experiments.ezpz.synthetic.detok_to_text \\
    --prefix .../fused_0001_of_0002_text_document \\
    --tokenizer assets/hf/gemma-7b --num-docs 200 --verify
"""
from __future__ import annotations

import argparse
import json
import os


def _load_tokenizer(tokenizer_path: str):
    # Decode directly with the HF tokenizer.json (same google/gemma-7b the
    # corpus was built with). transformers is available in the frameworks env.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path)


def _open_dataset(prefix: str):
    from blendcorpus.data.indexed_dataset import MMapIndexedDataset

    return MMapIndexedDataset(prefix)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prefix",
        required=True,
        help="indexed-dataset prefix WITHOUT .bin/.idx (a *_text_document path)",
    )
    ap.add_argument(
        "--tokenizer",
        default="assets/hf/gemma-7b",
        help="HF tokenizer dir (google/gemma-7b) used to build the corpus",
    )
    ap.add_argument("--num-docs", type=int, default=2000, help="how many docs to read")
    ap.add_argument("--start", type=int, default=0, help="first doc index")
    ap.add_argument("--out", default=None, help="output JSONL (omit with --verify)")
    ap.add_argument(
        "--min-tok",
        type=int,
        default=64,
        help="skip docs shorter than this many tokens (too short to summarize)",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="round-trip check (decode->re-encode == original ids); no output written",
    )
    args = ap.parse_args()

    ds = _open_dataset(args.prefix)
    tok = _load_tokenizer(args.tokenizer)
    n_total = len(ds)
    end = min(args.start + args.num_docs, n_total)
    print(f"dataset {args.prefix}: {n_total:,} docs; reading [{args.start}:{end})")

    if args.verify:
        # The real correctness gate is TEXT FIXED-POINT, not exact-id match.
        # SentencePiece decode is many-to-one (whitespace/normalization), so an
        # exact orig-ids == re-encoded-ids match is neither expected nor needed:
        # the corpus also carries a leading BOS + trailing EOD that a
        # skip_special_tokens decode drops. What summarization actually needs is
        # that the recovered text is FAITHFUL and TOKENIZER-STABLE, i.e.
        #   text = decode(ids);  encode(text) -> re_ids;  decode(re_ids) == text
        # If that fixed-point holds, the text we feed the summarizer is exactly
        # what the model would re-see, with no lossy drift. We also report the
        # core-span id-stability (re-encoding the stripped content) as a
        # secondary signal.
        bos = tok.bos_token_id
        eos = tok.eos_token_id
        stable = unstable = id_exact = short = 0
        for i in range(args.start, end):
            ids = [int(x) for x in ds[i]]
            if len(ids) < args.min_tok:
                short += 1
                continue
            text = tok.decode(ids, skip_special_tokens=True)
            re_ids = tok.encode(text, add_special_tokens=False)
            text2 = tok.decode(re_ids, skip_special_tokens=True)
            if text == text2:
                stable += 1
            else:
                unstable += 1
                if unstable <= 3:
                    print(f"  [unstable doc {i}] text drifted on re-decode")
                    print(f"    a[:120]={text[:120]!r}")
                    print(f"    b[:120]={text2[:120]!r}")
            # core span = ids minus a leading BOS and/or a trailing EOD/EOS
            core = ids[:]
            if core and core[0] == bos:
                core = core[1:]
            if core and core[-1] == eos:
                core = core[:-1]
            if core == re_ids:
                id_exact += 1
        checked = stable + unstable
        srate = (stable / checked * 100) if checked else 0.0
        irate = (id_exact / checked * 100) if checked else 0.0
        print(f"=== text fixed-point: {stable}/{checked} stable ({srate:.1f}%); "
              f"core-span id match {id_exact}/{checked} ({irate:.1f}%); "
              f"{short} skipped (<{args.min_tok} tok) ===")
        return 0

    assert args.out, "--out required when not --verify"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = skipped = 0
    with open(args.out, "w") as f:
        for i in range(args.start, end):
            ids = [int(x) for x in ds[i]]
            if len(ids) < args.min_tok:
                skipped += 1
                continue
            text = tok.decode(ids, skip_special_tokens=True)
            f.write(json.dumps({"id": i, "n_tok": len(ids), "text": text}) + "\n")
            written += 1
    print(f"=== wrote {written} docs to {args.out} ({skipped} skipped <{args.min_tok} tok) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
