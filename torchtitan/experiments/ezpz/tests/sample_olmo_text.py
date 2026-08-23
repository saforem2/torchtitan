#!/usr/bin/env python3
"""Extract a stratified plain-text sample from olmo-mix-1124 for tokenizer work.

READ-ONLY on the dataset. This script opens `data/**/*.json.gz` for reading and
writes ONLY under --out, which must live outside the dataset root. It never
touches `data_gemma_eod/` (the tokenized corpus backing the completed 4.674T 2B
flagship) and never writes inside `olmo-mix-1124/` at all.

Produces two disjoint splits per domain:
  train-<domain>.txt   -- for fitting a candidate tokenizer
  heldout-<domain>.txt -- for measuring fertility, never seen in training

Disjointness is by SOURCE FILE, not by line: a shard contributes to exactly one
split. Sampling within a shard is deterministic given --seed.

Usage (on Aurora, login node is fine):
    python3 sample_olmo_text.py --out /tmp/olmo-sample --mb-per-domain 64
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import shutil
import subprocess
import sys

# dclm is 94.8% of the corpus by tokens, but a fertility study needs each
# domain represented well enough to measure separately, so the sample is
# stratified (equal bytes per domain) rather than proportional. Per-domain
# numbers get combined with corpus weights afterwards.
DOMAINS = (
    "dclm",
    "starcoder",
    "pes2o",
    "arxiv",
    "open-web-math",
    "algebraic-stack",
    "wiki",
)

# Fields that hold document text in olmo/dolma-style jsonl.
TEXT_KEYS = ("text", "content", "raw_content")


def find_shards(root: str, domain: str) -> list[str]:
    """All .json.gz under a domain, sorted for determinism."""
    base = os.path.join(root, "data", domain)
    out = []
    for dirpath, _dirnames, filenames in os.walk(base):
        for fn in filenames:
            if fn.endswith((".json.gz", ".jsonl.gz", ".jsonl.zstd", ".json.zstd", ".jsonl.zst")):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def open_shard(path: str):
    """Line iterator for a .gz or .zstd shard.

    dclm -- 94.8% of the corpus -- ships as `.jsonl.zstd`, which an
    extension filter for `.gz` silently skips. That is exactly what happened
    on the first run: six domains sampled and the one that matters reported
    "NO SHARDS FOUND". The python `zstandard` module is not installed on the
    Aurora login nodes, so fall back to the `zstd` CLI, which is.
    """
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    try:
        import zstandard  # noqa: F401
    except ImportError:
        if not shutil.which("zstd"):
            raise RuntimeError(
                "need the `zstandard` module or the `zstd` CLI to read " + path
            )
        proc = subprocess.Popen(
            ["zstd", "-dc", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        return io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
    import zstandard

    fh = open(path, "rb")
    reader = zstandard.ZstdDecompressor().stream_reader(fh)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def extract_text(obj: dict) -> str | None:
    for k in TEXT_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def drain(shards: list[str], target_bytes: int, out_path: str) -> tuple[int, int]:
    """Write documents from `shards` into out_path until target_bytes.

    Returns (bytes_written, docs_written). Documents are separated by a blank
    line so a tokenizer trainer sees document boundaries.
    """
    written = 0
    docs = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for shard in shards:
            if written >= target_bytes:
                break
            try:
                with open_shard(shard) as fh:
                    for line in fh:
                        if written >= target_bytes:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = extract_text(obj)
                        if text is None:
                            continue
                        out.write(text)
                        out.write("\n\n")
                        written += len(text.encode("utf-8")) + 2
                        docs += 1
            except (OSError, EOFError, RuntimeError) as e:
                # A truncated shard should not kill the sample.
                print(f"  warn: {os.path.basename(shard)}: {e}", file=sys.stderr)
                continue
    return written, docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/lus/flare/projects/AuroraGPT/datasets/olmo-mix-1124",
        help="dataset root; opened READ-ONLY",
    )
    ap.add_argument("--out", required=True, help="output dir (must be OUTSIDE --root)")
    ap.add_argument(
        "--mb-per-domain",
        type=int,
        default=64,
        help="MB of training text per domain (held-out gets 1/4 of this)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = os.path.realpath(args.root)
    out = os.path.realpath(args.out)
    # Refuse to write anywhere inside the dataset. The gemma-tokenized corpus
    # is irreplaceable and the raw text is the only copy of the source.
    if out == root or out.startswith(root + os.sep):
        print(
            f"REFUSING: --out {out} is inside the dataset root {root}.\n"
            "This dataset is read-only; write the sample to scratch instead.",
            file=sys.stderr,
        )
        return 2
    os.makedirs(out, exist_ok=True)

    random.seed(args.seed)
    train_bytes = args.mb_per_domain * 1024 * 1024
    held_bytes = train_bytes // 4

    print(f"root (read-only): {root}")
    print(f"out:              {out}")
    print(f"per domain: {args.mb_per_domain} MB train + {args.mb_per_domain//4} MB held-out\n")

    totals = {}
    for domain in DOMAINS:
        shards = find_shards(root, domain)
        if not shards:
            print(f"{domain:18s} NO SHARDS FOUND -- skipping")
            continue
        # Shuffle deterministically, then split by file so train and held-out
        # never share a source shard.
        rng = random.Random(args.seed)
        rng.shuffle(shards)
        cut = max(1, len(shards) // 5)
        held_shards, train_shards = shards[:cut], shards[cut:]

        tb, td = drain(train_shards, train_bytes, os.path.join(out, f"train-{domain}.txt"))
        hb, hd = drain(held_shards, held_bytes, os.path.join(out, f"heldout-{domain}.txt"))
        totals[domain] = (tb, td, hb, hd)
        print(
            f"{domain:18s} shards={len(shards):5d}  "
            f"train {tb/1e6:6.1f} MB / {td:6d} docs   "
            f"heldout {hb/1e6:5.1f} MB / {hd:5d} docs"
        )

    print()
    tot_t = sum(v[0] for v in totals.values())
    tot_h = sum(v[2] for v in totals.values())
    print(f"TOTAL train {tot_t/1e6:.1f} MB, heldout {tot_h/1e6:.1f} MB")
    print(f"wrote to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
