#!/usr/bin/env python3
"""Build 2B CPT data-list mixes (olmo x dolmino) by renormalizing each source
block to its target fraction, so the blendcorpus weight column encodes the
intended olmo:dolmino sampling ratio exactly.

The two source lists use different weight-sum conventions (torchtitan
olmo-mix-1124 sums to ~2.717; the Megatron-DeepSpeed dolmino list sums to 1.0),
so a naive concat would NOT give the intended ratio. We rescale each block so
its weights sum to the target fraction, then concatenate -> total sums to 1.0.

Usage:
  build_cpt_mixes.py <olmo_src> <dolmino_src> <out_dir>

Writes: dolmino-mix-1124.txt, olmo50-dolmino50.txt, olmo25-dolmino75.txt
"""
import sys
from pathlib import Path


def read_list(path):
    """Return list of (weight, path, tag) tuples from a blendcorpus data-list."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # weight path tag  (tag may be absent in some lists; keep as-is)
            w = float(parts[0])
            rest = parts[1:]
            rows.append((w, rest))
    return rows


def scaled_block(rows, target_sum):
    """Rescale weights so they sum to target_sum; return formatted lines."""
    cur = sum(w for w, _ in rows)
    factor = target_sum / cur
    out = []
    for w, rest in rows:
        out.append(f"{w * factor:.10f} " + " ".join(rest))
    return out


def write_list(out_path, lines):
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    # sanity: re-read and sum
    total = sum(float(l.split()[0]) for l in lines)
    print(f"  wrote {out_path}  ({len(lines)} files, weight sum {total:.6f})")


def main():
    olmo_src, dolmino_src, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    out = Path(out_dir)
    olmo = read_list(olmo_src)
    dolmino = read_list(dolmino_src)
    print(f"olmo src: {len(olmo)} files; dolmino src: {len(dolmino)} files")

    # 1. dolmino-100 (port; already sums to 1.0, but renormalize to be exact)
    write_list(out / "dolmino-mix-1124.txt", scaled_block(dolmino, 1.0))

    # 2. olmo50-dolmino50
    lines = scaled_block(olmo, 0.50) + scaled_block(dolmino, 0.50)
    write_list(out / "olmo50-dolmino50.txt", lines)

    # 3. olmo25-dolmino75
    lines = scaled_block(olmo, 0.25) + scaled_block(dolmino, 0.75)
    write_list(out / "olmo25-dolmino75.txt", lines)


if __name__ == "__main__":
    main()
