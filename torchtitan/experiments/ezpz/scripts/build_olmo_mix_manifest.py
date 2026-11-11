#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build a raw olmo-mix-1124 file manifest for ``agpt_v2_*_4k_olmo_mix``.

HF ``split_dataset_by_node`` gives each DP rank whole files only when the
file count is divisible by the DP world size. Otherwise every rank streams
the entire corpus and discards (N-1)/N of it, which is unusable at
thousands of ranks. This writes the largest file list that is a multiple of
``--dp``: every non-DCLM shard is kept, and DCLM shards are sampled evenly
across their sorted order to fill the remainder.

Usage:
    build_olmo_mix_manifest.py --dp 6144 --out manifest-4b.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_ROOT = "/flare/AuroraGPT/datasets/olmo-mix-1124/data"
SUBSETS = (
    "algebraic-stack",
    "arxiv",
    "dclm",
    "open-web-math",
    "pes2o",
    "starcoder",
    "wiki",
)
SUFFIXES = (".json.gz", ".json.zst", ".jsonl.zst", ".jsonl.zstd", ".jsonl.gz")


def _shards(root: Path, subset: str) -> list[str]:
    files = sorted(
        str(p.resolve())
        for p in (root / subset).rglob("*")
        if p.is_file() and p.name.endswith(SUFFIXES)
    )
    if not files:
        raise SystemExit(f"no shards under {root / subset}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dp", type=int, required=True, help="DP world size")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    args = parser.parse_args()
    if args.dp <= 0:
        raise SystemExit("--dp must be positive")

    by_subset = {s: _shards(args.root, s) for s in SUBSETS}
    other = [f for s in SUBSETS if s != "dclm" for f in by_subset[s]]
    dclm = by_subset["dclm"]
    total = len(other) + len(dclm)
    target = (total // args.dp) * args.dp
    if target < len(other) or target == 0:
        raise SystemExit(f"{total} shards cannot fill a multiple of dp={args.dp}")

    num_dclm = target - len(other)
    # Evenly spaced indices across the sorted DCLM list (no RNG, reproducible).
    picked = [dclm[(i * len(dclm)) // num_dclm] for i in range(num_dclm)]
    files = sorted(other + picked)
    assert len(files) == len(set(files)) == target

    args.out.write_text("\n".join(files) + "\n")
    counts = ", ".join(f"{s}={len(by_subset[s])}" for s in SUBSETS)
    print(f"available: {counts}")
    print(
        f"wrote {target} files ({target // args.dp} per rank at dp={args.dp}; "
        f"dclm {num_dclm}/{len(dclm)}) -> {args.out}"
    )


if __name__ == "__main__":
    main()
