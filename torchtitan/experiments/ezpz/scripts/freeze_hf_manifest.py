#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Freeze a Hugging Face dataset's file list into a manifest.

Resolves the repo's current commit (or ``--revision``) and lists its files
once, so training ranks stream from the manifest without Hub API calls. See
``torchtitan/experiments/ezpz/hf_manifest.py``.

    python torchtitan/experiments/ezpz/scripts/freeze_hf_manifest.py \
        allenai/olmo-mix-1124 --include 'data/*'

    python torchtitan/experiments/ezpz/scripts/freeze_hf_manifest.py \
        HuggingFaceTB/cosmopedia --include 'data/auto_math_text/*' \
        --include 'data/khanacademy/*' --name cosmopedia-textbooks

Pass ``--row-groups`` for parquet repos with fewer files than DP ranks; it
records each file's row-group count (one footer read per file) so ranks split
row groups instead of files.

Writes ``data-lists/hf-manifests/<name>.json.gz`` (``<name>`` defaults to
``<org>--<repo>``). Re-freeze to move to a newer commit.
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import json
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

_DATA_SUFFIXES = (".parquet", ".json", ".jsonl", ".json.gz", ".jsonl.zst", ".zstd")
_OUT_DIR = Path(__file__).resolve().parents[1] / "data-lists" / "hf-manifests"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repo")
    parser.add_argument(
        "--include",
        action="append",
        required=True,
        help="fnmatch pattern on repo paths; repeatable ('*' crosses '/')",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--row-groups",
        action="store_true",
        help="record parquet row-group counts so ranks shard by row group",
    )
    args = parser.parse_args()

    api = HfApi()
    revision = api.dataset_info(args.repo, revision=args.revision).sha
    files = sorted(
        entry.path
        for entry in api.list_repo_tree(
            args.repo, repo_type="dataset", recursive=True, revision=revision
        )
        if isinstance(entry, RepoFile)
        and entry.path.endswith(_DATA_SUFFIXES)
        and any(fnmatch.fnmatch(entry.path, p) for p in args.include)
    )
    if not files:
        raise SystemExit(f"no data files in {args.repo} match {args.include}")

    name = args.name or args.repo.replace("/", "--")
    out = _OUT_DIR / f"{name}.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"repo": args.repo, "revision": revision, "files": files}
    if args.row_groups:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        if not all(f.endswith(".parquet") for f in files):
            raise SystemExit("--row-groups needs parquet files only")
        fs = HfFileSystem()
        payload["row_groups"] = []
        for f in files:
            with fs.open(f"datasets/{args.repo}@{revision}/{f}", "rb") as fh:
                payload["row_groups"].append(pq.ParquetFile(fh).metadata.num_row_groups)
    # mtime=0 keeps the gzip bytes identical across re-freezes of one commit.
    with gzip.GzipFile(out, "wb", mtime=0) as f:
        f.write(json.dumps(payload, indent=0).encode())
    units = sum(payload.get("row_groups", [])) or len(files)
    print(f"{out}: {args.repo}@{revision[:12]} {len(files)} files, {units} units")


if __name__ == "__main__":
    main()
