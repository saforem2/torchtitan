# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stream a Hugging Face dataset from a frozen file manifest.

``datasets.load_dataset(repo, streaming=True)`` resolves the split on every
rank through the Hub API (``/api/datasets/<repo>/revision`` and one paginated
``/tree`` listing per directory). Explicit ``huggingface.co`` URLs in
``data_files`` do not avoid it: ``datasets`` rewrites them to ``hf://`` paths
and lists their parent directories. For ``allenai/olmo-mix-1124`` (28,880
files in ~100 directories) at 24 ranks that exceeds the Hub's 1000 API
requests per 5 minutes before the first sample is read.

A manifest pins the repo commit and its file list once, offline, with
``scripts/freeze_hf_manifest.py``. At runtime each DP rank takes every
``dp_world_size``-th file and streams it with one plain HTTP GET to
``https://huggingface.co/datasets/<repo>/resolve/<commit>/<path>``. No rank
calls the Hub API. Tokenization still happens at runtime, downstream.

Manifest format (gzip JSON)::

    {"repo": "allenai/olmo-mix-1124",
     "revision": "<40-hex commit>",
     "files": ["data/wiki/wiki-0000.json.gz", ...]}
"""

from __future__ import annotations

import dataclasses
import gzip
import io
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import grain
import requests

from torchtitan.components.data.sources import HuggingFaceStreamingSource
from torchtitan.components.data.types import DatasetIterationPolicy

MANIFEST_DIR = Path(__file__).resolve().parent / "data-lists" / "hf-manifests"

_HF_ENDPOINT = "https://huggingface.co"
_MAX_ATTEMPTS = 5


def load_manifest(path: str | Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as f:
        manifest = json.load(f)
    if len(manifest.get("revision", "")) != 40:
        raise ValueError(f"manifest {path} must pin a full commit sha")
    if not manifest.get("files"):
        raise ValueError(f"manifest {path} lists no files")
    return manifest


def resolve_url(repo: str, revision: str, path: str) -> str:
    return f"{_HF_ENDPOINT}/datasets/{repo}/resolve/{revision}/{path}"


def _get(url: str) -> requests.Response:
    """GET with retry on 429/5xx, honoring ``Retry-After``."""
    from huggingface_hub import get_token

    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(_MAX_ATTEMPTS):
        resp = requests.get(url, headers=headers, stream=True, timeout=120)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == _MAX_ATTEMPTS - 1:
                resp.raise_for_status()
            wait = float(resp.headers.get("Retry-After", 2 ** (attempt + 2)))
            resp.close()
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    raise AssertionError("unreachable")


def _read_rows(url: str) -> Iterator[dict[str, Any]]:
    resp = _get(url)
    resp.raw.decode_content = True
    with resp:
        if url.endswith(".parquet"):
            import pyarrow.parquet as pq

            # Parquet needs random access; one file is a few hundred MB.
            table_file = pq.ParquetFile(io.BytesIO(resp.raw.read()))
            for batch in table_file.iter_batches():
                yield from batch.to_pylist()
            return
        if url.endswith(".gz"):
            stream: Any = gzip.GzipFile(fileobj=resp.raw)
        elif url.endswith((".zst", ".zstd")):
            import zstandard

            stream = zstandard.ZstdDecompressor().stream_reader(resp.raw)
        elif url.endswith((".json", ".jsonl")):
            stream = resp.raw
        else:
            raise ValueError(f"unsupported manifest file format: {url}")
        for line in io.TextIOWrapper(stream, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def _generate(urls: list[str]) -> Iterator[dict[str, Any]]:
    for url in urls:
        yield from _read_rows(url)


class HuggingFaceManifestStreamingSource(HuggingFaceStreamingSource):
    """``HuggingFaceStreamingSource`` that reads a frozen manifest, not the Hub API.

    ``path`` is the manifest file; ``split`` is unused and kept only to
    satisfy the parent config. Each DP rank owns ``files[dp_rank::dp_world_size]``,
    so the manifest must list at least ``dp_world_size`` files.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceStreamingSource.Config):
        split: str = "train"

    def __init__(self, config: Config) -> None:
        if config.name or config.revision or config.load_dataset_kwargs:
            raise ValueError(
                "manifest source takes the commit and files from the manifest; "
                "name, revision and load_dataset_kwargs must be unset"
            )
        manifest = load_manifest(config.path)
        self._urls = [
            resolve_url(manifest["repo"], manifest["revision"], f)
            for f in manifest["files"]
        ]

    def build_iter_dataset(
        self, *, dataset_iteration_policy: DatasetIterationPolicy
    ) -> grain.IterDataset:
        rank = dataset_iteration_policy.dp_rank
        world = dataset_iteration_policy.dp_world_size
        if len(self._urls) < world:
            raise ValueError(
                f"manifest lists {len(self._urls)} files for {world} DP ranks"
            )
        urls = self._urls[rank::world]
        # One generator shard per file, so the cursor checkpoints as
        # (file index, row offset) and resumes without re-listing anything.
        self._dataset = datasets.IterableDataset.from_generator(
            _generate, gen_kwargs={"urls": urls}
        )
        # This rank's files are already selected; the parent's DP split is
        # the identity at world size 1.
        return super().build_iter_dataset(
            dataset_iteration_policy=dataclasses.replace(
                dataset_iteration_policy, dp_rank=0, dp_world_size=1
            )
        )
