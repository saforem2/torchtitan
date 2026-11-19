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
     "files": ["data/wiki/wiki-0000.json.gz", ...],
     "row_groups": [19, 19, ...]}   # optional, parquet only

Without ``row_groups`` the unit of DP sharding is a file. With it (freeze with
``--row-groups``) the unit is one parquet row group, read with HTTP range
requests, so a repo with fewer files than DP ranks still splits evenly instead
of every rank reading every row (``split_dataset_by_node``'s fallback).
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
    row_groups = manifest.get("row_groups")
    if row_groups is not None:
        if len(row_groups) != len(manifest["files"]):
            raise ValueError(f"manifest {path}: row_groups and files differ in length")
        if not all(isinstance(n, int) and n > 0 for n in row_groups):
            raise ValueError(f"manifest {path}: row_groups must be positive integers")
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


def _read_row_group(url: str, row_group: int) -> list[dict[str, Any]]:
    """Read one parquet row group, retrying transient network failures.

    The whole row group is materialized before any row is yielded, so a retry
    never repeats rows. The ``resolve`` redirect is followed per row group, so
    the signed CDN URL cannot expire mid-run.
    """
    from urllib.parse import urljoin

    import aiohttp
    import fsspec
    import pyarrow.parquet as pq
    from huggingface_hub import get_token

    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    # trust_env: aiohttp ignores HTTP(S)_PROXY otherwise, and ALCF compute nodes
    # reach the internet only through the proxy (direct connects time out).
    http = fsspec.filesystem("http", client_kwargs={"trust_env": True})
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = requests.head(
                url, headers=headers, allow_redirects=False, timeout=60
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code} resolving {url}")
            if resp.is_redirect:
                cdn_url = urljoin(url, resp.headers["Location"])
            else:
                resp.raise_for_status()
                cdn_url = url
            with http.open(cdn_url, "rb") as f:
                return pq.ParquetFile(f).read_row_group(row_group).to_pylist()
        except (requests.RequestException, aiohttp.ClientError, OSError):
            # fsspec reports any failed HEAD as FileNotFoundError (an OSError).
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            time.sleep(2 ** (attempt + 2))
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


def _generate(units: list[tuple[str, int]]) -> Iterator[dict[str, Any]]:
    """Yield rows for each ``(url, row_group)`` unit; ``-1`` means the whole file."""
    for url, row_group in units:
        if row_group < 0:
            yield from _read_rows(url)
        else:
            yield from _read_row_group(url, row_group)


class HuggingFaceManifestStreamingSource(HuggingFaceStreamingSource):
    """``HuggingFaceStreamingSource`` that reads a frozen manifest, not the Hub API.

    ``path`` is the manifest file; ``split`` is unused and kept only to
    satisfy the parent config. Each DP rank owns
    ``units[dp_rank::dp_world_size]``, where a unit is a file or, for a manifest
    with ``row_groups``, one parquet row group. There must be at least
    ``dp_world_size`` units.
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
        urls = [
            resolve_url(manifest["repo"], manifest["revision"], f)
            for f in manifest["files"]
        ]
        row_groups = manifest.get("row_groups")
        if row_groups is None:
            self._units = [(url, -1) for url in urls]
        else:
            self._units = [
                (url, rg) for url, n in zip(urls, row_groups) for rg in range(n)
            ]

    def build_iter_dataset(
        self, *, dataset_iteration_policy: DatasetIterationPolicy
    ) -> grain.IterDataset:
        rank = dataset_iteration_policy.dp_rank
        world = dataset_iteration_policy.dp_world_size
        if len(self._units) < world:
            raise ValueError(
                f"manifest has {len(self._units)} units for {world} DP ranks"
            )
        # One generator shard per unit, so the cursor checkpoints as
        # (unit index, row offset) and resumes without re-listing anything.
        self._dataset = datasets.IterableDataset.from_generator(
            _generate, gen_kwargs={"units": self._units[rank::world]}
        )
        # This rank's files are already selected; the parent's DP split is
        # the identity at world size 1.
        return super().build_iter_dataset(
            dataset_iteration_policy=dataclasses.replace(
                dataset_iteration_policy, dp_rank=0, dp_world_size=1
            )
        )
