# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Generic HuggingFace dataset support for torchtitan experiments.
#
# Import this module to enable arbitrary HF datasets:
#
#   import torchtitan.experiments.ezpz.datasets  # noqa: F401
#
# Then any HF dataset path works directly:
#
#   --dataloader.dataset stanfordnlp/imdb
#   --dataloader.dataset HuggingFaceFW/fineweb-edu
#   --dataloader.dataset eliplutchok/fineweb-small-sample
#
# Or register one explicitly for custom text columns / configs:
#
#   from torchtitan.experiments.ezpz.datasets import register_hf_dataset
#   register_hf_dataset("my_data", "my-org/my-data", text_column="content")

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from typing import Any

from datasets import load_dataset

from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets import text_datasets
from torchtitan.hf_datasets.text_datasets import DATASETS

log = logging.getLogger(__name__)

# Silence noisy HTTP request logs from huggingface_hub / httpx / urllib3
for _noisy in ("httpx", "huggingface_hub", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _rank0_prefetch_then_barrier(dataset_path: str, **kwargs: Any) -> Any:
    """Call ``load_dataset`` on rank 0 first to populate the HF cache
    (dataset_info.json + parquet file-listing manifests), barrier, then
    let worker ranks call it and hit the warm local cache.

    Why: at 32N (384 ranks) HF Hub's per-IP rate limit (1000 req / 5min)
    is blown immediately because every rank does a fresh
    ``GET /api/datasets/<repo>/tree?recursive=true`` listing. Job 12468290
    died here with 429s on ``allenai/olmo-mix-1124``. Pre-warming the
    cache from one rank collapses the metadata-API surface to ~O(1)
    requests; the actual data streaming after that is per-rank but the
    rate-limit-prone metadata calls are local-cache hits.

    Falls back to plain ``load_dataset`` when torch.distributed isn't
    initialized (single-process / interactive runs).
    """
    try:
        import torch.distributed as dist
    except Exception:
        return load_dataset(dataset_path, **kwargs)

    if not dist.is_available() or not dist.is_initialized():
        return load_dataset(dataset_path, **kwargs)

    rank = dist.get_rank()
    if rank == 0:
        log.info(
            f"[ezpz/datasets] rank 0 prefetching {dataset_path!r} "
            f"to warm HF cache (avoids 429 storms at scale)"
        )
        # Eagerly resolve the dataset on rank 0. For streaming=True this
        # is cheap (just metadata + first shard listing); for non-streaming
        # this downloads the whole thing — which is also fine since the
        # cache will be reused by everyone.
        ds = load_dataset(dataset_path, **kwargs)
        log.info(f"[ezpz/datasets] rank 0 cache warm for {dataset_path!r}")
        dist.barrier()
        return ds

    # Worker ranks wait, then hit the warm cache.
    dist.barrier()
    return load_dataset(dataset_path, **kwargs)


def _make_loader(
    *,
    config_name: str | None = None,
    split: str = "train",
    streaming: bool = True,
    trust_remote_code: bool = False,
) -> Callable:
    """Build a dataset loader callable for the DATASETS registry.

    Returns a callable that accepts a dataset path (from the registry or
    CLI override) and returns a HF dataset object.
    """

    def _load(dataset_path: str) -> Any:
        kwargs: dict[str, Any] = {
            "split": split,
            "streaming": streaming,
            "trust_remote_code": trust_remote_code,
        }
        if config_name is not None:
            kwargs["name"] = config_name
        return _rank0_prefetch_then_barrier(dataset_path, **kwargs)

    return _load


def _make_local_loader(
    *,
    data_dir: str,
    split: str = "train",
    streaming: bool = True,
    format: str = "parquet",
) -> Callable:
    """Build a loader for local parquet/arrow/json files."""

    def _load(dataset_path: str) -> Any:
        # Local files don't hit HF Hub rate limits but we still funnel
        # through the same prefetch+barrier path so all ranks see a
        # consistent file listing (avoids any rank-vs-rank disagreement
        # about which parquet shards exist).
        return _rank0_prefetch_then_barrier(
            format,
            data_dir=data_dir,
            split=split,
            streaming=streaming,
        )

    return _load


def register_local_dataset(
    name: str,
    data_dir: str,
    *,
    split: str = "train",
    text_column: str = "text",
    streaming: bool = True,
    format: str = "parquet",
) -> DatasetConfig:
    """Register a local dataset for use with torchtitan.

    Args:
        name: Registry key (used as --dataloader.dataset <name>).
        data_dir: Local directory containing the dataset files.
        split: Dataset split (default "train").
        text_column: Column containing the text to train on.
        streaming: Use streaming mode (default True).
        format: One of "parquet", "json", "arrow", "csv", "text" (default "parquet").

    Returns:
        The registered DatasetConfig.
    """
    config = DatasetConfig(
        path=data_dir,
        loader=_make_local_loader(
            data_dir=data_dir, split=split, streaming=streaming, format=format
        ),
        sample_processor=_make_text_processor(text_column),
    )
    DATASETS[name] = config
    log.debug(f"Registered local dataset {name!r} -> {data_dir}")
    return config


def _extract_text_column(sample: dict[str, Any], text_column: str) -> str:
    """Module-level helper for ``_make_text_processor``.

    Defined at module scope (not as a nested closure) so it can be
    pickled by ``multiprocessing.reduction`` when ``--dataloader.num-workers
    > 0`` causes the PyTorch DataLoader to fork worker processes via
    forkserver. Pickling a local closure here previously raised
    ``PicklingError: Can't pickle local object _make_text_processor.<locals>._process``
    and crashed every HF-dataset run with ``num_workers > 0``.
    """
    return sample[text_column]


def _make_text_processor(text_column: str = "text") -> Callable:
    """Build a sample processor that extracts text from a given column.

    Uses ``functools.partial`` over ``_extract_text_column`` (module-scope)
    instead of returning a local closure so the resulting callable is
    pickleable for DataLoader worker processes.
    """
    return functools.partial(_extract_text_column, text_column=text_column)


def register_hf_dataset(
    name: str,
    path: str,
    *,
    config_name: str | None = None,
    split: str = "train",
    text_column: str = "text",
    streaming: bool = True,
    trust_remote_code: bool = False,
) -> DatasetConfig:
    """Register a HuggingFace dataset for use with torchtitan's data pipeline.

    The registered dataset can be used with the existing
    HuggingFaceTextDataset/HuggingFaceTextDataLoader infrastructure,
    which handles streaming, distributed sharding, tokenization,
    checkpointing, and infinite looping.

    Args:
        name: Registry key (used as --dataloader.dataset <name>).
        path: HuggingFace dataset path (e.g. "HuggingFaceFW/fineweb-edu").
        config_name: Dataset config/subset (e.g. "default", "en").
        split: Dataset split (default "train").
        text_column: Column containing the text to train on.
        streaming: Use HF streaming mode (default True).
        trust_remote_code: Allow running dataset scripts from the hub.

    Returns:
        The registered DatasetConfig.

    Example:
        >>> register_hf_dataset("fineweb_edu", "HuggingFaceFW/fineweb-edu")
        >>> # Then in config: --dataloader.dataset fineweb_edu
    """
    config = DatasetConfig(
        path=path,
        loader=_make_loader(
            config_name=config_name,
            split=split,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
        ),
        sample_processor=_make_text_processor(text_column),
    )
    DATASETS[name] = config
    log.debug(f"Registered HF dataset {name!r} -> {path}")
    return config


# ---------------------------------------------------------------------------
# Precached HF dataset -> local parquet resolution
# ---------------------------------------------------------------------------


def resolve_precached_parquet_dir(repo_id: str) -> str | None:
    """Return the local ``data/`` parquet dir of a snapshot_download'd HF
    dataset under ``$HF_HOME/hub``, or None if not fully cached.

    HF-hub STREAMING at many ranks 429-storms the metadata API, and OFFLINE
    ``load_dataset(<repo_id>)`` fails for script/parquet datasets because it
    still tries to resolve the loader against the hub. But once
    ``precache_hf_dataset.py`` has snapshot_download'd the repo, the parquet
    shards live locally at
    ``$HF_HOME/hub/datasets--<org>--<name>/snapshots/<rev>/data`` and can be
    read with ``load_dataset("parquet", data_dir=...)`` -- zero hub calls, at
    any rank count. This resolves that dir so callers can register the corpus
    as a LOCAL parquet dataset instead of a hub-streaming one.
    """
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    cache_key = "datasets--" + repo_id.replace("/", "--")
    snapshots = os.path.join(hf_home, "hub", cache_key, "snapshots")
    if not os.path.isdir(snapshots):
        return None
    # Prefer the revision pointed to by refs/main; else the first snapshot dir.
    revs = sorted(
        d for d in os.listdir(snapshots) if os.path.isdir(os.path.join(snapshots, d))
    )
    for rev in revs:
        data_dir = os.path.join(snapshots, rev, "data")
        if os.path.isdir(data_dir) and any(
            f.endswith(".parquet") for f in os.listdir(data_dir)
        ):
            return data_dir
    return None


def register_precached_or_hf(
    name: str,
    repo_id: str,
    *,
    config_name: str | None = None,
    text_column: str = "text",
) -> DatasetConfig:
    """Register ``name`` as a LOCAL parquet dataset if ``repo_id`` is fully
    precached under ``$HF_HOME``, else fall back to HF-hub streaming.

    Use this for corpora consumed at large rank counts (the anneal's
    open-web-math at 384 ranks): a precache turns it into a local-parquet
    load with no hub interaction, dodging the 429 storm; without a precache it
    still works (streaming) for small-scale / interactive runs.
    """
    data_dir = resolve_precached_parquet_dir(repo_id)
    if data_dir is not None:
        log.info(
            f"[ezpz/datasets] {name!r} ({repo_id}) resolved to LOCAL parquet "
            f"at {data_dir} (precached; no hub calls)"
        )
        return register_local_dataset(
            name, data_dir, text_column=text_column, format="parquet"
        )
    log.info(
        f"[ezpz/datasets] {name!r} ({repo_id}) not precached -- registering as "
        f"HF-hub streaming (fine for small scale; precache for 100+ ranks)"
    )
    return register_hf_dataset(
        name, repo_id, config_name=config_name, text_column=text_column
    )


# ---------------------------------------------------------------------------
# Auto-registration fallback: treat unknown dataset names as HF hub paths
# ---------------------------------------------------------------------------

# Save the original validator so we can call it for known datasets
_original_validate_dataset = text_datasets._validate_dataset


def _validate_dataset_with_fallback(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset, auto-registering unknown names as HF hub datasets.

    If dataset_name is in the registry, behaves identically to the original.
    Otherwise, treats dataset_name as a HuggingFace hub path (e.g.
    "stanfordnlp/imdb", "eliplutchok/fineweb-small-sample") and auto-registers
    it as a streaming dataset with text_column="text".
    """
    if dataset_name in DATASETS:
        return _original_validate_dataset(dataset_name, dataset_path)

    # Precache-first: if the name looks like an HF repo id (<org>/<name>) and
    # it has been snapshot_download'd under $HF_HOME, register it as a LOCAL
    # parquet dataset (zero hub calls, safe at 384 ranks) instead of hub
    # streaming (which 429-storms at scale). Falls through to hub streaming if
    # not precached. This is what lets the anneal's open-web-math run at 32N.
    if "/" in dataset_name and resolve_precached_parquet_dir(dataset_name):
        register_precached_or_hf(dataset_name, dataset_name)
        return _original_validate_dataset(dataset_name, dataset_path=None)

    # Auto-register: treat the dataset name as a HF hub path.
    # Force dataset_path=None so the registered hub path is used
    # instead of any stale local path from the config (e.g. books.txt).
    if dataset_path is not None:
        # The user explicitly passed both --dataloader.dataset=user/repo
        # AND --dataloader.dataset-path=/some/local/file. The local path
        # is silently ignored when streaming from the hub — surface that
        # so the user can either drop the path arg or pick a different
        # dataset name.
        log.warning(
            f"Dataset {dataset_name!r} is being auto-registered as a "
            f"streaming HF hub dataset, but --dataloader.dataset-path="
            f"{dataset_path!r} was also passed. The local path will be "
            f"IGNORED — drop --dataloader.dataset-path or use a "
            f"local-file dataset name (e.g. 'blendcorpus') instead."
        )
    else:
        log.info(
            f"Dataset {dataset_name!r} not in registry — "
            f"auto-registering as streaming HF dataset"
        )
    register_hf_dataset(
        name=dataset_name,
        path=dataset_name,
        streaming=True,
    )
    return _original_validate_dataset(dataset_name, dataset_path=None)


# Patch the validator so all code paths (HuggingFaceTextDataset,
# HuggingFaceTextDataLoader, BlendCorpusDataLoader) benefit.
text_datasets._validate_dataset = _validate_dataset_with_fallback


# ---------------------------------------------------------------------------
# Pre-registered common datasets
# ---------------------------------------------------------------------------

register_hf_dataset(
    "fineweb_edu",
    "HuggingFaceFW/fineweb-edu",
    config_name="default",
)

register_hf_dataset(
    "fineweb",
    "HuggingFaceFW/fineweb",
    config_name="default",
)

register_hf_dataset(
    "slimpajama",
    "cerebras/SlimPajama-627B",
)

register_hf_dataset(
    "pile",
    "monology/pile-uncopyrighted",
)

register_hf_dataset(
    "openwebtext",
    "Skylion007/openwebtext",
    streaming=False,
)

register_hf_dataset(
    "wikitext",
    "wikitext",
    config_name="wikitext-103-raw-v1",
    streaming=False,
)

register_hf_dataset(
    "c4_streaming",
    "allenai/c4",
    config_name="en",
)

# ---------------------------------------------------------------------------
# Local cached datasets
# ---------------------------------------------------------------------------

register_local_dataset(
    "fineweb_edu_local",
    "/lus/tegu/projects/datasets/datasets/fineweb-edu-100BT/sample/100BT/",
)

register_local_dataset(
    "olmo_mix_wiki_local",
    "/lus/tegu/projects/datasets/hf_cache/hub/datasets--allenai--olmo-mix-1124/snapshots/99ee6aaace88779d1ef099d36251b91101c1679b/data/wiki/",
    format="json",
)

# Cosmopedia synthetic-science-TEXTBOOK slice for the data-mix Wave 3 blend:
# the 5 curated STEM/textbook subsets (auto_math_text, khanacademy, openstax,
# stanford, wikihow), symlinked flat into one dir (36 parquet) so a single
# local-parquet load covers them all -- excludes cosmopedia's stories +
# web_samples (generic web/fiction). Built by symlinking the precached
# HuggingFaceTB/cosmopedia snapshot subdirs; 'text' column confirmed.
register_local_dataset(
    "cosmopedia_science_local",
    "/lus/tegu/projects/datasets/cosmopedia-science/data/",
)
