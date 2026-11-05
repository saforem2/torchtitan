#!/usr/bin/env python3
"""Fingerprint the expected AGPT token stream for TT and MDS configs.

The audit builds each framework's exact training dataset and hashes the
sequential single-sampler rows assigned to every DP rank. It does not launch a
model or capture the live dataloader. Index caches use a temporary directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist


DEFAULT_MDS_ROOT = Path(
    "/lus/tegu/home/foremans/.cache/agpt-mds-baseline-20261023/source"
)
DEFAULT_TT_ROOT = Path(
    "/lus/tegu/projects/datascience/foremans/projects/saforem2/"
    "torchtitan-agpt-matched-20261024"
)
def initialize_cpu_dist(tmp: Path) -> None:
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmp / 'gloo-init'}",
            rank=0,
            world_size=1,
        )


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def construct_tt_dataset(
    cache: Path,
    *,
    tt_root: Path,
    data_list: Path,
    dp_world_size: int,
    global_batch_size: int,
    micro_batch_size: int,
    seq_length: int,
    steps: int,
    split: str,
    seed: int,
):
    sys.path.insert(0, str(tt_root))
    import blendcorpus.parallel_state as mpu
    from blendcorpus.data.config import set_config
    from blendcorpus.data.gpt_dataset import build_gpt_datasets

    cfg = SimpleNamespace(
        data_file_list=str(data_list),
        global_batch_size=global_batch_size,
        train_iters=steps,
        eval_iters=0,
        seed=seed,
        data_impl="mmap",
        mmap_warmup=False,
        seq_length=seq_length,
        split=split,
        shuffle_sample_in_corpus=True,
        blend_sample_in_corpus=False,
        train_data_exact_num_epochs=None,
        train_desc_path=None,
        train_doc_idx_path=None,
        train_sample_idx_path=None,
        train_shuffle_idx_path=None,
        use_dataset_only=False,
        return_data_index=False,
        data_cache_path=str(cache),
        dataloader_type="single",
        micro_batch_size=micro_batch_size,
        num_workers=0,
        data_sharding=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel_size=1,
    )
    set_config(cfg)
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            sequence_parallel_size=1,
        )
    dataset = build_gpt_datasets(cfg)[0]
    dataset_module = importlib.import_module("blendcorpus.data.gpt_dataset")
    sampler_module = importlib.import_module("blendcorpus.data.data_samplers")
    return dataset, mpu, {
        "dataset_module": str(Path(dataset_module.__file__).resolve()),
        "sampler_module": str(Path(sampler_module.__file__).resolve()),
        "blendcorpus_version": importlib.metadata.version("blendcorpus"),
        "configured_dp_world_size": dp_world_size,
        "dataset_build_process_group_world_size": dist.get_world_size(),
    }


def construct_mds_dataset(
    cache: Path,
    *,
    mds_root: Path,
    data_list: Path,
    dp_world_size: int,
    global_batch_size: int,
    seq_length: int,
    steps: int,
    split: str,
    seed: int,
):
    sys.path.insert(0, str(mds_root))
    from megatron.global_vars import set_args
    from megatron.core import mpu
    from megatron.data.gpt_dataset import build_train_valid_test_datasets

    args = SimpleNamespace(
        blend_sample_in_corpus=False,
        return_data_index=False,
        shuffle_sample_in_corpus=True,
        train_data_exact_num_epochs=None,
        train_desc_path=None,
        train_doc_idx_path=None,
        train_sample_idx_path=None,
        train_shuffle_idx_path=None,
        use_dataset_only=False,
    )
    set_args(args)
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
    if not mpu.model_parallel_is_initialized():
        raise RuntimeError(
            "MDS model-parallel groups failed to initialize for world-size-1 data build"
        )
    file_specs: list[object] = []
    for line in data_list.read_text().splitlines():
        if not line.strip():
            continue
        weight, prefix, corpus = line.split()
        file_specs.extend((float(weight), prefix, corpus))

    dataset, _, _ = build_train_valid_test_datasets(
        data_prefix=file_specs,
        data_impl="mmap",
        splits_string=split,
        train_valid_test_num_samples=[
            global_batch_size * steps,
            global_batch_size,
            global_batch_size,
        ],
        seq_length=seq_length,
        seed=seed,
        skip_warmup=True,
        return_doc_ids=False,
        data_cache_path=str(cache),
    )
    module = importlib.import_module("megatron.data.gpt_dataset")
    return dataset, mpu, {
        "dataset_module": str(Path(module.__file__).resolve()),
        "sampler_module": "MegatronPretrainingSampler sequential single-mode mapping",
        "blendcorpus_version": None,
        "configured_dp_world_size": dp_world_size,
        "dataset_build_process_group_world_size": dist.get_world_size(),
    }


def iter_sample_rows(
    dataset,
    *,
    dp_world_size: int,
    micro_batch_size: int,
    seq_length: int,
    num_microbatches: int,
    include_token_samples: bool,
):
    """Yield rows using single-sampler global order, one DP slice per rank."""
    for microbatch in range(num_microbatches):
        for dp_rank in range(dp_world_size):
            first_index = (
                microbatch * dp_world_size + dp_rank
            ) * micro_batch_size
            for local_sample_index in range(micro_batch_size):
                dataset_index = first_index + local_sample_index
                sample = dataset[dataset_index]
                raw = np.asarray(sample["text"], dtype=np.int64).reshape(-1)
                if raw.size != seq_length + 1:
                    raise RuntimeError(
                        f"expected {seq_length + 1} tokens at dataset index "
                        f"{dataset_index}, got {raw.size}"
                    )
                inputs = np.ascontiguousarray(raw[:-1])
                labels = np.ascontiguousarray(raw[1:])
                row = {
                    "microbatch": microbatch,
                    "dp_rank": dp_rank,
                    "microbatch_sample_index": local_sample_index,
                    "dataset_index": dataset_index,
                    "raw_shape": list(raw.shape),
                    "input_shape": list(inputs.shape),
                    "label_shape": list(labels.shape),
                    "dtype": "int64",
                    "input_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
                    "label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                    "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                }
                if include_token_samples:
                    row["first_16_tokens"] = raw[:16].tolist()
                    row["last_4_tokens"] = raw[-4:].tolist()
                yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("tt", "mds"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dp-world-size", type=int, default=24)
    parser.add_argument("--global-batch-size", type=int, default=48)
    parser.add_argument("--gas", type=int, default=2)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--split", default="95,5,0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tt-root", type=Path, default=DEFAULT_TT_ROOT)
    parser.add_argument("--mds-root", type=Path, default=DEFAULT_MDS_ROOT)
    parser.add_argument("--data-list", type=Path, default=None)
    parser.add_argument(
        "--include-token-samples",
        action="store_true",
        help="Include first/last token IDs in each row; omitted by default.",
    )
    args = parser.parse_args()
    data_list = args.data_list or (
        args.tt_root / "torchtitan/experiments/ezpz/data-lists/sunspot/books.txt"
    )
    num_microbatches = (
        args.gas * args.steps if args.microbatches is None else args.microbatches
    )
    if min(
        args.dp_world_size,
        args.global_batch_size,
        args.gas,
        args.steps,
        args.micro_batch_size,
        args.seq_length,
    ) <= 0:
        parser.error("world size, batch sizes, GAS, steps, and sequence length must be positive")
    if args.global_batch_size != args.dp_world_size * args.micro_batch_size * args.gas:
        parser.error(
            "global batch size must equal dp_world_size * micro_batch_size * gas"
        )
    if num_microbatches != args.gas * args.steps:
        parser.error(
            "microbatches must equal gas * steps to audit the complete update window"
        )
    if not data_list.is_file():
        raise FileNotFoundError(data_list)

    root = args.tt_root if args.framework == "tt" else args.mds_root
    if not root.is_dir():
        raise FileNotFoundError(root)

    with tempfile.TemporaryDirectory(
        prefix=f"agpt-{args.framework}-dataset-hash-"
    ) as temp_dir:
        tmp = Path(temp_dir)
        cache = tmp / "index-cache"
        cache.mkdir()
        initialize_cpu_dist(tmp)
        if args.framework == "tt":
            dataset, mpu, package_info = construct_tt_dataset(
                cache,
                tt_root=args.tt_root,
                data_list=data_list,
                dp_world_size=args.dp_world_size,
                global_batch_size=args.global_batch_size,
                micro_batch_size=args.micro_batch_size,
                seq_length=args.seq_length,
                steps=args.steps,
                split=args.split,
                seed=args.seed,
            )
        else:
            dataset, mpu, package_info = construct_mds_dataset(
                cache,
                mds_root=args.mds_root,
                data_list=data_list,
                dp_world_size=args.dp_world_size,
                global_batch_size=args.global_batch_size,
                seq_length=args.seq_length,
                steps=args.steps,
                split=args.split,
                seed=args.seed,
            )

        required_samples = (
            num_microbatches * args.dp_world_size * args.micro_batch_size
        )
        if len(dataset) < required_samples:
            raise RuntimeError(
                f"dataset has {len(dataset)} rows; the requested audit needs "
                f"{required_samples}"
            )
        rows_path = args.output.with_suffix(".rows.jsonl")
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        row_count = 0
        with rows_path.open("w", encoding="utf-8") as rows_file:
            for row in iter_sample_rows(
                dataset,
                dp_world_size=args.dp_world_size,
                micro_batch_size=args.micro_batch_size,
                seq_length=args.seq_length,
                num_microbatches=num_microbatches,
                include_token_samples=args.include_token_samples,
            ):
                rows_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                row_count += 1

        if dist.is_initialized():
            mpu.destroy_model_parallel()
            dist.destroy_process_group()

    expected_rows = required_samples
    if row_count != expected_rows:
        raise RuntimeError(f"wrote {row_count} rows, expected {expected_rows}")
    result = {
        "format": "agpt_expected_stream_fingerprint_v2",
        "framework": args.framework,
        "source_root": str(root),
        "source_git_head": git_head(root),
        "dataset_list": str(data_list),
        "dataset_list_sha256": sha256_file(data_list),
        "dataset_module": package_info["dataset_module"],
        "sampler_module": package_info["sampler_module"],
        "blendcorpus_version": package_info["blendcorpus_version"],
        "dataset_build_process_group_world_size": package_info[
            "dataset_build_process_group_world_size"
        ],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "data_config": {
            "global_batch_size": args.global_batch_size,
            "micro_batch_size": args.micro_batch_size,
            "dp_world_size": args.dp_world_size,
            "gradient_accumulation_steps": args.gas,
            "microbatches_hashed": num_microbatches,
            "updates_hashed": args.steps,
            "seq_length": args.seq_length,
            "split": args.split,
            "seed": args.seed,
            "shuffle_sample_in_corpus": True,
            "consumed_samples": 0,
            "dataloader_type": "single",
            "complete_update_window": True,
        },
        "dataset_length": len(dataset),
        "row_count": row_count,
        "rows_jsonl": str(rows_path),
        "rows_jsonl_sha256": sha256_file(rows_path),
        "row_key": ["microbatch", "dp_rank", "microbatch_sample_index"],
        "token_samples_included": args.include_token_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.output} and {rows_path} ({row_count} row hashes)")


if __name__ == "__main__":
    main()
