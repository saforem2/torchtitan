#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Interpolate two compatible Hugging Face safetensors checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")

    base_weights = args.base / "model-00001-of-00001.safetensors"
    if not base_weights.is_file():
        base_weights = args.base / "model.safetensors"
    tuned_weights = args.tuned / "model.safetensors"
    if not base_weights.is_file() or not tuned_weights.is_file():
        raise FileNotFoundError((base_weights, tuned_weights))

    args.output.mkdir(parents=True)
    for source in (args.tuned, args.base):
        for path in source.iterdir():
            if path.name.endswith(".safetensors") or path.name.endswith(".bin"):
                continue
            if path.name in {
                "model.safetensors.index.json",
                "optimizer.pt",
                "scheduler.pt",
                "trainer_state.json",
            }:
                continue
            target = args.output / path.name
            if not target.exists() and path.is_file():
                shutil.copy2(path, target)

    merged = {}
    with safe_open(base_weights, framework="pt", device="cpu") as base, safe_open(
        tuned_weights, framework="pt", device="cpu"
    ) as tuned:
        if set(base.keys()) != set(tuned.keys()):
            raise ValueError("checkpoint tensor keys differ")
        for key in base.keys():
            left = base.get_tensor(key)
            right = tuned.get_tensor(key)
            if left.shape != right.shape:
                raise ValueError(
                    f"shape mismatch for {key}: {left.shape} != {right.shape}"
                )
            if left.is_floating_point():
                merged[key] = torch.lerp(left.float(), right.float(), args.alpha).to(
                    right.dtype
                )
            elif torch.equal(left, right):
                merged[key] = right
            else:
                raise ValueError(f"non-floating tensor differs: {key}")
    save_file(merged, args.output / "model.safetensors")
    if (args.output / "model.safetensors.index.json").exists():
        raise RuntimeError("single-file export contains a stale shard index")
    metadata = {
        "base": str(args.base),
        "tuned": str(args.tuned),
        "alpha": args.alpha,
        "formula": "base + alpha * (tuned - base)",
        "tensor_count": len(merged),
    }
    (args.output / "merge_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("AGPT_MODEL_INTERPOLATION_DONE", json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
