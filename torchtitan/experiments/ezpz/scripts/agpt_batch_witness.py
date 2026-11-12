# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hash consumed AGPT token batches without retaining token IDs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class BatchInputWitness:
    """Append per-sample hashes for CPU token batches, when explicitly enabled.

    Each sample records hashes of the raw row, input row (all but the final
    token), and label row (all but the first token). Token values are never
    written. The witness only reads CPU storage and does not mutate its input.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        rank: int,
        world_size: int,
        max_microbatches: int = 64,
    ) -> None:
        if rank < 0 or world_size < 1 or rank >= world_size:
            raise ValueError(
                f"Invalid rank/world_size: rank={rank}, world_size={world_size}"
            )
        if max_microbatches < 1:
            raise ValueError("max_microbatches must be positive")
        self.output_dir = Path(output_dir)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.max_microbatches = int(max_microbatches)
        self._num_calls = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path = self.output_dir / f"rank-{self.rank:05d}.jsonl"

    @classmethod
    def from_env(
        cls,
        *,
        rank: int,
        world_size: int,
    ) -> BatchInputWitness | None:
        """Build a witness only when its output directory env var is set."""
        output_dir = os.environ.get("AGPT_BENCH_BATCH_HASH_DIR", "").strip()
        if not output_dir:
            return None
        limit = int(os.environ.get("AGPT_BENCH_BATCH_MICROBATCHES", "64"))
        return cls(
            output_dir,
            rank=rank,
            world_size=world_size,
            max_microbatches=limit,
        )

    @staticmethod
    def _canonical_int64_bytes(row: Any) -> bytes:
        array = np.asarray(row)
        if array.ndim != 1:
            raise ValueError(f"Expected a one-dimensional token row, got {array.shape}")
        canonical = np.ascontiguousarray(array, dtype=np.dtype("<i8"))
        return canonical.tobytes(order="C")

    def __call__(
        self,
        raw_tokens: Any,
        *,
        microbatch_index: int | None = None,
        consumed_samples: int | None = None,
    ) -> None:
        """Hash one CPU ``[batch, sequence]`` tensor or array, if under limit."""
        if self._num_calls >= self.max_microbatches:
            return

        if hasattr(raw_tokens, "device"):
            if raw_tokens.device.type != "cpu":
                raise ValueError("Batch input witness requires a CPU token batch")
            values = raw_tokens.detach().numpy()
        else:
            values = np.asarray(raw_tokens)
        if values.ndim != 2:
            raise ValueError(f"Expected [batch, sequence] tokens, got {values.shape}")
        if values.shape[1] < 2:
            raise ValueError("Token rows must contain at least two tokens")

        current_index = (
            self._num_calls if microbatch_index is None else int(microbatch_index)
        )
        if current_index < 0:
            raise ValueError("microbatch_index must be nonnegative")
        raw_shape = [int(v) for v in values.shape]
        input_shape = [raw_shape[0], raw_shape[1] - 1]
        label_shape = [raw_shape[0], raw_shape[1] - 1]
        records = []
        for local_sample_index, row in enumerate(values):
            raw_bytes = self._canonical_int64_bytes(row)
            input_bytes = self._canonical_int64_bytes(row[:-1])
            label_bytes = self._canonical_int64_bytes(row[1:])
            records.append(
                {
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "microbatch": current_index,
                    "local_sample_index": local_sample_index,
                    "consumed_samples": consumed_samples,
                    "raw_shape": raw_shape,
                    "input_shape": input_shape,
                    "label_shape": label_shape,
                    "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                    "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
                }
            )

        with self._output_path.open("a", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
        self._num_calls += 1
