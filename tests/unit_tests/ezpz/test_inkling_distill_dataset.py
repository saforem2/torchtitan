# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest

from torchtitan.experiments.ezpz.rl.datasets_sft import (
    get_sft_dataset,
    INKLING_DISTILL_PATH_ENV,
)


def _row(i: int) -> dict:
    return {
        "id": f"row-{i}",
        "category": "instruction_flan",
        "source": "test",
        "prompt_hash": f"hash-{i}",
        "prompt": [{"role": "user", "content": f"question {i}"}],
        "completion": [{"role": "assistant", "content": f"answer {i}"}],
    }


def test_inkling_distill_is_registered():
    assert get_sft_dataset("inkling-distill").name == "inkling-distill"


def test_inkling_distill_requires_path(monkeypatch):
    monkeypatch.delenv(INKLING_DISTILL_PATH_ENV, raising=False)
    with pytest.raises(ValueError, match=INKLING_DISTILL_PATH_ENV):
        get_sft_dataset("inkling-distill").build()


def test_inkling_distill_loads_valid_unique_rows(tmp_path, monkeypatch):
    path = tmp_path / "accepted.jsonl"
    with path.open("w") as f:
        for i in range(4_000):
            f.write(json.dumps(_row(i)) + "\n")
    monkeypatch.setenv(INKLING_DISTILL_PATH_ENV, str(path))
    dataset = get_sft_dataset("inkling-distill").build()
    assert len(dataset) == 4_000
    assert dataset.column_names == ["prompt", "completion"]


def test_inkling_distill_rejects_duplicate_hashes(tmp_path, monkeypatch):
    path = tmp_path / "accepted.jsonl"
    with path.open("w") as f:
        for i in range(4_000):
            row = _row(i)
            if i == 3_999:
                row["prompt_hash"] = "hash-0"
            f.write(json.dumps(row) + "\n")
    monkeypatch.setenv(INKLING_DISTILL_PATH_ENV, str(path))
    with pytest.raises(ValueError, match="duplicate prompt hashes"):
        get_sft_dataset("inkling-distill").build()
