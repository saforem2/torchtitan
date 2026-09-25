# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest

from torchtitan.experiments.ezpz.rl.datasets_sft import (
    _build_stage3_star,
    get_sft_dataset,
    STAGE3_STAR_PATH_ENV,
)


def test_stage3_star_is_registered():
    assert get_sft_dataset("stage3-star").name == "stage3-star"


def test_stage3_star_requires_configured_path(monkeypatch):
    monkeypatch.delenv(STAGE3_STAR_PATH_ENV, raising=False)
    with pytest.raises(ValueError, match=STAGE3_STAR_PATH_ENV):
        _build_stage3_star()


def test_stage3_star_rejects_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv(STAGE3_STAR_PATH_ENV, str(tmp_path / "absent.jsonl"))
    with pytest.raises(FileNotFoundError):
        _build_stage3_star()


def test_stage3_star_loads_accepted_rows(monkeypatch, tmp_path):
    path = tmp_path / "accepted_sft.jsonl"
    row = {
        "source_index": 7,
        "prompt": [{"role": "user", "content": "q"}],
        "completion": [{"role": "assistant", "content": "a"}],
        "generation_sha256": "abc",
    }
    path.write_text(json.dumps(row) + "\n")
    monkeypatch.setenv(STAGE3_STAR_PATH_ENV, str(path))

    dataset = _build_stage3_star()

    assert len(dataset) == 1
    assert sorted(dataset.column_names) == ["completion", "prompt"]
