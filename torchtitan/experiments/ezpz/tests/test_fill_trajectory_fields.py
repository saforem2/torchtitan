#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for exact trajectory field/row selection."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "utils" / "fill_trajectory_fields.py"
MODULE_NAME = "torchtitan.experiments.ezpz.utils.fill_trajectory_fields"
TRAJECTORIES_MODULE = "torchtitan.experiments.ezpz.utils.trajectories"


def _load_module(monkeypatch: pytest.MonkeyPatch, repo_root: Path):
    trajectories = types.ModuleType(TRAJECTORIES_MODULE)
    trajectories.REPO_ROOT = repo_root  # pyrefly: ignore [missing-attribute]
    trajectories.by_key = lambda key: None  # pyrefly: ignore [missing-attribute]
    trajectories.live_trajectories = lambda: []  # pyrefly: ignore [missing-attribute]
    monkeypatch.setitem(sys.modules, TRAJECTORIES_MODULE, trajectories)

    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE_NAME, module)
    spec.loader.exec_module(module)
    return module


def _valid_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"step-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").touch()
    (checkpoint / "shard.distcp").touch()
    return root


def _trajectory(key: str, checkpoint_dir: Path) -> dict:
    trajectory = {
        "key": key,
        "model": "2b",
        "num_nodes": 512,
        "readme": "README.md",
        "ckpt_dir": str(checkpoint_dir),
        "gbs": 12_288,
        "seq_len": 8_192,
        "token_target": 4_673_780_159_710,
    }
    if "stage2" in key:
        trajectory["auto_fill_leaf"] = False
        trajectory["auto_fill_rollups"] = False
    return trajectory


def test_continuation_cannot_replace_base_leaf_fields(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    readme = tmp_path / "README.md"
    original = """> Last updated: 2026-09-20
| GBS | 12,288 |
**Latest checkpoint:** step-46429 -- FINAL base checkpoint
**Cumulative steps:** 46,429 / 46,429 -- base target
**Tokens consumed:** 46,429 x 12,288 x 8,192 = 4.674T tokens (100.0%)
**Loss:** 2.68687 (base)
"""
    readme.write_text(original)
    continuation = _trajectory(
        "2b_v2_512_stage2_dolmino", _valid_checkpoint(tmp_path / "continuation", 22_300)
    )

    changes, warnings = module.fill_one(
        continuation, today="2026-09-21", wandb_loss=2.48, dry_run=False
    )

    assert readme.read_text() == original
    assert changes == []
    assert any("shares README" in warning for warning in warnings)


def test_base_row_is_not_matched_by_continuation_values(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    base = _trajectory("2b_v2_512", tmp_path / "base")
    continuation = _trajectory("2b_v2_512_stage2_dolmino", tmp_path / "continuation")
    base_values = {
        "step": 46_429,
        "loss": "2.6869",
        "tokens": "4.674T",
        "pct": "100.0%",
    }
    continuation_values = {
        "step": 22_300,
        "loss": "2.480",
        "tokens": "2.245T",
        "pct": "47.9%",
    }
    base_row = (
        "| 2B | 512 | **46,000** (FINAL) | **2.700** | **4.60T** (98.0%) "
        "| -- chain complete | base status |"
    )
    continuation_row = (
        "| 2B | 512 | **22,000** (last checkpoint) | **2.500** | stage-2 "
        "| job | continuation status |"
    )

    assert (
        module._rewrite_rollup_row_for_trajectory(
            base_row, continuation, continuation_values, force_shape_b=True
        )
        == base_row
    )
    assert (
        module._rewrite_rollup_row_for_trajectory(
            continuation_row, base, base_values, force_shape_b=True
        )
        == continuation_row
    )
    assert (
        module._rewrite_rollup_row_for_trajectory(
            base_row, base, base_values, force_shape_b=True
        )
        == base_row
    )


def test_exact_linked_base_row_updates_only_numeric_cells(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    base = _trajectory("2b_v2_512", tmp_path / "base")
    values = {"step": 46_429, "loss": "2.6869", "tokens": "4.674T", "pct": "100.0%"}
    row = (
        "| [v2 512N](n512/README.md) | base status mentions step 22,300 "
        "| **46,000** | **2.700** | **~4.60T (98.0%)** |"
    )

    rewritten = module._rewrite_rollup_row_for_trajectory(row, base, values)

    assert rewritten == (
        "| [v2 512N](n512/README.md) | base status mentions step 22,300 "
        "| **46,429** | **2.6869** | **~4.674T (100.0%)** |"
    )


def test_numeric_replacements_are_anchored_to_expected_cell_tokens(
    monkeypatch, tmp_path
):
    module = _load_module(monkeypatch, tmp_path)
    values = {"step": 86_200, "loss": "2.656", "tokens": "4.339T", "pct": "92.9%"}
    malformed = (
        "| [v2 256N](n256/README.md) | status | prefix 1,000 | loss 2.7 "
        "| note 4.0T and 80.0% |"
    )

    assert module._rewrite_rollup_row(malformed, values) == malformed
