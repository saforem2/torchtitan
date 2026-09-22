# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Focused contracts for LR-finder stage selection and fail-closed results."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
VALIDATION_PATH = REPO_ROOT / "torchtitan/experiments/ezpz/lr_finder_validation.py"
SPEC = importlib.util.spec_from_file_location("lr_finder_validation", VALIDATION_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATION)
validate_sweep_results = VALIDATION.validate_sweep_results
validate_sweep_config = VALIDATION.validate_sweep_config
exponential_lr_schedule = VALIDATION.exponential_lr_schedule


@pytest.mark.parametrize(
    ("init_lr", "max_lr", "fraction"),
    [(0.0, 1e-3, 0.1), (1e-3, 1e-3, 0.1), (1e-3, 1e-4, 0.1), (1e-6, 1e-3, 0.0)],
)
def test_validate_sweep_config_rejects_invalid_ranges(init_lr, max_lr, fraction):
    with pytest.raises(ValueError):
        validate_sweep_config(init_lr, max_lr, fraction, 100, 0.0)


def test_validate_sweep_config_rejects_too_few_planned_points():
    with pytest.raises(ValueError, match="at least 5 planned sweep points"):
        validate_sweep_config(1e-8, 1e-3, 0.1, 10, 0.0)


def test_validate_sweep_config_returns_post_warmup_point_count():
    assert validate_sweep_config(1e-8, 1e-3, 0.1, 1000, 0.1) == 90


@pytest.mark.parametrize(
    ("lrs", "losses", "message"),
    [
        ([1e-6, 1e-5], [1.0], "different learning-rate and loss counts"),
        (
            [1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
            [5.0, 4.0, float("nan"), 3.0, 4.0],
            "non-finite",
        ),
        (
            [1e-8, 1e-7, float("inf"), 1e-5, 1e-4],
            [5.0, 4.0, 3.0, 3.5, 5.0],
            "non-finite",
        ),
        ([1e-6] * 4, [1.0] * 4, "at least 5"),
    ],
)
def test_validate_sweep_results_rejects_unusable_data(lrs, losses, message):
    with pytest.raises(RuntimeError, match=message):
        validate_sweep_results(lrs, losses)


def test_validate_sweep_results_accepts_finite_curve():
    validate_sweep_results(
        [1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
        [5.0, 4.0, 3.0, 3.5, 5.0],
    )


def test_exponential_schedule_includes_both_endpoints():
    schedule = exponential_lr_schedule(1e-8, 1e-1, 100)
    assert len(schedule) == 100
    assert schedule[0] == 1e-8
    assert schedule[-1] == 1e-1
    assert all(left < right for left, right in zip(schedule, schedule[1:]))


def test_olmo_submitter_preserves_caller_timeout():
    script = (
        REPO_ROOT
        / "torchtitan/experiments/ezpz/scripts/submit_lr_finder_olmo2tok_aurora.sh"
    )
    text = script.read_text()
    assert 'export LRF_TIMEOUT="${LRF_TIMEOUT:-6000}"' in text
    assert "export LRF_TIMEOUT=6000" not in text


def test_runner_exposes_coarse_and_fine_modes_with_fine_bounds_required():
    script = REPO_ROOT / "torchtitan/experiments/ezpz/scripts/run_lr_finder.sh"
    text = script.read_text()
    assert 'LRF_MODE="${LRF_MODE:-custom}"' in text
    assert "coarse)" in text
    assert "fine)" in text
    assert "fine mode requires caller-supplied LRF_INIT_LR and LRF_MAX_LR" in text


def test_olmo_submitter_preserves_caller_fine_bounds():
    script = (
        REPO_ROOT
        / "torchtitan/experiments/ezpz/scripts/submit_lr_finder_olmo2tok_aurora.sh"
    )
    text = script.read_text()
    custom_block = text.split('if [[ "${LRF_MODE}" == custom ]]', 1)[1]
    assert 'export LRF_INIT_LR="${LRF_INIT_LR:-1e-8}"' in custom_block
    assert 'export LRF_MAX_LR="${LRF_MAX_LR:-1e-1}"' in custom_block
    assert 'export LRF_MAX_LR="${LRF_MAX_LR:-1e-3}"' in custom_block
    assert 'LRF_DUMP_FOLDER="${LRF_DUMP_FOLDER:-outputs/lr_finder_${LRF_MODE}' in text


def test_runner_forces_no_checkpoint_after_caller_arguments():
    script = REPO_ROOT / "torchtitan/experiments/ezpz/scripts/run_lr_finder.sh"
    text = script.read_text()
    caller_args = text.index('            "$@" \\\n')
    checkpoint_off = text.index("            --checkpoint.no-enable \\\n", caller_args)
    assert checkpoint_off > caller_args


def test_runner_uses_top_level_hf_assets_option():
    script = REPO_ROOT / "torchtitan/experiments/ezpz/scripts/run_lr_finder.sh"
    text = script.read_text()
    assert 'asset_args=(--hf-assets-path "${LRF_HF_ASSETS_PATH}")' in text
    assert "--job.hf-assets-path" not in text
