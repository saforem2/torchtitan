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


def test_validate_sweep_results_rejects_partial_sweep():
    with pytest.raises(RuntimeError, match="partial sweep"):
        validate_sweep_results([1e-8] * 5, [1.0] * 5, expected_points=6)


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
    assert 'export LRF_IDLE_TIMEOUT="${LRF_IDLE_TIMEOUT:-1800}"' in text
    assert "export LRF_TIMEOUT=6000" not in text
    assert "export LRF_IDLE_TIMEOUT=1800" not in text


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


def test_runner_forces_resumable_checkpoint_contract_after_caller_arguments():
    script = REPO_ROOT / "torchtitan/experiments/ezpz/scripts/run_lr_finder.sh"
    text = script.read_text()
    caller_args = text.index('            "$@" \\\n')
    checkpoint_on = text.index("            --checkpoint.enable \\\n", caller_args)
    assert checkpoint_on > caller_args
    assert "--checkpoint.no-last-save-model-only" in text[checkpoint_on:]
    assert 'LRF_CHECKPOINT_INTERVAL="${LRF_CHECKPOINT_INTERVAL:-5}"' in text
    assert '--checkpoint.folder "checkpoints/lr_finder_${label}"' in text
    assert "--checkpoint.no-enable" not in text[caller_args:]


def test_ezpz_translation_preserves_checkpoint_component_selection_flags():
    from torchtitan.experiments.ezpz.train import _translate_legacy_args

    assert _translate_legacy_args(["--checkpoint.enable"]) == ["--checkpoint.enable"]
    assert _translate_legacy_args(["--checkpoint.no-enable"]) == [
        "--checkpoint.no-enable"
    ]
    assert _translate_legacy_args(["--checkpoint.interval", "7"]) == [
        "--checkpointer.interval",
        "7",
    ]


def test_runner_supports_stable_run_identity_for_walltime_resume():
    script = REPO_ROOT / "torchtitan/experiments/ezpz/scripts/run_lr_finder.sh"
    text = script.read_text()
    assert 'LRF_RUN_ID="${LRF_RUN_ID:-${TIMESTAMP}${_JOBTAG:+_${_JOBTAG}}}"' in text
    assert (
        'LRF_DUMP_FOLDER="${LRF_DUMP_FOLDER:-outputs/lr_finder_${LRF_MODE}_${LRF_RUN_ID}}"'
        in text
    )
    for name in (
        "submit_lr_finder_olmo2tok_aurora.sh",
        "submit_lr_finder_30b_aurora.sh",
    ):
        submitter_text = (script.parent / name).read_text()
        assert '_run_tag="${LRF_RUN_ID:-' in submitter_text
        assert 'export LRF_RUN_ID="${_run_tag}"' in submitter_text


def test_runner_uses_top_level_hf_assets_option():
    script = REPO_ROOT / "torchtitan/experiments/ezpz/scripts/run_lr_finder.sh"
    text = script.read_text()
    assert 'asset_args=(--hf-assets-path "${LRF_HF_ASSETS_PATH}")' in text
    assert "--job.hf-assets-path" not in text


def test_runner_owns_ccl_op_sync_policy_and_preserves_explicit_values():
    script = REPO_ROOT / "torchtitan/experiments/ezpz/scripts/run_lr_finder.sh"
    text = script.read_text()
    assert 'export CCL_OP_SYNC="${CCL_OP_SYNC:-1}"' in text
    assert "lr-finder: CCL_OP_SYNC=${CCL_OP_SYNC}" in text
    assert "unset CCL_OP_SYNC" not in text


def test_all_submitter_output_defaults_are_stage_and_run_specific():
    scripts = REPO_ROOT / "torchtitan/experiments/ezpz/scripts"
    for name in (
        "submit_lr_finder_olmo2tok_aurora.sh",
        "submit_lr_finder_30b_aurora.sh",
    ):
        text = (scripts / name).read_text()
        assert (
            'LRF_DUMP_FOLDER="${LRF_DUMP_FOLDER:-outputs/lr_finder_${LRF_MODE}' in text
        )
        assert "_run_tag" in text


def test_ladder_summarizer_accepts_stage_specific_output_prefixes():
    script = (
        REPO_ROOT / "torchtitan/experiments/ezpz/scripts/summarize_lr_finder_ladder.py"
    )
    assert 'f"lr_finder_*{size}_olmo2tok_gbs6144*{opt}*"' in script.read_text()
