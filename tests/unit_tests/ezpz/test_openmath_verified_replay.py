# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.ezpz.rl.datasets_sft import (
    _format_openmath_verified_row,
    get_sft_dataset,
)


def test_openmath_verified_replay_is_registered():
    assert get_sft_dataset("openmath-verified-replay").name == "openmath-verified-replay"


def test_openmath_row_requires_boxed_gold_agreement():
    row = _format_openmath_verified_row(
        {
            "problem": "What is 6 times 7?",
            "generated_solution": "Compute 6\\cdot7=\\boxed{42}.",
            "expected_answer": "42",
        }
    )
    assert row is not None
    assert row["completion"][0]["content"].endswith("<answer>\\boxed{42}</answer>")
    assert _format_openmath_verified_row(
        {
            "problem": "What is 6 times 7?",
            "generated_solution": "Compute 6\\cdot7=\\boxed{41}.",
            "expected_answer": "42",
        }
    ) is None


def test_openmath_row_rejects_unboxed_and_long_solution():
    assert _format_openmath_verified_row(
        {"problem": "x", "generated_solution": "answer 1", "expected_answer": "1"}
    ) is None
    assert _format_openmath_verified_row(
        {"problem": "x", "generated_solution": "a" * 1201 + "\\boxed{1}", "expected_answer": "1"}
    ) is None
