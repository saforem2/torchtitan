# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.ezpz.rl.datasets_sft import (
    _format_metamath_math_row,
    get_sft_dataset,
)


def test_metamath_math_distill_is_registered():
    assert get_sft_dataset("metamath-math-distill").name == "metamath-math-distill"


def test_metamath_math_row_uses_boxed_answer_and_cot_envelope():
    row = _format_metamath_math_row(
        {
            "query": "Convert $101_2$ to base ten.",
            "response": (
                "$101_2 = 1\\cdot 2^2 + 0\\cdot2 + 1 = \\boxed{5}$.\n"
                "The answer is: 5"
            ),
        }
    )
    assert row is not None
    assert row["prompt"][0]["content"].endswith(
        "inside <answer>\\boxed{}</answer>."
    )
    assert row["completion"][0]["content"] == (
        "<think>$101_2 = 1\\cdot 2^2 + 0\\cdot2 + 1 = \\boxed{5}$.</think>\n"
        "<answer>\\boxed{5}</answer>"
    )


def test_metamath_math_row_rejects_unboxed_or_long_trace():
    assert _format_metamath_math_row({"query": "x", "response": "answer 5"}) is None
    assert (
        _format_metamath_math_row(
            {"query": "x", "response": "a" * 1201 + " \\boxed{1}"}
        )
        is None
    )
