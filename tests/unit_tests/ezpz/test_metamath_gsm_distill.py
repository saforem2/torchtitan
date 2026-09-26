# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.ezpz.rl.datasets_sft import (
    _format_metamath_gsm_row,
    get_sft_dataset,
)


def test_metamath_gsm_distill_is_registered():
    assert get_sft_dataset("metamath-gsm-distill").name == "metamath-gsm-distill"


def test_metamath_gsm_row_uses_canonical_cot_envelope():
    row = _format_metamath_gsm_row(
        {
            "query": "A box has 12 balls and loses 5. How many remain?",
            "response": "Subtract the lost balls: 12 - 5 = 7.\n#### 7\nThe answer is: 7",
        }
    )
    assert row is not None
    assert row["prompt"][0]["content"].endswith("inside <answer>\\boxed{}</answer>.")
    assert row["completion"][0]["content"] == (
        "<think>Subtract the lost balls: 12 - 5 = 7.</think>\n"
        "<answer>\\boxed{7}</answer>"
    )


def test_metamath_gsm_row_rejects_missing_answer_or_long_trace():
    assert _format_metamath_gsm_row({"query": "x", "response": "no answer"}) is None
    assert (
        _format_metamath_gsm_row({"query": "x", "response": "a" * 1201 + "\n#### 1"})
        is None
    )
