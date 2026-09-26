# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.ezpz.rl.scripts.stage3_star_generate import (
    assess_trace,
    reasoning_step_count,
)


def test_reasoning_step_count_requires_substantive_segments():
    text = "<think>First compute 2 + 2.\nThen verify that it is 4.</think>"
    assert reasoning_step_count(text) == 2


def test_assess_trace_accepts_strict_correct_stopped_trace():
    text = (
        "<think>First compute 2 + 2.\nThen verify that it is 4.</think>\n"
        "<answer>\\boxed{4}</answer>"
    )
    assert assess_trace(text, "stop", "4") == (True, "accepted")


def test_assess_trace_rejects_length_format_answer_and_short_reasoning():
    valid = (
        "<think>First compute 2 + 2.\nThen verify that it is 4.</think>\n"
        "<answer>\\boxed{4}</answer>"
    )
    assert assess_trace(valid, "length", "4") == (False, "not_stopped")
    assert assess_trace("4", "stop", "4") == (False, "invalid_envelope")
    wrong = valid.replace("boxed{4}", "boxed{5}")
    assert assess_trace(wrong, "stop", "4") == (False, "incorrect")
    short = "<think>Compute it.</think><answer>\\boxed{4}</answer>"
    assert assess_trace(short, "stop", "4") == (
        False,
        "fewer_than_two_reasoning_steps",
    )
