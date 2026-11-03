# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Reward-shaped variant of the multiply task.
#
# Extends ``multiply`` with a dense digit-proximity reward so GRPO gets
# gradient even on wrong answers that are numerically close -- the
# ceiling-attack lesson (componentized reward beats hyperparameter tuning).
#
# Example: 7 x 8 = 56, model says 54 -> 0.25 proximity bonus (1/2 digits).
# A binary accuracy-only reward gives 0 for both 54 and 99; proximity
# distinguishes "close" from "wildly wrong" and points the gradient at
# the right digit positions.
#
# Reuse the multiply dataset (same prompts/answers); only the reward
# functions differ.

import re

from datasets import Dataset

from torchtitan.experiments.ezpz.rl.tasks import RLTask, register_task
from torchtitan.experiments.ezpz.rl.tasks.common import (
    extract_answer,
    get_completion_text,
)
from torchtitan.experiments.ezpz.rl.tasks.multiply import build_dataset  # noqa: F401 -- re-export

# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def accuracy_reward(completions, answer, **kwargs) -> list[float]:
    """1.0 if extracted answer matches ground truth, 0.0 otherwise (unchanged)."""
    rewards = []
    for completion, expected in zip(completions, answer):
        text = get_completion_text(completion)
        extracted = extract_answer(text)
        if extracted is not None and extracted == str(expected):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def digit_proximity_reward(completions, answer, **kwargs) -> list[float]:
    """Dense partial credit: 0.5 * fraction of matching digit positions.

    Right-aligned comparison (least-significant digit aligned), so
    e.g. expected=56, extracted=54 -> 1/2 digits match -> 0.25.
    Returns 0 if no number is extracted or the answer is already exact
    (accuracy_reward already gives 1.0 for exact; this adds nothing extra
    there -- max total for exact is 1.0 + 0.5 = 1.5).
    """
    rewards = []
    for completion, expected in zip(completions, answer):
        text = get_completion_text(completion)
        extracted = extract_answer(text)
        if extracted is None:
            rewards.append(0.0)
            continue
        exp_str = str(expected)
        if extracted == exp_str:
            rewards.append(0.5)  # exact: also earns full proximity
            continue
        max_len = max(len(exp_str), len(extracted))
        exp_padded = exp_str.zfill(max_len)
        ext_padded = extracted.zfill(max_len)
        correct = sum(1 for a, b in zip(exp_padded, ext_padded) if a == b)
        rewards.append(0.5 * correct / max_len)
    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """0.3 bonus if completion shows the multiplication expression."""
    rewards = []
    for completion in completions:
        text = get_completion_text(completion)
        if re.search(r"\d+\s*[x*×]\s*\d+", text):
            rewards.append(0.3)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_task(
    RLTask(
        name="multiply_shaped",
        build_dataset=build_dataset,
        reward_funcs=[accuracy_reward, digit_proximity_reward, format_reward],
        description="Multiplication with dense digit-proximity reward (shaped variant)",
    )
)
