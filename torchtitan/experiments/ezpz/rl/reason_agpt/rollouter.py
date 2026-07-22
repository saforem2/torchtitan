# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GSM8K chain-of-thought rollouter: datasets + env + componentized rubric.

Pure config, like ``search_r1/rollouter.py`` -- all behavior (``make_env_group``,
``run_group_rollouts``, ``score_group``) is inherited from ``Rollouter``. This
only supplies the default configs: the GSM8K CoT train/val datasets, the one-shot
CoT env, the four-component reward rubric, and the token budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan.experiments.ezpz.rl.reason_agpt.data import GSM8KReasonDataset
from torchtitan.experiments.ezpz.rl.reason_agpt.env import GSM8KReasonEnv
from torchtitan.experiments.ezpz.rl.reason_agpt.reward import (
    AnswerCloseReward,
    AnswerCorrectReward,
    AnswerExtractableReward,
    ThinkFormatReward,
)
from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rubrics import Rubric


class GSM8KReasonRollouter(Rollouter):
    """Wires the GSM8K CoT task: train/val datasets, the one-shot env, and the
    four-component reward. All behavior is inherited from ``Rollouter``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: GSM8KReasonDataset.Config = field(
            default_factory=lambda: GSM8KReasonDataset.Config(seed=42, split="train")
        )
        validation_dataset: GSM8KReasonDataset.Config = field(
            default_factory=lambda: GSM8KReasonDataset.Config(
                # test split, deterministic order (no shuffle) so every validation
                # pass draws the same held-out samples.
                seed=99,
                split="test",
                shuffle=False,
            )
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                # Weights carried over from rl/tasks/gsm8k_reason.py (0.05 / 0.05 /
                # 0.20 / 0.70). The Rubric normalizes them to sum to 1.0, preserving
                # the ratio. Each fn is separate so reward_breakdown keeps per-
                # component metrics (the CoT plan's Path B requirement).
                reward_fns=[
                    ThinkFormatReward.Config(weight=0.05),
                    AnswerExtractableReward.Config(weight=0.05),
                    AnswerCloseReward.Config(weight=0.20),
                    AnswerCorrectReward.Config(weight=0.70),
                ],
                # A truncated rollout has no <answer> -> no reward / learning signal.
                truncation_reward=0.0,
            )
        )
        message_env: GSM8KReasonEnv.Config = field(
            default_factory=GSM8KReasonEnv.Config
        )
        token_env: TokenEnv.Config = field(
            default_factory=lambda: TokenEnv.Config(
                max_rollout_tokens=2048, max_num_turns=1
            )
        )
