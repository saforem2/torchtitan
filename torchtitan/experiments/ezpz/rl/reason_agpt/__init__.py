# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""agpt-2b GSM8K chain-of-thought GRPO+LoRA overlay for the upstream RL engine (XPU).

Sibling of ``alphabet_sort_agpt/``: a thin ``--module`` target that provides
``rl_grpo_lora_agpt_2b_gsm8k*`` configs backing the upstream
``torchtitan.experiments.rl`` engine with ``ezpz.agpt``, for Stage 2 (GRPO-RLVR)
of the CoT plan (docs/production/rl/plans/cot.md). The model reasons inside
``<think></think>`` and answers inside ``<answer>\\boxed{}</answer>``, scored by a
componentized dense reward ported from ``rl/tasks/gsm8k_reason.py``.

See ``config_registry.py`` for the entry points, ``data.py`` for the GSM8K stream
+ difficulty curriculum, ``env.py`` for the one-shot CoT prompt, and ``reward.py``
for the four-component rubric.
"""

from torchtitan.experiments.ezpz.rl.reason_agpt.data import (
    GSM8KReasonDataset,
    GSM8KReasonSample,
)
from torchtitan.experiments.ezpz.rl.reason_agpt.env import GSM8KReasonEnv
from torchtitan.experiments.ezpz.rl.reason_agpt.reward import (
    AnswerCloseReward,
    AnswerCorrectReward,
    AnswerExtractableReward,
    ThinkFormatReward,
)
from torchtitan.experiments.ezpz.rl.reason_agpt.rollouter import GSM8KReasonRollouter

__all__ = [
    "GSM8KReasonDataset",
    "GSM8KReasonSample",
    "GSM8KReasonEnv",
    "GSM8KReasonRollouter",
    "ThinkFormatReward",
    "AnswerExtractableReward",
    "AnswerCloseReward",
    "AnswerCorrectReward",
]
