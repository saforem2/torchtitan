# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Single-turn GSM8K chain-of-thought env with a one-shot format exemplar.

A ``MessageEnv`` that asks the model to solve one GSM8K problem inside the
``<think></think><answer>\\boxed{}</answer>`` envelope. Like
``alphabet_sort_agpt/few_shot_env.py``, the turn-0 prompt carries a one-shot
exemplar so the SFT cold-start model reliably emits the envelope (the reward keys
on it, so format adherence gives the rubric variance to optimize). The exemplar
uses a DIFFERENT toy problem with an explicit "do not reuse" instruction so the
model does not parrot it.

Single-turn: ``init`` poses the problem; the first ``step`` ends the rollout (the
model's one completion is graded). No tools, no multi-turn -- pure CoT, per the
CoT plan's decision to drop search_r1's retrieval tool
(docs/live/chains/rl/plans/cot.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.experiments.ezpz.rl.reason_agpt.data import (
    GSM8KReasonSample,
    PROMPT_SUFFIX,
)
from torchtitan.experiments.rl.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)

# One-shot exemplar: a DISTINCT toy problem, with an explicit "do not reuse"
# instruction so the model solves the given problem rather than parroting this.
# Kept short (the reward only needs the envelope shape demonstrated) to preserve
# the completion token budget for the model's own reasoning.
_ONE_SHOT = (
    "Example (a DIFFERENT problem -- do not reuse these numbers; solve the "
    "problem given above instead):\n"
    "Question: A basket has 3 apples. You add 2 more. How many apples now?\n"
    "<think>Start with 3 apples, add 2 more: 3 + 2 = 5.</think>\n"
    "<answer>\\boxed{5}</answer>"
)


class GSM8KReasonEnv(MessageEnv):
    """Single-turn GSM8K CoT env; turn-0 prompt carries a one-shot format exemplar."""

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        # One-shot exemplar OFF by default: a weak cold-start copies the
        # exemplar's literal answer instead of reasoning (95% of rollouts emitted
        # no <think>, 20% echoed the exemplar's \boxed{5}). Turn on only with a
        # strong cold-start that won't parrot it.
        one_shot: bool = False

    def __init__(self, config: Config, *, env_input: GSM8KReasonSample) -> None:
        self._env_input = env_input
        self._one_shot = config.one_shot

    async def init(self) -> MessageEnvInitOutput:
        prompt = f"{self._env_input.question}{PROMPT_SUFFIX}"
        if self._one_shot:
            prompt = f"{prompt}\n\n{_ONE_SHOT}"
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": prompt}]
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        # Single-turn: the model's first completion is the final answer; end here.
        del completion_message
        return MessageEnvStepOutput(done=True)
