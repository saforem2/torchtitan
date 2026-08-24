# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AlphabetSort env with a one-shot format example in the turn-0 prompt.

The SFT agpt-2b model can sort but rarely emits the exact
``<alphabetical_sorted>`` block, starving GRPO of signal. A one-shot example
raises format adherence (~30% -> ~50-65% hit rate in the runs) so partial-credit
reward has variance to optimize.

Design notes (v2, learned the hard way -- see docs/live/chains/rl/
grpo-lora-agpt2b-repro.md):
  - BARE name lines (no ``<name>`` tags) -- the rubric reads bare lines.
  - example names clearly DISTINCT from any task names, with an explicit
    "do not reuse these" instruction, so the model does not parrot the example.

Only ``init()`` (turn 0) is overridden; multi-turn ``step()`` is inherited
unchanged from the upstream ``AlphabetSortEnv``.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.experiments.rl.environment import MessageEnvInitOutput
from torchtitan.experiments.rl.examples.alphabet_sort.env import AlphabetSortEnv


class AgptFewShotAlphabetSortEnv(AlphabetSortEnv):
    """``AlphabetSortEnv`` whose turn-0 prompt carries a one-shot format example."""

    @dataclass(kw_only=True, slots=True)
    class Config(AlphabetSortEnv.Config):
        pass

    async def init(self) -> MessageEnvInitOutput:
        # init asks for turn 0; step asks for turns 1, 2, ... in order.
        self._next_turn = 1
        sort_type = "FIRST" if self._env_input.sort_by_first_name else "LAST"
        names = ", ".join(self._env_input.new_names_per_turn[0])
        prompt = (
            f"Sort these names in alphabetical order by {sort_type} name: {names}\n\n"
            "Reply with ONLY this block, one name per line, nothing before or after:\n"
            "<alphabetical_sorted>\nName1\nName2\n...\n</alphabetical_sorted>\n\n"
            "Formatting example (DIFFERENT names -- do not reuse these; sort the "
            "names given above instead):\n"
            "<alphabetical_sorted>\nQuinnRivera\nOmarSaito\nPiaValdez\n"
            "</alphabetical_sorted>"
        )
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": prompt}]
        )
