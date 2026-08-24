# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""agpt-2b GRPO+LoRA overlay for the upstream RL engine (XPU).

A thin `--module` target that provides `rl_grpo_lora_agpt_2b*` configs backing the
upstream `torchtitan.experiments.rl` engine with `ezpz.agpt`. See
`config_registry.py` for the entry points and `few_shot_env.py` for the one-shot
prompt. Docs: docs/live/chains/rl/history/grpo-lora-agpt2b-repro.md.
"""

from torchtitan.experiments.ezpz.rl.alphabet_sort_agpt.few_shot_env import (
    AgptFewShotAlphabetSortEnv,
)

__all__ = ["AgptFewShotAlphabetSortEnv"]
