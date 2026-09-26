# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_OVERLAYS = (
    "alphabet_sort_agpt/config_registry.py",
    "reason_agpt/config_registry.py",
)
_RL_ROOT = Path(__file__).parents[3] / "torchtitan/experiments/ezpz/rl"


@pytest.mark.parametrize("relative_path", _OVERLAYS)
def test_agpt_rl_overlays_use_controller_model_field(relative_path: str) -> None:
    tree = ast.parse((_RL_ROOT / relative_path).read_text())
    controller_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "Controller"
        and node.func.attr == "Config"
    ]

    assert controller_calls
    for call in controller_calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "model" in keywords
        assert "model_config" not in keywords
        assert "model_spec" not in keywords


@pytest.mark.parametrize("relative_path", _OVERLAYS)
def test_agpt_rl_overlays_keep_direct_model_and_runtime_contracts(
    relative_path: str,
) -> None:
    source = (_RL_ROOT / relative_path).read_text()

    assert "model_spec" not in source
    assert 'target_modules=["wqkv", "wo"]' in source
    assert "LMHeadCastConverter.Config()" in source
    assert 'model_dtype="float32"' in source
    assert "checkpointer=None" in source
    assert "extra_stop_token_ids=(1, 107)" in source


def test_reasoning_overlay_keeps_reward_bearing_rollout_settings() -> None:
    source = (_RL_ROOT / "reason_agpt/config_registry.py").read_text()

    assert "drop_zero_std_reward_groups=False" in source
    assert "max_rollout_tokens=2048" in source
    assert "max_num_turns=1" in source
    assert "truncation_reward=0.0" in source
    assert "should_std_normalize=True" in source
    assert "max_tokens=max_tokens" in source
