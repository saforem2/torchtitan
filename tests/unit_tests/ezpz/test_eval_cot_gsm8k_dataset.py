# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
from pathlib import Path


def test_cot_gsm8k_eval_uses_canonical_dataset_id():
    path = (
        Path(__file__).parents[3]
        / "torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dataset_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_dataset"
    ]
    assert any(
        call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "openai/gsm8k"
        for call in dataset_calls
    )
    assert not any(
        call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "gsm8k"
        for call in dataset_calls
    )
