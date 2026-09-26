# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
EZPZ_ROOT = REPO_ROOT / "torchtitan" / "experiments" / "ezpz"


def _source(relative_path: str) -> str:
    return (EZPZ_ROOT / relative_path).read_text()


def test_non_moe_non_rl_ezpz_python_has_no_model_spec_references():
    root = EZPZ_ROOT
    excluded = {
        root / "agpt",
        root / "moe",
        root / "rl",
        root / "trainer.py",
        root / "tests",
    }
    offenders = []
    for path in root.rglob("*.py"):
        if any(parent == path or parent in path.parents for parent in excluded):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in {"ModelSpec", "model_spec"}:
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
            if isinstance(node, ast.Attribute) and node.attr == "model_spec":
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []


def test_hf_converters_use_direct_config_and_model_adapter():
    for relative_path in ("eval/convert_to_hf.py", "eval/convert_to_hf_legacy.py"):
        source = _source(relative_path)
        assert "model_config = model_module.model_registry(model_flavor)" in source
        assert "adapter_cls = type(model).state_dict_adapter_cls" in source


def test_checkpoint_conversion_is_strict_except_for_legacy_key_renames():
    assert "allow_partial_load" not in _source("eval/convert_to_hf.py")
    legacy_source = _source("eval/convert_to_hf_legacy.py")
    assert legacy_source.count("allow_partial_load=True") == 1


def test_lora_merge_uses_rope_aware_agpt_adapter():
    source = _source("scripts/eval/merge_lora_dcp_to_hf.py")
    assert "AgptStateDictAdapter(model_config, base_hf)" in source
    assert "Llama3StateDictAdapter" not in source
