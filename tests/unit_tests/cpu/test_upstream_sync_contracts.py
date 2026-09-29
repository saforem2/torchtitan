# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.unit_tests.cpu.upstream_sync_baseline import (
    model_state_signature,
    signature_digest,
    training_step_signature,
)
from torchtitan.config import CommConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3 import Qwen3Model, model_registry, qwen3_configs

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EZPZ_ROOT = _REPO_ROOT / "torchtitan" / "experiments" / "ezpz"
_EZPZ_ACTIVE_ROOTS = (
    _EZPZ_ROOT / "train.py",
    _EZPZ_ROOT / "trainer.py",
    _EZPZ_ROOT / "agpt",
    _EZPZ_ROOT / "moe",
    _EZPZ_ROOT / "rl",
)

# Importing these from the package root has broken repeatedly when exports moved.
_REQUIRED_CONFIG_EXPORTS = {
    "CommConfig",
    "CompileConfig",
    "ConfigManager",
    "Configurable",
    "TORCH_DTYPE_MAP",
    "TrainingConfig",
    "apply_overrides",
}
_CONFIG_SUBMODULES = {
    "ParallelismConfig": "torchtitan.config.parallelism",
}
_STALE_RUNTIME_ATTRIBUTES = {
    ("comm", "mode"),
    ("comm", "fake_backend"),
    ("comm", "local_tensor"),
}
_RENAMED_PIPELINE_OPTIONS = {
    "num_pipeline_parallel_microbatches": "num_pp_microbatches",
    "pipeline_parallel_schedule_name": "pipeline_parallel_schedule",
    "pipeline_parallel_split_points": "pipeline_parallel_module_fqns_per_model_part",
}


def _active_python_files() -> list[Path]:
    files: set[Path] = set()
    for root in _EZPZ_ACTIVE_ROOTS:
        if root.is_file():
            files.add(root)
        else:
            files.update(root.rglob("*.py"))
    return sorted(
        path
        for path in files
        if "tests" not in path.parts
        and "vendor" not in path.parts
        and "scripts" not in path.parts
    )


def _attribute_path(node: ast.Attribute) -> tuple[str, ...]:
    parts = [node.attr]
    value = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return tuple(reversed(parts))


def _display_path(path: Path) -> str:
    return (
        str(path.relative_to(_REPO_ROOT))
        if path.is_relative_to(_REPO_ROOT)
        else str(path)
    )


@pytest.mark.xfail(reason="active ezpz imports still need upstream-sync migration")
def test_ezpz_uses_valid_config_imports() -> None:
    failures: list[str] = []
    for path in _active_python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.ImportFrom)
                or node.module != "torchtitan.config"
            ):
                continue
            for alias in node.names:
                expected_module = _CONFIG_SUBMODULES.get(alias.name)
                if alias.name not in _REQUIRED_CONFIG_EXPORTS:
                    failures.append(
                        f"{_display_path(path)}:{node.lineno}: "
                        f"{alias.name} is not a supported torchtitan.config export"
                    )
                if expected_module is not None:
                    failures[-1] += f"; import it from {expected_module}"
    assert not failures, "\n" + "\n".join(failures)


@pytest.mark.xfail(reason="active ezpz runtime still reads removed comm.mode")
def test_ezpz_runtime_has_no_removed_comm_or_pipeline_options() -> None:
    failures: list[str] = []
    for path in _active_python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                attr_path = _attribute_path(node)
                if attr_path[-2:] in _STALE_RUNTIME_ATTRIBUTES:
                    failures.append(
                        f"{_display_path(path)}:{node.lineno}: "
                        f"removed runtime option {'.'.join(attr_path[-2:])}"
                    )
                if node.attr in _RENAMED_PIPELINE_OPTIONS:
                    failures.append(
                        f"{_display_path(path)}:{node.lineno}: "
                        f"{node.attr} was renamed to "
                        f"{_RENAMED_PIPELINE_OPTIONS[node.attr]}"
                    )
    assert not failures, "\n" + "\n".join(failures)


@pytest.mark.xfail(strict=True, reason="negative control: invalid root config import")
def test_invalid_config_import_negative_control() -> None:
    tree = ast.parse("from torchtitan.config import ParallelismConfig")
    imported = next(node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    assert all(alias.name in _REQUIRED_CONFIG_EXPORTS for alias in imported.names)


@pytest.mark.xfail(strict=True, reason="negative control: removed comm.mode option")
def test_removed_comm_option_negative_control() -> None:
    tree = ast.parse("value = config.comm.mode")
    paths = {
        _attribute_path(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert not any(path[-2:] in _STALE_RUNTIME_ATTRIBUTES for path in paths)


def test_config_schema_keeps_synced_runtime_options() -> None:
    assert {field.name for field in dataclasses.fields(CommConfig)} >= {
        "train_timeout_seconds"
    }
    fields = {field.name for field in dataclasses.fields(ParallelismConfig)}
    assert fields >= {
        "pipeline_parallel_module_fqns_per_model_part",
        "pipeline_parallel_schedule",
        "num_pp_microbatches",
    }
    assert not fields.intersection(_RENAMED_PIPELINE_OPTIONS)


@pytest.mark.parametrize("flavor", sorted(qwen3_configs))
def test_qwen3_registry_builds_every_config(flavor: str) -> None:
    config = model_registry(
        flavor,
        seq_len=16,
        moe_comm_backend="standard"
        if "moe" in flavor.lower() or "-A" in flavor
        else None,
    )
    assert isinstance(config, Qwen3Model.Config)
    with torch.device("meta"):
        assert config.build().__class__ is Qwen3Model


_MODEL_SNAPSHOTS = {
    "debugmodel": {
        "parameter_count": 31987968,
        "state_key_count": 67,
        "first_keys": (
            "tok_embeddings.weight",
            "layers.0.attention.qkv_linear.wqkv.weight",
            "layers.0.attention.wo.weight",
        ),
        "last_keys": ("layers.7.ffn_norm.weight", "norm.weight", "lm_head.weight"),
    },
    "debugmodel_moe": {
        "parameter_count": 315758848,
        "state_key_count": 83,
        "first_keys": (
            "tok_embeddings.weight",
            "layers.0.attention.qkv_linear.wqkv.weight",
            "layers.0.attention.wo.weight",
        ),
        "last_keys": ("layers.7.ffn_norm.weight", "norm.weight", "lm_head.weight"),
    },
}


@pytest.mark.parametrize("flavor", tuple(_MODEL_SNAPSHOTS))
def test_qwen3_debug_model_state_snapshot(flavor: str) -> None:
    with torch.device("meta"):
        model = model_registry(flavor, seq_len=16).build()
    signature = model_state_signature(model)
    expected = _MODEL_SNAPSHOTS[flavor]
    assert signature["parameter_count"] == expected["parameter_count"]
    assert len(signature["keys"]) == expected["state_key_count"]
    assert tuple(signature["keys"][:3]) == expected["first_keys"]
    assert tuple(signature["keys"][-3:]) == expected["last_keys"]


def _run_tiny_baseline() -> dict:
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    inputs = torch.tensor([[0.25, -0.5, 0.75], [-0.1, 0.2, 0.3]])
    targets = torch.tensor([1, 0])
    return training_step_signature(model, optimizer, inputs, targets)


def test_training_baseline_helper_is_deterministic_and_complete() -> None:
    first = _run_tiny_baseline()
    second = _run_tiny_baseline()
    assert signature_digest(first) == signature_digest(second)
    assert set(first) == {
        "output",
        "loss",
        "gradients",
        "optimizer",
        "model_after_step",
    }
    assert first["gradients"]
    assert first["optimizer"]


def test_static_gate_negative_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from torchtitan.config import ParallelismConfig\n"
        "def run(config):\n"
        "    return config.comm.mode\n"
    )
    monkeypatch.setattr(
        "tests.unit_tests.cpu.test_upstream_sync_contracts._EZPZ_ACTIVE_ROOTS",
        (bad,),
    )
    with pytest.raises(AssertionError, match="ParallelismConfig"):
        test_ezpz_uses_valid_config_imports()
    with pytest.raises(AssertionError, match="comm.mode"):
        test_ezpz_runtime_has_no_removed_comm_or_pipeline_options()
