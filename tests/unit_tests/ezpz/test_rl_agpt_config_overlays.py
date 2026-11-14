# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
import importlib
import inspect
import pickle
from pathlib import Path
from typing import Any

import pytest
import spmd_types as spmd
import torch

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.experiments.ezpz.rl.model_configs import agpt_rl_model_configs
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig

_OVERLAY_MODULES = (
    "torchtitan.experiments.ezpz.rl.alphabet_sort_agpt.config_registry",
    "torchtitan.experiments.ezpz.rl.reason_agpt.config_registry",
)
_OVERLAYS = tuple(
    f"{name.rsplit('.', 2)[-2]}/config_registry.py" for name in _OVERLAY_MODULES
)
_RL_ROOT = Path(__file__).parents[3] / "torchtitan/experiments/ezpz/rl"


def _rl_model_config() -> Any:
    """The overlays' trainer model: agpt 2b-rl, fp32 lm_head, LoRA rank 8."""
    trainer_model, _ = agpt_rl_model_configs(
        trainer_parallelism=ParallelismConfig(
            data_parallel_shard_degree=1, tensor_parallel_degree=1
        ),
        generator_parallelism=InferenceParallelismConfig(
            data_parallel_degree=1, tensor_parallel_degree=1
        ),
        lora_rank=8,
        lora_alpha=16.0,
    )
    return trainer_model


def _state_dict_layouts(model: torch.nn.Module) -> dict[str, spmd.SpmdType]:
    layouts = {}
    for module_fqn, module in model.named_modules():
        prefix = f"{module_fqn}." if module_fqn else ""
        sharding_config = getattr(module, "_sharding_config", None)
        if sharding_config is not None:
            for state_name, layout in sharding_config.state_shardings.items():
                layouts[f"{prefix}{state_name}"] = layout
    return layouts


@pytest.mark.parametrize("relative_path", _OVERLAYS)
def test_agpt_rl_overlays_put_models_on_trainer_and_generator(
    relative_path: str,
) -> None:
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
        # Upstream split Controller.Config.model into trainer.model and
        # generator.model; the root has no model field.
        assert {"trainer", "generator", "precision"} <= keywords
        assert not keywords & {"model", "model_config", "model_spec"}


@pytest.mark.parametrize("relative_path", _OVERLAYS)
def test_agpt_rl_overlays_keep_direct_model_and_runtime_contracts(
    relative_path: str,
) -> None:
    source = (_RL_ROOT / relative_path).read_text()
    model_source = (_RL_ROOT / "model_configs.py").read_text()

    assert "model_spec" not in source
    assert "precision=AGPT_RL_PRECISION" in source
    assert "engine=xpu_vllm_engine_config()" in source
    assert "checkpointer=None" in source
    assert "extra_stop_token_ids=(1, 107)" in source
    assert 'target_modules=["wqkv", "wo"]' in model_source
    assert 'PrecisionConfig(dtype="float32", param_dtype="float32")' in model_source


def test_reasoning_overlay_keeps_reward_bearing_rollout_settings() -> None:
    source = (_RL_ROOT / "reason_agpt/config_registry.py").read_text()

    assert "drop_zero_std_reward_groups=False" in source
    assert "max_rollout_tokens=2048" in source
    assert "max_num_turns=1" in source
    assert "truncation_reward=0.0" in source
    assert "should_std_normalize=True" in source
    assert "max_tokens=max_tokens" in source


@pytest.mark.parametrize("relative_path", _OVERLAYS)
def test_parallelism_config_uses_its_canonical_module(relative_path: str) -> None:
    tree = ast.parse((_RL_ROOT / relative_path).read_text())
    imports = {
        node.module: {alias.name for alias in node.names}
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "ParallelismConfig" in imports.get("torchtitan.config.parallelism", set())
    assert "ParallelismConfig" not in imports.get("torchtitan.config", set())


@pytest.mark.parametrize("module_name", _OVERLAY_MODULES)
def test_every_retained_rl_factory_constructs(module_name: str) -> None:
    pytest.importorskip("vllm")
    pytest.importorskip("torchstore")
    module = importlib.import_module(module_name)
    factories = [
        factory
        for name, factory in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("rl_") and factory.__module__ == module_name
    ]

    assert factories
    for factory in factories:
        config = factory()
        assert config.trainer.model is not None
        assert config.generator.model is not None
        assert config.trainer.parallelism.tensor_parallel_degree == 1
        assert config.trainer.model.max_context_length == 2048
        assert config.precision.dtype == "float32"
        assert config.precision.param_dtype == "float32"
        assert config.generator.engine.cuda_graph.mode == "NONE"


def test_agpt_rl_model_key_and_layout_contracts_are_stable() -> None:
    config = _rl_model_config()
    with torch.device("meta"):
        model = config.build()
    state_dict = model.state_dict()
    layouts = _state_dict_layouts(model)

    assert len(state_dict) == 123
    assert set(state_dict).issubset(layouts)
    assert {
        "tok_embeddings.weight": ((256000, 2048), torch.float32),
        "layers.0.attention.qkv_linear.wqkv.weight": (
            (3072, 2048),
            torch.float32,
        ),
        "layers.0.attention.qkv_linear.wqkv.lora_a.weight": (
            (8, 2048),
            torch.float32,
        ),
        "layers.0.attention.qkv_linear.wqkv.lora_b.weight": (
            (3072, 8),
            torch.float32,
        ),
        "layers.0.attention.wo.lora_a.weight": ((8, 2048), torch.float32),
        "layers.0.attention.wo.lora_b.weight": ((2048, 8), torch.float32),
        "layers.0.feed_forward.w13.weight": (
            (2, 11008, 2048),
            torch.float32,
        ),
        "layers.0.feed_forward.w2.weight": ((2048, 11008), torch.float32),
        "norm.weight": ((2048,), torch.float32),
        "lm_head.weight": ((256000, 2048), torch.float32),
    } == {
        name: (tuple(state_dict[name].shape), state_dict[name].dtype)
        for name in (
            "tok_embeddings.weight",
            "layers.0.attention.qkv_linear.wqkv.weight",
            "layers.0.attention.qkv_linear.wqkv.lora_a.weight",
            "layers.0.attention.qkv_linear.wqkv.lora_b.weight",
            "layers.0.attention.wo.lora_a.weight",
            "layers.0.attention.wo.lora_b.weight",
            "layers.0.feed_forward.w13.weight",
            "layers.0.feed_forward.w2.weight",
            "norm.weight",
            "lm_head.weight",
        )
    }


def test_buffer_canonical_fqns_are_covered_by_layouts() -> None:
    config = _rl_model_config()
    with torch.device("meta"):
        model = config.build()
    layouts = _state_dict_layouts(model)
    wrapped_buffer_fqns = {
        name.replace(".attention.", ".\u005fcheckpoint_wrapped_module.attention.")
        # The RoPE cache may be one tensor shared by every layer; count each FQN.
        for name, _ in model.named_buffers(remove_duplicate=False)
    }

    assert len(wrapped_buffer_fqns) == 12
    assert {canonical_fqn(name) for name in wrapped_buffer_fqns}.issubset(layouts)


def test_parallelism_context_with_none_process_group_is_picklable() -> None:
    parallelism_context = ParallelismContext(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=1,
        enable_sequence_parallel=False,
        _real_pp_group_for_fake_spmd=None,
    )

    restored = pickle.loads(pickle.dumps(parallelism_context))

    assert restored == parallelism_context
    assert restored._real_pp_group_for_fake_spmd is None
