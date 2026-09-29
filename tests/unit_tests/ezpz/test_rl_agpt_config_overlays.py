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
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    LinearLoRAHandler,
    LoRATransform,
    transform_model_config_,
)
from torchtitan.config.transform.cast_linear import LMHeadCastConverter
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import (
    dtensor_to_plain_tensor_state_dict,
    plain_tensor_to_dtensor_state_dict,
)
from torchtitan.experiments.ezpz.agpt import model_registry

_OVERLAY_MODULES = (
    "torchtitan.experiments.ezpz.rl.alphabet_sort_agpt.config_registry",
    "torchtitan.experiments.ezpz.rl.reason_agpt.config_registry",
)
_OVERLAYS = tuple(
    f"{name.rsplit('.', 2)[-2]}/config_registry.py" for name in _OVERLAY_MODULES
)
_RL_ROOT = Path(__file__).parents[3] / "torchtitan/experiments/ezpz/rl"


def _rl_model_config():
    config = model_registry("2b-rl", converters=[LMHeadCastConverter.Config()])
    return cast(
        Any,
        transform_model_config_(
            config,
            [
                LoRATransform(
                    handlers=(LinearLoRAHandler(),),
                    rank=8,
                    alpha=16,
                    target_modules=["wqkv", "wo"],
                )
            ],
        ),
    )


def _state_dict_layouts(model: torch.nn.Module) -> dict[str, spmd.SpmdType]:
    layouts = {}
    for module_fqn, module in model.named_modules():
        prefix = f"{module_fqn}." if module_fqn else ""
        sharding_config = getattr(module, "_sharding_config", None)
        if sharding_config is not None:
            for state_name, layout in sharding_config.state_shardings.items():
                layouts[f"{prefix}{state_name}"] = layout
    return layouts


def _check_plain_dtensor_roundtrip(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            world_size=2,
            enable_sequence_parallel=False,
        )
        parallel_dims.build_mesh()
        plain = {
            "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) + rank,
            "buffer": torch.arange(4, dtype=torch.int64),
            "metadata": "unchanged",
        }
        layouts = {
            "weight": spmd.SpmdType({"tp": spmd.S(0)}),
            "buffer": spmd.SpmdType({"tp": spmd.R}),
        }

        dtensor_state = plain_tensor_to_dtensor_state_dict(
            plain,
            state_dict_layouts=layouts,
            parallel_dims=parallel_dims,
        )
        restored = dtensor_to_plain_tensor_state_dict(dtensor_state)

        assert restored.keys() == plain.keys()
        assert isinstance(dtensor_state["weight"], DTensor)
        assert isinstance(dtensor_state["buffer"], DTensor)
        assert restored["metadata"] == plain["metadata"]
        for name in ("weight", "buffer"):
            assert restored[name].shape == plain[name].shape
            assert restored[name].dtype == plain[name].dtype
            torch.testing.assert_close(
                restored[name].cpu(), plain[name], check_device=False
            )
    finally:
        dist.destroy_process_group()


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
    module = importlib.import_module(module_name)
    factories = [
        factory
        for name, factory in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("rl_") and factory.__module__ == module_name
    ]

    assert factories
    for factory in factories:
        config = factory()
        assert config.model is not None
        assert config.trainer.parallelism.tensor_parallel_degree == 1


def test_agpt_rl_model_key_and_layout_contracts_are_stable() -> None:
    config = _rl_model_config()
    config.update_from_config(
        config=SimpleNamespace(
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=1,
            )
        )
    )
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
    config.update_from_config(config=SimpleNamespace(parallelism=ParallelismConfig()))
    with torch.device("meta"):
        model = config.build()
    layouts = _state_dict_layouts(model)
    wrapped_buffer_fqns = {
        name.replace(".attention.", ".\u005fcheckpoint_wrapped_module.attention.")
        for name, _ in model.named_buffers()
    }

    assert len(wrapped_buffer_fqns) == 12
    assert {canonical_fqn(name) for name in wrapped_buffer_fqns}.issubset(layouts)


def test_parallel_dims_with_none_process_group_is_picklable() -> None:
    parallel_dims = ParallelDims(
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

    restored = pickle.loads(pickle.dumps(parallel_dims))

    assert restored == parallel_dims
    assert restored._real_pp_group_for_fake_spmd is None


def test_plain_dtensor_roundtrip_preserves_keys_shapes_and_dtypes(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _check_plain_dtensor_roundtrip,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )
