# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression coverage for AGPT's direct BaseModel.Config contract."""

from unittest.mock import patch

import torch

from torchtitan.experiments.ezpz.agpt import model_registry
from torchtitan.experiments.ezpz.agpt.model import AgptModel
from torchtitan.experiments.ezpz.agpt.state_dict_adapter import AgptStateDictAdapter
from torchtitan.protocols.model import BaseModel


def test_registry_returns_independent_direct_model_configs() -> None:
    first = model_registry("debugmodel")
    second = model_registry("debugmodel")

    assert isinstance(first, BaseModel.Config)
    assert isinstance(first, AgptModel.Config)
    assert first is not second
    assert first.layers is not second.layers
    assert first.max_context_length == first.layers[0].attention.rope.max_context_length


def test_model_owns_adapter_pipeline_fragment_and_xpu_parallelization() -> None:
    config = model_registry("debugmodel")
    with torch.device("meta"):
        model = config.build()

    assert isinstance(model, AgptModel)
    assert type(model).state_dict_adapter_cls is AgptStateDictAdapter
    assert model.supports_pipeline_parallel
    assert callable(model.pipeline)
    assert callable(model._fragment)
    # pipeline_llm parallelizes each partition through the model-owned method,
    # so AGPT's override remains active for pipeline stages as well.
    assert type(model).parallelize is AgptModel.parallelize

    sentinel = object()
    with patch(
        "torchtitan.experiments.ezpz.agpt.parallelize.parallelize_llama",
        return_value=sentinel,
    ) as parallelize:
        assert model.parallelize(marker="xpu") is sentinel
    parallelize.assert_called_once_with(model, marker="xpu")


def test_cos_sin_hf_export_splits_native_fused_linears() -> None:
    config = model_registry("2b_real")
    adapter = AgptStateDictAdapter(config, hf_assets_path=None)
    with torch.device("meta"):
        model = config.build()

    native_state_dict = model.state_dict()
    hf_state_dict = adapter.to_hf(native_state_dict)

    assert "model.layers.0.self_attn.q_proj.weight" in hf_state_dict
    assert "model.layers.0.self_attn.k_proj.weight" in hf_state_dict
    assert "model.layers.0.self_attn.v_proj.weight" in hf_state_dict
    assert "model.layers.0.mlp.gate_proj.weight" in hf_state_dict
    assert "model.layers.0.mlp.up_proj.weight" in hf_state_dict
    assert sum(value.numel() for value in hf_state_dict.values()) == sum(
        value.numel() for value in native_state_dict.values()
    )


def test_qknorm_sharding_and_parameter_flop_accounting_survive() -> None:
    config = model_registry("debugmodel_qknorm")

    from torchtitan.experiments.ezpz.agpt.sharding import set_agpt_sharding_config

    set_agpt_sharding_config(config, enable_sp=True)
    for layer in config.layers:
        assert layer.attention.sharding_config is not None
        assert layer.attention.qkv_linear.wqkv.sharding_config is not None
        assert layer.attention.wo.sharding_config is not None
        assert layer.attention.qk_norm is not None
        assert layer.attention.qk_norm.sharding_config is not None

    with torch.device("meta"):
        model = config.build()
    nparams, flops = config.get_nparams_and_flops(model, seq_len=512)
    assert nparams > 0
    assert flops > 0
