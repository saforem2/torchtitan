# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Current AGPT/MoE registry contracts and host-testable Sonic coverage.

The historical ``AGPT_2B_50K_MOE_sdpa_aurora_full_sonic`` architecture is now
restored on the current routed-expert API. These tests cover its registry and
meta-model contracts alongside the smaller Sonic flavors; executing Sonic
kernels remains XPU-only.
"""

from typing import get_args

import pytest
import torch

from torchtitan.experiments.ezpz.agpt import agpt_configs
from torchtitan.experiments.ezpz.moe import model_registry, moe_configs
from torchtitan.experiments.ezpz.moe.config_registry import (
    agpt_2b_50k_moe_sdpa_aurora_full_sonic,
    moe_10b_2b_sdpa_sonic_ep,
    moe_2b_sonic_ep2,
    moe_4b_sonic_ep12,
    moe_7b_sonic_ep12,
    moe_debugmodel_sonic,
)
from torchtitan.experiments.ezpz.moe.experts import ExpertComputeBackend
from torchtitan.experiments.ezpz.moe.model import Attention
from torchtitan.experiments.ezpz.moe.routed_experts import EzpzRoutedExperts
from torchtitan.experiments.ezpz.moe.sharding import set_moe_sharding_config
from torchtitan.experiments.ezpz.moe.state_dict_adapter import moeStateDictAdapter
from torchtitan.models.common.attention import GQAttention


SONIC_CONFIGS = (
    (agpt_2b_50k_moe_sdpa_aurora_full_sonic, 12, 2048, 2112),
    (moe_debugmodel_sonic, 2, 256, 256),
    (moe_2b_sonic_ep2, 2, 1024, 1024),
    # Non-square D/F layouts catch transposition bugs hidden by debugmodel/2B.
    (moe_4b_sonic_ep12, 12, 1536, 1024),
    (moe_7b_sonic_ep12, 12, 2048, 1280),
    (moe_10b_2b_sdpa_sonic_ep, 12, 2048, 1408),
)


def _moe_layers(model_config):
    return [layer.moe for layer in model_config.layers if layer.moe is not None]


def test_registered_expert_backends_match_the_implemented_selector():
    assert set(get_args(ExpertComputeBackend)) == {
        "for_loop",
        "grouped_mm",
        "bmm",
        "bmm_nodrop",
        "aurora_sycl",
        "aurora_full_sonic",
    }


@pytest.mark.parametrize(
    "config_factory,ep_degree,dim,hidden_dim",
    SONIC_CONFIGS,
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_current_sonic_configs_wire_routing_and_layout(
    config_factory, ep_degree, dim, hidden_dim
):
    trainer_config = config_factory()
    model_config = trainer_config.model
    layers = _moe_layers(model_config)

    assert trainer_config.parallelism.expert_parallel_degree == ep_degree
    assert model_config.dim == dim
    assert layers
    assert {layer.routed_experts.compute_backend for layer in layers} == {
        "aurora_full_sonic"
    }
    assert {layer.routed_experts.w13.out_features for layer in layers} == {hidden_dim}
    assert all(
        isinstance(layer.routed_experts, EzpzRoutedExperts.Config) for layer in layers
    )


@pytest.mark.parametrize(
    "config_factory", [moe_4b_sonic_ep12, moe_7b_sonic_ep12, moe_10b_2b_sdpa_sonic_ep]
)
def test_non_square_sonic_meta_models_preserve_weight_shapes(config_factory):
    config = config_factory().model
    with torch.device("meta"):
        model = config.build()

    routed = [
        module for module in model.modules() if isinstance(module, EzpzRoutedExperts)
    ]
    assert routed
    for module in routed:
        experts = module
        num_experts, projections, hidden_dim, dim = experts.w13.weight.shape
        assert projections == 2
        assert dim != hidden_dim
        assert tuple(experts.w2.weight.shape) == (num_experts, dim, hidden_dim)
        assert tuple(experts.w13.weight[:, 1].shape) == (num_experts, hidden_dim, dim)
        assert module._wants_routing()


def test_moe_model_registry_contract_for_sonic_base_flavor():
    model_config = model_registry("10B_2B_sdpa")

    assert model_config.__class__.__qualname__ == "moeModel.Config"
    assert model_config.build().state_dict_adapter_cls is moeStateDictAdapter
    layers = model_config.layers
    assert layers[0].attention.rope is not None


def test_agpt_and_moe_model_registries_are_current_and_distinct():
    """Guard the intentionally restored 50K flavors and registry ownership."""
    moe_flavor = "AGPT_2B_50K_MOE_sdpa_aurora_full_sonic"

    assert moe_flavor in moe_configs
    assert moe_flavor not in agpt_configs
    assert "2b_50k" in agpt_configs
    assert "10B_2B_sdpa" in moe_configs
    assert "2b" in agpt_configs
    # Generic names intentionally overlap, but they must be family-specific.
    assert moe_configs["debugmodel"] is not agpt_configs["debugmodel"]


@pytest.mark.parametrize("enable_sp", [False, True])
def test_moe_mla_sharding_contract(enable_sp):
    """DeepSeek-derived MoE flavors retain the MLA-specific sharding path."""
    model_config = moe_debugmodel_sonic().model

    set_moe_sharding_config(  # pyrefly: ignore [bad-argument-type]
        model_config, enable_sp=enable_sp, enable_ep=True
    )

    layers = model_config.layers  # pyrefly: ignore [missing-attribute]
    for layer in layers:
        attention = layer.attention
        assert isinstance(attention, Attention.Config)
        assert attention.sharding_config is not None
        assert attention.wkv_a.sharding_config is not None
        assert attention.wkv_b.sharding_config is not None
        if attention.q_lora_rank == 0:
            assert attention.wq is not None
            assert attention.wq.sharding_config is not None
        else:
            assert attention.wq_a is not None
            assert attention.wq_b is not None
            assert attention.wq_a.sharding_config is not None
            assert attention.wq_b.sharding_config is not None
        if layer.moe is not None:
            assert layer.moe.routed_experts.sharding_config is not None


@pytest.mark.parametrize("enable_sp", [False, True])
def test_agpt_50k_gqa_moe_sharding_contract(enable_sp):
    """AGPT uses GQA sharding while DeepSeek-style MoE retains MLA sharding."""
    model_config = agpt_2b_50k_moe_sdpa_aurora_full_sonic().model

    set_moe_sharding_config(  # pyrefly: ignore [bad-argument-type]
        model_config, enable_sp=enable_sp, enable_ep=True
    )

    layers = model_config.layers  # pyrefly: ignore [missing-attribute]
    for layer in layers:
        assert isinstance(layer.attention, GQAttention.Config)
        assert layer.attention.sharding_config is not None
        assert layer.attention.qkv_linear.wqkv.sharding_config is not None
        assert layer.attention.wo.sharding_config is not None
        if layer.moe is not None:
            assert layer.moe.routed_experts.sharding_config is not None


def test_agpt_50k_gqa_moe_flop_accounting():
    model_config = agpt_2b_50k_moe_sdpa_aurora_full_sonic().model
    with torch.device("meta"):
        model = model_config.build()

    nparams, flops = model_config.get_nparams_and_flops(model, seq_len=512)

    assert nparams > 0
    assert flops > 0
