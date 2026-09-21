# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Current AGPT/MoE registry contracts and host-testable Sonic coverage.

The historical ``AGPT_2B_50K_MOE_sdpa_aurora_full_sonic`` flavor was removed
when the MoE model API moved to routed-expert configs. These tests deliberately
exercise the Sonic flavors that current HEAD actually registers. They stop at
config/meta-model boundaries; executing Sonic kernels remains XPU-only.
"""

from typing import get_args

import pytest
import torch

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.experiments.ezpz.agpt import agpt_configs
from torchtitan.experiments.ezpz.moe import model_registry, moe_configs
from torchtitan.experiments.ezpz.moe.config_registry import (
    moe_10b_2b_sdpa_sonic_ep,
    moe_2b_sonic_ep2,
    moe_4b_sonic_ep12,
    moe_7b_sonic_ep12,
    moe_debugmodel_sonic,
)
from torchtitan.experiments.ezpz.moe.experts import ExpertComputeBackend
from torchtitan.experiments.ezpz.moe.routed_experts import EzpzRoutedExperts
from torchtitan.experiments.ezpz.moe.state_dict_adapter import moeStateDictAdapter


SONIC_CONFIGS = (
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
        "for_loop", "grouped_mm", "bmm", "bmm_nodrop", "aurora_sycl",
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
    model_config = trainer_config.model_spec.model
    layers = _moe_layers(model_config)

    assert trainer_config.parallelism.expert_parallel_degree == ep_degree
    assert model_config.dim == dim
    assert layers
    assert {layer.routed_experts.inner_experts.compute_backend for layer in layers} == {
        "aurora_full_sonic"
    }
    assert {layer.routed_experts.inner_experts.hidden_dim for layer in layers} == {
        hidden_dim
    }
    assert all(
        isinstance(layer.routed_experts, EzpzRoutedExperts.Config) for layer in layers
    )


@pytest.mark.parametrize(
    "config_factory", [moe_4b_sonic_ep12, moe_7b_sonic_ep12, moe_10b_2b_sdpa_sonic_ep]
)
def test_non_square_sonic_meta_models_preserve_weight_shapes(config_factory):
    config = config_factory().model_spec.model
    with torch.device("meta"):
        model = config.build()

    routed = [module for module in model.modules() if isinstance(module, EzpzRoutedExperts)]
    assert routed
    for module in routed:
        experts = module.inner_experts
        num_experts, hidden_dim, dim = experts.w1_EFD.shape
        assert dim != hidden_dim
        assert tuple(experts.w2_EDF.shape) == (num_experts, dim, hidden_dim)
        assert tuple(experts.w3_EFD.shape) == (num_experts, hidden_dim, dim)
        assert experts._wants_routing()


def test_moe_model_registry_contract_for_sonic_base_flavor():
    spec = model_registry("10B_2B_sdpa")

    assert spec.name == "moe"
    assert spec.flavor == "10B_2B_sdpa"
    assert spec.post_optimizer_build_fn is register_moe_load_balancing_hook
    assert spec.state_dict_adapter is moeStateDictAdapter
    model_config = spec.model
    layers = getattr(model_config, "layers")
    assert spec.max_context_length == layers[0].attention.rope.max_context_length


def test_agpt_and_moe_model_registries_are_current_and_distinct():
    """Guard current registry ownership without reviving removed 50K flavors."""
    removed_flavor = "AGPT_2B_50K_MOE_sdpa_aurora_full_sonic"

    assert removed_flavor not in moe_configs
    assert removed_flavor not in agpt_configs
    assert "10B_2B_sdpa" in moe_configs
    assert "2b" in agpt_configs
    # Generic names intentionally overlap, but they must be family-specific.
    assert moe_configs["debugmodel"] is not agpt_configs["debugmodel"]
