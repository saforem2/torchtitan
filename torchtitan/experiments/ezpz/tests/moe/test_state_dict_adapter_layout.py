# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import types
from types import SimpleNamespace

import pytest
import torch

ezpz_stub = types.ModuleType("ezpz")
ezpz_stub.get_rank = lambda: 0
ezpz_stub.distributed = SimpleNamespace(verify_wandb=lambda: False)
sys.modules.setdefault("ezpz", ezpz_stub)

from torchtitan.experiments.ezpz.moe import moe_configs
from torchtitan.experiments.ezpz.moe.sharding import set_moe_sharding_config
from torchtitan.experiments.ezpz.moe.state_dict_adapter import moeStateDictAdapter
from torchtitan.models.common.linear import Linear, RowParallelLinear


def test_ezpz_moe_sharding_uses_upstream_owned_expert_layout():
    config = moe_configs["debugmodel"]()

    set_moe_sharding_config(config, enable_sp=False, enable_ep=True)

    moe_layers = [layer.moe for layer in config.layers if layer.moe is not None]
    assert moe_layers
    for moe in moe_layers:
        assert moe.routed_experts.w13.sharding_config is not None
        assert set(moe.routed_experts.w13.sharding_config.state_shardings) == {
            "weight",
        }


def test_ezpz_moe_sharding_replicates_grouped_linears_without_ep():
    config = moe_configs["debugmodel"]()

    set_moe_sharding_config(config, enable_sp=False, enable_ep=False)

    moe_layers = [layer.moe for layer in config.layers if layer.moe is not None]
    assert moe_layers
    for moe in moe_layers:
        for linear in (moe.routed_experts.w13, moe.routed_experts.w2):
            assert linear.sharding_config is not None
            assert set(linear.sharding_config.state_shardings) == {"weight"}


@pytest.mark.parametrize(
    ("enable_sp", "expected_w2_type"),
    [(False, Linear.Config), (True, RowParallelLinear.Config)],
)
def test_ezpz_shared_expert_w2_follows_sp_contract(enable_sp, expected_w2_type):
    config = moe_configs["debugmodel"]()

    set_moe_sharding_config(config, enable_sp=enable_sp, enable_ep=True)

    shared_experts = [
        layer.moe.shared_experts
        for layer in config.layers
        if layer.moe is not None and layer.moe.shared_experts is not None
    ]
    assert shared_experts
    for shared in shared_experts:
        assert type(shared.w2) is expected_w2_type
        assert shared.w13.__class__.__qualname__.startswith(
            "_LegacyInterleavedColumnParallelLinear."
        )
        assert shared.w13.param_init is not None
        assert shared.w2.param_init is not None


def test_ezpz_moe_adapter_roundtrips_shared_experts_as_native_stacked_w13():
    config = moe_configs["debugmodel"]()
    adapter = moeStateDictAdapter(config, hf_assets_path=None)
    w1 = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    w3 = w1 + 100
    w2 = torch.arange(3 * 2, dtype=torch.float32).reshape(3, 2)
    native = {
        "layers.1.moe.shared_experts.w13.weight": torch.stack((w1, w3), dim=0),
        "layers.1.moe.shared_experts.w2.weight": w2,
    }

    hf_state_dict = adapter.to_hf(native)

    torch.testing.assert_close(
        hf_state_dict["model.layers.1.mlp.shared_experts.gate_proj.weight"], w1
    )
    torch.testing.assert_close(
        hf_state_dict["model.layers.1.mlp.shared_experts.up_proj.weight"], w3
    )
    torch.testing.assert_close(
        hf_state_dict["model.layers.1.mlp.shared_experts.down_proj.weight"], w2
    )

    restored = adapter.from_hf(hf_state_dict)

    assert set(restored) == set(native)
    torch.testing.assert_close(
        restored["layers.1.moe.shared_experts.w13.weight"],
        native["layers.1.moe.shared_experts.w13.weight"],
    )
    torch.testing.assert_close(restored["layers.1.moe.shared_experts.w2.weight"], w2)
