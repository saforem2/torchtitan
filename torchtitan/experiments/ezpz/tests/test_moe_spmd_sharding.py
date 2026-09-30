# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.ezpz.moe.config_registry import moe_debugmodel


def test_moe_routed_weights_have_spmd_placement_without_ep() -> None:
    """FSDP with dp_mesh_dims requires every parameter on the full SPMD mesh."""
    config = moe_debugmodel()
    config.model.update_from_config(config=config)
    layer = next(layer for layer in config.model.layers if layer.moe is not None)

    for grouped_linear in (
        layer.moe.routed_experts.w13,
        layer.moe.routed_experts.w2,
    ):
        assert grouped_linear.sharding_config is not None
        assert "weight" in grouped_linear.sharding_config.state_shardings
