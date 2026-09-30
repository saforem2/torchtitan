# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.ezpz.moe.config_registry import moe_debugmodel
from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer


def test_moe_routed_weights_have_spmd_placement_without_ep() -> None:
    """FSDP with dp_mesh_dims requires every parameter on the full SPMD mesh."""
    config = moe_debugmodel()
    config.model.set_sharding_(config.parallelism)
    layer = next(layer for layer in config.model.layers if layer.moe is not None)

    for grouped_linear in (
        layer.moe.routed_experts.w13,
        layer.moe.routed_experts.w2,
    ):
        assert grouped_linear.sharding_config is not None
        assert "weight" in grouped_linear.sharding_config.state_shardings


def test_moe_sharding_resolves_after_ep_override() -> None:
    from torchtitan.experiments.ezpz.legacy_config_loader import LegacyConfigLoader

    config = LegacyConfigLoader().parse_args(
        [
            "--module=ezpz.moe",
            "--config=moe_debugmodel",
            "--parallelism.expert-parallel-degree=1",
        ]
    )
    assert isinstance(config, FaultTolerantTrainer.Config)
    model = config.model
    model.set_sharding_(config.parallelism)
    layer = next(layer for layer in model.layers if layer.moe is not None)  # type: ignore[attr-defined]

    assert layer.moe.routed_experts.w13.sharding_config is not None
