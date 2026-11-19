# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz must finish its model template before upstream validates the config.

``Trainer.Config.__post_init__`` runs ``validate_model_training_config``,
which checks that every sharding layout names each active mesh axis and that
accelerator graphs are not combined with a CPU-synchronizing EP dispatcher.
ezpz flavors are templates built for the default parallelism, so
``FaultTolerantTrainer.Config.__post_init__`` must apply the final sharding
and ezpz's accelerator-graph opt-out first. Before that, every MoE config
with ``expert_parallel_degree > 1`` was rejected at parse time.
"""

import dataclasses

import pytest

from torchtitan.distributed import accelerator_graph
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.experiments.ezpz import train, trainer as ezpz_trainer
from torchtitan.experiments.ezpz.config_manager import ConfigManager
from torchtitan.models.common.moe import MoE


def _parse(argv):
    args = train._inject_default_module_and_config(train._translate_legacy_args(argv))
    _, _, args = train._extract_optimizer_args(args)
    return ConfigManager().parse_args(args)


@pytest.fixture
def xpu_with_graphs(monkeypatch):
    """The Sunspot/Aurora case: torch.accelerator.Graph works on XPU."""
    for module in (accelerator_graph, ezpz_trainer):
        monkeypatch.setattr(module, "accelerator_graphs_supported", lambda: True)
    monkeypatch.setattr(ezpz_trainer, "dist_utils_device_type", lambda: "xpu")


@pytest.mark.parametrize(
    "argv",
    [
        ["--module=ezpz.moe", "--config=moe_debugmodel_ep"],
        [
            "--module=ezpz.moe",
            "--config=moe_debugmodel",
            "--parallelism.expert-parallel-degree=2",
            "--parallelism.data-parallel-shard-degree=12",
            "--parallelism.data-parallel-replicate-degree=2",
        ],
    ],
)
def test_moe_ep_config_parses(xpu_with_graphs, argv):
    cfg = _parse(argv)
    assert cfg.parallelism.expert_parallel_degree == 2
    assert cfg.training.disable_accelerator_graphs is True
    # Configurable.build constructs the model from dataclasses.replace(config),
    # which re-runs moeModel.Config.__post_init__. The EP expert placement must
    # survive that copy; resetting it to the dense template made FSDP reject
    # the experts at init (Aurora 9101603, "annotation on axis default_pg").
    for model_cfg in (cfg.model, dataclasses.replace(cfg.model)):
        for _, moe, _, _ in model_cfg.traverse(MoE.Config):
            for linear in (moe.routed_experts.w13, moe.routed_experts.w2):
                weight = linear.sharding_config.state_shardings["weight"]
                assert MeshAxisName.EP in weight.local_type
