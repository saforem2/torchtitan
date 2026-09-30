# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from typing import Any, cast
from unittest import mock

import pytest

from torchtitan.experiments.ezpz.legacy_config_loader import LegacyConfigLoader


def test_loads_recipe_through_upstream_config_loader() -> None:
    config = LegacyConfigLoader().parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel"]
    )

    assert type(config.model).__qualname__ == "Llama3Model.Config"
    assert config.training.steps == 10


def test_supports_equals_form_and_current_sys_argv() -> None:
    argv = ["train.py", "--module=llama3", "--config=llama3_debugmodel"]
    with mock.patch.object(sys, "argv", argv):
        config = LegacyConfigLoader().parse_args()

    assert type(config.model).__qualname__ == "Llama3Model.Config"


def test_applies_validated_dotted_mutations() -> None:
    config = LegacyConfigLoader().parse_args(
        [
            "--module",
            "llama3",
            "--config",
            "llama3_debugmodel",
            "--training.steps",
            "5",
            "--metrics.no-enable-wandb",
            "--parallelism.tensor-parallel-degree=1",
        ]
    )

    assert config.training.steps == 5
    assert not config.metrics.enable_wandb
    assert config.parallelism.tensor_parallel_degree == 1


def test_checkpoint_aliases_select_and_mutate_checkpointer() -> None:
    config = LegacyConfigLoader().parse_args(
        [
            "--module",
            "llama3",
            "--config",
            "llama3_debugmodel",
            "--checkpoint.enable",
            "--checkpoint.interval=7",
        ]
    )

    assert config.checkpointer is not None
    assert config.checkpointer.interval == 7


def test_supported_operational_options_match_config_loader() -> None:
    config = LegacyConfigLoader().parse_args(
        [
            "--module",
            "llama3",
            "--config",
            "llama3_debugmodel",
            "--comm-backend",
            "fake",
            "--output-dir",
            "/tmp/torchtitan-ezpz-test",
        ]
    )

    assert config.comm.backend == "fake"
    assert config.dump_folder == "/tmp/torchtitan-ezpz-test"


def test_unknown_dotted_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown config option"):
        LegacyConfigLoader().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--training.not-real",
                "1",
            ]
        )


def test_invalid_mutation_is_revalidated() -> None:
    with pytest.raises(ValueError, match="must be greater than 0"):
        LegacyConfigLoader().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--training.num-tokens-per-microbatch-per-dp-rank",
                "0",
            ]
        )


@pytest.mark.parametrize(
    ("config_name", "global_batch_size", "steps"),
    [
        ("agpt_20b_real", 12288, 46429),
        ("agpt_20b_real", 6144, 92859),
        ("agpt_2b", 6144, 47492),
    ],
)
def test_aurora_production_geometry_and_scheduler_contract(
    config_name: str,
    global_batch_size: int,
    steps: int,
) -> None:
    # Three active Aurora umbrella seat geometries audited in PR #53.
    # Optimizer selection/LR is extracted by train.py before this loader.
    config = LegacyConfigLoader().parse_args(
        [
            "--module=ezpz.agpt",
            f"--config={config_name}",
            "--checkpoint.enable=true",
            "--checkpoint.interval=100",
            "--checkpoint.keep-latest-k=0",
            "--checkpoint.last-save-model-only=false",
            "--checkpoint.async-mode=disabled",
            "--dataloader.dataset=blendcorpus",
            "--lr-scheduler.decay-ratio=0.0",
            "--lr-scheduler.min-lr-factor=1.0",
            "--lr-scheduler.warmup-steps=20",
            "--training.local-batch-size=2",
            f"--training.global-batch-size={global_batch_size}",
            "--training.seq-len=8192",
            f"--training.steps={steps}",
            "--validator.enable=true",
            "--validator.freq=100",
            "--validator.steps=10",
        ]
    )
    config = cast(Any, config)

    assert config.training.steps == steps
    assert config.training.max_context_length == 8192
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 2 * 8192
    assert config.training.num_tokens_per_train_step == global_batch_size * 8192
    assert config.model.max_context_length == 8192
    assert all(
        layer.attention.rope.max_context_length == 8192
        for layer in config.model.layers
        if getattr(layer, "attention", None) is not None
        and getattr(layer.attention, "rope", None) is not None
    )
    assert config.checkpointer is not None
    assert config.checkpointer.interval == 100
    assert config.checkpointer.keep_latest_k == 0
    assert config.checkpointer.last_save_model_only is False
    assert config.checkpointer.async_mode == "disabled"
    assert config.optim.lr_scheduler.decay_ratio == 0.0
    assert config.optim.lr_scheduler.min_lr_factor == 1.0
    assert config.optim.lr_scheduler.warmup_steps == 20
    assert config.validator.enable is True
    assert config.validator.freq == 100
    assert config.validator.steps == 10


def test_legacy_training_geometry_is_order_independent() -> None:
    config = LegacyConfigLoader().parse_args(
        [
            "--module=ezpz.agpt",
            "--config=agpt_2b",
            "--training.global-batch-size=32",
            "--training.seq-len=4096",
            "--training.local-batch-size=3",
        ]
    )
    config = cast(Any, config)

    assert config.training.max_context_length == 4096
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 3 * 4096
    assert config.training.num_tokens_per_train_step == 32 * 4096


def test_direct_context_length_override_synchronizes_model_and_rope() -> None:
    config = LegacyConfigLoader().parse_args(
        [
            "--module=ezpz.agpt",
            "--config=agpt_debugmodel",
            "--training.max-context-length=512",
            "--training.num-tokens-per-microbatch-per-dp-rank=512",
        ]
    )
    config = cast(Any, config)

    assert config.training.max_context_length == 512
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 512
    assert config.model.max_context_length == 512
    assert all(
        layer.attention.rope.max_context_length == 512
        for layer in config.model.layers
        if getattr(layer, "attention", None) is not None
        and getattr(layer.attention, "rope", None) is not None
    )
