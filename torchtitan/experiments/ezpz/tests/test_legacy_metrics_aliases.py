# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Legacy ``metrics.enable_{wandb,tensorboard}`` surface after #5775.

Upstream replaced the two booleans with ``MetricsProcessor.Config.loggers``.
ezpz keeps the CLI flags and JSON keys as one-way aliases on
``EzpzMetricsConfig``; these tests pin the mapping. CPU only, no distributed.
"""

import json

import pytest

from torchtitan.experiments.ezpz.config import (
    apply_legacy_metrics_fields,
    EzpzMetricsConfig,
)
from torchtitan.experiments.ezpz.config_manager import ConfigManager
from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer
from torchtitan.observability.metrics import (
    MetricsProcessor,
    TensorBoardLogger,
    WandBLogger,
)


def _types(metrics) -> list[str]:
    return [type(c).__qualname__ for c in metrics.loggers]


def test_build_owner_is_metrics_processor():
    assert EzpzMetricsConfig._owner is MetricsProcessor
    metrics_field = FaultTolerantTrainer.Config.__dataclass_fields__["metrics"]
    assert metrics_field.default_factory is EzpzMetricsConfig


@pytest.mark.parametrize(
    ("start", "wandb", "tensorboard", "expected"),
    [
        ([], True, None, ["WandBLogger.Config"]),
        ([], True, True, ["WandBLogger.Config", "TensorBoardLogger.Config"]),
        ([WandBLogger.Config()], True, None, ["WandBLogger.Config"]),
        ([WandBLogger.Config()], False, None, []),
        (
            [WandBLogger.Config(), TensorBoardLogger.Config()],
            False,
            None,
            ["TensorBoardLogger.Config"],
        ),
        ([TensorBoardLogger.Config()], None, None, ["TensorBoardLogger.Config"]),
    ],
)
def test_apply_legacy_metrics_fields(start, wandb, tensorboard, expected):
    metrics = EzpzMetricsConfig(
        loggers=list(start), enable_wandb=wandb, enable_tensorboard=tensorboard
    )
    apply_legacy_metrics_fields(metrics)
    assert _types(metrics) == expected
    assert metrics.enable_wandb is None and metrics.enable_tensorboard is None
    # Consumed: a second pass changes nothing.
    apply_legacy_metrics_fields(metrics)
    assert _types(metrics) == expected


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], ["WandBLogger.Config"]),
        (["--metrics.enable-wandb"], ["WandBLogger.Config"]),
        (["--metrics.enable_wandb"], ["WandBLogger.Config"]),
        (["--metrics.no-enable-wandb"], []),
        (["--metrics.no-enable_wandb"], []),
        (
            ["--metrics.enable-tensorboard"],
            ["WandBLogger.Config", "TensorBoardLogger.Config"],
        ),
        (
            ["--metrics.no-enable-wandb", "--metrics.enable-tensorboard"],
            ["TensorBoardLogger.Config"],
        ),
    ],
)
def test_cli_flags(flags, expected):
    # ``agpt_debugmodel`` enables W&B in the registry.
    cfg = ConfigManager().parse_args(
        ["--module", "ezpz.agpt", "--config", "agpt_debugmodel", *flags]
    )
    cfg.__post_init__()
    assert _types(cfg.metrics) == expected


def test_json_enable_wandb(tmp_path, monkeypatch):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"metrics": {"enable_wandb": True, "log_freq": 3}}))
    monkeypatch.setenv("TT_CONFIG_JSON", str(path))
    from torchtitan.experiments.ezpz.agpt import config_registry

    cfg = config_registry.ezpz_agpt_debugmodel_from_json()
    cfg.__post_init__()
    assert _types(cfg.metrics) == ["WandBLogger.Config"]
    assert cfg.metrics.log_freq == 3
