# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EzpzWandBLogger: a failed W&B start degrades to a no-op; success logs as before.

Uses a fake ``wandb`` module, so no network and no wandb install are needed.
CPU only, no distributed.
"""

import sys
import types

import pytest

from torchtitan.experiments.ezpz.wandb_logger import EzpzWandBLogger
from torchtitan.observability.metrics import MetricsProcessor, WandBLogger


class _FakeRun:
    pass


def _fake_wandb(*, init_error: Exception | None = None):
    mod = types.ModuleType("wandb")
    mod.calls = []
    mod.run = None

    def init(**kwargs):
        mod.calls.append(("init", kwargs))
        if init_error is not None:
            raise init_error
        mod.run = _FakeRun()
        return mod.run

    def define_metric(*args, **kwargs):
        mod.calls.append(("define_metric", args, kwargs))

    def log(data, **kwargs):
        mod.calls.append(("log", data, kwargs))

    def finish(**kwargs):
        mod.calls.append(("finish", kwargs))
        mod.run = None

    mod.init, mod.define_metric, mod.log, mod.finish = init, define_metric, log, finish
    return mod


def _build(tmp_path, monkeypatch, fake, tag=None):
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return EzpzWandBLogger.Config().build(
        log_dir=str(tmp_path / "tb"), config_dict={"a": 1}, tag=tag
    )


def test_config_builds_ezpz_logger_and_is_a_wandb_logger_config():
    assert EzpzWandBLogger.Config._owner is EzpzWandBLogger
    assert isinstance(EzpzWandBLogger.Config(), WandBLogger.Config)


def test_success_path_env_settings_metric_keys_and_step(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.setenv("WANDB_TEAM", "team")
    monkeypatch.setenv("WANDB_RUN_NAME", "name")
    fake = _fake_wandb()
    logger = _build(tmp_path, monkeypatch, fake)
    assert logger.enabled

    (_, init_kwargs) = fake.calls[0]
    assert init_kwargs["project"] == "proj"
    assert init_kwargs["entity"] == "team"
    assert init_kwargs["name"] == "name"
    assert init_kwargs["config"] == {"a": 1}

    logger.log({"loss_metrics/global_avg_loss": 2.5, "grad_norm": 1.0}, step=7)
    logger.log({"validation_metrics/loss": 2.6}, step=7)
    logs = [c for c in fake.calls if c[0] == "log"]
    # _step is the training step (step=), trainer/step is kept, and nothing is
    # force-committed, so the validation row merges into step 7.
    assert logs[0] == (
        "log",
        {"loss_metrics/global_avg_loss": 2.5, "grad_norm": 1.0, "trainer/step": 7},
        {"step": 7},
    )
    assert logs[1] == (
        "log",
        {"validation_metrics/loss": 2.6, "trainer/step": 7},
        {"step": 7},
    )
    logger.close()
    assert fake.calls[-1][0] == "finish"


def test_success_path_tag_prefix(tmp_path, monkeypatch):
    fake = _fake_wandb()
    logger = _build(tmp_path, monkeypatch, fake, tag="val")
    logger.log({"loss": 1.0}, step=3)
    assert fake.calls[-1] == ("log", {"val/loss": 1.0, "trainer/step": 3}, {"step": 3})


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("api key not configured"),
        ConnectionError("network unreachable"),
        TimeoutError("init timed out"),
    ],
)
def test_init_failure_degrades_to_noop(tmp_path, monkeypatch, caplog, error):
    fake = _fake_wandb(init_error=error)
    with caplog.at_level("WARNING"):
        logger = _build(tmp_path, monkeypatch, fake)
    assert not logger.enabled
    assert "Failed to create WandB logger" in caplog.text
    assert type(error).__name__ in caplog.text
    n_calls = len(fake.calls)
    logger.log({"loss": 1.0}, step=1)
    logger.close()
    assert len(fake.calls) == n_calls


def test_failure_after_init_finishes_the_partial_run(tmp_path, monkeypatch):
    fake = _fake_wandb()

    def define_metric(*args, **kwargs):
        raise RuntimeError("define_metric failed")

    fake.define_metric = define_metric
    logger = _build(tmp_path, monkeypatch, fake)
    assert not logger.enabled
    assert fake.calls[-1] == ("finish", {"exit_code": 1})
    assert fake.run is None


def test_missing_wandb_package_degrades(tmp_path, monkeypatch):
    # None in sys.modules makes ``import wandb`` raise ImportError.
    monkeypatch.setitem(sys.modules, "wandb", None)
    logger = EzpzWandBLogger.Config().build(log_dir=str(tmp_path / "tb"))
    assert not logger.enabled
    logger.log({"loss": 1.0}, step=1)
    logger.close()


def test_metrics_processor_survives_wandb_failure(tmp_path, monkeypatch):
    """The processor's own construction loop does not see the failure."""
    import torch

    monkeypatch.setitem(
        sys.modules, "wandb", _fake_wandb(init_error=RuntimeError("no key"))
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *a, **k: 0)
    monkeypatch.setattr(
        "torchtitan.observability.metrics._get_metrics_rank",
        lambda **kwargs: 0,
    )

    class _Monitor:
        device_name = "cpu"

        def reset_peak_stats(self):
            pass

    processor = MetricsProcessor.Config(loggers=[EzpzWandBLogger.Config()]).build(
        parallelism_context=None,
        device_memory_monitor=_Monitor(),
        dump_folder=str(tmp_path),
    )
    assert len(processor.loggers) == 1
    assert not processor.loggers[0].enabled
