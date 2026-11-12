# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz drops to eager when the run cannot use upstream accelerator graphs."""

from types import SimpleNamespace

import pytest

from torchtitan.experiments.ezpz import trainer as ezpz_trainer


def _config(*, native_ddp=False, backends=(), optim_graph=False, per_group=False):
    moes = [
        (
            f"layers.{i}.moe",
            SimpleNamespace(routed_experts=SimpleNamespace(compute_backend=b)),
            None,
            None,
        )
        for i, b in enumerate(backends)
    ]
    return SimpleNamespace(
        parallelism=SimpleNamespace(enable_data_parallel_native_ddp=native_ddp),
        model=SimpleNamespace(traverse=lambda cls: iter(moes)),
        optim=SimpleNamespace(enable_accelerator_graph=optim_graph),
        training=SimpleNamespace(
            disable_accelerator_graphs=False,
            accelerator_graph_per_microbatch_group=per_group,
        ),
    )


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setattr(ezpz_trainer, "accelerator_graphs_supported", lambda: True)
    monkeypatch.setattr(ezpz_trainer, "dist_utils_device_type", lambda: "cuda")


def test_dense_fsdp_keeps_graphs(cuda):
    cfg = _config()
    ezpz_trainer.maybe_disable_accelerator_graphs(cfg)
    assert cfg.training.disable_accelerator_graphs is False


def test_grouped_mm_moe_keeps_graphs(cuda):
    cfg = _config(backends=("grouped_mm", "grouped_mm"))
    ezpz_trainer.maybe_disable_accelerator_graphs(cfg)
    assert cfg.training.disable_accelerator_graphs is False


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"native_ddp": True}, "native DDP"),
        ({"backends": ("for_loop", "for_loop")}, "MoE expert backend ['for_loop']"),
        ({"backends": ("grouped_mm", "bmm")}, "MoE expert backend ['bmm']"),
    ],
)
def test_host_syncing_paths_disable_graphs(cuda, kwargs, reason):
    cfg = _config(**kwargs)
    assert ezpz_trainer.accelerator_graph_blockers(cfg, "cuda") == [reason]
    ezpz_trainer.maybe_disable_accelerator_graphs(cfg)
    assert cfg.training.disable_accelerator_graphs is True


def test_xpu_keeps_upstream_capture_off(monkeypatch):
    """XPU capture stays with ezpz's opt-in wrapper, never upstream's."""
    monkeypatch.setattr(ezpz_trainer, "accelerator_graphs_supported", lambda: True)
    monkeypatch.setattr(ezpz_trainer, "dist_utils_device_type", lambda: "xpu")
    cfg = _config()
    ezpz_trainer.maybe_disable_accelerator_graphs(cfg)
    assert cfg.training.disable_accelerator_graphs is True


@pytest.mark.parametrize("flag", ["optim_graph", "per_group"])
def test_explicit_graph_features_raise(cuda, flag):
    cfg = _config(native_ddp=True, **{flag: True})
    with pytest.raises(ValueError, match="native DDP"):
        ezpz_trainer.maybe_disable_accelerator_graphs(cfg)


def test_noop_without_accelerator_graph_support(monkeypatch):
    monkeypatch.setattr(ezpz_trainer, "accelerator_graphs_supported", lambda: False)
    cfg = _config(native_ddp=True)
    ezpz_trainer.maybe_disable_accelerator_graphs(cfg)
    assert cfg.training.disable_accelerator_graphs is False
