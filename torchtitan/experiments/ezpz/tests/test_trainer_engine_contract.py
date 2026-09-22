# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ezpz_stub = types.ModuleType("ezpz")
ezpz_stub.get_rank = lambda: 0
ezpz_stub.distributed = SimpleNamespace(verify_wandb=lambda: False)
sys.modules.setdefault("ezpz", ezpz_stub)

from torchtitan.experiments.ezpz.trainer import EzpzTrainingEngine, FaultTolerantTrainer
from torchtitan.training_engine import TrainingEngine


class _EngineLike:
    def __init__(self, step: int):
        self.num_completed_steps = step
        self.ntokens_seen = 17

    def state_dict(self):
        return {"step": self.num_completed_steps, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict):
        self.num_completed_steps = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]


def test_ezpz_trainer_uses_training_engine_state_contract():
    trainer = object.__new__(FaultTolerantTrainer)
    trainer.engine = _EngineLike(step=4)
    trainer.config = SimpleNamespace(training=SimpleNamespace(steps=5))

    assert issubclass(FaultTolerantTrainer.engine_cls, TrainingEngine)
    assert trainer.state_dict() == {"step": 4, "ntokens_seen": 17}


def test_ezpz_batch_ramp_reads_engine_completed_steps():
    trainer = object.__new__(FaultTolerantTrainer)
    trainer.engine = _EngineLike(step=1)
    trainer.gradient_accumulation_steps = 8
    trainer.batch_ramp_steps = 4
    trainer.batch_ramp_start_gas = 2

    assert trainer._effective_gas() == 4

    trainer.engine.num_completed_steps = 4
    assert trainer._effective_gas() == 8


def test_ezpz_trainer_load_state_restores_engine_state():
    trainer = object.__new__(FaultTolerantTrainer)
    trainer.engine = _EngineLike(step=0)

    trainer.load_state_dict({"step": 9, "ntokens_seen": torch.tensor(123)})

    assert trainer.engine.num_completed_steps == 9
    assert trainer.engine.ntokens_seen == torch.tensor(123)


def test_ezpz_trainer_has_no_dead_legacy_class():
    import torchtitan.experiments.ezpz.trainer as trainer_mod

    assert not hasattr(trainer_mod, "_LegacyFaultTolerantTrainer")


def test_ezpz_dataloader_build_preserves_sequence_and_token_kwargs():
    class DataloaderConfig:
        max_num_documents = None

        def build(self, **kwargs):
            built_kwargs.update(kwargs)
            return SimpleNamespace(max_num_documents=None)

    built_kwargs = {}
    trainer = object.__new__(FaultTolerantTrainer)
    trainer.config = SimpleNamespace(dataloader=DataloaderConfig())

    dataloader = trainer._build_dataloader(
        dp_degree=2,
        dp_rank=1,
        tokenizer=None,
        max_context_length=8,
        num_tokens_per_microbatch=16,
        training_steps=3,
        parallel_dims=SimpleNamespace(),
    )

    assert dataloader.max_num_documents is None
    assert built_kwargs["dp_world_size"] == 2
    assert built_kwargs["dp_rank"] == 1
    assert built_kwargs["seq_len"] == 8
    assert built_kwargs["local_batch_size"] == 2
    assert built_kwargs["max_context_length"] == 8
    assert built_kwargs["num_tokens_per_batch"] == 16
    assert built_kwargs["training_steps"] == 3


def test_ezpz_dataloader_build_rejects_subsequence_microbatch():
    trainer = object.__new__(FaultTolerantTrainer)
    trainer.config = SimpleNamespace(dataloader=SimpleNamespace())

    with pytest.raises(ValueError, match="must be at least one full sequence"):
        trainer._build_dataloader(
            dp_degree=1,
            dp_rank=0,
            tokenizer=None,
            max_context_length=8,
            num_tokens_per_microbatch=4,
            training_steps=1,
            parallel_dims=SimpleNamespace(),
        )


def test_ezpz_engine_installs_xpu_graph_wrapper(monkeypatch):
    engine = object.__new__(EzpzTrainingEngine)
    engine.config = SimpleNamespace(
        sdc_replayer=None,
        training=SimpleNamespace(disable_cuda_graphs=True),
        debug=SimpleNamespace(spmd_typechecking=False),
        parallelism=SimpleNamespace(enable_data_parallel_native_ddp=False),
    )
    engine.model_parts = [object()]
    engine.parallel_dims = SimpleNamespace(pp_enabled=False)
    engine.device = torch.device("cpu")
    wrapped = Mock(return_value="wrapped")
    monkeypatch.setattr(
        "torchtitan.experiments.ezpz.trainer.maybe_wrap_with_xpu_graph",
        wrapped,
    )
    monkeypatch.setattr(
        "torchtitan.experiments.ezpz.trainer.native_ddp_autocast_context",
        Mock(side_effect=AssertionError("native DDP should be disabled")),
    )
    monkeypatch.setattr(
        "torchtitan.experiments.ezpz.trainer.dist_utils.get_spmd_context",
        Mock(return_value="context"),
    )

    engine._initialize_forward_backward()

    wrapped.assert_called_once_with(engine._non_pp_forward_backward_body)
    assert engine.forward_backward_body_fn == "wrapped"
