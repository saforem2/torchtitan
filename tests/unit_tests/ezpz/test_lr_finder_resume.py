# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Focused state and deterministic-replay contracts for resumable LR sweeps."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from torchtitan.experiments.ezpz.lr_finder import (
    _seed_finder_iteration,
    _stable_config_value,
    LRFinderState,
    run_lr_finder,
)
from torchtitan.experiments.ezpz.lr_finder_validation import exponential_lr_schedule


def _state() -> LRFinderState:
    return LRFinderState(
        init_lr=3e-7,
        max_lr=3e-4,
        beta=0.98,
        total_iters=75,
        warmup_steps=0,
        sweep_steps=75,
        world_size=768,
        trajectory_fingerprint="test-trajectory",
        base_seeds=[1234] * 768,
    )


def test_lr_finder_state_round_trip_preserves_resume_trajectory():
    source = _state()
    schedule = exponential_lr_schedule(source.init_lr, source.max_lr, 75)
    source.next_iter = 23
    source.curr_lr = schedule[23]
    source.avg_loss = 7.75
    source.best_loss = 7.5
    source.batch_num = 23
    source.lrs = schedule[:23]
    source.losses = [9.0 - index / 100 for index in range(23)]

    resumed = _state()
    resumed.load_state_dict(deepcopy(source.state_dict()))

    assert resumed.loaded
    assert resumed.state_dict() == source.state_dict()
    assert resumed.curr_lr == schedule[23]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_lr", 4e-4, "configuration mismatch"),
        ("world_size", 384, "configuration mismatch"),
        ("next_iter", 76, "invalid LR Finder resume cursor"),
        ("curr_lr", float("nan"), "invalid LR Finder resume LR"),
    ],
)
def test_lr_finder_state_rejects_incompatible_or_corrupt_resume(field, value, message):
    saved = _state().state_dict()
    saved[field] = value
    with pytest.raises(RuntimeError, match=message):
        _state().load_state_dict(saved)


def test_lr_finder_iteration_rng_replays_exactly_after_restart():
    expected = []
    for iteration in range(8):
        _seed_finder_iteration(991, iteration)
        expected.append(torch.rand(4))

    # Simulate a fresh process resuming at the first uncommitted iteration.
    resumed = []
    for iteration in range(3, 8):
        _seed_finder_iteration(991, iteration)
        resumed.append(torch.rand(4))

    assert all(torch.equal(left, right) for left, right in zip(expected[3:], resumed))


def test_config_fingerprint_input_does_not_use_object_addresses():
    left = SimpleNamespace(dataset="test", transform=lambda value: value)
    right = SimpleNamespace(dataset="test", transform=lambda value: value)
    assert _stable_config_value(left) == _stable_config_value(right)


def test_75_point_three_decade_schedule_is_endpoint_inclusive():
    schedule = exponential_lr_schedule(3e-7, 3e-4, 75)
    assert len(schedule) == 75
    assert schedule[0] == 3e-7
    assert schedule[-1] == 3e-4
    ratio = schedule[1] / schedule[0]
    assert ratio == pytest.approx(1000 ** (1 / 74))


class _Loader:
    def __init__(self, losses):
        self.losses = losses
        self.cursor = 0


class _Checkpointer:
    def __init__(self, trainer, saved=None):
        self.enable = True
        self.trainer = trainer
        self.states = {}
        self.saved = saved

    def load(self, step=-1):
        del step
        if self.saved is None:
            return False
        self.trainer.step = self.saved["step"]
        self.trainer.dataloader.cursor = self.saved["cursor"]
        self.states["lr_finder"].load_state_dict(deepcopy(self.saved["lr_finder"]))
        return True

    def save(self, step, last_step=False):
        del last_step
        self.saved = {
            "step": step,
            "cursor": self.trainer.dataloader.cursor,
            "lr_finder": deepcopy(self.states["lr_finder"].state_dict()),
        }
        return True

    def maybe_wait_for_staging(self):
        pass

    def maybe_wait_for_saving(self):
        pass


class _Trainer:
    def __init__(self, losses, saved=None, fail_before=None):
        self.step = 0
        self.fail_before = fail_before
        self.dataloader = _Loader(losses)
        parameter = torch.nn.Parameter(torch.zeros(()))
        self.optimizers = SimpleNamespace(
            optimizers=[torch.optim.SGD([parameter], lr=1e-6)]
        )
        self.lr_schedulers = SimpleNamespace(step=lambda: None)
        self.config = SimpleNamespace(
            lr_finder=SimpleNamespace(
                init_lr=1e-6,
                max_lr=1e-3,
                beta=0.8,
                fraction=1.0,
                warmup_fraction=0.0,
                smooth_frac=0.05,
            ),
            training=SimpleNamespace(
                steps=len(losses),
                num_tokens_per_microbatch_per_dp_rank=1,
                num_tokens_per_train_step=1,
                max_context_length=1,
            ),
            checkpoint=SimpleNamespace(load_step=-1),
            metrics=SimpleNamespace(log_freq=1000),
            model_spec=None,
            optimizer=SimpleNamespace(name="sgd"),
            dataloader=SimpleNamespace(dataset="test"),
            parallelism=SimpleNamespace(
                data_parallel_replicate_degree=1,
                data_parallel_shard_degree=1,
            ),
            dump_folder="unused",
        )
        self.checkpointer = _Checkpointer(self, saved)

    def batch_generator(self, dataloader):
        assert dataloader is self.dataloader
        while dataloader.cursor < len(dataloader.losses):
            yield dataloader.losses[dataloader.cursor]

    def train_step(self, iterator):
        if self.fail_before == self.dataloader.cursor:
            raise RuntimeError("injected interruption")
        loss = next(iterator)
        self.dataloader.cursor += 1
        return loss


def test_interrupted_resume_matches_uninterrupted_lr_and_ema(monkeypatch):
    monkeypatch.setenv("RANK", "1")  # suppress rank-zero artifact publication
    monkeypatch.setattr(
        "torchtitan.experiments.ezpz.lr_finder.find_optimal_lr",
        lambda lrs, losses, smooth_frac: [lrs[-1]],
    )
    losses = [8.0, 7.0, 6.0, 5.0, 5.5, 6.0]

    torch.manual_seed(2026)
    baseline = _Trainer(losses)
    run_lr_finder(baseline)

    torch.manual_seed(2026)
    interrupted = _Trainer(losses, fail_before=3)
    with pytest.raises(RuntimeError, match="injected interruption"):
        run_lr_finder(interrupted)
    resumed = _Trainer(losses, saved=interrupted.checkpointer.saved)
    run_lr_finder(resumed)

    assert resumed.checkpointer.saved["step"] == len(losses)
    assert resumed.checkpointer.saved["cursor"] == len(losses)
    assert (
        resumed.checkpointer.saved["lr_finder"]
        == baseline.checkpointer.saved["lr_finder"]
    )
