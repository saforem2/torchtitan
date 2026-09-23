# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dependency-free validation for learning-rate finder output."""

from __future__ import annotations

import math


def exponential_lr_schedule(init_lr: float, max_lr: float, points: int) -> list[float]:
    """Return a logarithmic schedule that includes both requested endpoints."""
    if points < 2:
        raise ValueError("an exponential LR schedule requires at least 2 points")
    multiplier = (max_lr / init_lr) ** (1.0 / (points - 1))
    values = [init_lr * multiplier**index for index in range(points)]
    values[-1] = max_lr  # avoid endpoint drift from repeated floating-point powers
    return values


def validate_sweep_config(
    init_lr: float,
    max_lr: float,
    fraction: float,
    training_steps: int,
    warmup_fraction: float,
) -> int:
    """Reject invalid ranges before spending accelerator time."""
    if not math.isfinite(init_lr) or init_lr <= 0:
        raise ValueError("lr_finder.init_lr must be finite and greater than zero")
    if not math.isfinite(max_lr) or max_lr <= init_lr:
        raise ValueError("lr_finder.max_lr must be finite and greater than init_lr")
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("lr_finder.fraction must be finite and in (0, 1]")
    if training_steps < 1:
        raise ValueError("training.steps must be at least 1")
    if not math.isfinite(warmup_fraction) or not 0 <= warmup_fraction < 1:
        raise ValueError("lr_finder.warmup_fraction must be finite and in [0, 1)")
    total_iters = max(1, int(training_steps * fraction))
    sweep_steps = total_iters - int(total_iters * warmup_fraction)
    if sweep_steps < 5:
        raise ValueError(
            f"LR Finder requires at least 5 planned sweep points; got {sweep_steps}"
        )
    return sweep_steps


def validate_sweep_results(
    lrs: list[float], losses: list[float], expected_points: int | None = None
) -> None:
    """Raise when a sweep cannot safely be treated as a successful result."""
    if len(lrs) != len(losses):
        raise RuntimeError(
            "LR Finder produced different learning-rate and loss counts; "
            f"got {len(lrs)} and {len(losses)}"
        )
    if expected_points is not None and len(lrs) != expected_points:
        raise RuntimeError(
            "LR Finder produced a partial sweep; "
            f"expected {expected_points} points, got {len(lrs)}"
        )
    if len(lrs) < 5:
        raise RuntimeError(
            f"LR Finder requires at least 5 measured points; got {len(lrs)}"
        )
    if any(not math.isfinite(value) or value <= 0 for value in lrs):
        raise RuntimeError(
            "LR Finder produced a non-finite or non-positive learning rate"
        )
    if any(not math.isfinite(value) for value in losses):
        raise RuntimeError("LR Finder produced a non-finite loss; refusing artifacts")
