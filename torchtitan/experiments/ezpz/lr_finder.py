# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Learning rate finder for torchtitan-ezpz.

Sweeps learning rates exponentially from init_lr to max_lr, recording
EMA-smoothed loss at each step. The resulting curve identifies the optimal
learning rate (steepest descent point or blow-up point / 10).

Reference:
  - Smith 2015: https://arxiv.org/abs/1506.01186
  - Gugger: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
  - Megatron-DeepSpeed: argonne-lcf/Megatron-DeepSpeed
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist

from torchtitan.experiments.ezpz.logging import logger
from torchtitan.experiments.ezpz.lr_finder_validation import (
    exponential_lr_schedule,
    validate_sweep_config,
    validate_sweep_results,
)

if TYPE_CHECKING:
    from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer


@dataclass
class LRFinderConfig:
    """Configuration for the learning rate finder."""

    enable: bool = False
    """Run LR finder sweep instead of normal training."""

    init_lr: float = 1e-6
    """Starting learning rate for the sweep."""

    max_lr: float = 1.0
    """Maximum learning rate for the sweep."""

    fraction: float = 0.1
    """Fraction of training.steps to use for the sweep."""

    beta: float = 0.98
    """EMA smoothing factor for loss."""

    warmup_fraction: float = 0.0
    """Fraction of finder steps to hold at init_lr before sweeping.
    Lets the model settle before measuring LR sensitivity."""

    smooth_frac: float = 0.05
    """Moving-average window fraction for derivative-based analysis.
    Increase for noisier curves or fewer steps (e.g. 0.1 for <50 steps)."""


@dataclass
class LRFinderState:
    """Checkpointed cursor and statistics for one logical LR sweep."""

    init_lr: float
    max_lr: float
    beta: float
    total_iters: int
    warmup_steps: int
    sweep_steps: int
    world_size: int
    trajectory_fingerprint: str
    next_iter: int = 0
    curr_lr: float = 0.0
    avg_loss: float = 0.0
    best_loss: float = float("inf")
    batch_num: int = 0
    lrs: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    base_seeds: list[int] = field(default_factory=list)
    loaded: bool = field(default=False, init=False, repr=False)

    VERSION = 1

    def __post_init__(self) -> None:
        if self.curr_lr == 0.0:
            self.curr_lr = self.init_lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "init_lr": self.init_lr,
            "max_lr": self.max_lr,
            "beta": self.beta,
            "total_iters": self.total_iters,
            "warmup_steps": self.warmup_steps,
            "sweep_steps": self.sweep_steps,
            "world_size": self.world_size,
            "trajectory_fingerprint": self.trajectory_fingerprint,
            "next_iter": self.next_iter,
            "curr_lr": self.curr_lr,
            "avg_loss": self.avg_loss,
            "best_loss": self.best_loss,
            "batch_num": self.batch_num,
            "lrs": list(self.lrs),
            "losses": list(self.losses),
            "base_seeds": list(self.base_seeds),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        expected = {
            "init_lr": self.init_lr,
            "max_lr": self.max_lr,
            "beta": self.beta,
            "total_iters": self.total_iters,
            "warmup_steps": self.warmup_steps,
            "sweep_steps": self.sweep_steps,
            "world_size": self.world_size,
            "trajectory_fingerprint": self.trajectory_fingerprint,
        }
        if state_dict.get("version") != self.VERSION:
            raise RuntimeError(
                "LR Finder checkpoint version mismatch: "
                f"expected {self.VERSION}, got {state_dict.get('version')}"
            )
        for key, value in expected.items():
            if state_dict.get(key) != value:
                raise RuntimeError(
                    f"LR Finder checkpoint configuration mismatch for {key}: "
                    f"expected {value}, got {state_dict.get(key)}"
                )

        next_iter = int(state_dict["next_iter"])
        batch_num = int(state_dict["batch_num"])
        curr_lr = float(state_dict["curr_lr"])
        lrs = [float(value) for value in state_dict["lrs"]]
        losses = [float(value) for value in state_dict["losses"]]
        if not 0 <= next_iter <= self.total_iters:
            raise RuntimeError(f"invalid LR Finder resume cursor: {next_iter}")
        if not 0 <= batch_num <= next_iter:
            raise RuntimeError(f"invalid LR Finder EMA sample count: {batch_num}")
        if len(lrs) != len(losses):
            raise RuntimeError("LR Finder checkpoint has mismatched LR/loss counts")
        expected_points = max(0, next_iter - self.warmup_steps)
        if len(lrs) != expected_points:
            raise RuntimeError(
                "LR Finder checkpoint point count does not match its cursor: "
                f"expected {expected_points}, got {len(lrs)}"
            )
        if not math.isfinite(curr_lr) or curr_lr <= 0:
            raise RuntimeError(f"invalid LR Finder resume LR: {curr_lr}")
        if any(not math.isfinite(value) for value in [*lrs, *losses]):
            raise RuntimeError("LR Finder checkpoint contains non-finite curve data")

        self.next_iter = next_iter
        self.curr_lr = curr_lr
        self.avg_loss = float(state_dict["avg_loss"])
        self.best_loss = float(state_dict["best_loss"])
        self.batch_num = batch_num
        self.lrs = lrs
        self.losses = losses
        base_seeds = [int(value) for value in state_dict["base_seeds"]]
        if len(base_seeds) != self.world_size:
            raise RuntimeError(
                "LR Finder checkpoint RNG seed count does not match world size: "
                f"expected {self.world_size}, got {len(base_seeds)}"
            )
        self.base_seeds = base_seeds
        self.loaded = True


def _seed_finder_iteration(
    base_seed: int, iteration: int, parallel_dims: Any | None = None
) -> None:
    """Make process-local model RNG reproducible across job restarts."""
    seed = (base_seed + iteration * 1_000_003) % 2**64
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % 2**32)
    except ImportError:
        pass
    if parallel_dims is not None and parallel_dims.world_size > parallel_dims.pp:
        torch.distributed.tensor._random.manual_seed(  # pyrefly: ignore[missing-attribute]
            seed, parallel_dims.world_mesh
        )


def _stable_config_value(value: Any) -> Any:
    """Convert nested config objects to deterministic JSON-compatible data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _stable_config_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _stable_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_config_value(item) for item in value]
    if callable(value):
        return {
            "callable": f"{getattr(value, '__module__', '')}."
            f"{getattr(value, '__qualname__', type(value).__qualname__)}"
        }
    if hasattr(value, "__dict__"):
        return {
            key: _stable_config_value(item)
            for key, item in sorted(vars(value).items())
            if not key.startswith("_")
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _trajectory_fingerprint(trainer: FaultTolerantTrainer) -> str:
    """Fingerprint configuration that defines an exact LR-finder trajectory."""
    config = trainer.config
    # Production configs own the model directly. Lightweight utility/test
    # trainers may intentionally omit it; fingerprint that absence rather than
    # failing before resume state can be validated.
    model_config = getattr(config, "model", None)
    payload = {
        "model": _stable_config_value(model_config),
        "optimizer_container": type(trainer.optimizers).__qualname__,
        "optimizer_config": _stable_config_value(config.optimizer),
        "dataloader_config": _stable_config_value(config.dataloader),
        "microbatch_tokens": config.training.num_tokens_per_microbatch_per_dp_rank,
        "train_step_tokens": config.training.num_tokens_per_train_step,
        "context_length": config.training.max_context_length,
        "parallelism": _stable_config_value(config.parallelism),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def find_optimal_lr(
    lrs: list[float],
    losses: list[float],
    smooth_frac: float = 0.05,
) -> list[float]:
    """Find optimal learning rates using derivative analysis.

    Smooths the loss curve, computes derivative vs log10(LR), and finds
    zero-crossings from negative to positive (local minima = blow-up points).

    Returns sorted list of candidate LRs (blow-up points). Divide by 10
    for suggested training LR.

    Ported from argonne-lcf/Megatron-DeepSpeed.
    """
    import numpy as np

    lr_arr = np.array(lrs)
    loss_arr = np.array(losses)
    n = len(lr_arr)

    if n < 5:
        return []

    # Moving-average smoothing
    window = max(1, int(n * smooth_frac))
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(loss_arr, kernel, mode="same")
        # Fix boundary effects
        for i in range(window // 2):
            smoothed[i] = loss_arr[: i + window // 2 + 1].mean()
            smoothed[-(i + 1)] = loss_arr[-(i + window // 2 + 1) :].mean()
    else:
        smoothed = loss_arr.copy()

    # Derivative of smoothed loss vs log10(LR)
    log_lr = np.log10(lr_arr)
    dloss = np.gradient(smoothed, log_lr)

    # Find zero-crossings: negative -> positive (local minima)
    minima_lrs = []
    for i in range(1, len(dloss)):
        if dloss[i - 1] < 0 and dloss[i] >= 0:
            # Interpolate the crossing point
            frac = -dloss[i - 1] / (dloss[i] - dloss[i - 1] + 1e-12)
            crossing_log_lr = log_lr[i - 1] + frac * (log_lr[i] - log_lr[i - 1])
            minima_lrs.append(10**crossing_log_lr)

    return sorted(minima_lrs)


def run_lr_finder(trainer: FaultTolerantTrainer) -> None:
    """Run an exponential LR sweep and save results.

    Sweeps LR from init_lr to max_lr over (fraction * training.steps) steps,
    recording EMA-smoothed loss at each LR. Saves CSV, NPZ, and a plot.
    Does not continue to normal training.
    """
    config = trainer.config.lr_finder
    training_steps = trainer.config.training.steps
    total_iters = max(1, int(training_steps * config.fraction))
    warmup_steps = int(total_iters * config.warmup_fraction)
    sweep_steps = validate_sweep_config(
        config.init_lr,
        config.max_lr,
        config.fraction,
        training_steps,
        config.warmup_fraction,
    )
    lr_schedule = exponential_lr_schedule(config.init_lr, config.max_lr, sweep_steps)
    mult = lr_schedule[1] / lr_schedule[0]
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    initial_seed = torch.initial_seed()
    if dist.is_initialized():
        gathered_seeds: list[int | None] = [None] * world_size
        dist.all_gather_object(gathered_seeds, initial_seed)
        base_seeds = [int(seed) for seed in gathered_seeds if seed is not None]
        if len(base_seeds) != world_size:
            raise RuntimeError("failed to gather every rank's LR Finder RNG seed")
    else:
        base_seeds = [initial_seed]
    finder_state = LRFinderState(
        init_lr=config.init_lr,
        max_lr=config.max_lr,
        beta=config.beta,
        total_iters=total_iters,
        warmup_steps=warmup_steps,
        sweep_steps=sweep_steps,
        world_size=world_size,
        trajectory_fingerprint=_trajectory_fingerprint(trainer),
        base_seeds=base_seeds,
    )
    if not getattr(trainer.checkpointer, "enable", False):
        raise RuntimeError(
            "LR Finder requires checkpointing so interrupted sweeps can resume"
        )
    trainer.checkpointer.states["lr_finder"] = finder_state
    checkpoint_loaded = trainer.checkpointer.load(
        step=trainer.config.checkpoint.load_step
    )
    if checkpoint_loaded and not finder_state.loaded and trainer.step != 0:
        raise RuntimeError(
            "Loaded a non-seed checkpoint without LR Finder state; refusing to "
            "continue from already-updated weights"
        )

    logger.info(
        f"LR Finder: sweeping from {config.init_lr:.2e} to {config.max_lr:.2e} "
        f"over {sweep_steps} steps (mult={mult:.6f})"
    )
    if warmup_steps > 0:
        logger.info(
            f"LR Finder: warmup {warmup_steps} steps at lr={config.init_lr:.2e} "
            f"before sweep"
        )

    # Reapply the authoritative LR-finder value after optimizer/scheduler load.
    curr_lr = finder_state.curr_lr
    for optimizer in trainer.optimizers.optimizers:
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

    # Build the iterator only after checkpoint load restores dataloader state.
    data_iterator = trainer.batch_generator(trainer.dataloader)

    # SUSPEND THE LR SCHEDULER FOR THE DURATION OF THE SWEEP.
    #
    # Without this the finder does not sweep the learning rate at all. It
    # writes param_group["lr"], then trainer.train_step() runs and ends with an
    # unconditional self.lr_schedulers.step() (trainer.py:1073) that recomputes
    # base_lr * lambda(step) and discards the write. This file previously had
    # no reference to the scheduler at all.
    #
    # MEASURED on the muon run's own settings (base_lr 3.0e-4, warmup_steps
    # 200): the plotted axis spans 1e-6 -> 1e-1, five decades, while the LR
    # actually used ramped 3.0e-6 -> 1.5e-4 -- about one decade, never
    # exceeding base_lr. Step 11 ran at 18x its recorded value, step 100 at
    # 0.002x. Every curve produced before this fix is a warmup ramp plotted
    # against a fictional axis, so EVERY LR-finder number in this repo
    # predating it has to be re-measured.
    #
    _sched = getattr(trainer, "lr_schedulers", None)
    _orig_step = getattr(_sched, "step", None) if _sched is not None else None
    if _orig_step is not None:
        _sched.step = lambda *a, **k: None
        logger.info(
            "LR Finder: LR scheduler suspended for the sweep "
            "(it would otherwise overwrite every swept LR)"
        )
    else:
        # Not fatal, but if a scheduler IS active the curve is meaningless in
        # exactly the way described above. Say so rather than emit a fiction.
        logger.warning(
            "LR Finder: no trainer.lr_schedulers.step to suspend. If a "
            "scheduler is active it will overwrite every swept LR and the "
            "resulting curve will be meaningless."
        )

    try:
        for i in range(finder_state.next_iter, total_iters):
            _seed_finder_iteration(
                finder_state.base_seeds[
                    dist.get_rank() if dist.is_initialized() else 0
                ],
                i,
                getattr(trainer, "parallel_dims", None),
            )
            trainer.step += 1
            in_warmup = i < warmup_steps

            # Run one training step; returns global_avg_loss
            loss_val = trainer.train_step(data_iterator)
            if loss_val is None:
                raise RuntimeError(
                    "LR Finder produced no loss; refusing to advance a resumable sweep"
                )

            if isinstance(loss_val, torch.Tensor):
                loss_val = float(loss_val.item())

            finder_state.batch_num += 1

            # EMA-smoothed loss with bias correction
            finder_state.avg_loss = (
                config.beta * finder_state.avg_loss + (1.0 - config.beta) * loss_val
            )
            smoothed_loss = finder_state.avg_loss / (
                1.0 - config.beta**finder_state.batch_num
            )

            if smoothed_loss < finder_state.best_loss or finder_state.batch_num == 1:
                finder_state.best_loss = smoothed_loss

            # Only record data points during the sweep phase
            if not in_warmup:
                finder_state.lrs.append(curr_lr)
                finder_state.losses.append(smoothed_loss)

            # Log progress
            log_freq = trainer.config.metrics.log_freq
            if (i + 1) % log_freq == 0:
                phase = "warmup" if in_warmup else "sweep"
                logger.info(
                    f"LR Finder [{phase}]: step {i + 1}/{total_iters}, "
                    f"lr={curr_lr:.8f}, smoothed_loss={smoothed_loss:.4f}"
                )

            # Advance LR exponentially only during sweep. Avoid assigning one
            # step beyond max_lr after the final recorded endpoint.
            if not in_warmup and len(finder_state.lrs) < sweep_steps:
                curr_lr = lr_schedule[len(finder_state.lrs)]
                finder_state.curr_lr = curr_lr
                for optimizer in trainer.optimizers.optimizers:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr
            finder_state.next_iter = i + 1
            if finder_state.next_iter < total_iters:
                trainer.checkpointer.save(trainer.step)
    finally:
        if _orig_step is not None:
            assert _sched is not None
            _sched.step = _orig_step
            logger.info("LR Finder: LR scheduler restored")

    # Validate before creating or appending any artifact. A partial, empty, or
    # non-finite curve is not a successful finder run and must make the launcher
    # fail instead of leaving success-looking CSV/NPZ/PNG files behind.
    trainer.checkpointer.maybe_wait_for_staging()
    trainer.checkpointer.maybe_wait_for_saving()
    lrs = finder_state.lrs
    losses = finder_state.losses
    validate_sweep_results(lrs, losses, expected_points=sweep_steps)
    blow_up_lrs = find_optimal_lr(lrs, losses, smooth_frac=config.smooth_frac)
    if not blow_up_lrs:
        raise RuntimeError(
            "LR Finder could not detect a blow-up point; refusing to mark "
            "the sweep successful. Run a wider coarse sweep or inspect the curve."
        )
    suggested = blow_up_lrs[0] / 10
    # Only a semantically valid curve earns the terminal checkpoint. Otherwise
    # the latest automatic resume point remains the preceding periodic save.
    trainer.checkpointer.save(trainer.step, last_step=True)
    trainer.checkpointer.maybe_wait_for_staging()
    trainer.checkpointer.maybe_wait_for_saving()

    # Save results on rank 0
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        # Group outputs by the direct model config type and optimizer.
        model_config = trainer.config.model
        # Derive optimizer name from container class
        opt_cls = type(trainer.optimizers).__name__
        opt_name = (
            opt_cls.removesuffix("OptimizersContainer")
            .removesuffix("Container")
            .lower()
            or "adamw"
        )
        model_name = type(model_config).__qualname__.removesuffix(".Config")
        sub_path = os.path.join("ezpz", model_name, opt_name)
        out_dir = os.path.join(trainer.config.dump_folder, "lr_finder", sub_path)
        os.makedirs(out_dir, exist_ok=True)

        # Metadata for this run
        from datetime import datetime, UTC

        run_timestamp = datetime.now(UTC).isoformat()
        job_id = os.environ.get(
            "PBS_JOBID",
            os.environ.get("SLURM_JOB_ID", "local"),
        )
        hostname = os.environ.get("HOSTNAME", "unknown")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_batch_size = trainer.config.training.num_tokens_per_train_step
        seq_len = trainer.config.training.max_context_length

        # Publish a complete per-run CSV atomically. A retry after interruption
        # replaces this logical run instead of appending duplicate rows.
        csv_path = os.path.join(out_dir, "lr_finder_data.csv")
        new_header = [
            "learning_rate",
            "loss",
            "timestamp",
            "job_id",
            "hostname",
            "world_size",
            "global_batch_size",
            "seq_len",
        ]
        fd, csv_tmp = tempfile.mkstemp(prefix=".lr_finder_data.", dir=out_dir)
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(new_header)
            for lr, loss in zip(lrs, losses):
                writer.writerow(
                    [
                        lr,
                        loss,
                        run_timestamp,
                        job_id,
                        hostname,
                        world_size,
                        global_batch_size,
                        seq_len,
                    ]
                )
            f.flush()
            os.fsync(f.fileno())
        os.replace(csv_tmp, csv_path)
        logger.info(f"LR Finder: atomically wrote {len(lrs)} rows to {csv_path}")

        # NPZ
        try:
            import numpy as np

            npz_path = os.path.join(out_dir, "lr_finder_data.npz")
            fd, npz_tmp = tempfile.mkstemp(prefix=".lr_finder_data.", dir=out_dir)
            with os.fdopen(fd, "wb") as f:
                np.savez(
                    f,
                    learning_rates=np.array(lrs),
                    losses=np.array(losses),
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(npz_tmp, npz_path)
            logger.info(f"LR Finder: saved NPZ to {npz_path}")
        except ImportError:
            logger.warning("numpy not available, skipping NPZ output")

        # Derivative-based optimal LR analysis (validated before artifacts).
        logger.info(
            f"LR Finder: suggested LR = {suggested:.2e} "
            f"(blow-up at {blow_up_lrs[0]:.2e})"
        )
        if len(blow_up_lrs) > 1:
            logger.info(
                f"LR Finder: all blow-up points: "
                f"{[f'{lr:.2e}' for lr in blow_up_lrs]}"
            )

        # Plot
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # House style (ambivalent + Iosevka), matching the production /
            # eval / docs charts. apply_style is import-safe in the XPU
            # training .venv (loads the stylesheet from file rather than
            # importing ambivalent, which would pull IPython).
            from torchtitan.experiments.ezpz.utils.plot_style import apply_style

            apply_style()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(lrs, losses, linewidth=1.5)
            ax.set_xscale("log")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Smoothed Loss")
            ax.set_title("LR Finder: Learning Rate vs. Loss")
            ax.grid(True, alpha=0.3)

            # Mark the minimum loss point
            min_idx = losses.index(min(losses))
            ax.axvline(
                x=lrs[min_idx],
                color="b",
                linestyle="--",
                alpha=0.7,
                label=f"Min loss @ lr={lrs[min_idx]:.2e}",
            )

            # Mark blow-up and suggested LR
            if blow_up_lrs:
                ax.axvline(
                    x=blow_up_lrs[0],
                    color="r",
                    linestyle="-.",
                    alpha=0.7,
                    label=f"Blow-up @ lr={blow_up_lrs[0]:.2e}",
                )
            if suggested is not None:
                ax.axvline(
                    x=suggested,
                    color="g",
                    linestyle=":",
                    linewidth=2,
                    alpha=0.8,
                    label=f"Suggested lr={suggested:.2e}",
                )

            ax.legend()

            plot_path = os.path.join(out_dir, "lr_vs_loss.png")
            fd, plot_tmp = tempfile.mkstemp(
                prefix=".lr_vs_loss.", suffix=".png", dir=out_dir
            )
            os.close(fd)
            fig.savefig(plot_tmp, dpi=150, bbox_inches="tight")
            plt.close(fig)
            os.replace(plot_tmp, plot_path)
            logger.info(f"LR Finder: saved plot to {plot_path}")
        except ImportError:
            logger.warning("matplotlib not available, skipping plot output")

        logger.info(f"LR Finder complete. {len(lrs)} data points saved to {out_dir}/")

    # Synchronize all ranks before exit
    if dist.is_initialized():
        dist.barrier()
