# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ablation driver: is a NORMS-ONLY fp32 master sufficient?

Runs one arm of the master-weight-dtype ablation on the agpt debugmodel and
dumps per-parameter statistics to JSON so the arms can be compared offline.

Arms (select with --arm):
  A  bf16 master everywhere      (training.dtype=bfloat16)
  B  fp32 master for norms only  (training.dtype=bfloat16 + EZPZ_FP32_NORMS=1)
  C  fp32 master everywhere      (training.dtype=float32, production default)

Every arm uses the same seed, data, optimizer, and step count, so any
difference in the dumped parameters is attributable to the master dtype.

What gets dumped, per parameter (post-training, plus the init snapshot):
  dtype, numel, mean, std, var, min, max, and -- the load-bearing one --
  ``frac_changed``, the fraction of elements that differ from their value at
  step 0. For a frozen bf16 norm weight this is exactly 0.0.

Usage (single process; the debugmodel is ~20M params):

    TORCH_DEVICE=cpu RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \\
      MASTER_ADDR=127.0.0.1 MASTER_PORT=29610 \\
      python3 -m torchtitan.experiments.ezpz.scripts.oneoff.fp32_norms_ablation \\
        --arm A --steps 300 --out /tmp/armA.json

On XPU drop ``TORCH_DEVICE=cpu`` and launch under ``ezpz launch``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

# EZPZ_FP32_NORMS must be set before the model is parallelized; we set it in
# main() from --arm, well before config.build().


def _param_stats(name: str, tensor: Any, init: Any = None) -> dict[str, Any]:
    import torch

    t = tensor
    if hasattr(t, "full_tensor"):  # DTensor
        t = t.full_tensor()
    t = t.detach().to(torch.float64).flatten()
    out = {
        "name": name,
        "dtype": str(tensor.dtype),
        "numel": int(t.numel()),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "var": float(t.var(unbiased=False)),
        "min": float(t.min()),
        "max": float(t.max()),
    }
    if init is not None:
        i = init
        if hasattr(i, "full_tensor"):
            i = i.full_tensor()
        i = i.detach().to(torch.float64).flatten()
        delta = (t - i).abs()
        out["frac_changed"] = float((delta > 0).to(torch.float64).mean())
        out["max_abs_delta"] = float(delta.max())
        out["mean_abs_delta"] = float(delta.mean())
    return out


def _snapshot(model_parts) -> dict[str, Any]:
    """Detached fp64 copy of every parameter, keyed by FQN."""
    import torch

    snap = {}
    for part in model_parts:
        for name, p in part.named_parameters():
            t = p.detach()
            if hasattr(t, "full_tensor"):
                t = t.full_tensor()
            snap[name] = t.to(torch.float64).clone()
    return snap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--config", default="agpt_debugmodel_local")
    ap.add_argument(
        "--dump-params",
        action="store_true",
        help=(
            "Also save the final parameters as <out>.pt (fp32). Needed for the "
            "exact arm-B-vs-arm-C divergence measurement -- summary statistics "
            "like std are far too coarse to see it."
        ),
    )
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    arm_dtype = {"A": "bfloat16", "B": "bfloat16", "C": "float32"}[args.arm]
    if args.arm == "B":
        os.environ["EZPZ_FP32_NORMS"] = "1"
    else:
        os.environ.pop("EZPZ_FP32_NORMS", None)

    import ezpz.distributed
    import torch

    from torchtitan.config import ConfigManager
    from torchtitan.experiments.ezpz.logging import init_logger

    init_logger()

    argv = [
        "--module",
        "ezpz.agpt",
        "--config",
        args.config,
        f"--training.dtype={arm_dtype}",
        f"--training.steps={args.steps}",
        "--compile.no-enable",
        "--checkpoint.no-enable",
        "--validator.no-enable",
        "--debug.seed",
        str(args.seed),
        "--debug.deterministic",
        "--metrics.log-freq=25",
        "--metrics.no-enable-wandb",
        # LR warmup shorter than the run so the arms reach a steady state.
        "--lr-scheduler.warmup-steps=20",
        *args.extra,
    ]

    config = ConfigManager().parse_args(argv)

    # Set the optimizer programmatically: `--optimizer adamw` is consumed by
    # ezpz train.py's own pre-parser, and tyro (which we call directly here)
    # rejects it.
    from torchtitan.components.optimizer import default_adamw

    config.optimizer = default_adamw(lr=args.lr)

    trainer = config.build()

    init_snap = _snapshot(trainer.model_parts)

    # Capture the per-step loss without touching the trainer: train_step
    # returns the reduced loss for the step.
    loss_curve: list[float] = []
    _orig_train_step = trainer.train_step

    def _recording_train_step(*a, **kw):
        out = _orig_train_step(*a, **kw)
        try:
            loss_curve.append(float(out))
        except (TypeError, ValueError):
            pass
        return out

    trainer.train_step = _recording_train_step

    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0

    stats = []
    for part in trainer.model_parts:
        for name, p in part.named_parameters():
            stats.append(_param_stats(name, p, init_snap.get(name)))

    losses = loss_curve

    peak_mem = None
    try:
        mm = trainer.metrics_processor.device_memory_monitor.get_peak_stats()
        peak_mem = {
            "max_reserved_gib": float(mm.max_reserved_gib),
            "max_reserved_pct": float(mm.max_reserved_pct),
        }
    except Exception:
        pass

    result = {
        "arm": args.arm,
        "training_dtype": arm_dtype,
        "fp32_norms": os.environ.get("EZPZ_FP32_NORMS", "0"),
        "steps": args.steps,
        "seed": args.seed,
        "lr": args.lr,
        "config": args.config,
        "wall_seconds": wall,
        "sec_per_step": wall / max(args.steps, 1),
        "peak_memory": peak_mem,
        "final_step": int(trainer.step),
        "losses": losses,
        "params": stats,
        "torch": torch.__version__,
        "argv": argv,
    }

    # Stats are gathered via full_tensor(), so every rank holds the same
    # values; only rank 0 writes.
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        if args.dump_params:
            flat = {}
            for part in trainer.model_parts:
                for name, p in part.named_parameters():
                    t = p.detach()
                    if hasattr(t, "full_tensor"):
                        t = t.full_tensor()
                    flat[name] = t.to(torch.float32).cpu().clone()
            torch.save(flat, args.out + ".pt")
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"[ablation] arm={args.arm} wrote {args.out} "
            f"({len(stats)} params, {wall:.1f}s)"
        )

    trainer.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import ezpz.distributed

    ezpz.distributed.setup_torch()
    os.environ.setdefault("RANK", str(ezpz.distributed.get_rank()))
    os.environ.setdefault("WORLD_SIZE", str(ezpz.distributed.get_world_size()))
    sys.exit(main())
