#!/usr/bin/env python3
"""Render the current OLMo-3-vocab coarse-to-fine LR campaign.

The raw CSVs remain cluster artifacts. Pass a directory containing paths of the
form ``<model>/<optimizer>/lr_finder_data.csv``. The script writes stable
per-model charts and a comparison containing only the available valid arms.

Example:
  python plot_olmo2tok_coarse_fine.py \
      --data-dir /tmp/olmo2tok-lrf \
      --output-dir docs/experiments/lr-finder/agpt/figures/olmo2tok-gbs6144
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"adamw": "#1769aa", "sophiag": "#2e7d32"}
LABELS = {"adamw": "AdamW", "sophiag": "SophiaG"}
MODEL_LABELS = {"5b_olmo2tok": "5B", "10b_olmo2tok": "10B", "30b_olmo2tok": "30B"}


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lrs: list[float] = []
    losses: list[float] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            lr = float(row["learning_rate"])
            loss = float(row["loss"])
            if np.isfinite(lr) and np.isfinite(loss):
                lrs.append(lr)
                losses.append(loss)
    if not lrs:
        raise ValueError(f"no finite rows in {path}")
    return np.asarray(lrs), np.asarray(losses)


def available(data_dir: Path) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    result: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for path in sorted(data_dir.glob("*/*/lr_finder_data.csv")):
        model, optimizer = path.parts[-3:-1]
        if optimizer not in LABELS:
            continue
        result.setdefault(model, {})[optimizer] = load(path)
    return result


def suggested_lr(model: str, optimizer: str) -> float | None:
    # Application-validated recommendations from the current fine artifacts.
    return {
        ("5b_olmo2tok", "adamw"): 7.17e-5,
        ("10b_olmo2tok", "adamw"): 4.16e-5,
    }.get((model, optimizer))


def style_ax(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Smoothed loss (EMA)")
    ax.grid(True, alpha=0.2)
    ax.set_facecolor("white")


def draw_model(ax: plt.Axes, model: str, arms: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    for optimizer, (lrs, losses) in sorted(arms.items()):
        color = COLORS[optimizer]
        label = LABELS[optimizer]
        minimum = int(np.argmin(losses))
        ax.plot(lrs, losses, color=color, linewidth=2, label=label)
        ax.scatter(
            [lrs[minimum]], [losses[minimum]], color=color, edgecolors="white",
            linewidths=1.25, s=70, zorder=4, label=f"{label} minimum",
        )
        recommendation = suggested_lr(model, optimizer)
        if recommendation is not None:
            ax.axvline(
                recommendation, color=color, linestyle=":", linewidth=1.8,
                label=f"{label} suggested LR = {recommendation:.2e}",
            )
    style_ax(ax)
    ax.set_title(f"{MODEL_LABELS.get(model, model)} OLMo-3-vocab")
    ax.legend(framealpha=0.95, fontsize=9)


def save(fig: plt.Figure, stem: Path) -> None:
    fig.patch.set_facecolor("white")
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results = available(args.data_dir)
    if not results:
        raise SystemExit(f"no valid campaign CSVs below {args.data_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    order = [m for m in ("5b_olmo2tok", "10b_olmo2tok", "30b_olmo2tok") if m in results]
    for model in order:
        fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
        draw_model(ax, model, results[model])
        fig.suptitle("Coarse-to-fine LR finder — current valid arms", fontweight="bold")
        save(fig, args.output_dir / f"lr_finder_{model}")

    fig, axes = plt.subplots(1, len(order), figsize=(7 * len(order), 5.5), squeeze=False, facecolor="white")
    for ax, model in zip(axes[0], order):
        draw_model(ax, model, results[model])
    fig.suptitle("AdamW fine sweeps — completed models", fontweight="bold")
    fig.tight_layout()
    save(fig, args.output_dir / "lr_finder_comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
