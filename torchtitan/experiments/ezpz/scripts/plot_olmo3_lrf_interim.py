#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render an interim snapshot of the 2026-09 OLMo-3 LR-finder campaign."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from torchtitan.experiments.ezpz.utils.plot_style import apply_style


@dataclass(frozen=True)
class Run:
    model: str
    optimizer: str
    machine: str
    job_id: str
    csv_path: Path


RUNS = (
    Run("5B", "AdamW", "Sunspot", "12478315", Path("sunspot-12478315-5b-adamw.csv")),
    Run("5B", "Mano", "Sunspot", "12478327", Path("sunspot-12478327-5b-mano.csv")),
    Run("10B", "Mano", "Sunspot", "12478328", Path("sunspot-12478328-10b-mano.csv")),
)

COLORS = {"AdamW": "#1E88E5", "Mano": "#7B1FA2", "Muon": "#D32F2F"}
MARKERS = {"Sunspot": "o", "Aurora": "s"}


def load_run(run: Run, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only rows belonging to the declared job from a possibly reused CSV."""
    lrs: list[float] = []
    losses: list[float] = []
    with (data_dir / run.csv_path).open() as fh:
        for row in csv.DictReader(fh):
            if row.get("job_id", "").split(".", 1)[0] != run.job_id:
                continue
            lr = float(row["learning_rate"])
            loss = float(row["loss"])
            if np.isfinite(lr) and np.isfinite(loss):
                lrs.append(lr)
                losses.append(loss)
    return np.asarray(lrs), np.asarray(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    plotted = 0
    for ax, model in zip(axes, ("5B", "10B")):
        for run in RUNS:
            if run.model != model:
                continue
            lrs, losses = load_run(run, args.data_dir)
            if not len(lrs):
                continue
            plotted += 1
            label = f"{run.optimizer} — {run.machine} {run.job_id} (n={len(lrs)})"
            ax.plot(
                lrs,
                losses,
                color=COLORS[run.optimizer],
                marker=MARKERS[run.machine],
                markevery=max(1, len(lrs) // 15),
                markersize=4,
                linewidth=1.8,
                label=label,
            )
            i = int(np.argmin(losses))
            ax.scatter(lrs[i], losses[i], color=COLORS[run.optimizer], s=45, zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model} at GBS=6144")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    if plotted == 0:
        raise SystemExit("no matching LR-finder rows")
    fig.suptitle(
        "OLMo-3-vocab LR finder — interim completed-run snapshot",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Interim: only complete 150-point sweeps with finite optimizer updates are shown; regenerate as remaining arms finish.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
