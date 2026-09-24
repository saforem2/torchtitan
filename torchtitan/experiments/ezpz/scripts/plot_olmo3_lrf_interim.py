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
import hashlib
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
    sha256: str


RUNS = (
    Run(
        "5B",
        "AdamW",
        "Sunspot",
        "12478315",
        Path("sunspot-12478315-5b-adamw.csv"),
        "71d1e2ab9da41b24c668bba9a51a9c7a7bbd78a5a60f55ac3626a9977ac3587c",
    ),
    Run(
        "5B",
        "Mano",
        "Sunspot",
        "12478327",
        Path("sunspot-12478327-5b-mano.csv"),
        "78be71cf0c0d80745b1175337f1f94904159780f3177e89e1cd9f54950f783cf",
    ),
    Run(
        "10B",
        "Mano",
        "Sunspot",
        "12478328",
        Path("sunspot-12478328-10b-mano.csv"),
        "10ccc0c5e207432b85c6678d1b7e8269e8b98b0ddf71ec0fa78b4f1963da7e88",
    ),
)

COLORS = {"AdamW": "#1E88E5", "Mano": "#7B1FA2", "Muon": "#D32F2F"}
MARKERS = {"Sunspot": "o", "Aurora": "s"}


def load_run(run: Run, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one approved, complete campaign artifact and fail closed otherwise."""
    csv_file = data_dir / run.csv_path
    digest = hashlib.sha256(csv_file.read_bytes()).hexdigest()
    if digest != run.sha256:
        raise ValueError(
            f"{csv_file}: SHA-256 {digest} does not match approved source {run.sha256}"
        )
    lrs: list[float] = []
    losses: list[float] = []
    with csv_file.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("job_id", "").split(".", 1)[0] != run.job_id:
                continue
            lr = float(row["learning_rate"])
            loss = float(row["loss"])
            if not (np.isfinite(lr) and np.isfinite(loss)):
                raise ValueError(f"{csv_file}: job {run.job_id} has a non-finite row")
            lrs.append(lr)
            losses.append(loss)
    if len(lrs) != 150:
        raise ValueError(
            f"{csv_file}: job {run.job_id} has {len(lrs)} rows; expected exactly 150"
        )
    if len(set(lrs)) != len(lrs):
        raise ValueError(f"{csv_file}: job {run.job_id} has duplicate LR values")
    return np.asarray(lrs), np.asarray(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parents[1]
        / "docs"
        / "experiments"
        / "lr-finder"
        / "agpt"
        / "agpt-v2"
        / "data"
        / "2026-09-21-olmo3-gbs6144-interim",
    )
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
        "Interim: approved complete 150-point sweeps only; regenerate as remaining arms finish.",
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
