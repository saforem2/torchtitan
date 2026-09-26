# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate figures for the 2026-09-26 AGPT-2B Stage-4 report.

The checkpoint/full-eval values are terminal hardware results recorded in the
companion Markdown report. Training curves are parsed from the retained job
12478743 log when ``--training-log`` is supplied; otherwise the checked-in
compact trajectory below is used so figures remain reproducible off-cluster.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import ambivalent  # noqa: E402

    plt.style.use(ambivalent.STYLES["ambivalent"])
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "axes.titlecolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
    }
)

HERE = Path(__file__).parent
FIGURES = HERE / "figures"

# Fallback points sampled from job 12478743's 100-step log.
FALLBACK = [
    (1, 0.3705, 0.8906, 14.29),
    (5, 0.2976, 0.9023, 12.31),
    (10, 0.2818, 0.9062, 10.05),
    (25, 0.2683, 0.9141, 8.158),
    (50, 0.2308, 0.9258, 8.062),
    (75, 0.2154, 0.9258, 7.181),
    (100, 0.2283, 0.9219, 8.020),
]


def parse_training_log(path: Path | None):
    if path is None:
        return FALLBACK
    pattern = re.compile(r"\{'loss':.*?'epoch':\s*'[^']+'\}")
    rows = []
    for text in pattern.findall(path.read_text(errors="replace")):
        item = ast.literal_eval(text)
        rows.append(
            (
                len(rows) + 1,
                float(item["loss"]),
                float(item["mean_token_accuracy"]),
                float(item["grad_norm"]),
            )
        )
    if len(rows) != 100:
        raise ValueError(f"expected 100 updates in {path}, found {len(rows)}")
    return rows


def save(fig, stem: str):
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(
        FIGURES / f"{stem}.png", dpi=170, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def training_curves(rows):
    steps, loss, accuracy, grad_norm = map(list, zip(*rows))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(steps, loss, color="#3366cc", lw=1.8)
    axes[0].set(title="Training loss", xlabel="optimizer step", ylabel="loss")
    axes[1].plot(steps, accuracy, color="#2ca02c", lw=1.8)
    axes[1].set(
        title="Assistant-token accuracy", xlabel="optimizer step", ylabel="accuracy"
    )
    axes[1].set_ylim(0.88, 0.94)
    axes[2].plot(steps, grad_norm, color="#ff7f0e", lw=1.5)
    axes[2].set(title="Global gradient norm", xlabel="optimizer step", ylabel="L2 norm")
    fig.suptitle("MetaMath GSM distillation — job 12478743")
    save(fig, "stage4_training_curves")


def checkpoint_sweep():
    names = ["Stage-2", "step 25", "step 50", "step 75", "step 100"]
    accuracy = [21.5, 29.0, 26.5, 29.0, 29.5]
    fmt = [98.5, 99.0, 98.5, 100.0, 98.0]
    trunc = [3, 2, 2, 0, 4]
    colors = ["#7f7f7f", "#3366cc", "#3366cc", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].bar(names, accuracy, color=colors)
    axes[0].axhline(21.5, color="#7f7f7f", ls="--", lw=1)
    axes[0].text(3.9, 22.1, "Stage-2 baseline", ha="right", fontsize=8)
    axes[0].set(title="GSM8K-200 accuracy", ylabel="percent", ylim=(0, 34))
    axes[1].bar(names, fmt, color=colors)
    axes[1].set(title="Strict output format", ylabel="percent", ylim=(94, 101))
    axes[2].bar(names, trunc, color=colors)
    axes[2].set(title="Length truncations", ylabel="count", ylim=(0, 5))
    for axis, values in zip(axes, (accuracy, fmt, trunc)):
        for i, value in enumerate(values):
            axis.text(
                i,
                value + (0.5 if axis is axes[0] else 0.1),
                f"{value:g}",
                ha="center",
                fontsize=8,
            )
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Distillation checkpoint selection")
    save(fig, "stage4_checkpoint_sweep")


def full_eval_and_pareto():
    labels = ["Stage-2\nbaseline", "math\nstep 75", "accepted\nα=0.65"]
    correct = [269, 375, 385]
    colors = ["#7f7f7f", "#3366cc", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, [x / 1319 * 100 for x in correct], color=colors)
    ax.set(
        title="Full GSM8K test (1,319 problems)", ylabel="accuracy (%)", ylim=(0, 34)
    )
    for bar, count in zip(bars, correct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.7,
            f"{count}/1319\n{count / 1319 * 100:.2f}%",
            ha="center",
            fontsize=9,
        )
    save(fig, "stage4_full_gsm8k")

    alpha = [0.50, 0.60, 0.65, 0.70, 0.75]
    gsm = [27.75, 29.64, 29.19, 27.90, 28.43]
    fmt = [94.84, 97.95, 98.18, 97.88, 98.33]
    bounded = [8, 7, 8, 8, 7]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    offsets = {
        0.50: (8, 8),
        0.60: (-48, -36),
        0.65: (-20, -44),
        0.70: (-70, -30),
        0.75: (-62, -40),
    }
    for a, x, y, b in zip(alpha, gsm, fmt, bounded):
        selected = a == 0.65
        ax.scatter(
            x,
            y,
            s=190 if selected else 90,
            marker="*" if selected else "o",
            color="#2ca02c" if selected else ("#3366cc" if b == 8 else "#d62728"),
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )
        ax.annotate(
            f"α={a:.2f}",
            (x, y),
            xytext=offsets[a],
            textcoords="offset points",
            fontsize=8,
        )
    # The nondominated accuracy/format frontier is alpha 0.75 -> 0.65 -> 0.60.
    frontier = [(28.43, 98.33), (29.19, 98.18), (29.64, 97.95)]
    ax.plot(
        [point[0] for point in frontier],
        [point[1] for point in frontier],
        color="#555555",
        ls="--",
        lw=1,
        label="nondominated frontier",
    )
    ax.set(
        title="Instruction/math interpolation candidates",
        xlabel="GSM8K full accuracy (%)",
        ylabel="strict format rate (%)",
    )
    ax.axvline(20.39, color="#7f7f7f", ls=":", lw=1, label="Stage-2 accuracy (20.39%)")
    fig.text(
        0.5,
        0.01,
        "color: green=selected, red=7/8 bounded, blue=8/8 bounded",
        fontsize=8,
        ha="center",
    )
    ax.legend(loc="upper left")
    save(fig, "stage4_interpolation_pareto")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-log", type=Path)
    args = parser.parse_args()
    training_curves(parse_training_log(args.training_log))
    checkpoint_sweep()
    full_eval_and_pareto()


if __name__ == "__main__":
    main()
