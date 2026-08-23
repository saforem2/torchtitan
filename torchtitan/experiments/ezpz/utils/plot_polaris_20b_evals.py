#!/usr/bin/env python3
"""Plot the Polaris 20B lm-eval trajectory (Llama2 tokenizer).

Companion to ``plot_polaris_20b.py`` (which plots the W&B *training*
trajectory). This one reads the lm-eval ``results_*.json`` files produced
by ``scripts/eval/polaris_20b_eval_llama2tok_sweep.sh`` and plots
downstream benchmark accuracy vs training step.

Why a separate eval chart: the Polaris 20B dolma run trained on
Llama2-tokenized data, so it must be evaluated with the Llama2 tokenizer,
not the gemma one the model config declares (see
docs/reference/known-bugs/polaris-20b-tokenizer-mismatch.md). These results
come from the ``results-llama2tok`` dirs; the earlier ``results`` dirs
(gemma tokenizer) are all at chance and should be ignored.

Metric convention (matches the Aurora 20B eval overview): ``acc`` for
arc_easy / boolq / piqa / winogrande, ``acc_norm`` for hellaswag /
arc_challenge / openbookqa (the length-normalized metric is the standard
headline number for those multiple-choice tasks).

Run from the repo root (on Polaris, where the eval outputs live):

    python3 torchtitan/experiments/ezpz/utils/plot_polaris_20b_evals.py \
        --eval-base outputs/evals/agpt-20b-dolma-n128 \
        --out torchtitan/experiments/ezpz/docs/production/polaris/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from torchtitan.experiments.ezpz.utils.plot_style import apply_style  # noqa: E402

apply_style()

from torchtitan.experiments.ezpz.utils.plot_production_wandb import (  # noqa: E402
    _savefig_both,
)

# Task -> (headline metric key in results json, chance baseline for the plot).
# acc_norm for the multiple-choice tasks whose options differ in length;
# acc for the binary / two-choice tasks.
TASKS: dict[str, tuple[str, float]] = {
    "arc_easy": ("acc,none", 0.25),
    "arc_challenge": ("acc_norm,none", 0.25),
    "piqa": ("acc,none", 0.50),
    "hellaswag": ("acc_norm,none", 0.25),
    "boolq": ("acc,none", 0.50),
    "openbookqa": ("acc_norm,none", 0.25),
    "winogrande": ("acc,none", 0.50),
}

# GBS x seq_len for the Polaris 20B chain -> tokens per step.
GBS = 1024
SEQ_LEN = 8192

# Distinct colors per task (Material palette, matching the dashboards).
TASK_COLORS = {
    "arc_easy": "#D32F2F",
    "arc_challenge": "#F06292",
    "piqa": "#1976D2",
    "hellaswag": "#43A047",
    "boolq": "#FF9800",
    "openbookqa": "#8E24AA",
    "winogrande": "#00897B",
}


def _load_step(eval_base: str, step: int, subdir: str) -> dict | None:
    """Return the ``results`` dict for one step, or None if absent."""
    d = os.path.join(eval_base, f"step-{step}", subdir)
    js = glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True)
    if not js:
        return None
    return json.load(open(sorted(js)[-1]))["results"]


def collect(
    eval_base: str, steps: list[int], subdir: str
) -> dict[str, list[tuple[int, float]]]:
    """Return {task: [(step, score), ...]} across all steps that have results."""
    series: dict[str, list[tuple[int, float]]] = {t: [] for t in TASKS}
    for step in steps:
        r = _load_step(eval_base, step, subdir)
        if r is None:
            continue
        for task, (metric, _chance) in TASKS.items():
            rt = r.get(task)
            if rt is None or metric not in rt:
                continue
            series[task].append((step, float(rt[metric])))
    return series


def plot_evals(
    series: dict[str, list[tuple[int, float]]], output_path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 7))
    for task, (metric, chance) in TASKS.items():
        pts = series[task]
        if not pts:
            continue
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        color = TASK_COLORS[task]
        label = f"{task} ({metric.split(',')[0]})"
        ax.plot(xs, ys, "-o", color=color, linewidth=1.8, markersize=5, label=label)
        # chance baseline (thin dashed, same color)
        ax.axhline(chance, color=color, linewidth=0.6, linestyle=":", alpha=0.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        "AuroraGPT 20B (Polaris, dolma) -- lm-eval trajectory (Llama2 tokenizer)\n"
        "dotted lines = chance baselines",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig_both(fig, output_path, dpi=200)
    plt.close(fig)
    return output_path


def emit_markdown_table(series: dict[str, list[tuple[int, float]]]) -> str:
    """Build a markdown table: rows = steps, cols = tasks (headline metric)."""
    all_steps = sorted({s for pts in series.values() for s, _ in pts})
    lookup = {t: dict(pts) for t, pts in series.items()}
    tasks = list(TASKS)
    header = (
        "| Step | Tokens | "
        + " | ".join(f"{t}<br>({TASKS[t][0].split(',')[0]})" for t in tasks)
        + " |"
    )
    sep = "|---|---:|" + "|".join("---:" for _ in tasks) + "|"
    rows = [header, sep]
    for s in all_steps:
        toks = s * GBS * SEQ_LEN / 1e9
        cells = []
        for t in tasks:
            v = lookup[t].get(s)
            cells.append(f"{v:.3f}" if v is not None else "-")
        rows.append(f"| {s:,} | {toks:.1f}B | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-base",
        default="outputs/evals/agpt-20b-dolma-n128",
        help="Dir containing step-*/results-llama2tok/",
    )
    parser.add_argument(
        "--subdir",
        default="results-llama2tok",
        help="Per-step results subdir (Llama2-tokenizer results)",
    )
    parser.add_argument(
        "--steps",
        default="100,500,1000,1500,2000,2400",
        help="Comma-separated steps to include",
    )
    parser.add_argument(
        "--out",
        default="torchtitan/experiments/ezpz/docs/production/polaris/figures",
        help="Output dir for the chart",
    )
    args = parser.parse_args()

    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    series = collect(args.eval_base, steps, args.subdir)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    chart = plot_evals(series, out_dir / "evals_20b_polaris_128n_llama2tok.png")
    print(f"wrote {chart}")
    print()
    print(emit_markdown_table(series))


if __name__ == "__main__":
    main()
