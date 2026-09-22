#!/usr/bin/env python3
"""Aggregate lm-eval results across training steps and generate plots/tables.

Two layouts supported:

DCP (current torchtitan training):
    outputs/evals/agpt-{model}/step-{N}/results/results.json
    -> single curve per task

MDS (Megatron-DeepSpeed AuroraGPT-2B optimizer-experiments runs):
    outputs/evals/agpt-{model}-mds/{stage}/step-{N}/results/results.json
    -> one curve per (stage, task), since each stage is a different
       continuation branch off the AdamW parent run

Generates per-model plots in `docs/evals/agpt/{model}/figures/eval_{model}.png`
and prints a markdown table of accuracies.

Usage:
    python aggregate_evals.py --model 2b
    python aggregate_evals.py --model 20b
    python aggregate_evals.py --model 2b-mds
    python aggregate_evals.py --model both
    python aggregate_evals.py --model 2b --csv outputs/evals/eval_results_2b.csv
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ambivalent is required — silent fallback hides style regressions.
# Install with: uv pip install --no-deps "git+https://github.com/saforem2/ambivalent"
import ambivalent  # noqa: F401

plt.style.use(ambivalent.STYLES["ambivalent"])


TASK_COLORS = {
    "hellaswag": "#e74c3c",
    "arc_easy": "#2ecc71",
    "arc_challenge": "#3498db",
    "winogrande": "#f39c12",
    "piqa": "#9b59b6",
    "openbookqa": "#1abc9c",
    "boolq": "#34495e",
    "mmlu": "#e67e22",
    "gsm8k": "#16a085",
}

TASK_ORDER = [
    # legacy commonsense dashboard
    "hellaswag", "arc_easy", "arc_challenge", "winogrande",
    "piqa", "openbookqa", "boolq",
    # modern suite (2026-07 landscape review)
    "mmlu", "gsm8k",
]

SHOTS = "0shot"

DCP_MODEL_OVERRIDES = {
    "20b": [
        {
            "base": "agpt-20b-v2-256n",
            "corrected": "agpt-20b-v2-256n-ropefix",
            "switch_step": 3101,
            "trusted_base_steps": {16_000},
        },
        {
            "base": "agpt-20b-v2-512n",
            "corrected": "agpt-20b-v2-512n-ropefix",
            "switch_step": 4401,
            "extra_bases": ["agpt-20b-v2-512n-constlr"],
        },
    ],
}


def _ordered_tasks(found):
    """TASK_ORDER first (those present), then any extras alphabetically."""
    present = [t for t in TASK_ORDER if t in found]
    extras = sorted(t for t in found if t not in TASK_ORDER)
    return present + extras


def _task_metrics(results: dict, task: str) -> dict | None:
    tagged = results.get(f"{task}@{SHOTS}")
    if isinstance(tagged, dict):
        return tagged
    if any(k.startswith(f"{task}@") for k in results):
        return None
    if task == "arc_challenge" and any(k.startswith("mmlu") for k in results):
        return None
    metrics = results.get(task)
    return metrics if isinstance(metrics, dict) else None


RANDOM_BASELINES = {
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "winogrande": 0.5,
    "piqa": 0.5,        # binary choice
    "openbookqa": 0.25, # 4-way MCQ
    "boolq": 0.5,       # yes/no
    "mmlu": 0.25,      # 4-way MCQ
    "gsm8k": 0.0,      # generative exact-match
}


def _read_one(path: Path) -> dict[str, float]:
    with open(path) as f:
        payload = json.load(f)
    results = payload.get("results", payload)
    scores: dict[str, float] = {}
    for task in results:
        # mmlu logs a 'mmlu' aggregate PLUS ~57 'mmlu_<subject>' subtasks;
        # keep only the aggregate so the table/plot are not swamped.
        if task.startswith("mmlu_"):
            continue
        m = _task_metrics(results, task)
        if m is None:
            continue
        acc = (m.get("acc_norm,none")
               or m.get("acc,none")
               or m.get("exact_match,strict-match")
               or m.get("exact_match,none")
               or m.get("exact_match,flexible-extract"))
        if acc is not None:
            scores[task] = acc
    return scores


def _load_base_dir(base: Path) -> dict[int, dict[str, float]]:
    data: dict[int, dict[str, float]] = {}
    if not base.is_dir():
        return data
    for results_path in sorted(base.glob("step-*/results/results.json")):
        step = int(results_path.parent.parent.name.split("-")[1])
        scores = _read_one(results_path)
        if scores:
            data[step] = scores
    return data


def _load_corrected_dcp(
    evals_dir: Path,
    *,
    base: str,
    corrected: str | None = None,
    switch_step: int | None = None,
    extra_bases: list[str] | None = None,
    trusted_base_steps: set[int] | None = None,
) -> dict[int, dict[str, float]]:
    original = _load_base_dir(evals_dir / base)
    extra: dict[int, dict[str, float]] = {}
    for extra_base in extra_bases or []:
        extra.update(_load_base_dir(evals_dir / extra_base))
    if corrected is None or switch_step is None:
        original.update(extra)
        return original

    corrected_steps = _load_base_dir(evals_dir / corrected)
    merged = {step: scores for step, scores in original.items() if step < switch_step}
    for step, scores in original.items():
        if step < switch_step:
            continue
        fixed = corrected_steps.get(step, {})
        if fixed:
            merged[step] = dict(fixed)
    for step, scores in corrected_steps.items():
        if step >= switch_step:
            merged.setdefault(step, scores)
    for step in trusted_base_steps or set():
        if step in original:
            merged[step] = original[step]
    merged.update(extra)
    return merged


def load_results(model: str, evals_dir: Path) -> dict[int, dict[str, float]]:
    """Load DCP-layout results (single sweep) keyed by training step.

    Reads the legacy agpt-{model}/ dir AND the per-trajectory
    agpt-{model}-v2-{LABEL}/ dirs (e.g. -v2-512n, -v2-256n) where the
    current driver writes. Per step, scores from all matching dirs are merged
    (later dirs win on key collisions), so modern tasks (mmlu/gsm8k/arc_c)
    that live in the -v2-{LABEL} dirs show up alongside the legacy suite.
    """
    data: dict[int, dict[str, float]] = {}
    for step, scores in _load_base_dir(evals_dir / f"agpt-{model}").items():
        data.setdefault(step, {}).update(scores)

    overrides = DCP_MODEL_OVERRIDES.get(model)
    if overrides is not None:
        for override in overrides:
            for step, scores in _load_corrected_dcp(evals_dir, **override).items():
                data.setdefault(step, {}).update(scores)
        return data

    for base in sorted(evals_dir.glob(f"agpt-{model}-v2-*")):
        for step, scores in _load_base_dir(base).items():
            data.setdefault(step, {}).update(scores)
    return data


def load_results_mds(
    model: str, evals_dir: Path
) -> dict[int, dict[str, list[float]]]:
    """Load MDS-layout results aggregated across replicate eval runs.

    The on-disk layout has three sibling directories
    (`ntok4673B`, `ntok7064B`, `ntok7770B`). For our SophiaG sweep these
    are symlinks to the same physical checkpoint directory, so the
    eval was effectively run three times against the same 28
    checkpoints — 3 measurement replicates per step (XPU lm-eval is
    not bitwise-deterministic, so the replicates differ at the ~1pp
    level).

    Returns ``{step: {task: [acc1, acc2, ...]}}`` with one entry per
    replicate found at that (step, task).
    """
    base = evals_dir / f"agpt-{model}"  # caller passes "2b-mds" -> agpt-2b-mds
    by_step: dict[int, dict[str, list[float]]] = {}
    for step_path in sorted(base.glob("*/step-*/results/results.json")):
        step = int(step_path.parent.parent.name.split("-")[1])
        scores = _read_one(step_path)
        for task, acc in scores.items():
            by_step.setdefault(step, {}).setdefault(task, []).append(acc)
    return dict(sorted(by_step.items()))


def make_plot(data: dict, model: str, outpath: Path) -> None:
    """Generate accuracy-vs-step plot for a model."""
    if not data:
        print(f"[skip] no data for {model}")
        return

    _, ax = plt.subplots(1, 1, figsize=(12, 7))

    for task, color in TASK_COLORS.items():
        steps = sorted([s for s in data if task in data[s]])
        accs = [data[s][task] for s in steps]
        if steps:
            ax.plot(
                steps,
                accs,
                "o-",
                color=color,
                label=task,
                markersize=6,
                linewidth=2,
            )
            ax.axhline(
                y=RANDOM_BASELINES[task],
                color=color,
                linestyle="--",
                alpha=0.3,
                linewidth=1,
            )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"agpt_{model} — Benchmark Accuracy vs Training Step ({len(data)} checkpoints)"
    )
    ax.legend(loc="upper left")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {outpath}")


def _mean_stderr(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), 0.0
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, (var / n) ** 0.5


def make_plot_mds(
    by_step: dict[int, dict[str, list[float]]],
    model: str,
    outpath: Path,
) -> None:
    """4-panel figure (one per task), mean ± stderr across replicates."""
    if not by_step:
        print(f"[skip] no MDS data for {model}")
        return

    tasks = sorted({t for s in by_step.values() for t in s})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    n_replicates = max(len(by_step[s].get(tasks[0], [])) for s in by_step)

    for ax, task in zip(axes, tasks):
        color = TASK_COLORS.get(task, "#666666")
        steps = sorted(s for s in by_step if task in by_step[s])
        means = []
        errs = []
        for s in steps:
            m, e = _mean_stderr(by_step[s][task])
            means.append(m)
            errs.append(e)
        ax.errorbar(
            steps,
            means,
            yerr=errs,
            fmt="o-",
            color=color,
            markersize=4,
            linewidth=1.5,
            capsize=2,
            alpha=0.9,
            label=f"{task} (mean ± SE, n={n_replicates})",
        )
        ax.axhline(
            y=RANDOM_BASELINES.get(task, 0.25),
            color="gray",
            linestyle=":",
            alpha=0.5,
            linewidth=1,
            label="random baseline",
        )
        ax.set_title(task)
        ax.set_ylabel("Accuracy")
        ax.legend(loc="best", fontsize=9)

    for ax in axes[len(tasks):]:
        ax.set_visible(False)
    for ax in axes[-2:]:
        ax.set_xlabel("global_step")

    fig.suptitle(
        f"agpt_{model} — MDS SophiaG sweep ({len(by_step)} unique checkpoints, "
        f"{n_replicates}x replicate evals each)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {outpath}")


def print_table_mds(
    by_step: dict[int, dict[str, list[float]]], model: str
) -> None:
    """Print a markdown table of mean accuracies per (step, task)."""
    if not by_step:
        print(f"\n## agpt_{model}: no MDS results")
        return
    tasks = sorted({t for s in by_step.values() for t in s})
    n_replicates = max(len(by_step[s].get(tasks[0], [])) for s in by_step)
    header = "| Step | " + " | ".join(tasks) + " |"
    sep = "|------|" + "|".join(["------"] * len(tasks)) + "|"
    print(
        f"\n## agpt_{model} — {len(by_step)} unique steps "
        f"({n_replicates}x replicate evals; values shown are means)\n"
    )
    print(header)
    print(sep)
    for step in sorted(by_step):
        row = f"| {step:>6} |"
        for task in tasks:
            xs = by_step[step].get(task, [])
            if xs:
                m, _ = _mean_stderr(xs)
                row += f" {m:.4f} |"
            else:
                row += " — |"
        print(row)


def write_csv_mds(
    by_step: dict[int, dict[str, list[float]]], model: str, outpath: Path
) -> None:
    """Write per-replicate MDS results to CSV.

    One row per (step, task, replicate) so downstream analysis can
    recompute means/stderrs or plot replicate spread.
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "step", "task", "replicate", "acc"])
        for step in sorted(by_step):
            for task in sorted(by_step[step]):
                for i, acc in enumerate(by_step[step][task]):
                    w.writerow([model, step, task, i, acc])
    print(f"Wrote CSV: {outpath}")


def print_table(data: dict, model: str) -> None:
    """Print a markdown table of results."""
    if not data:
        print(f"\n## agpt_{model}: no results")
        return

    tasks = _ordered_tasks({t for s in data for t in data[s]})
    header = "| Step | " + " | ".join(tasks) + " |"
    sep = "|------|" + "|".join(["------"] * len(tasks)) + "|"
    print(f"\n## agpt_{model} — {len(data)} checkpoints\n")
    print(header)
    print(sep)
    for step in sorted(data):
        row = f"| {step:>5} |"
        for task in tasks:
            v = data[step].get(task)
            row += f" {v:.4f} |" if v is not None else " — |"
        print(row)


def write_csv(data: dict, model: str, outpath: Path) -> None:
    """Write results to CSV for downstream analysis."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "step", "task", "acc"])
        for step in sorted(data):
            for task, acc in sorted(data[step].items()):
                w.writerow([model, step, task, acc])
    print(f"Wrote CSV: {outpath}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]

    parser = argparse.ArgumentParser(description="Aggregate lm-eval results")
    parser.add_argument(
        "--model",
        choices=["2b", "20b", "2b-mds", "both"],
        default="both",
        help="Which model to process. Use `2b-mds` for the Megatron-DeepSpeed sweep.",
    )
    parser.add_argument(
        "--evals-dir",
        type=Path,
        default=repo_root / "outputs/evals",
        help="Base directory containing per-step eval results",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=repo_root / "torchtitan/experiments/ezpz/docs/evals/agpt",
        help="Base docs directory for plot output",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path (default: skip)",
    )
    args = parser.parse_args()

    models = ["2b", "20b"] if args.model == "both" else [args.model]

    for model in models:
        if model.endswith("-mds"):
            stages = load_results_mds(model, args.evals_dir)
            print_table_mds(stages, model)
            plot_path = args.docs_dir / model / "figures" / f"eval_{model}.png"
            make_plot_mds(stages, model, plot_path)
            if args.csv:
                csv_path = (
                    args.csv
                    if args.model != "both"
                    else args.csv.with_stem(f"{args.csv.stem}_{model}")
                )
                write_csv_mds(stages, model, csv_path)
        else:
            data = load_results(model, args.evals_dir)
            print_table(data, model)
            plot_path = args.docs_dir / model / "figures" / f"eval_{model}.png"
            make_plot(data, model, plot_path)
            if args.csv:
                csv_path = (
                    args.csv
                    if args.model != "both"
                    else args.csv.with_stem(f"{args.csv.stem}_{model}")
                )
                write_csv(data, model, csv_path)


if __name__ == "__main__":
    main()
