"""Eval charts for the full-mix 8N SFT (gs138650 x tulu_math_uc_mix_full).

Three panels (one figure -> eval-curves.svg), all from the same result JSONs /
logs the evals README tables were built from:

  1. base-LM collapse trajectory: 7 base-LM tasks vs SFT step (baseline->8672).
     The headline -- shows the cliff between ~step 1500-4500 as checkpoint-8672
     overfits the math-CoT distribution and forgets everything else.
  2. IFEval grouped bars: prompt/inst x strict/loose, for baseline / full-mix
     step-900 / full-mix step-8672 / metamathqa-729.
  3. GRPO accuracy_reward vs step for checkpoint-900 (the vLLM server-mode run).

Reproduce (Sunspot):
  module load frameworks/2025.3.1   # or source the frameworks venv
  PYTHONPATH=<repo> python3 plot_eval_curves.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    # Walk up to the repo root instead of counting parents: a depth-counted
    # path silently resolves to the wrong directory if this file ever moves,
    # and the plot then renders empty rather than failing.
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "torchtitan").is_dir():
            return parent
    raise RuntimeError("could not locate torchtitan repo root from " + __file__)


# Resolve the repo from this file's location, not a hardcoded machine path:
# the plotter is authored on Sunspot but refresh_all.sh runs it on Aurora too,
# where /lus/tegu does not exist (it failed every Aurora refresh until 2026-08-05).
REPO = _repo_root()
EVALS = REPO / "outputs" / "evals"

# Load plot_style DIRECTLY by file path. Importing it via the package
# (torchtitan.experiments.ezpz.utils...) triggers ezpz/__init__ -> import torch,
# which needs MKL libs not present on the login/plot host. plot_style.py itself
# is torch-free (matplotlib + stdlib only), so load it standalone.
import importlib.util as _ilu

_ps_path = REPO / "torchtitan" / "experiments" / "ezpz" / "utils" / "plot_style.py"
_spec = _ilu.spec_from_file_location("_ezpz_plot_style", _ps_path)
_ps = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ps)
_ps.apply_style()
OUT_DIR = Path(__file__).resolve().parent.parent / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- base-LM sweep ---------------------------------------------------------
# steps 300/600/900 live in the 12470365 sweep; 1500..8672 in 12470886.
# baseline (gs138650) is in 12470365 under label "baseline-gs138650".
SWEEP_365 = EVALS / "fullmix-8n-sweep-12470365"
SWEEP_886 = EVALS / "fullmix-8n-sweep-12470886"

# The eval sweeps this reads live under outputs/evals/ on SUNSPOT. Without
# them the series are empty and plotting raises IndexError on x[-1]; skip
# cleanly so the catch-all refresh does not fail on an absent-data host.
if not (SWEEP_365.is_dir() and SWEEP_886.is_dir()):
    print(f"skip: fullmix-8n sweeps not found under {EVALS} -- nothing to plot")
    raise SystemExit(0)

TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "piqa",
    "openbookqa",
    "boolq",
]
# random-chance reference per task (n-way multiple choice)
CHANCE = {
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "winogrande": 0.50,
    "piqa": 0.50,
    "openbookqa": 0.25,
    "boolq": 0.50,
}


def _score(task_results: dict) -> float | None:
    """acc_norm where present else acc (matches the evals README convention)."""
    if "acc_norm,none" in task_results:
        return task_results["acc_norm,none"]
    if "acc,none" in task_results:
        return task_results["acc,none"]
    for k, v in task_results.items():
        if k.startswith("acc,"):
            return v
    return None


def _read_task(sweep: Path, label: str, task: str) -> float | None:
    hits = glob.glob(str(sweep / label / task / "**" / "results*.json"), recursive=True)
    if not hits:
        return None
    res = json.load(open(hits[0]))["results"].get(task, {})
    return _score(res)


def base_lm_series() -> tuple[list[int], dict[str, list[float]]]:
    """Return (steps, {task: [scores aligned to steps]}). step 0 = baseline."""
    # (step, sweep, label)
    points = [
        (0, SWEEP_365, "baseline-gs138650"),
        (300, SWEEP_365, "sft-step300"),
        (600, SWEEP_365, "sft-step600"),
        (900, SWEEP_365, "sft-step900"),
        (1500, SWEEP_886, "sft-step1500"),
        (3000, SWEEP_886, "sft-step3000"),
        (4500, SWEEP_886, "sft-step4500"),
        (6000, SWEEP_886, "sft-step6000"),
        (7500, SWEEP_886, "sft-step7500"),
        (8672, SWEEP_886, "sft-step8672"),
    ]
    steps: list[int] = []
    series: dict[str, list[float]] = {t: [] for t in TASKS}
    for step, sweep, label in points:
        row = {t: _read_task(sweep, label, t) for t in TASKS}
        if all(v is None for v in row.values()):
            continue  # label absent (e.g. baseline only in one sweep)
        steps.append(step)
        for t in TASKS:
            series[t].append(row[t] if row[t] is not None else np.nan)
    return steps, series


# --- IFEval ----------------------------------------------------------------
IFEVAL_KEYS = [
    ("prompt_level_strict_acc,none", "prompt/strict"),
    ("inst_level_strict_acc,none", "inst/strict"),
    ("prompt_level_loose_acc,none", "prompt/loose"),
    ("inst_level_loose_acc,none", "inst/loose"),
]
IFEVAL_MODELS = [
    ("baseline", EVALS / "aurora2b-ifeval-20260717-111225", "baseline"),
    ("full-mix 900", EVALS / "aurora2b-ifeval-20260717-121518", "sft-step900"),
    ("full-mix 8672", EVALS / "aurora2b-ifeval-20260717-111225", "sft-step8672"),
    ("metamathqa 729", EVALS / "aurora2b-ifeval-20260610-204909", "sft-step729"),
]


def _ifeval(base: Path, label: str) -> dict | None:
    hits = glob.glob(str(base / label / "**" / "results*.json"), recursive=True)
    if not hits:
        return None
    try:
        return json.load(open(hits[0]))["results"]["ifeval"]
    except Exception:
        return None


# --- GRPO reward -----------------------------------------------------------
GRPO_LOG = REPO / "logs" / "grpo-aurora2b-sft-arithmetic-vllm-xnode-12470959" / "trainer.log"


def grpo_reward() -> list[float]:
    if not GRPO_LOG.exists():
        return []
    txt = GRPO_LOG.read_text(errors="replace")
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    return [float(m) for m in re.findall(r"rewards/accuracy_reward/mean':\s*'([0-9.]+)", txt)]


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # panel 1: base-LM collapse
    ax = axes[0]
    steps, series = base_lm_series()
    x = np.array(steps)
    for t in TASKS:
        y = np.array(series[t], dtype=float)
        ax.plot(x, y, marker="o", ms=3, lw=1.4, label=t)
    # chance band (min..max of per-task chance)
    lo, hi = min(CHANCE.values()), max(CHANCE.values())
    ax.axhspan(lo, hi, color="gray", alpha=0.12, zorder=0)
    ax.axhline(lo, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(hi, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.text(x[-1], hi, " chance", va="bottom", ha="right", fontsize=7, color="gray")
    ax.set_title("base-LM collapse (acc_norm/acc vs SFT step)")
    ax.set_xlabel("SFT step (0 = gs138650 baseline)")
    ax.set_ylabel("score")
    ax.legend(fontsize=6, ncol=2, loc="upper right")

    # panel 2: IFEval grouped bars
    ax = axes[1]
    model_names = [m[0] for m in IFEVAL_MODELS]
    data = {}
    for name, base, label in IFEVAL_MODELS:
        r = _ifeval(base, label)
        data[name] = [r.get(k[0]) if r else np.nan for k in IFEVAL_KEYS]
    nmetrics = len(IFEVAL_KEYS)
    nmodels = len(IFEVAL_MODELS)
    bw = 0.8 / nmodels
    xm = np.arange(nmetrics)
    for i, name in enumerate(model_names):
        ax.bar(xm + i * bw, data[name], width=bw, label=name)
    ax.set_xticks(xm + bw * (nmodels - 1) / 2)
    ax.set_xticklabels([k[1] for k in IFEVAL_KEYS], fontsize=7)
    ax.set_title("IFEval (higher = better instruction following)")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=6, loc="upper left")

    # panel 3: GRPO reward
    ax = axes[2]
    r = grpo_reward()
    if r:
        xs = np.arange(1, len(r) + 1)
        ax.plot(xs, r, marker="o", ms=4, lw=1.5, color="tab:green")
        ax.set_ylim(0, 1)
    ax.set_title("GRPO accuracy_reward -- ckpt-900 (sum_digits, vLLM)")
    ax.set_xlabel("GRPO step (~20; hit 6h walltime)")
    ax.set_ylabel("accuracy_reward/mean")

    fig.suptitle(
        "AuroraGPT-2B (MDS) x tulu_math_uc_mix_full SFT -- eval summary "
        "(deliverable = checkpoint-900; 8672 overfit/forgot)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "eval-curves.svg"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    # echo the plotted values for cross-check vs the README tables
    print("base-LM steps:", steps)
    for t in TASKS:
        vals = [f"{v:.3f}" for v in series[t]]
        print(f"  {t:14s} {vals}")
    print("IFEval:")
    for name in model_names:
        print(f"  {name:16s} {[round(v,4) if v==v else None for v in data[name]]}")
    print("GRPO accuracy_reward:", [round(v, 3) for v in r])


if __name__ == "__main__":
    main()
