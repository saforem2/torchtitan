#!/usr/bin/env python3
"""Plot the Polaris 20B production trajectory from W&B.

Standalone companion to ``plot_production_wandb.py``. That script is
Aurora-tuned (12 GPU/node, the 4.67T olmo-mix token target, run-ids in
``trajectories.py``); this one keeps the Polaris 20B chain separate so
neither set of charts perturbs the other:

  - Polaris nodes are **4 GPU/node** (not 12).
  - Corpus is ``dolma`` with no fixed token target, so tokens are
    reported absolute (no "% of 4.67T").
  - The chain is a sequence of ``ezpz launch`` legs, each a distinct
    W&B run (one crashes at walltime / on a fault, the next resumes
    from the last checkpoint). Listed oldest-first; ``concat_runs``
    keeps the latest run's row per step (resume semantics).

Reuses ``concat_runs`` / ``smooth`` / ``_savefig_both`` from
``plot_production_wandb.py`` so the fetch + styling match the Aurora
dashboards.

Run from the repo root (needs W&B creds; Polaris venv has them):

    python3 torchtitan/experiments/ezpz/utils/plot_polaris_20b.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from torchtitan.experiments.ezpz.utils.plot_style import apply_style  # noqa: E402

apply_style()

from torchtitan.experiments.ezpz.utils.plot_production_wandb import (  # noqa: E402
    _savefig_both,
    concat_runs,
    smooth,
)

import wandb  # noqa: E402

# W&B run-ids for the Polaris 20B chain, oldest first (resume order).
# leg1 7237948 -> leg2 7237949 -> leg3 7243413 -> leg4 7247525 -> leg5 7252666
# -> leg6 7260480 (post-NVLink-fault restart) -> leg7 7260483.
# Append the next leg's run-id here as each leg starts logging.
RUN_IDS = [
    "jgjd0qbf",  # leg1  steps 1-398
    "h8uzg2om",  # leg2  steps 301-704
    "u81yhgtb",  # leg3  steps 701-1099
    "j56diiz9",  # leg4  steps 1001-1399
    "nm41nsbj",  # leg5  steps 1301-1448
    "nvmf9hnj",  # leg6  steps 1401-1804 (restart after NVLink fault)
    "j0jnww7s",  # leg7  steps 1701-2096
    "8snyuaxk",  # leg8  steps 2001-2405
    "rnf9tfhw",  # leg9  steps 2401-2500 (then stuck-resume -> chain went dry)
]

# Some legs' W&B history() carries all rows but scan_history(keys=...) --
# what plot_production_wandb.fetch_run uses -- returns 0 rows (a wandb
# server-side quirk seen on the leg6/leg7 runs). concat_runs falls back to
# parsing the PBS .o log when a run's history is empty; the Polaris per-step
# lines match plot_production_wandb._OLOG_STEP_RE. Point those legs at their
# .o logs on the Polaris eagle filesystem (this script runs on Polaris).
_REPO = "/eagle/AuroraGPT/foremans/projects/saforem2/torchtitan"
OLOG_FALLBACKS = {
    "nvmf9hnj": f"{_REPO}/agpt-20b-autoretry.o7260480",
    "j0jnww7s": f"{_REPO}/agpt-20b-autoretry.o7260483",
    "8snyuaxk": f"{_REPO}/agpt-20b-autoretry.o7262999",
    "rnf9tfhw": f"{_REPO}/agpt-20b-autoretry.o7263000",
}

NUM_NODES = 128
GPUS_PER_NODE = 4  # Polaris A100 (Aurora is 12)
COLOR = "#D32F2F"  # 20B red, matching MODEL_COLORS in plot_production_wandb


def plot_dashboard(data: dict[str, np.ndarray], output_path: Path) -> Path:
    """3-panel loss / TPS-per-GPU / MFU vs step for the Polaris 20B chain."""
    num_gpus = NUM_NODES * GPUS_PER_NODE
    steps = data["_step"].astype(float)
    loss = data["loss_metrics/global_avg_loss"].astype(float)
    tps_per_gpu = data["throughput(tps)"].astype(float)
    mfu = data["mfu(%)"].astype(float)

    valid = ~np.isnan(steps)
    steps, loss = steps[valid], loss[valid]
    tps_per_gpu, mfu = tps_per_gpu[valid], mfu[valid]

    tokens_b = steps[-1] * data.get("_gbs", 1024) * 8192 / 1e9 if len(steps) else 0.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"AuroraGPT 20B Production (Polaris)  |  "
        f"{NUM_NODES} nodes ({num_gpus} A100)  |  "
        f"step {int(steps[-1]):,}  |  ~{tokens_b:,.1f}B tokens (dolma)",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(steps, loss, color=COLOR, alpha=0.25, linewidth=0.5, rasterized=True)
    ax.plot(steps, smooth(loss), color=COLOR, linewidth=1.8, label="Loss (smoothed)")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, tps_per_gpu, color="#43A047", alpha=0.25, linewidth=0.5,
            rasterized=True)
    ax.plot(steps, smooth(tps_per_gpu), color="#43A047", linewidth=1.8,
            label="TPS/GPU (smoothed)")
    ax.set_ylabel("Tokens/sec/GPU")
    ax.set_title("Throughput per GPU")
    ax.legend()

    ax = axes[2]
    ax.plot(steps, mfu, color="#FF9800", alpha=0.25, linewidth=0.5, rasterized=True)
    ax.plot(steps, smooth(mfu), color="#FF9800", linewidth=1.8, label="MFU (smoothed)")
    ax.set_ylabel("MFU (%)")
    ax.set_xlabel("Training Step")
    ax.set_title("Model FLOPs Utilization")
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig_both(fig, output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_diagnostics(data: dict[str, np.ndarray], output_path: Path) -> Path:
    """grad_norm (log) + lr vs step -- the two diagnostics W&B carries here."""
    steps = data["_step"].astype(float)
    grad_norm = data["grad_norm"].astype(float)
    lr = data["lr"].astype(float)

    valid = ~np.isnan(steps)
    steps, grad_norm, lr = steps[valid], grad_norm[valid], lr[valid]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f"AuroraGPT 20B Diagnostics (Polaris)  |  {NUM_NODES} nodes  |  "
        f"step {int(steps[-1]):,}",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(steps, grad_norm, color=COLOR, alpha=0.25, linewidth=0.5, rasterized=True)
    ax.plot(steps, smooth(grad_norm), color=COLOR, linewidth=1.8,
            label="grad_norm (smoothed)")
    ax.set_ylabel("grad_norm")
    ax.set_yscale("log")
    ax.set_title("Gradient Norm")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, lr, color="#FF9800", linewidth=1.8, label="lr")
    ax.set_ylabel("lr")
    ax.set_xlabel("Training Step")
    ax.set_title("Learning Rate")
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig_both(fig, output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs" / "live" / "chains" / "polaris" / "figures",
        help="Directory to write the SVG/PNG charts into.",
    )
    args = parser.parse_args()

    api = wandb.Api()
    print(f"=== Pulling Polaris 20B chain ({len(RUN_IDS)} legs) ===")
    data = concat_runs(api, RUN_IDS, OLOG_FALLBACKS)
    print(f"  Concatenated: {len(data['_step'])} unique steps")
    if len(data["_step"]) == 0:
        raise SystemExit("no W&B data pulled -- nothing to plot")

    plot_dashboard(data, args.output_dir / "production_20b_polaris_128n.svg")
    plot_diagnostics(data, args.output_dir / "diagnostics_20b_polaris_128n.svg")


if __name__ == "__main__":
    main()
