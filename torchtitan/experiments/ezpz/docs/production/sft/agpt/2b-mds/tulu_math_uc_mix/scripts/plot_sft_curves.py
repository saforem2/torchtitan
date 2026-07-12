"""Plot SFT training curves with ambivalent + Iosevka style.

Reads trainer_state.json (cumulative TRL metrics) and writes an SVG
to ../charts/sft-curves.svg.

Reproduce:
  source /home/foremans/venvs/sunspot/foremans-aurora_frameworks-2025.3.1/bin/activate
  python3 plot_sft_curves.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from torchtitan.experiments.ezpz.utils.plot_style import apply_style

apply_style()

ROOT = Path("/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan")
TRAINER_STATE = ROOT / "outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729/trainer_state.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JOB_SPANS = [
    ("12468404", 1,  10, 140, "#fde0dc"),
    ("12468404", 2,  10, 100, "#fdcec5"),
    ("12468409", 1, 110, 200, "#dde9fb"),
    ("12468409", 2, 210, 300, "#c7dcfa"),
    ("12468437", 1, 310, 400, "#d8f0d8"),
    ("12468437", 2, 410, 500, "#c5e9c5"),
    ("12468437", 3, 510, 600, "#b1e2b1"),
    ("12468437", 4, 610, 729, "#9bdb9b"),
]


def add_job_shading(ax, smin, smax):
    for job_id, attempt, lo, hi, color in JOB_SPANS:
        if hi < smin or lo > smax:
            continue
        ax.axvspan(lo, hi, alpha=0.35, color=color, zorder=0)


def cumulative_tokens(entries):
    """Stitch TRL's per-attempt num_tokens counter into a monotonic total.

    num_tokens is a per-Trainer-instance running count. Each auto-retry
    resumes from the last checkpoint and re-inits the counter to ~0, so the
    raw series drops back at every attempt boundary (the sawtooth). Whenever
    the counter falls below the previous entry, a new attempt has started, so
    carry the prior cumulative total forward as an offset. global_step is
    already continuous (restored from checkpoint), so this realigns the token
    axis with the real, cumulative amount of data the model has seen.
    """
    raw = np.array([e["num_tokens"] for e in entries], dtype=float)
    cum = np.empty_like(raw)
    offset = prev = 0.0
    for i, v in enumerate(raw):
        if v < prev:  # counter reset -> new attempt began
            offset += prev
        cum[i] = offset + v
        prev = v
    return cum


def main():
    # TRAINER_STATE lives on Sunspot (/lus/tegu/...); on any other host the
    # data is absent. Skip cleanly (exit 0) so the refresh catch-all's parallel
    # worker stays green instead of reporting a FileNotFoundError failure.
    if not TRAINER_STATE.exists():
        print(f"skip: TRAINER_STATE not found ({TRAINER_STATE}) -- not on this host")
        return
    state = json.load(open(TRAINER_STATE))
    h = state["log_history"]
    steps = np.array([e["step"] for e in h])
    loss = np.array([e["loss"] for e in h])
    grad_norm = np.array([e["grad_norm"] for e in h])
    lr = np.array([e["learning_rate"] for e in h])
    acc = np.array([e["mean_token_accuracy"] for e in h])
    entropy = np.array([e["entropy"] for e in h])
    tokens_b = cumulative_tokens(h) / 1e9

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    fig.suptitle(
        f"AuroraGPT-2B x tulu_math_uc_mix SFT trajectory  (step {steps[-1]}, 3 epochs, GBS=6144)",
        fontsize=13, y=0.995,
    )
    smin, smax = int(steps.min()), int(steps.max())
    panels = [
        (axes[0, 0], loss,       "loss",                 "loss"),
        (axes[0, 1], grad_norm,  "grad_norm",            "grad_norm"),
        (axes[0, 2], lr * 1e5,   "learning rate",        "lr (x 1e-5)"),
        (axes[1, 0], acc,        "mean token accuracy",  "mean_token_accuracy"),
        (axes[1, 1], entropy,    "entropy",              "entropy (nats)"),
        (axes[1, 2], tokens_b,   "tokens seen (cumulative)", "tokens (B)"),
    ]
    for ax, y, title, ylabel in panels:
        add_job_shading(ax, smin, smax)
        ax.plot(steps, y, lw=1.5)
        ax.scatter(steps, y, s=10)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    for ax in axes[1]:
        ax.set_xlabel("global step")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.5, label=f"{jid} attempt {a}")
        for jid, a, _, _, color in JOB_SPANS
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = OUT_DIR / "sft-curves.svg"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
