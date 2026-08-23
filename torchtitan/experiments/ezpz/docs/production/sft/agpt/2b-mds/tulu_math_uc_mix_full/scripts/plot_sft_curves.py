"""Plot the FULL-mix 8N SFT training curves (ambivalent + Iosevka style).

Reads the LATEST checkpoint's trainer_state.json (cumulative TRL metrics
across the afterany chain) and writes an SVG to ../charts/sft-curves.svg.

This is the ongoing 8N run (gs138650 x tulu_math_uc_mix_full, ~54B tokens,
1 epoch). Unlike the small-mix plotter (which reads a fixed final
checkpoint-729), this auto-selects the highest-numbered checkpoint-N so the
chart tracks the live trajectory as the chain advances. Chain links each
relaunch a fresh Trainer (afterany + resume_from_checkpoint), so num_tokens
resets per link -- the same cumulative-token stitching as the small plotter
applies.

Reproduce (on Sunspot):
  source /home/foremans/venvs/sunspot/foremans-aurora_frameworks-2025.3.1/bin/activate
  python3 plot_sft_curves.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from torchtitan.experiments.ezpz.utils.plot_style import apply_style

apply_style()


def _repo_root() -> Path:
    # Walk up to the repo root instead of counting parents: a depth-counted
    # path silently resolves to the wrong directory if this file ever moves,
    # and the plot then renders empty rather than failing.
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "torchtitan").is_dir():
            return parent
    raise RuntimeError("could not locate torchtitan repo root from " + __file__)


# Resolve the repo from this file's location, not a hardcoded machine path:
# authored on Sunspot, but refresh_all.sh also runs this on Aurora where
# /lus/tegu does not exist (the run then silently plotted nothing).
ROOT = _repo_root()
CKPT_DIR = ROOT / "outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144"
OUT_DIR = Path(__file__).resolve().parent.parent / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_CKPT_RE = re.compile(r"checkpoint-(\d+)$")


def latest_trainer_state():
    """Return the trainer_state.json of the highest-numbered checkpoint-N,
    or None if the ckpt dir isn't on this host yet."""
    if not CKPT_DIR.exists():
        return None
    best_n, best = -1, None
    for d in CKPT_DIR.glob("checkpoint-*"):
        m = _CKPT_RE.search(d.name)
        if not m:
            continue
        ts = d / "trainer_state.json"
        if ts.exists() and int(m.group(1)) > best_n:
            best_n, best = int(m.group(1)), ts
    return best


def cumulative_tokens(entries):
    """Stitch TRL's per-link num_tokens counter into a monotonic total.

    num_tokens is a per-Trainer running count; each afterany continuation
    resumes from the last checkpoint and re-inits the counter, so the raw
    series drops back at every chain-link boundary. Carry the prior
    cumulative total forward whenever the counter falls. global_step is
    continuous (restored from checkpoint), so this realigns the token axis
    with the real cumulative data seen.
    """
    raw = np.array([e.get("num_tokens", 0.0) for e in entries], dtype=float)
    cum = np.empty_like(raw)
    offset = prev = 0.0
    for i, v in enumerate(raw):
        if v < prev:
            offset += prev
        cum[i] = offset + v
        prev = v
    return cum


def main():
    ts = latest_trainer_state()
    if ts is None:
        print(f"skip: no checkpoint trainer_state under {CKPT_DIR} -- not on this host")
        return
    state = json.load(open(ts))
    h = [e for e in state["log_history"] if "loss" in e]
    if not h:
        print(f"skip: no loss entries in {ts}")
        return
    steps = np.array([e["step"] for e in h])
    loss = np.array([e["loss"] for e in h])
    grad_norm = np.array([e.get("grad_norm", np.nan) for e in h])
    lr = np.array([e.get("learning_rate", np.nan) for e in h])
    acc = np.array([e.get("mean_token_accuracy", np.nan) for e in h])
    entropy = np.array([e.get("entropy", np.nan) for e in h])
    tokens_b = cumulative_tokens(h) / 1e9

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    _status = "COMPLETE, 1 epoch" if int(steps[-1]) >= 8672 else "in progress"
    fig.suptitle(
        f"AuroraGPT-2B (MDS) x tulu_math_uc_mix_full SFT -- 8N, GBS=6144, 1 epoch "
        f"(step {int(steps[-1])}/8672, {_status})",
        fontsize=13, y=0.995,
    )
    panels = [
        (axes[0, 0], loss,       "loss",                     "loss"),
        (axes[0, 1], grad_norm,  "grad_norm",                "grad_norm"),
        (axes[0, 2], lr * 1e5,   "learning rate",            "lr (x 1e-5)"),
        (axes[1, 0], acc,        "mean token accuracy",      "mean_token_accuracy"),
        (axes[1, 1], entropy,    "entropy",                  "entropy (nats)"),
        (axes[1, 2], tokens_b,   "tokens seen (cumulative)", "tokens (B)"),
    ]
    for ax, y, title, ylabel in panels:
        ax.plot(steps, y, lw=1.5)
        ax.scatter(steps, y, s=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    for ax in axes[1]:
        ax.set_xlabel("global step")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = OUT_DIR / "sft-curves.svg"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (from {ts.parent.name}, {len(steps)} logged steps)")


if __name__ == "__main__":
    main()
