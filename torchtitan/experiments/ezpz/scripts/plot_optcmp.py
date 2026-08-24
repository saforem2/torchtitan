#!/usr/bin/env python3
"""Plot the fixed-batch optimizer comparison (AdamW / Mano / SophiaG).

Reads loss + grad_norm straight from the arms' train.log files rather than
W&B, so the figure can be regenerated from a checkout with no network and no
run-id bookkeeping. Emits SVG under the doc's figures/ dir, matching the
convention used by docs/experiments/lr-finder/moe/*/figures/.

Usage:
    python3 plot_optcmp.py [--out <dir>]
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ambivalent is required -- silent fallback hides style regressions.
# Install with: uv pip install --no-deps "git+https://github.com/saforem2/ambivalent"
import ambivalent  # noqa: F401

plt.style.use(ambivalent.STYLES["ambivalent"])

# ambivalent sets IBM Plex Sans; these figures want Iosevka to match the rest
# of the docs. Prepend rather than replace so the style's own fallback chain
# still applies on a machine without Iosevka installed.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Iosevka"] + list(
    plt.rcParams["font.sans-serif"]
)
# Math text otherwise renders in DejaVu and visibly disagrees with the labels.
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Iosevka"
plt.rcParams["mathtext.it"] = "Iosevka:italic"
plt.rcParams["mathtext.bf"] = "Iosevka:bold"

REPO = "/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan"
SEQ_LEN = 4096
GBS = 960  # sequences per train step; tokens/step = GBS * SEQ_LEN

# Colors held fixed per optimizer so every figure in this experiment agrees.
ARMS = {
    "adamw": ("AdamW  (lr 3.05e-05)", "#1f77b4"),
    "mano": ("Mano  (lr 5.61e-05)", "#d62728"),
    "sophiag": ("SophiaG  (lr 3.55e-05)", "#2ca02c"),
}

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(
    r"step:\s+(\d+)\s+loss:\s+([0-9.]+)\s+grad_norm:\s+([0-9.]+)"
)


def read_arm(arm: str) -> tuple[list[int], list[float], list[float]]:
    """Collect (step, loss, grad_norm) across every chain link for one arm.

    Multiple job dirs exist per arm because the run is chained; later links
    resume from a checkpoint and REPEAT steps the earlier link already logged.
    Keyed by step with last-write-wins so a resumed step reflects the run that
    actually continued, not the one that was killed.
    """
    by_step: dict[int, tuple[float, float]] = {}
    logs = sorted(glob.glob(f"{REPO}/outputs/logs/optcmp-{arm}/*/train.log"))
    for path in logs:
        with open(path, errors="ignore") as fh:
            for line in fh:
                m = STEP_RE.search(ANSI.sub("", line))
                if m:
                    by_step[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    steps = sorted(by_step)
    return steps, [by_step[s][0] for s in steps], [by_step[s][1] for s in steps]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=f"{REPO}/torchtitan/experiments/ezpz/docs/experiments/"
        "optimizer-comparison/figures",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = {a: read_arm(a) for a in ARMS}
    if not any(d[0] for d in data.values()):
        print("no step data found -- are the logs present?")
        return 1

    fig, (ax_loss, ax_gn) = plt.subplots(1, 2, figsize=(13, 4.8))

    for arm, (label, color) in ARMS.items():
        steps, loss, gn = data[arm]
        if not steps:
            continue
        # Tokens, not steps, is the only fair x-axis across optimizers -- all
        # three share a batch here, but plotting tokens keeps the figure
        # comparable to runs that do not.
        tokens = [s * GBS * SEQ_LEN / 1e9 for s in steps]
        ax_loss.plot(tokens, loss, label=label, color=color, lw=1.6)
        ax_gn.plot(tokens, gn, label=label, color=color, lw=1.0, alpha=0.75)
        print(f"{arm:9} {len(steps):5} pts  last: step {steps[-1]} loss {loss[-1]:.4f}")

    ax_loss.set_xlabel("tokens (B)")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("30B optimizer comparison, GBS=960, constant LR")
    ax_loss.legend(frameon=False, fontsize=9)

    ax_gn.set_xlabel("tokens (B)")
    ax_gn.set_ylabel("grad_norm")
    ax_gn.set_title("gradient norm (pre-clip)")
    ax_gn.set_yscale("log")

    fig.tight_layout()
    for ext in ("svg", "png"):
        p = os.path.join(args.out, f"optcmp_30b_gbs960.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"wrote {p}")

    # Second figure: the crossover, which is the actual finding and is
    # invisible on the full-range plot once the curves converge visually.
    fig2, ax = plt.subplots(figsize=(7, 4.6))
    a_steps, a_loss, _ = data["adamw"]
    m_steps, m_loss, _ = data["mano"]
    common = sorted(set(a_steps) & set(m_steps))
    if common:
        ai = {s: v for s, v in zip(a_steps, a_loss)}
        mi = {s: v for s, v in zip(m_steps, m_loss)}
        tokens = [s * GBS * SEQ_LEN / 1e9 for s in common]
        delta = [mi[s] - ai[s] for s in common]
        ax.axhline(0.0, color="#888", lw=1.0, ls="--")
        ax.plot(tokens, delta, color="#d62728", lw=1.8)
        ax.fill_between(tokens, delta, 0, where=[d < 0 for d in delta],
                        color="#d62728", alpha=0.15)
        ax.set_xlabel("tokens (B)")
        ax.set_ylabel("Mano loss  -  AdamW loss  (nats)")
        ax.set_title("Mano overtakes AdamW  (below zero = Mano ahead)")
        fig2.tight_layout()
        for ext in ("svg", "png"):
            p = os.path.join(args.out, f"optcmp_30b_crossover.{ext}")
            fig2.savefig(p, bbox_inches="tight", dpi=150)
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
