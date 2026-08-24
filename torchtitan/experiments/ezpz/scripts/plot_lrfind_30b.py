#!/usr/bin/env python3
"""Plot the 30B GBS=960 LR-finder sweeps for AdamW / Mano / SophiaG.

The finder writes a per-arm lr_vs_loss.png under outputs/, but outputs/ is
gitignored, so those never reach the repo -- the doc's paths are dead links off
Sunspot. This rebuilds one COMBINED figure from the committed-adjacent CSVs and
writes SVG into the doc's figures/ dir.

One axis for all three arms is also more useful than three separate plots: the
whole point is comparing where each optimizer's usable band sits.

Usage:
    python3 plot_lrfind.py [--out <dir>]
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ambivalent is required -- silent fallback hides style regressions.
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

# Same colors as plot_optcmp.py so the two figures read as one experiment.
ARMS = {
    "adamw": ("AdamW", "#1f77b4", 3.05e-05),
    "mano": ("Mano", "#d62728", 5.61e-05),
    "sophiag": ("SophiaG", "#2ca02c", 3.55e-05),
}


def read_csv(arm: str) -> tuple[list[float], list[float]]:
    hits = glob.glob(
        f"{REPO}/outputs/lrfind-30b-gbs960-{arm}/**/lr_finder_data.csv",
        recursive=True,
    )
    if not hits:
        return [], []
    rows = [
        r for r in csv.DictReader(open(hits[0])) if r["learning_rate"] and r["loss"]
    ]
    return (
        [float(r["learning_rate"]) for r in rows],
        [float(r["loss"]) for r in rows],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=f"{REPO}/torchtitan/experiments/ezpz/docs/experiments/"
        "lr-finder/agpt/figures",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    any_data = False

    for arm, (label, color, suggested) in ARMS.items():
        lrs, losses = read_csv(arm)
        if not lrs:
            print(f"{arm}: NO CSV -- skipped")
            continue
        any_data = True
        lo = min(losses)
        at = lrs[losses.index(lo)]
        ax.plot(lrs, losses, color=color, lw=1.7,
                label=f"{label}  (min {lo:.3f} @ {at:.2e})")
        # Mark the minimum and the Smith-2015 suggestion (blow-up / 10). The
        # suggestion sits well LEFT of the minimum by construction, which is
        # the point -- it buys stability margin, it is not the best-loss LR.
        ax.plot([at], [lo], marker="o", ms=6, color=color, zorder=5)
        ax.axvline(suggested, color=color, ls=":", lw=1.2, alpha=0.7)

    if not any_data:
        print("no finder CSVs found")
        return 1

    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("EMA-smoothed loss")
    ax.set_title("30B LR finder, GBS=960, fineweb-edu  (dotted = suggested LR)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    for ext in ("svg", "png"):
        p = os.path.join(args.out, f"lrfind_30b_gbs960.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
