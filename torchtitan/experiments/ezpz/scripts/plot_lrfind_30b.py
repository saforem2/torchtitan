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
# fontset="custom" makes matplotlib resolve EVERY mathtext family, including
# mathtext.cal, which defaults to "cursive" -- not installed here, so it warns
# and falls back to DejaVu on every render. Point it at Iosevka too.
plt.rcParams["mathtext.cal"] = "Iosevka:italic"
plt.rcParams["mathtext.sf"] = "Iosevka"
plt.rcParams["mathtext.tt"] = "Iosevka"

REPO = "/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan"

# Same colors as plot_optcmp.py so the two figures read as one experiment.
# Third element is the Smith-2015 suggestion (blow-up/10), drawn as a dotted
# vertical. Use None for an arm whose sweep has not been read out yet: the
# curve still plots, only the marker line is skipped. Do NOT guess a value --
# a wrong dotted line is worse than a missing one.
#
# An arm missing from this dict is missing from the FIGURE, silently, at
# exit 0. Adding a config + a PBS arm is not enough; register it here too.
ARMS = {
    "adamw": ("AdamW", "#1f77b4", 3.05e-05),
    "mano": ("Mano", "#d62728", 5.61e-05),
    "sophiag": ("SophiaG", "#2ca02c", 3.55e-05),
    # Muon, job 12474326 (2026-08-30). Suggestion pending; fill it in from the
    # job summary. NOTE the number that goes here is the BASE lr the finder
    # reports -- Muon rescales by a uniform 15.677x on its 21.5% of params
    # (optimizer/muon.py:130-139), so this dotted line is NOT comparable
    # like-for-like with the other three. See
    # docs/experiments/lr-finder/agpt/2026-08-30-30b-gbs960-muon.md.
    "muon": ("Muon", "#9467bd", 5.68e-04),
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

    # Two panels: the full sweep proves each arm actually blew up (a sweep that
    # never diverges yields no suggestion at all), but the blow-up runs to ~84
    # while every minimum sits in 8.84-9.34 -- a 0.5-nat spread squashed flat on
    # an 84-unit axis. The zoom is where the result is legible.
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(14, 5.2))
    any_data = False

    for arm, (label, color, suggested) in ARMS.items():
        lrs, losses = read_csv(arm)
        if not lrs:
            print(f"{arm}: NO CSV -- skipped")
            continue
        any_data = True
        lo = min(losses)
        at = lrs[losses.index(lo)]
        for a_ in (ax, axz):
            a_.plot(lrs, losses, color=color, lw=1.7,
                    label=f"{label}  (min {lo:.3f} @ {at:.2e})")
            # Mark the minimum and the Smith-2015 suggestion (blow-up / 10).
            # The suggestion sits well LEFT of the minimum by construction --
            # it buys stability margin, it is not the best-loss LR.
            a_.plot([at], [lo], marker="o", ms=6, color=color, zorder=5)
            if suggested is not None:
                a_.axvline(suggested, color=color, ls=":", lw=1.2, alpha=0.7)

    if not any_data:
        print("no finder CSVs found")
        return 1

    # Derive the zoom window from the observed minima rather than hardcoding:
    # a rerun that shifts the curves should move the window with them.
    mins = []
    for arm in ARMS:
        _, ls_ = read_csv(arm)
        if ls_:
            mins.append(min(ls_))
    y_lo = min(mins) - 0.15
    y_hi = max(mins) + 1.20
    print(f"zoom window: y in [{y_lo:.2f}, {y_hi:.2f}] from minima {[round(m,3) for m in mins]}")

    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("EMA-smoothed loss")
    ax.set_title("full sweep  (dotted = suggested LR)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # Zoom: y clipped just around the minima, x to where any arm is still in
    # the band. Limits derive from the DATA, not hardcoded, so the panel stays
    # honest if a rerun shifts the curves.
    axz.set_xscale("log")
    # Derive the x-window from the data. It was hardcoded 1e-5..3e-3, which fits
    # arms whose suggestions cluster at 3-6e-05; an arm reporting a much smaller
    # BASE lr (Muon, whose 15.677x rescale shifts the whole curve left) would
    # fall off the left edge and vanish while the figure still saved at exit 0.
    x_at_min = []
    for arm_ in ARMS:
        lr_, ls_ = read_csv(arm_)
        if ls_:
            x_at_min.append(lr_[ls_.index(min(ls_))])
    x_lo = min(min(x_at_min) / 10.0, 1e-5) if x_at_min else 1e-5
    x_hi = max(max(x_at_min) * 6.0, 3e-3) if x_at_min else 3e-3
    axz.set_xlim(x_lo, x_hi)
    axz.set_ylim(y_lo, y_hi)
    axz.set_xlabel("learning rate")
    axz.set_ylabel("EMA-smoothed loss")
    axz.set_title(f"zoom: minima ({y_lo:.2f}-{y_hi:.2f} nats)")
    axz.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("30B LR finder, GBS=960, fineweb-edu", y=1.02,
                 fontfamily="sans-serif")

    fig.tight_layout()
    for ext in ("svg", "png"):
        p = os.path.join(args.out, f"lrfind_30b_gbs960.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
