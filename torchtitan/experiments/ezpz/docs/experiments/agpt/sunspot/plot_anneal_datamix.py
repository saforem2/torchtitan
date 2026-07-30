#!/usr/bin/env python3
"""Charts for the 2B MDS anneal-schedule + data-mix experiment.

Renders two SVGs into ./figures/ (ambivalent style, reads well on light+dark):
  1. anneal_flat_vs_wsd.svg  -- grouped bars: flat vs WSD-decay-to-0 held-out
     FineMath NLL on both bases (flat wins on both -> schedule is not the lever).
  2. datamix_tradeoff.svg    -- FineMath (math) vs wikitext (general) held-out
     NLL scatter across mixes (owm-100 -> 75/25 -> 50/50 -> edu-100), showing the
     math-heavy sweet spot.

All numbers are the FINAL held-out val-loss (mean NLL, lower=better) from the
eval jobs (see the companion .md). Hardcoded here so the figure is reproducible
and self-documenting. 50/50 is filled in once job 12472037 evals; until then it
is plotted from the *predicted* trend and flagged in the label.

Run:  python3 torchtitan/experiments/ezpz/docs/experiments/agpt/sunspot/plot_anneal_datamix.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# ambivalent is required -- silent fallback hides style regressions.
# uv pip install --no-deps "git+https://github.com/saforem2/ambivalent"
import ambivalent  # noqa: E402

plt.style.use(ambivalent.STYLES["ambivalent"])

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# --- Experiment 1: anneal schedule A/B (held-out FineMath NLL) ---
# base, flat-1600, wsd-1600
ANNEAL = {
    "MDS (val 2.05)": {"base": 1.8476, "flat": 1.8039, "wsd": 1.8183},
    "olmo (val 2.65)": {"base": 1.9033, "flat": 1.8807, "wsd": 1.9003},
}

# --- Experiment 2: data mix (held-out FineMath vs wikitext NLL) ---
# arm: (finemath, wikitext, is_measured)
DATAMIX = {
    "owm-100\n(control)": (1.8039, 2.6828, True),
    "owm75/edu25": (1.8089, 2.6597, True),
    "owm50/edu50": (1.8155, 2.6519, True),  # measured (job 12472203, 2026-07-30)
    "edu-100": (2.1122, 2.6585, True),
}


def plot_anneal() -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bases = list(ANNEAL)
    x = range(len(bases))
    w = 0.38
    flat = [ANNEAL[b]["flat"] for b in bases]
    wsd = [ANNEAL[b]["wsd"] for b in bases]
    ax.bar([i - w / 2 for i in x], flat, w, label="flat (constant LR)", color="C0")
    ax.bar([i + w / 2 for i in x], wsd, w, label="WSD decay-to-0", color="C1")
    # base reference lines
    for i, b in enumerate(bases):
        ax.hlines(
            ANNEAL[b]["base"], i - 0.5, i + 0.5,
            color="0.5", linestyle="--", linewidth=1,
        )
    ax.text(
        0.98, 0.02, "dashed = un-annealed base", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8, color="0.5",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(bases)
    ax.set_ylabel("held-out FineMath NLL (lower = better)")
    ax.set_title("Anneal schedule A/B: flat beats WSD on both bases\n"
                 "(the LR schedule is not the lever at 10B tokens)")
    ax.set_ylim(1.78, 1.92)
    ax.legend(frameon=False)
    out = FIGDIR / "anneal_flat_vs_wsd.svg"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_datamix() -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (fm, wt, measured) in DATAMIX.items():
        ax.scatter(fm, wt, s=90, zorder=3,
                   edgecolor="black" if measured else "0.6",
                   facecolor="C0" if measured else "none",
                   linewidth=1.2)
        label = name if measured else name + "\n(predicted)"
        ax.annotate(label, (fm, wt), textcoords="offset points",
                    xytext=(8, 6), fontsize=8)
    # connect the mix ladder in order (owm -> 75/25 -> 50/50 -> edu)
    order = list(DATAMIX)
    ax.plot([DATAMIX[k][0] for k in order], [DATAMIX[k][1] for k in order],
            color="0.6", linewidth=1, linestyle=":", zorder=1)
    ax.set_xlabel("held-out FineMath NLL  (math generalization, lower = better)")
    ax.set_ylabel("held-out wikitext NLL  (general, lower = better)")
    ax.set_title("Data mix trade-off: 75/25 math/edu keeps math intact\n"
                 "while capturing ~all of edu's general-domain gain")
    # annotate the sweet-spot region
    ax.annotate(
        "sweet spot:\nmath intact + general gained",
        xy=(1.8089, 2.6597), xytext=(1.86, 2.665), fontsize=8, color="C0",
        arrowprops=dict(arrowstyle="->", color="C0", lw=1),
    )
    out = FIGDIR / "datamix_tradeoff.svg"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    for f in (plot_anneal(), plot_datamix()):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
