"""Charts for the B-series CoT cold-start SFT result.

Two figures, each making one point that the results table alone does not:

  cot_ladder.svg      -- accuracy vs the two failure modes, side by side. The
                         whole finding is that accuracy tracks the two-stage
                         STRUCTURE, not the mix / filtering / seq-len knobs, and
                         that the run-on symptom (gen_len, unclosed) is a
                         SEPARATE axis that B4b fixed without moving accuracy.
  accuracy_vs_genlen.svg -- the same four models in (gen_len, accuracy) space,
                         showing B2 alone in the short-and-accurate corner while
                         every single-stage rebuild sits long-and-inaccurate.

Numbers are the 200-problem GSM8K CoT eval (fp32 vLLM, identical harness) from
README.md in this directory. Run from repo root with the .venv active:
    python3 torchtitan/experiments/ezpz/docs/live/chains/sft/agpt/2b-mds/b4-finish-and-reweight/plot_b4.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# ambivalent is required -- a silent fallback would hide style regressions.
# uv pip install --no-deps "git+https://github.com/saforem2/ambivalent"
import ambivalent  # noqa: E402

plt.style.use(ambivalent.STYLES["ambivalent"])

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# model -> (cot_accuracy, format_hit_rate, mean_gen_len, n_unclosed, is_two_stage)
RESULTS = {
    "B2\ntwo-stage": (0.205, 0.985, 282, 0, True),
    "B3\nsingle, broad mix": (0.05, 0.86, 601, 26, False),
    "B4a\nfinish on B3": (0.02, 0.26, 1378, 133, False),
    "B4b\nreweight+filter": (0.065, 0.86, 611, 27, False),
}

WIN = "#2ca02c"     # the two-stage winner
LOSE = "#d62728"    # single-stage rebuilds
NEUTRAL = "#7f7f7f"


def _colors():
    return [WIN if v[4] else LOSE for v in RESULTS.values()]


def plot_ladder(out: Path) -> None:
    """Accuracy next to the two run-on metrics: they move independently."""
    names = list(RESULTS)
    acc = [v[0] for v in RESULTS.values()]
    fmt = [v[1] for v in RESULTS.values()]
    glen = [v[2] for v in RESULTS.values()]
    cols = _colors()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].bar(names, acc, color=cols)
    axes[0].axhline(0.205, ls="--", lw=1, color=NEUTRAL)
    axes[0].text(2.6, 0.212, "B2 = 0.205", fontsize=8, color=NEUTRAL, ha="right")
    axes[0].set_ylabel("GSM8K-CoT accuracy  (higher = better)")
    axes[0].set_title("Accuracy: only the two-stage recipe wins")
    axes[0].set_ylim(0, 0.24)
    for i, a in enumerate(acc):
        axes[0].text(i, a + 0.006, f"{a:.3f}", ha="center", fontsize=9)

    axes[1].bar(names, fmt, color=cols)
    axes[1].set_ylabel("format hit rate  (higher = better)")
    axes[1].set_title("Format: B4a's envelope collapsed")
    axes[1].set_ylim(0, 1.05)
    for i, f in enumerate(fmt):
        axes[1].text(i, f + 0.02, f"{f:.2f}", ha="center", fontsize=9)

    axes[2].bar(names, glen, color=cols)
    axes[2].axhline(282, ls="--", lw=1, color=NEUTRAL)
    axes[2].set_ylabel("mean generation length  (lower = tighter)")
    axes[2].set_title("Verbosity: B4b fixed this, accuracy did not follow")
    for i, g in enumerate(glen):
        axes[2].text(i, g + 30, f"{g}", ha="center", fontsize=9)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        "B-series CoT cold-start SFT (agpt-2b, MDS stage-3 base) -- "
        "200-problem GSM8K CoT, fp32 vLLM",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def plot_acc_vs_genlen(out: Path) -> None:
    """One scatter: short+accurate is a corner only B2 reaches."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for (name, (acc, _fmt, glen, unclosed, two)) in RESULTS.items():
        ax.scatter(
            glen,
            acc,
            s=90 + 6 * unclosed,          # marker grows with unclosed generations
            color=WIN if two else LOSE,
            zorder=3,
            edgecolor="white",
            linewidth=0.8,
        )
        # B4a sits at the right edge -- label it leftward so it stays in frame.
        right_edge = glen > 1000
        ax.annotate(
            name.replace("\n", " "),
            (glen, acc),
            textcoords="offset points",
            xytext=(-14, 14) if right_edge else (10, 6),
            ha="right" if right_edge else "left",
            fontsize=9,
        )
    ax.axhline(0.205, ls="--", lw=1, color=NEUTRAL)
    ax.set_xlim(100, 1750)         # room for the B4a annotation at gen_len 1378
    ax.set_ylim(0, 0.235)
    ax.text(1730, 0.209, "B2 target", fontsize=8, color=NEUTRAL, ha="right")
    ax.set_xlabel("mean generation length  (lower = tighter)")
    ax.set_ylabel("GSM8K-CoT accuracy")
    ax.set_title(
        "Short AND accurate is a corner only the two-stage recipe reaches\n"
        "(marker size ~ unclosed generations; green = two-stage, red = single-stage)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_ladder(FIGDIR / "cot_ladder.svg")
    plot_acc_vs_genlen(FIGDIR / "accuracy_vs_genlen.svg")
