#!/usr/bin/env python3
"""Combined-overlay eval chart across all 5 production trajectories.

One figure, 4 subplot panels (HellaSwag acc_norm, ARC-Easy acc, ARC-C acc_norm,
Winogrande acc), with 5 curves per panel — one per production trajectory —
plotted against **tokens consumed** so trajectories with different GBS can
be compared directly:

    - 2B-MDS (Megatron-DeepSpeed reference, ~7.77T tokens)
    - 2B 256N async (current production comparator)
    - 2B 512N sync (canonical 2B chain)
    - 20B 256N (per-token comparator)
    - 20B 512N sync (canonical 20B chain)

Writes to docs/evals/figures/all_production_evals.svg (single artifact
referenced from docs/evals/README.md as the landing-page chart).

Run:
    python3 -m torchtitan.experiments.ezpz.eval.plot_evals_combined
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Shared ambivalent + Iosevka styling (identical to the production
# charts). apply_style registers Iosevka when installed and falls back
# to a monospace chain otherwise; ambivalent itself is a hard dep there
# (silent fallback previously gave us months of wrong-style charts).
import matplotlib.pyplot as plt  # noqa: F401,E402

from torchtitan.experiments.ezpz.utils.plot_style import apply_style  # noqa: E402

apply_style()

REPO_ROOT = Path(__file__).resolve().parents[4]
EVALS_DIR = REPO_ROOT / "outputs" / "evals"
OUT_PATH = (
    REPO_ROOT
    / "torchtitan/experiments/ezpz/docs/evals/figures/all_production_evals.svg"
)

# Tokens-per-step for each trajectory (computed from GBS × SEQ_LEN where
# SEQ_LEN=8192 across the board).
# Canonical per-trajectory palette, shared across eval + training charts.
# Single source of truth in utils/palette.py -- this used to be a hand-copied
# block under a "keep these consistent with ..." comment, and it drifted: it
# never even defined COLOR_20B_TT_256N, so the 20B family had no consistent
# color at all.
from torchtitan.experiments.ezpz.utils import palette as _pal  # noqa: E402

COLOR_2B_MDS      = _pal.COLOR_MDS
COLOR_2B_TT_256N  = _pal.COLOR_2B_256N
COLOR_2B_TT_512N  = _pal.COLOR_2B_512N
COLOR_20B_TT_256N = _pal.COLOR_20B_256N
COLOR_20B_TT_512N = _pal.COLOR_20B_512N
COLOR_RANDOM      = _pal.COLOR_RANDOM

TRAJECTORIES: list[dict] = [
    {
        "label": "2B-MDS (SophiaG, n256)",
        "eval_subdir": "agpt-2b-mds",
        "layout": "mds",
        "tokens_per_step": 7_770e9 / 140_000,
        "color": COLOR_2B_MDS,
        "linestyle": "--",
        "marker": "x",
    },
    {
        "label": "2B 256N async (GBS=6144)",
        "eval_subdir": "agpt-2b-v2-256n",
        "layout": "dcp",
        "tokens_per_step": 6144 * 8192,
        "color": COLOR_2B_TT_256N,
        "linestyle": "-",
        "marker": "o",
    },
    {
        "label": "2B 512N sync (GBS=12288)",
        "eval_subdir": "agpt-2b-v2-512n",
        "corrected_subdir": "agpt-2b-v2-512n-ropefix",
        "switch_step": 30401,
        "layout": "dcp",
        "tokens_per_step": 12288 * 8192,
        "color": COLOR_2B_TT_512N,
        "linestyle": "-",
        "marker": "s",
    },
    # 20B-256 was excluded here with the note "a one-off 364-step NODE_FAIL
    # run, 18.3B tokens ... a noisy 3-pt cluster that crowded the legend."
    # That was true when written and has not been true for months: it is a
    # full production chain (cls="live" in trajectories.py, 8,333 steps) with
    # 43 eval points on disk. The exclusion silently kept all of them off the
    # chart. Restored 2026-08-16.
    {
        "label": "20B 256N (GBS=3072)",
        "eval_subdir": "agpt-20b-v2-256n",
        "corrected_subdir": "agpt-20b-v2-256n-ropefix",
        "switch_step": 3101,
        "layout": "dcp",
        "tokens_per_step": 3072 * 8192,
        "color": COLOR_20B_TT_256N,
        "linestyle": "--",
        "marker": "^",
    },
    {
        "label": "20B 512N sync (GBS=12288)",
        "eval_subdir": "agpt-20b-v2-512n",
        "corrected_subdir": "agpt-20b-v2-512n-ropefix",
        "switch_step": 4401,
        "layout": "dcp",
        "tokens_per_step": 12288 * 8192,
        "color": COLOR_20B_TT_512N,
        "linestyle": "-",
        "marker": "D",
    },
]

PANELS: list[tuple[str, str, str]] = [
    ("hellaswag", "acc_norm,none", "HellaSwag (acc_norm)"),
    ("arc_easy", "acc,none", "ARC-Easy (acc)"),
    ("arc_challenge", "acc_norm,none", "ARC-Challenge (acc_norm)"),
    ("winogrande", "acc,none", "Winogrande (acc)"),
    ("piqa", "acc_norm,none", "PIQA (acc_norm)"),
    ("openbookqa", "acc_norm,none", "OpenBookQA (acc_norm)"),
    ("boolq", "acc,none", "BoolQ (acc)"),
]

RANDOM_BASELINE = {
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "winogrande": 0.5,
    "piqa": 0.5,         # binary choice
    "openbookqa": 0.25,  # 4-way MCQ
    "boolq": 0.5,        # yes/no
}


# Every number on this chart must be the SAME shot count, or the curve compares
# two different measurements. That is not hypothetical here: the eval scripts
# write `<task>@<N>shot` keys AND a bare `<task>` alias for whichever group ran
# last, so on steps where a 25-shot ARC-C pass followed the 0-shot pass, the
# bare key IS the 25-shot number. MEASURED on the affected chains: 14 of 36
# post-switch steps on 20b_v2_512 and 15 of 37 on 20b_v2_256 have a bare
# arc_challenge equal to arc_challenge@25shot, with an unused @0shot alias
# sitting right beside it.
#
# The corrected (-ropefix) sweep ran plain 0-shot throughout. So reading the
# bare key would have spliced a 25-shot pre-switch segment onto a 0-shot
# post-switch one -- in the ARC-Challenge panel, the exact panel this work
# exists to fix.
SHOTS = "0shot"


def _read_metric(path: Path, task: str, metric: str) -> float | None:
    """Read one metric, pinned to SHOTS when the file records shot counts.

    Prefers the explicit `<task>@<SHOTS>` key. Falls back to the bare key ONLY
    when no `<task>@` variant exists at all -- i.e. the file predates shot
    tagging and is unambiguous. If shot-tagged keys exist but not the one we
    want, returns None rather than silently substituting a different shot
    count: a missing point is visible, a wrong point is not.
    """
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None

    tagged = f"{task}@{SHOTS}"
    if tagged in d:
        t = d[tagged]
        return t.get(metric) if isinstance(t, dict) else None

    if any(k.startswith(f"{task}@") for k in d):
        # Shot-tagged results exist for this task, but not at SHOTS. The bare
        # key aliases whichever group ran last, so trusting it here is what
        # produces a spliced curve.
        return None

    # UNTAGGED few-shot results. Some batches ran SHOTS_SPEC="5:mmlu;25:arc_challenge"
    # WITHOUT writing @Nshot keys, so the file looks 0-shot by key shape while
    # holding 25-shot numbers. The tell is the company it keeps: that spec always
    # pulls mmlu in alongside, and a plain 0-shot commonsense pass never does.
    # MEASURED on 2b_v2_256: the 8 mmlu-bearing files average 0.3638 on ARC-C
    # against 0.320-0.333 for the 185 without -- a ~4pp step that read as the
    # curve "oscillating" between two levels across the whole plateau.
    if task == "arc_challenge" and any(k.startswith("mmlu") for k in d):
        return None

    t = d.get(task)
    if not isinstance(t, dict):
        return None
    return t.get(metric)


def load_dcp(
    subdir: str,
    task: str,
    metric: str,
    corrected_subdir: str | None = None,
    switch_step: int | None = None,
) -> list[tuple[int, float]]:
    """Load a chain's eval series, splicing in corrected results if it has any.

    A chain that changed RoPE convention mid-flight has TWO valid sources: its
    original dir is correct BELOW ``switch_step`` and wrongly-permuted at or
    above it, while ``corrected_subdir`` holds re-exported results for exactly
    the post-switch steps. Neither alone is right -- the original loses the
    corrected numbers, and the corrected dir alone throws away all pre-switch
    history. So take pre-switch from the original and post-switch from the
    corrected dir.

    Chains that never switched pass ``corrected_subdir=None`` and read one dir,
    unchanged. ``agpt-2b-v2-256n`` is deliberately in that group: it used the
    complex convention end to end, so its results were never corrupted and
    there is nothing to correct.
    """
    def _series(d: str) -> dict[int, float]:
        base = EVALS_DIR / d
        out: dict[int, float] = {}
        for p in sorted(base.glob("step-*/results/results.json")):
            step = int(p.parent.parent.name.split("-")[1])
            val = _read_metric(p, task, metric)
            if val is not None:
                out[step] = val
        return out

    original = _series(subdir)
    if corrected_subdir is None or switch_step is None:
        return sorted(original.items())

    corrected = _series(corrected_subdir)

    # The corrected sweep only re-ran arc_challenge, arc_easy and hellaswag.
    # For any OTHER task the corrected dir is empty, so splicing would silently
    # delete the whole post-switch history -- MEASURED: 36 winogrande points on
    # 20b_v2_512 and 31 on 20b_v2_256. Falling back to the original would be
    # worse (those numbers are the wrongly-permuted ones this exists to remove),
    # so drop them, but say so on stderr. A silent gap in a published chart is
    # exactly the failure this workstream started from.
    post_original = {s for s in original if s >= switch_step}
    post_corrected = {s for s in corrected if s >= switch_step}
    if post_original and not post_corrected:
        print(
            f"  NOTE [{subdir}/{task}]: {len(post_original)} post-switch point(s) "
            f"dropped -- the corrected sweep did not cover this task, and the "
            f"original values are wrongly permuted. Re-run the sweep with "
            f"{task} to restore them.",
        )

    # Pre-switch from the original; at/after the switch, corrected only. A
    # post-switch step with no corrected result is DROPPED rather than back-
    # filled from the original -- showing a known-wrong point would defeat the
    # entire exercise, and a gap is at least visible.
    merged = {s: v for s, v in original.items() if s < switch_step}
    merged.update({s: v for s, v in corrected.items() if s >= switch_step})
    return sorted(merged.items())


def load_mds(subdir: str, task: str, metric: str) -> list[tuple[int, float]]:
    """MDS layout: 3 stage dirs containing step-{N}/results/results.json.

    The three stages all symlink to the same physical ckpt dir for the
    SophiaG sweep — same step appears up to 3 times. Average across
    replicates (XPU lm-eval isn't bit-deterministic).
    """
    base = EVALS_DIR / subdir
    by_step: dict[int, list[float]] = {}
    for p in sorted(base.glob("*/step-*/results/results.json")):
        step = int(p.parent.parent.name.split("-")[1])
        val = _read_metric(p, task, metric)
        if val is not None:
            by_step.setdefault(step, []).append(val)
    return sorted((s, sum(v) / len(v)) for s, v in by_step.items())


def _assert_no_missing_live_chains() -> None:
    """Fail if a live chain with an eval_subdir is absent from TRAJECTORIES.

    Same failure mode as the training chart: this module keeps its own display
    list, so a chain can be fully registered in trajectories.py, have results
    sitting on disk, and simply never appear. 20B-256 was excluded here as "a
    noisy 3-pt cluster" and stayed excluded after it grew into a full
    production chain with 43 eval points -- an exclusion that was correct when
    written and silently wrong for months afterward. Reconcile the two lists
    rather than trusting them to stay in sync.
    """
    from torchtitan.experiments.ezpz.utils.trajectories import TRAJECTORIES as _ALL

    want = {
        t["eval_subdir"]
        for t in _ALL
        if t.get("cls") == "live" and t.get("eval_subdir")
    }
    have = {t["eval_subdir"] for t in TRAJECTORIES}
    missing = sorted(want - have)
    if missing:
        raise SystemExit(
            "Live chains with eval results missing from this chart:\n"
            + "".join(f"  - {k}\n" for k in missing)
            + "\nAdd an entry to TRAJECTORIES here, or clear that chain's\n"
            "eval_subdir in utils/trajectories.py if it is intentionally\n"
            "not plotted. Do not leave it registered-but-undrawn."
        )


def main() -> None:
    _assert_no_missing_live_chains()
    # Grid: 2 columns, enough rows to fit every panel. With 7 panels
    # that's a 4x2 (one empty cell, hidden below).
    ncols = 2
    nrows = (len(PANELS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5.5 * nrows))
    axes = axes.flatten()

    # Guard against the silent-empty-chart failure. EVALS_DIR holds results
    # that live on Aurora, NOT in the repo -- so running this on a laptop
    # (or anywhere the evals are not mounted) used to emit a clean, plausible
    # chart containing only axes and the gray random-chance lines. One such
    # figure was committed and pushed on 2026-08-16 before anyone noticed.
    # Count what actually gets drawn and fail loudly if the answer is nothing.
    n_series = 0

    for ax, (task, metric, title) in zip(axes, PANELS):
        for traj in TRAJECTORIES:
            if traj["layout"] == "mds":
                pts = load_mds(traj["eval_subdir"], task, metric)
            else:
                pts = load_dcp(
                    traj["eval_subdir"],
                    task,
                    metric,
                    traj.get("corrected_subdir"),
                    traj.get("switch_step"),
                )
            if not pts:
                print(f"  [{title}] no data for {traj['label']}")
                continue
            n_series += 1
            steps, accs = zip(*pts)
            tokens_b = [s * traj["tokens_per_step"] / 1e9 for s in steps]
            ax.plot(
                tokens_b,
                accs,
                marker=traj["marker"],
                color=traj["color"],
                linestyle=traj["linestyle"],
                label=traj["label"],
                markersize=5,
                linewidth=1.8,
                alpha=0.9,
            )
            print(
                f"  [{title}] {traj['label']}: "
                f"{len(pts)} pts, tokens {tokens_b[0]:.1f}B → {tokens_b[-1]:.1f}B, "
                f"acc {accs[0]:.3f} → {accs[-1]:.3f}"
            )

        ax.axhline(
            y=RANDOM_BASELINE[task],
            color=COLOR_RANDOM,
            linestyle=":",
            alpha=0.6,
            linewidth=1,
            label="random",
        )
        ax.set_xlabel("Tokens consumed (B)")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Hide any unused cells when len(PANELS) doesn't fill the grid evenly.
    for ax in axes[len(PANELS):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "AuroraGPT v2 — Eval Benchmarks vs Training Tokens (all production trajectories)",
        y=1.06,
        fontsize=14,
    )
    plt.tight_layout()

    if n_series == 0:
        plt.close()
        raise SystemExit(
            "REFUSING to write an empty chart: not one trajectory yielded a\n"
            f"single point. Looked under: {EVALS_DIR}\n"
            "\n"
            "Eval results live on Aurora, not in the repo. Regenerate there:\n"
            "  ssh aurora; cd .../torchtitan-ezpz\n"
            "  ./.venv/bin/python3 torchtitan/experiments/ezpz/eval/"
            "plot_evals_combined.py\n"
            "\n"
            "The previous behaviour was to emit axes plus the gray\n"
            "random-chance lines and exit 0, which is indistinguishable from a\n"
            "real chart at a glance -- that is how an empty figure reached the\n"
            "branch on 2026-08-16. Do not 'fix' this by removing the check."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"\nSaved: {OUT_PATH}  ({n_series} series drawn)")


if __name__ == "__main__":
    main()
