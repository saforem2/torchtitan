#!/usr/bin/env python3
"""Plot 2B CPT downstream-benchmark curves per data mix vs CPT step.

The decisive CPT screen: does continued-pretraining on olmo x dolmino lift the
plateaued 2B base's benchmarks, or hurt them? Reads per-mix TSVs
(step, hellaswag, arc_easy, arc_challenge, winogrande, piqa) and overlays each
mix against the olmo-100 base plateau (dashed reference), one subplot per task.

Usage: plot_cpt_eval.py --data-dir <dir with cpt-eval-<mix>.tsv> --out figures/cpt_eval.svg
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import ambivalent

    plt.style.use(ambivalent.STYLES["ambivalent"])
except Exception:
    pass

# (mix label, tsv basename, color)
MIXES = [
    ("dolmino-100", "cpt-eval-dolmino100", "#264653"),
    ("olmo50-dolmino50", "cpt-eval-olmo50-dolmino50", "#e76f51"),
]

# olmo-100 base plateau (the completed step-92,859 base), the level CPT must beat.
# Columns: step, hellaswag(acc_norm), arc_easy(acc), arc_challenge(acc_norm),
#          winogrande(acc), piqa(acc_norm)
TASKS = [
    ("HellaSwag (acc_norm)", 1, 0.560),
    ("ARC-Easy (acc)", 2, 0.651),
    ("ARC-Challenge (acc_norm)", 3, 0.336),
    ("Winogrande (acc)", 4, 0.543),
    ("PIQA (acc_norm)", 5, 0.733),
]


def read_tsv(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                rows.append([float(x) for x in parts[:6]])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "figures"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figures", "cpt_eval.svg"))
    args = ap.parse_args()

    data = {}
    for label, base, _ in MIXES:
        p = os.path.join(args.data_dir, f"{base}.tsv")
        if os.path.exists(p):
            data[label] = read_tsv(p)

    fig, axes = plt.subplots(1, len(TASKS), figsize=(4 * len(TASKS), 4.2), sharex=True)
    for ax, (task_name, col, base_val) in zip(axes, TASKS):
        ax.axhline(base_val, color="gray", ls="--", lw=1.3,
                   label=f"olmo-100 base ({base_val})", zorder=1)
        for label, base, color in MIXES:
            rows = data.get(label)
            if not rows:
                continue
            steps = [r[0] for r in rows]
            ys = [r[col] for r in rows]
            ax.plot(steps, ys, "-o", color=color, lw=1.6, ms=4, label=label, zorder=3)
        ax.set_title(task_name, fontsize=10)
        ax.set_xlabel("CPT step")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="best", fontsize=7, framealpha=0.9)
    fig.suptitle(
        "2B CPT downstream eval -- olmo x dolmino (LR=2.28e-5 re-warm pilot): "
        "CPT DEGRADES benchmarks vs the base plateau",
        fontsize=12, fontweight="bold",
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
