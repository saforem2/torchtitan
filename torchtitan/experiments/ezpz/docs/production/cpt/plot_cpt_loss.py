#!/usr/bin/env python3
"""Plot 2B CPT (continued-pretraining) training-loss curves per data mix.

Reads per-run TSV files (columns: step<TAB>loss) produced from the CPT job
logs and renders a comparison SVG: loss vs CPT step, one line per mix, with
the olmo-100 base plateau drawn as a reference band.

Usage:
  plot_cpt_loss.py --data-dir <dir with cpt-<mix>.tsv> --out figures/cpt_loss.svg

The mixes + final validation losses are annotated from the run summary.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    # ezpz docs plots expect the ambivalent style; fall back silently if the
    # optional IPython/ambivalent stack is missing (see feedback note).
    import ambivalent

    plt.style.use(ambivalent.STYLES["ambivalent"])
except Exception:
    pass

# (label, tsv basename, final validation loss, color)
RUNS = [
    ("olmo25-dolmino75", "cpt-olmo25-dolmino75", None, "#2a9d8f"),
    ("olmo50-dolmino50", "cpt-olmo50-dolmino50", 2.601, "#e76f51"),
    ("dolmino-100", "cpt-dolmino100", 2.492, "#264653"),
]

# olmo-100 base plateau (the control): the completed 2B 256N base saturated
# here on its own distribution. Val-loss reference band.
OLMO100_PLATEAU = 2.80


def read_tsv(path):
    steps, losses = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                steps.append(int(float(parts[0])))
                losses.append(float(parts[1]))
            except ValueError:
                continue
    return steps, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "figures"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figures", "cpt_loss.svg"))
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # olmo-100 plateau reference line (the level CPT has to beat)
    ax.axhline(
        OLMO100_PLATEAU,
        color="gray",
        ls="--",
        lw=1.3,
        label=f"olmo-100 base plateau (~{OLMO100_PLATEAU})",
        zorder=1,
    )

    for label, base, final_val, color in RUNS:
        path = os.path.join(args.data_dir, f"{base}.tsv")
        if not os.path.exists(path):
            print(f"  [skip] {label}: {path} not found")
            continue
        steps, losses = read_tsv(path)
        if not steps:
            print(f"  [skip] {label}: no points in {path}")
            continue
        lab = label
        if final_val is not None:
            lab = f"{label} (val {final_val:.3f})"
        ax.plot(steps, losses, color=color, lw=1.4, label=lab, zorder=3)
        print(f"  {label}: {len(steps)} pts, loss {losses[0]:.3f} -> {losses[-1]:.3f}")

    ax.set_xlabel("CPT step (fork from 2B base step-92,859; GBS=6144)")
    ax.set_ylabel("training loss")
    ax.set_title("2B continued-pretraining: olmo x dolmino mixing-ratio sweep")
    ax.set_ylim(2.2, 8.0)
    ax.legend(loc="upper right", framealpha=0.9)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
