#!/usr/bin/env python3
"""Coordinate-check slopes: standard parametrization vs muP.

The number quoted in the sync notes ("max |slope| 0.020 against a 0.05
tolerance") is the whole muP result compressed to one scalar, and it hides
what makes it convincing: the SAME modules under standard parametrization run
to 1.42. A 140x difference is worth showing rather than asserting.

Data source: docs/experiments/mup/README.md section 5.6, the 4-widths x
8-steps CPU coordinate check. Slope is d log2(l1) / d log2(width); muP
promises 0, and the harness tolerance is 0.05.

Numbers are transcribed from that table rather than recomputed -- this is a
presentation script, not a measurement. If the table changes, change this too.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    import ambivalent  # noqa: F401
    plt.style.use(ambivalent.STYLES["ambivalent"])
    _STYLE = "ambivalent"
except Exception:
    _STYLE = "matplotlib default (ambivalent unavailable)"

# (module, SP slope, muP with d^-1 readout, muP with fan_in^-1/2 readout).
# None = not reported in the source table.
#
# The middle column is the FIRST muP attempt, reading Table 3 literally as
# std = d^-1. It flattened the hidden layers but left lm_head at -0.130.
# The right column keeps agpt's existing fan_in^-1/2 and lets the eta/m LR
# group supply the width scaling, which is abc-equivalent to Table 8 with the
# 1/m folded into the update. Only the right column is what ships.
ROWS = [
    ("layers.5.attention", 1.42, -0.010, -0.010),
    ("layers.4.attention", 1.32, 0.020, 0.020),
    ("layers.3.attention", 1.13, None, None),
    ("layers.5.feed_forward", None, -0.011, -0.011),
    ("lm_head", 0.22, -0.130, 0.001),
    ("tok_embeddings\n[control]", 0.001, 0.001, 0.001),
]
TOLERANCE = 0.05

labels = [r[0] for r in ROWS]
sp = [r[1] for r in ROWS]
mup = [r[2] for r in ROWS]
fixed = [r[3] for r in ROWS]
y = np.arange(len(ROWS))

fig, (ax_sp, ax_mup, ax_fix) = plt.subplots(
    1, 3, figsize=(14.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1, 1]}
)

# --- left: standard parametrization, linear scale, shows the blow-up --------
vals_sp = [v if v is not None else np.nan for v in sp]
ax_sp.barh(y, vals_sp, color="#c44e52", alpha=0.85)
ax_sp.axvline(0.0, color="0.4", lw=1)
ax_sp.axvspan(-TOLERANCE, TOLERANCE, color="#55a868", alpha=0.18, zorder=0)
ax_sp.set_yticks(y)
ax_sp.set_yticklabels(labels, fontsize=9)
ax_sp.invert_yaxis()
ax_sp.set_xlabel("slope  d log2(l1) / d log2(width)")
ax_sp.set_title("Standard parametrization", fontsize=11)
for i, v in enumerate(sp):
    if v is None:
        ax_sp.text(0.02, i, "not reported", va="center", fontsize=8, color="0.5")
    else:
        ax_sp.text(v + 0.03, i, f"{v:.3f}", va="center", fontsize=8.5)
ax_sp.set_xlim(-0.15, 1.75)

# --- right: muP, same rows, axis zoomed to the tolerance band ---------------
vals_mup = [v if v is not None else np.nan for v in mup]
colors = [
    "#55a868" if (v is not None and abs(v) <= TOLERANCE) else "#dd8452"
    for v in mup
]
ax_mup.barh(y, vals_mup, color=colors, alpha=0.9)
ax_mup.axvline(0.0, color="0.4", lw=1)
ax_mup.axvspan(-TOLERANCE, TOLERANCE, color="#55a868", alpha=0.18, zorder=0)
ax_mup.set_yticks(y)
ax_mup.set_yticklabels([])
ax_mup.invert_yaxis()
ax_mup.set_xlabel("slope (note the axis: 12x narrower)")
ax_mup.set_title("muP, readout std = $d^{-1}$\n(literal Table 3)", fontsize=10.5)
for i, v in enumerate(mup):
    if v is None:
        ax_mup.text(0.004, i, "flat", va="center", fontsize=8, color="0.5")
    else:
        off = 0.006 if v >= 0 else -0.006
        ha = "left" if v >= 0 else "right"
        ax_mup.text(v + off, i, f"{v:+.3f}", va="center", ha=ha, fontsize=8.5)
ax_mup.set_xlim(-0.16, 0.10)
ax_mup.text(
    TOLERANCE, len(ROWS) - 0.35, f" tolerance +/-{TOLERANCE}",
    fontsize=8, color="#3d7a4e", va="center",
)

# --- right: after the readout fix, same axis as the middle panel ----------
vals_fix = [v if v is not None else np.nan for v in fixed]
colors_fix = [
    "#55a868" if (v is not None and abs(v) <= TOLERANCE) else "#dd8452"
    for v in fixed
]
ax_fix.barh(y, vals_fix, color=colors_fix, alpha=0.9)
ax_fix.axvline(0.0, color="0.4", lw=1)
ax_fix.axvspan(-TOLERANCE, TOLERANCE, color="#55a868", alpha=0.18, zorder=0)
ax_fix.set_yticks(y)
ax_fix.set_yticklabels([])
ax_fix.invert_yaxis()
ax_fix.set_xlabel("slope (same axis as centre)")
ax_fix.set_title("muP, readout std = $fan\\_in^{-1/2}$\n(SHIPPED)", fontsize=10.5)
for i, v in enumerate(fixed):
    if v is None:
        ax_fix.text(0.004, i, "flat", va="center", fontsize=8, color="0.5")
    else:
        off = 0.006 if v >= 0 else -0.006
        ha = "left" if v >= 0 else "right"
        ax_fix.text(v + off, i, f"{v:+.3f}", va="center", ha=ha, fontsize=8.5)
ax_fix.set_xlim(-0.16, 0.10)

fig.suptitle(
    "muP coordinate check: the LR grouping flattens the hidden layers; the readout needed a measurement, not the paper",
    fontsize=12.5,
)
fig.text(
    0.5, 0.005,
    "4 widths x 8 steps, lr 1e-4. muP promises slope 0; tolerance 0.05. "
    "Reading Table 3 literally (std = $d^{-1}$) over-scales the readout DOWN to -0.130 "
    "-- and scaling its LR too made it -0.483, worse. Keeping $fan\\_in^{-1/2}$ and "
    "letting the eta/m LR group carry the width scaling gives 0.001.",
    ha="center", fontsize=8.5, color="0.35",
)
fig.tight_layout(rect=(0, 0.035, 1, 0.94))

out = Path("torchtitan/experiments/ezpz/docs/experiments/mup/figures")
out.mkdir(parents=True, exist_ok=True)
for ext in ("svg", "png"):
    fig.savefig(out / f"mup_coord_check_slopes.{ext}", dpi=150,
                bbox_inches="tight")
print(f"style: {_STYLE}")
print(f"wrote {out}/mup_coord_check_slopes.{{svg,png}}")
