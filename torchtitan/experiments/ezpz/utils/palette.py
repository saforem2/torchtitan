# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One canonical trajectory palette for every AuroraGPT chart.

The scheme is **hue per model family, lightness per scale**:

    2B   blues     n256 light -> n512 dark
    20B  oranges   n256 light -> n512 dark
    80B  greens    n256 light -> n512 dark
    MDS  slate     (the v1 Megatron-DeepSpeed reference, deliberately neutral)

Reading a chart then takes one step: hue says *which model*, lightness says
*how many nodes*. Two chains of the same model always look related.

Why this module exists: the palette used to be copy-pasted into
``utils/plot_production_combined.py`` and ``eval/plot_evals_combined.py``
under a "keep in sync with ..." comment, and it did not stay in sync. Only
the 2B pair ever got the lightness treatment; 20B-256 was handed an unrelated
orange with the comment "one-off NODE_FAIL run; no canonical color" while
20B-512 was green, so one model family read as two. (20B-256 has since become
a full production chain, which made the mismatch obvious.) A third palette in
``utils/plot_production_wandb.py`` mapped 2b->blue / 20b->red, directly
contradicting the first two, so 2B was blue in the per-run charts and red in
the overlay.

Import from here rather than redefining. ``CHAIN_COLORS`` is keyed by the
trajectory keys in ``utils/trajectories.py``; ``MODEL_COLORS`` gives the
family's mid-tone for single-model charts.
"""
from __future__ import annotations

# Family mid-tones, for charts that plot one model and need a single color.
MODEL_COLORS: dict[str, str] = {
    "2b": "#1E88E5",   # blue 600
    "20b": "#43A047",  # green 600
    "80b": "#FB8C00",  # orange 600
}

# Per-scale shades: light = fewer nodes, dark = more nodes.
#
# 20B moved orange -> green on 2026-08-16 (user request: the orange/yellow pair
# read as washed-out next to the blues). 80B took orange in the swap rather
# than a new hue: it has no plotted chain today (`80b_v2_4_smoke` carries no
# run-ids), so nothing re-colored, and orange keeps the three families
# maximally separated. Revisit if 80B ever plots alongside a fork curve, since
# COLOR_FORK is purple and orange/purple are the weakest pair here.
COLOR_2B_256N = "#64B5F6"   # blue 300
COLOR_2B_512N = "#0D47A1"   # blue 900
COLOR_20B_256N = "#81C784"  # green 300
COLOR_20B_512N = "#1B5E20"  # green 900
COLOR_80B_256N = "#FFB74D"  # orange 300
COLOR_80B_512N = "#E65100"  # orange 900

# Non-torchtitan / non-scale entries.
COLOR_MDS = "#78909C"     # blue-grey 400 -- the v1 MDS reference curve
COLOR_V1 = "#94A3B8"      # slate -- bf16-tainted v1 chains (drawn faded)
COLOR_RANDOM = "#808080"  # gray -- random-chance lines on eval charts
COLOR_FORK = "#8E24AA"    # purple 600 -- LR/schedule forks off a base chain
# Stage-2 continued pre-training on a DIFFERENT corpus. Deliberately not a 2B
# blue: it seeds from the 2B stage-1 endpoint but trains on dolmino-mix-1124,
# so reading it as "more of the same curve" would be wrong. Pink is unused by
# every other family (blues / greens / oranges / slate / purple), so the
# corpus change reads instantly. Was teal 600 until 2026-08-17; teal sat close
# enough to the 20B greens to be mistaken for one at a glance.
COLOR_STAGE2 = "#EC407A"  # pink 400

# Keyed by utils/trajectories.py trajectory keys.
CHAIN_COLORS: dict[str, str] = {
    "2b_v2_256": COLOR_2B_256N,
    "2b_v2_512": COLOR_2B_512N,
    "2b_v2_512_lr3.22e-5": COLOR_FORK,
    "2b_v2_512_stage2_dolmino": COLOR_STAGE2,
    "20b_v2_256": COLOR_20B_256N,
    "20b_v2_512": COLOR_20B_512N,
    "80b_v2_256": COLOR_80B_256N,
    "80b_v2_512": COLOR_80B_512N,
    "2b_v1_256": COLOR_V1,
    "20b_v1_256": COLOR_V1,
    "2b_mds": COLOR_MDS,
}

# v1 chains are drawn faded so v2 reads as the "real" curve.
CHAIN_ALPHA: dict[str, float] = {
    "2b_v1_256": 0.55,
    "20b_v1_256": 0.55,
}


def color_for(key: str, model: str | None = None) -> str:
    """Color for a trajectory key, falling back to the model mid-tone.

    ``model`` is used only when ``key`` is not in ``CHAIN_COLORS`` (a new
    chain that has not been added yet); an unknown model falls back to a
    neutral gray rather than silently reusing another family's hue.
    """
    if key in CHAIN_COLORS:
        return CHAIN_COLORS[key]
    return MODEL_COLORS.get(model or "", "#666666")


def alpha_for(key: str) -> float:
    """Line alpha for a trajectory key (1.0 unless it is a faded v1 chain)."""
    return CHAIN_ALPHA.get(key, 1.0)
