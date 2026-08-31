#!/usr/bin/env python3
"""Per-model and combined production overlay charts.

For each configured production trajectory we emit one or two artifacts:

    - ``all_production_training.{svg,png}`` — every trajectory overlaid
      (top-level ``docs/production/README.md`` + cross-model
      ``docs/production/agpt/README.md``).
    - ``production_2b_training.{svg,png}`` — 2B trajectories only
      (embedded on ``docs/production/agpt/2b/README.md``).
    - ``production_20b_training.{svg,png}`` — 20B only
      (embedded on ``docs/production/agpt/20b/README.md``).
    - ``production_80b_training.{svg,png}`` — 80B only
      (embedded on ``docs/production/agpt/80b/README.md``); skipped if
      no 80B trajectory has live W&B data yet.

Each chart is 3 subplot panels (Loss / TPS-per-GPU / MFU) vs tokens
consumed. The 2B-MDS reference is on the Loss + TPS panels only (MDS
doesn't log MFU). All charts pull from the same trajectory definitions
below and the same W&B fetch path as ``plot_production_wandb.py``, so
per-model and combined views can't drift from each other.

Run:
    python3 -m torchtitan.experiments.ezpz.utils.plot_production_combined
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Shared ambivalent + Iosevka style helper.
from torchtitan.experiments.ezpz.utils.plot_style import apply_style

apply_style()

# Reuse the W&B fetch + .o-log fallback already proven in
# plot_production_wandb.py — same data path that feeds the per-trajectory
# dashboards, so this chart can't diverge from those.
from torchtitan.experiments.ezpz.utils.plot_production_wandb import (  # noqa: E402
    PRODUCTION_RUNS,
    concat_runs,
)

import wandb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
FIGURES_DIR = REPO_ROOT / "torchtitan/experiments/ezpz/docs/production/figures"
OUT_PATH = FIGURES_DIR / "all_production_training.svg"

# MDS data lives as a CSV pulled separately from the MDS W&B project.
MDS_CSV = (
    REPO_ROOT
    / "torchtitan/experiments/ezpz/docs/production/agpt/2b-mds/loss_data/train_metrics.csv"
)
# MDS tokens/iter is CONSTANT across all 3 stages: micro=1 x grad-acc=2 x
# (256 nodes x 12 GPU) = GBS 6144, x seq 8192 = 50,331,648 tok/iter. (The old
# 7770e9/140000 ~= 55.5M constant was wrong -- it mis-scaled the curve to a
# phantom ~8.57T endpoint instead of the true 7.770T budget.) Verified against
# the Megatron-DeepSpeed train_aGPT_2B_*.sh TRAIN_TOKENS budgets: this constant
# reproduces the stage boundaries at exactly iter 92,859 / 140,353 / 154,391.
MDS_TOKENS_PER_STEP = 6144 * 8192  # 50,331,648 tok/iter (GBS 6144 x seq 8192)

# olmo-mix-1124 stage-1 token target, in billions (matches plot_tokens_vs_time's
# target_b). Used in the loss legend to report each chain's progress as a % of
# target -- far more meaningful than a raw sample count.
TARGET_TOKENS_B = 4670

# MDS reference run's 3 pre-training stages (Megatron-DeepSpeed
# train_aGPT_2B_{large_batch,sophiag_stage2,sophiag_stage3}.sh), by cumulative
# token budget in billions. Stage 1 = 0->4.670T (main mix), stage 2 =
# 4.670->7.064T (constant-LR continuation), stage 3 = 7.064->7.770T. Drawn as
# vertical boundary lines on the loss panel.
MDS_STAGE_BOUNDARIES_B = [
    (4673.780, "stage 1 -> 2"),
    (7064.156, "stage 2 -> 3"),
    (7770.766, "stage 3 end"),
]

# Canonical per-trajectory palette. Single source of truth in utils/palette.py
# (hue per model family, lightness per scale) -- the old copy here carried a
# "keep in sync with ..." comment and did not: only 2B got the lightness
# treatment, while 20B-256 was given an unrelated orange next to 20B-512's
# green, so one model family read as two.
from torchtitan.experiments.ezpz.utils import palette as _pal  # noqa: E402

# Stage-1's full token budget, imported rather than retyped: it is the x-axis
# offset for the stage-2 chain, which restarts its step counter at 1.
from torchtitan.experiments.ezpz.utils.trajectories import (  # noqa: E402
    OLMO_MIX_1124_TOKENS as STAGE1_2B_TOKENS,
)

COLOR_2B_MDS      = _pal.COLOR_MDS
COLOR_2B_TT_256N  = _pal.COLOR_2B_256N
COLOR_2B_TT_512N  = _pal.COLOR_2B_512N
COLOR_20B_TT_256N = _pal.COLOR_20B_256N
COLOR_20B_TT_512N = _pal.COLOR_20B_512N

TRAJECTORIES: list[dict] = [
    {
        "model": "2b",
        "label": "2B-MDS (n256, SophiaG ref)",
        "source": "mds",
        "csv_path": str(MDS_CSV),
        "tokens_per_step": MDS_TOKENS_PER_STEP,
        "color": COLOR_2B_MDS,
        "linestyle": "--",
        "marker": None,
    },
    {
        "model": "2b",
        "label": "2B-TT v2 (n256, async)",
        "source": "wandb",
        "key": "2b_v2_256",
        "tokens_per_step": 6144 * 8192,
        "color": COLOR_2B_TT_256N,
        "linestyle": "-",
        "marker": None,
    },
    {
        "model": "2b",
        "label": "2B-TT v2 (n512, sync)",
        "source": "wandb",
        "key": "2b_v2_512",
        "tokens_per_step": 12288 * 8192,
        "color": COLOR_2B_TT_512N,
        "linestyle": "-",
        "marker": None,
    },
    {
        # Stage-2 continued pre-training on dolmino-mix-1124, seeded from the
        # stage-1 n512 endpoint. Its W&B step numbering RESTARTS at 1, so on a
        # tokens axis it would otherwise draw back over the early stage-1 curve.
        # token_offset_b shifts it to where it actually sits: after the 4.674T
        # its base already consumed. Only this chain needs the offset; every
        # other trajectory starts from scratch at step 0.
        "model": "2b",
        "label": "2B-TT v2 stage-2 (n512, dolmino-mix)",
        "source": "wandb",
        "key": "2b_v2_512_stage2_dolmino",
        "tokens_per_step": 12288 * 8192,
        "token_offset_b": STAGE1_2B_TOKENS / 1e9,
        "color": _pal.COLOR_STAGE2,
        "linestyle": "-",
        "marker": None,
    },
    {
        "model": "20b",
        "label": "20B-TT v2 (n256)",
        "source": "wandb",
        "key": "20b_v2_256",
        "tokens_per_step": 3072 * 8192,
        "color": COLOR_20B_TT_256N,
        "linestyle": "--",
        "marker": None,
    },
    {
        "model": "20b",
        "label": "20B-TT v2 (n512, sync)",
        "source": "wandb",
        "key": "20b_v2_512",
        "tokens_per_step": 12288 * 8192,
        "color": COLOR_20B_TT_512N,
        "linestyle": "-",
        "marker": None,
    },
]


def smooth(values: np.ndarray, window: int = 100) -> np.ndarray:
    """Centered moving average; falls back to identity for short series."""
    values = np.asarray(values, dtype=float)
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")
    half = window // 2
    smoothed[:half] = values[:half]
    smoothed[-half:] = values[-half:]
    return smoothed


def load_wandb_trajectory(
    api: wandb.Api, key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull a trajectory from W&B via the existing concat_runs path.
    Returns (steps, loss, tps_per_gpu, mfu) arrays with NaN-rows filtered.
    """
    cfg = PRODUCTION_RUNS[key]
    data = concat_runs(api, cfg["run_ids"], cfg.get("olog_fallbacks"))
    steps = data["_step"].astype(float)
    loss = data["loss_metrics/global_avg_loss"].astype(float)
    tps = data["throughput(tps)"].astype(float)
    mfu = data["mfu(%)"].astype(float)
    valid = ~np.isnan(steps) & ~np.isnan(loss)
    return steps[valid], loss[valid], tps[valid], mfu[valid]


def load_mds_trajectory(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse train_metrics.csv (iteration, lm_loss, grad_norm, tflops,
    tps_per_gpu, run_id). Returns (iteration, loss, tps_per_gpu). MDS
    doesn't log MFU so the MFU panel just skips this trajectory.
    """
    iters, losses, tps_list = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                it = int(row["iteration"])
                lo = float(row["lm_loss"])
                tp = float(row["tps_per_gpu"])
            except (ValueError, KeyError):
                continue
            iters.append(it)
            losses.append(lo)
            tps_list.append(tp)
    return np.array(iters), np.array(losses), np.array(tps_list)


def _apply_loss_ylim(ax, series: list[dict]) -> None:
    """Crop the loss y-axis to the informative band.

    Bottom = ~0.1 below the global min loss. Top = the 95th percentile of
    each series' post-warmup losses (drops the step-0 warmup spike of ~12+
    nats AND the residual early-transient tail of the least-converged chain,
    so the well-converged chains don't get compressed into a thin band),
    hard-capped at ``LOSS_YMAX_CAP`` so a young chain's early values can never
    push the axis top above the informative band. This intentionally clips
    the very top of a young, still-descending chain. No-op if degenerate.
    """
    # Hard ceiling on the loss y-axis. The converged chains sit ~2.5-2.7 and
    # even early chains are well under this; capping keeps the panel zoomed on
    # the informative band regardless of any transient early spike.
    LOSS_YMAX_CAP = 8.0
    mins, tops = [], []
    for s in series:
        loss = np.asarray(s["loss"], dtype=float)
        loss = loss[np.isfinite(loss)]
        if loss.size == 0:
            continue
        mins.append(float(loss.min()))
        # Drop the leading 5% (warmup spike), then take the 95th percentile
        # of the remainder as the top (trims residual early-transient values
        # that a raw max would otherwise let dominate the axis).
        start = max(1, int(0.05 * loss.size))
        tail = loss[start:] if loss.size > start else loss
        tops.append(float(np.percentile(tail, 95)))
    if not mins or not tops:
        return
    lo = min(mins) - 0.1
    hi = max(tops)
    # A touch of headroom above the crop so curves don't kiss the top border.
    hi = hi + 0.05 * (hi - lo)
    # Hard cap so no early-transient value can push the top above the band.
    hi = min(hi, LOSS_YMAX_CAP)
    if hi > lo:
        ax.set_ylim(lo, hi)


def _series_label(s: dict) -> str:
    """Legend label reporting real progress, not a raw sample count.

    ``step <N> (<pct>% of 4.67T)`` -- so the overlay agrees with the prod_dash
    board (which reports step + % of target) instead of the old ``(n=<rows>)``
    that read like a step and made a completed chain look stuck. Falls back to
    the plain label if a series carries no step array.
    """
    steps = s.get("steps")
    if steps is None or len(steps) == 0:
        return s["label"]
    max_step = int(np.nanmax(steps))
    pct = 100.0 * float(np.nanmax(s["tokens_b"])) / TARGET_TOKENS_B
    return f"{s['label']}  (step {max_step:,}, {pct:.0f}% of 4.67T)"


def _draw_stage_boundaries(ax, series: list[dict]) -> None:
    """Draw the MDS reference run's stage boundaries as dashed verticals.

    Only meaningful when the MDS curve is on this figure (its stages are what
    the boundaries describe); no-op otherwise, so the 20B-only chart stays
    clean. Each line gets a small rotated label near the top of the axis. A
    boundary past the current x-limit is skipped so it never stretches the axis.
    """
    has_mds = any(s.get("source") == "mds" for s in series)
    if not has_mds:
        return
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    for tok_b, label in MDS_STAGE_BOUNDARIES_B:
        if tok_b > x_hi:
            continue  # boundary beyond plotted data -- don't extend the axis
        ax.axvline(tok_b, color="0.45", linestyle=":", linewidth=1.0,
                   alpha=0.8, zorder=1)
        ax.text(tok_b, y_hi - 0.02 * (y_hi - y_lo), f" {label}",
                rotation=90, va="top", ha="left", fontsize=6.5,
                color="0.35", alpha=0.9)


def render_figure(
    series: list[dict],
    *,
    suptitle: str,
    out_path: Path,
) -> None:
    """Render a 3-panel (Loss / TPS / MFU) figure for the given series
    list. Saves both SVG and PNG next to ``out_path``.
    """
    if not series:
        print(f"  (skipping {out_path.name}: no trajectories)")
        return
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(suptitle, fontsize=15, fontweight="bold")

    # Panel 1: Loss vs tokens
    ax = axes[0]
    for s in series:
        ax.plot(
            s["tokens_b"], s["loss"],
            color=s["color"], alpha=0.18, linewidth=0.5, rasterized=True,
        )
        ax.plot(
            s["tokens_b"], smooth(s["loss"], window=min(100, max(2, len(s["loss"]) // 20))),
            color=s["color"], linestyle=s["linestyle"], linewidth=1.8,
            label=_series_label(s),
        )
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    # Crop the y-axis so the converged region (where the trajectories
    # actually diverge) is legible instead of being squashed under the
    # step-0 spike (~12+ nats). Bottom: a hair below the global min.
    # Top: the highest loss reached AFTER each curve's first ~5% of
    # tokens, so the early warmup spike is cropped but the full descent
    # still shows. Falls back to autoscale if data is degenerate.
    _apply_loss_ylim(ax, series)
    # MDS reference stage boundaries (only meaningful when the MDS curve is on
    # this figure -- i.e. the 2B/combined charts, not the 20B-only one). Draw a
    # dashed vertical at each cumulative-token boundary with a rotated label.
    # AFTER _apply_loss_ylim so the label y-position uses the cropped y-range.
    _draw_stage_boundaries(ax, series)

    # Panel 2: TPS/GPU vs tokens
    ax = axes[1]
    for s in series:
        ax.plot(
            s["tokens_b"], s["tps"],
            color=s["color"], alpha=0.18, linewidth=0.5, rasterized=True,
        )
        ax.plot(
            s["tokens_b"], smooth(s["tps"], window=min(100, max(2, len(s["tps"]) // 20))),
            color=s["color"], linestyle=s["linestyle"], linewidth=1.8,
            label=f"{s['label']}",
        )
    ax.set_ylabel("Tokens / sec / GPU")
    ax.set_title("Throughput per GPU")
    ax.grid(alpha=0.25)

    # Panel 3: MFU vs tokens (skip MDS — no MFU column)
    ax = axes[2]
    have_mfu = False
    for s in series:
        if s["mfu"] is None:
            continue
        have_mfu = True
        ax.plot(
            s["tokens_b"], s["mfu"],
            color=s["color"], alpha=0.18, linewidth=0.5, rasterized=True,
        )
        ax.plot(
            s["tokens_b"], smooth(s["mfu"], window=min(100, max(2, len(s["mfu"]) // 20))),
            color=s["color"], linestyle=s["linestyle"], linewidth=1.8,
            label=f"{s['label']}",
        )
    ax.set_ylabel("MFU (%)")
    ax.set_xlabel("Tokens consumed (B)")
    mfu_title = (
        "Model FLOPs Utilization (TT only — MDS does not log MFU)"
        if any(s["mfu"] is None for s in series) and have_mfu
        else "Model FLOPs Utilization"
    )
    ax.set_title(mfu_title)
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  saved: {out_path}")
    print(f"  saved: {png_path}")


def _assert_no_missing_live_chains() -> None:
    """Fail if a live chain in trajectories.py is absent from TRAJECTORIES.

    This module keeps its own display list (labels, colors, linestyles, the
    stage-2 x-offset -- presentation choices that do not belong in the data
    module). The cost is that adding a chain to trajectories.py does NOT add it
    to this chart, and nothing used to say so: the stage-2 dolmino chain was
    registered on 2026-08-16, regenerated cleanly, and simply was not on the
    figure. "0 scripts failed" looked like success.

    So the two lists are reconciled here instead of trusted to stay in sync.
    Only ``cls == "live"`` chains are required -- historical/smoke/wandb_only
    entries are deliberately not overlaid.
    """
    from torchtitan.experiments.ezpz.utils.trajectories import TRAJECTORIES as _ALL

    # Aurora/Sunspot (Intel XPU, olmo-mix) only. The Polaris chain is a
    # different machine, a different corpus (dolma) and a different token
    # budget, so overlaying it on "all canonical chains" would invite a
    # cross-machine comparison the axes do not support. It has its own charts
    # under docs/production/polaris/. Excluded by key, not by weakening the
    # guard -- any OTHER live chain that goes missing must still hard-fail.
    _OTHER_MACHINE = {"20b_polaris_128"}
    live = {
        t["key"] for t in _ALL
        if t.get("cls") == "live" and t["key"] not in _OTHER_MACHINE
    }
    shown = {t.get("key") for t in TRAJECTORIES if t.get("source") == "wandb"}
    missing = sorted(live - shown)
    if missing:
        raise SystemExit(
            "Live chains missing from this chart's TRAJECTORIES list:\n"
            + "".join(f"  - {k}\n" for k in missing)
            + "\nThey are registered in utils/trajectories.py but will not be\n"
            "drawn. Add an entry here (label, color, linestyle -- plus\n"
            "token_offset_b if the chain restarts its step counter), or set\n"
            'cls to something other than "live" if it is not meant to appear.'
        )


def main() -> None:
    _assert_no_missing_live_chains()
    api = wandb.Api()

    # Pull all trajectories once and cache results.
    series: list[dict] = []
    for traj in TRAJECTORIES:
        print(f"\n=== loading {traj['label']} ===")
        if traj["source"] == "wandb":
            steps, loss, tps, mfu = load_wandb_trajectory(api, traj["key"])
            if len(steps) == 0:
                # A chain in the display list that fetched nothing is a wiring
                # bug (wrong key, unsynced run), not an empty chart -- say so
                # here rather than silently dropping the curve.
                print(f"  WARNING: no rows for {traj['key']} -- curve omitted")
                continue
            # Stage-2 chains restart step numbering, so shift them past the
            # tokens their base already consumed (0 for from-scratch chains).
            tokens_b = (
                steps * traj["tokens_per_step"] / 1e9 + traj.get("token_offset_b", 0.0)
            )
            print(f"  {len(steps)} rows, tokens [{tokens_b[0]:.1f}B, {tokens_b[-1]:.1f}B]")
            series.append({**traj, "steps": steps, "tokens_b": tokens_b,
                           "loss": loss, "tps": tps, "mfu": mfu})
        else:  # mds
            # The MDS CSV is gitignored -- that data is pulled from its own
            # W&B project and has never been tracked -- so off-cluster it is
            # legitimately absent. Drop the trajectory with a clear line
            # instead of dying in open(); the other trajectories still plot.
            if not os.path.exists(traj["csv_path"]):
                # Do NOT quietly drop the trajectory. The CSV is gitignored
                # (cluster-only), so off-cluster this fires every run -- and
                # skipping it writes a chart with one fewer series while
                # exiting 0, which is the silent-degradation failure this repo
                # has already been bitten by twice. Refuse to write instead.
                raise SystemExit(
                    f"REFUSING to write a thinner chart: "
                    f"{traj.get('label', 'the MDS trajectory')} needs "
                    f"{os.path.basename(traj['csv_path'])}, which is "
                    f"gitignored and lives on the cluster.\n"
                    f"  Rebuild it from W&B (works from anywhere):\n"
                    f"    python3 torchtitan/experiments/ezpz/utils/"
                    f"fetch_mds_metrics.py\n"
                    f"  The source is the aurora_gpt/AuroraGPT project -- a\n"
                    f"  CHAIN of ~356 runs, with namespaced metric keys\n"
                    f"  ('loss/lm loss', 'loss/iteration'), which is why it is\n"
                    f"  not findable by searching for one run or a bare key."
                )
            iters, loss, tps = load_mds_trajectory(traj["csv_path"])
            tokens_b = iters * traj["tokens_per_step"] / 1e9
            print(f"  {len(iters)} rows, tokens [{tokens_b[0]:.1f}B, {tokens_b[-1]:.1f}B]")
            series.append({**traj, "steps": iters, "tokens_b": tokens_b,
                           "loss": loss, "tps": tps, "mfu": None})

    # 1) Combined chart (all models)
    print("\n=== rendering all_production_training ===")
    render_figure(
        series,
        suptitle="AuroraGPT production training — all canonical chains overlaid",
        out_path=OUT_PATH,
    )

    # 2) Per-model charts (one per distinct `model` field)
    models_present = sorted({s.get("model", "?") for s in series})
    for model in models_present:
        model_series = [s for s in series if s.get("model") == model]
        suptitle = f"AuroraGPT-{model.upper()} production training"
        out_path = FIGURES_DIR / f"production_{model}_training.svg"
        print(f"\n=== rendering production_{model}_training ===")
        render_figure(model_series, suptitle=suptitle, out_path=out_path)


if __name__ == "__main__":
    main()
