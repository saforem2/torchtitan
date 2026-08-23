#!/usr/bin/env python3
"""Plot production training metrics pulled from W&B.

W&B is the single source of truth for production training history,
since PBS log files only exist for jobs that have already exited and
the active runs span multiple resumes. Generates three figures per
model size:

  - ``production_<model>_<num_nodes>n.svg``: loss / tps-per-gpu / mfu
    vs step.
  - ``training_diagnostics_<model>_<num_nodes>n.svg``: grad_norm, lr,
    and max_loss vs step.
  - ``tokens_vs_time_<model>_<num_nodes>n.svg``: cumulative
    ``n_tokens_seen`` vs wall-clock datetime.

The dense raw per-step traces (13K+ points) use ``rasterized=True`` so
matplotlib embeds them as a small PNG inside the otherwise-vector SVG.
Axes, labels, ticks, smoothed curves, and the legend all stay vector;
``savefig(..., dpi=200)`` controls the resolution of the rasterized
region. Net file size: ~50-150 KB SVG vs ~300-500 KB PNG.

Run from the repo root:

    python3 torchtitan/experiments/ezpz/utils/plot_production_wandb.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Shared ambivalent + Iosevka style helper.
from torchtitan.experiments.ezpz.utils.plot_style import apply_style

apply_style()

import wandb  # noqa: E402

PROJECT = "aurora_gpt/torchtitan.ezpz.train"


def _savefig_both(fig, svg_path, dpi=200):
    """Save figure as both .svg (vector + rasterized data) and .png.

    GitHub's markdown renderer doesn't reliably display SVGs over ~500KB
    (the production charts hit 700-800KB with their dense rasterized
    per-step traces), so we emit a PNG alongside that the README can
    reference for reliable rendering. SVG stays available for anyone
    who wants higher-fidelity inspection.
    """
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, dpi=dpi, bbox_inches="tight")
    png_path = svg_path.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")

# Production runs identified by step ranges (cross-checked with PBS logs).
# Listed oldest first so concatenation matches resume order.
# Each key here drives the figure filename + output dir:
#   docs/production/agpt/<model>/n<num_nodes>/figures/<scope>_<key>n.svg
# So the key MUST encode model + version + node count, e.g. "2b_v1_256",
# "20b_v2_512". Don't include the trailing "n" — the template adds it.
#
# v1 = original 2026-04-{14..29} runs (torch 2.10, LBS=1) trained with
#      `--training.dtype=bfloat16`. Sub-ULP master-weight updates froze
#      every RMSNorm.weight at its 1.0 init; loss curves are real but
#      the model has no trainable normalization. Tainted, superseded
#      by v2. Kept here for the historical record. See
#      docs/reference/guides/training-dtype-bf16-norm-freeze.md.
# v2 = fresh restarts on 2026-04-30 from /flare/AuroraGPT/foremans/runs/
#      agpt-{2b,20b}-v2/ (torch 2.13 venv, LBS=2,
#      `--training.dtype=float32`, plain CrossEntropyLoss). These are
#      the current production runs.
# PRODUCTION_RUNS now lives in trajectories.py (the single source of
# truth shared with check_stale_docs.sh, fill_trajectory_fields.py, and
# the eval scripts). It is imported here as a drop-in: same dict shape
# {key: {run_ids, num_nodes, model[, olog_fallbacks]}}, same key order.
# Add new W&B run-ids by appending to a trajectory's wandb_run_ids in
# trajectories.py — that single edit updates the charts, the stale-doc
# map, the field-filler, and the eval plots together.
from torchtitan.experiments.ezpz.utils.trajectories import (  # noqa: E402
    OLMO_MIX_1124_TOKENS,
    PRODUCTION_RUNS,
    by_key,
)

# The W&B/.o-log fetch + concat logic is shared with prod_dash.py via this
# dependency-light module (stdlib + lazy wandb; NO numpy/matplotlib/torch) so
# the chart plotters and the live board can never drift again. This file adapts
# its plain-dict records to the numpy/METRIC_KEYS contract the plotters expect.
from torchtitan.experiments.ezpz.utils import wandb_fetch  # noqa: E402

# Family mid-tones from the shared palette. The local copy here said
# 20b -> #D32F2F (red), which contradicted plot_production_combined.py where
# red WAS the 2B family -- so 2B rendered blue in the per-run charts and red
# in the overlay. utils/palette.py is now the only definition.
from torchtitan.experiments.ezpz.utils.palette import (  # noqa: E402
    alpha_for as _pal_alpha,
    CHAIN_COLORS as _pal_chain_colors,
    MODEL_COLORS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DOCS_BASE = REPO_ROOT / "torchtitan" / "experiments" / "ezpz" / "docs"

# Full metric set the charts consume. Identical to wandb_fetch.ALL_KEYS; the
# fetch/parse/concat logic lives in that dependency-light shared module (also
# used by prod_dash.py). The functions below are thin numpy adapters that map
# its plain-dict records to the {key: np.ndarray} contract the plotters expect.
METRIC_KEYS = wandb_fetch.ALL_KEYS


def _records_to_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    """Adapt wandb_fetch's plain-dict records to {METRIC_KEYS: np.ndarray}.

    Keys absent from a record (e.g. lr/_timestamp/max_loss/n_tokens_seen when a
    record came from a .o line) become NaN, so downstream ``~np.isnan(steps)``
    filtering degrades gracefully.
    """
    return {
        k: np.array([r.get(k, np.nan) for r in records], dtype=float)
        if k != "_step"
        else np.array([r.get("_step") for r in records])
        for k in METRIC_KEYS
    }


def fetch_run(api: wandb.Api, run_id: str) -> dict[str, np.ndarray]:
    """Pull one wandb run's full history for METRIC_KEYS as numpy arrays.

    Empty arrays if the run cannot be found in ``PROJECT`` (e.g. it logged to a
    different project); ``concat_runs`` then uses the run's ``olog_fallbacks``
    entry. Thin adapter over ``wandb_fetch.fetch_wandb_run``.
    """
    records = wandb_fetch.fetch_wandb_run(
        run_id, keys=METRIC_KEYS, project=PROJECT, api=api)
    return _records_to_arrays(records)


def fetch_from_olog(log_path: str) -> dict[str, np.ndarray]:
    """Parse per-step metric lines from a PBS .o log into the ``fetch_run``
    shape. Thin adapter over ``wandb_fetch.parse_olog``; keys unavailable from
    stdout (lr, _timestamp, max_loss, n_tokens_seen) come back as NaN.
    """
    records, _ = wandb_fetch.parse_olog([log_path])
    return _records_to_arrays(records)


def concat_runs(
    api: wandb.Api,
    run_ids: list[str],
    olog_fallbacks: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Fetch and concatenate a chain's wandb runs by ascending _step.

    Thin numpy adapter over ``wandb_fetch.concat_chain`` (the one canonical
    concat, shared with prod_dash.py): for a run with an ``olog_fallbacks``
    entry it prefers the PBS .o log whenever that reaches at least as far as
    W&B, so a partially/never-synced run's trajectory tail survives.
    """
    records = wandb_fetch.concat_chain(
        run_ids, olog_fallbacks=olog_fallbacks, keys=METRIC_KEYS,
        project=PROJECT, api=api, log=print)
    return _records_to_arrays(records)


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


def plot_dashboard(
    data: dict[str, np.ndarray],
    model_name: str,
    num_nodes: int,
    output_path: Path,
) -> Path:
    """3-panel figure: loss, tps/gpu, mfu vs step.

    Replaces the log-parsing version in ``plot_production.py``: pulls
    from W&B so it reflects in-progress runs whose PBS log files
    haven't been written yet.
    """
    color = MODEL_COLORS.get(model_name, "#1E88E5")
    num_gpus = num_nodes * 12
    steps = data["_step"].astype(float)
    loss = data["loss_metrics/global_avg_loss"].astype(float)
    # `throughput(tps)` is already logged per-rank (per-GPU) by torchtitan,
    # not aggregated — values for the 2B run sit around 1k-3k, matching
    # the per-GPU numbers in the PBS-log dashboards. Don't divide.
    tps_per_gpu = data["throughput(tps)"].astype(float)
    mfu = data["mfu(%)"].astype(float)

    valid = ~np.isnan(steps)
    steps = steps[valid]
    loss = loss[valid]
    tps_per_gpu = tps_per_gpu[valid]
    mfu = mfu[valid]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"AuroraGPT {model_name.upper()} Production Training  |  "
        f"{num_nodes} nodes ({num_gpus} GPUs)  |  "
        f"step {int(steps[-1]):,}",
        fontsize=14,
        fontweight="bold",
    )

    # rasterized=True on the dense raw lines keeps them as a small embedded
    # PNG inside the SVG (13K+ points would otherwise be 13K SVG path nodes).
    # Smoothed lines + axes + legend stay vector.
    ax = axes[0]
    ax.plot(steps, loss, color=color, alpha=0.25, linewidth=0.5, rasterized=True)
    ax.plot(steps, smooth(loss), color=color, linewidth=1.8, label="Loss (smoothed)")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, tps_per_gpu, color="#43A047", alpha=0.25, linewidth=0.5,
            rasterized=True)
    ax.plot(
        steps,
        smooth(tps_per_gpu),
        color="#43A047",
        linewidth=1.8,
        label="TPS/GPU (smoothed)",
    )
    ax.set_ylabel("Tokens/sec/GPU")
    ax.set_title("Throughput per GPU")
    ax.legend()

    ax = axes[2]
    ax.plot(steps, mfu, color="#FF9800", alpha=0.25, linewidth=0.5,
            rasterized=True)
    ax.plot(steps, smooth(mfu), color="#FF9800", linewidth=1.8, label="MFU (smoothed)")
    ax.set_ylabel("MFU (%)")
    ax.set_xlabel("Training Step")
    ax.set_title("Model FLOPs Utilization")
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # dpi here sets resolution of the rasterized region in the SVG.
    _savefig_both(fig, output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_diagnostics(
    data: dict[str, np.ndarray],
    model_name: str,
    num_nodes: int,
    output_path: Path,
) -> Path:
    """3-panel figure: grad_norm, lr, max_loss vs step."""
    color = MODEL_COLORS.get(model_name, "#1E88E5")
    steps = data["_step"].astype(float)
    grad_norm = data["grad_norm"].astype(float)
    lr = data["lr"].astype(float)
    max_loss = data["loss_metrics/global_max_loss"].astype(float)

    # Filter out NaN-only rows that some logging configs emit
    valid = ~np.isnan(steps)
    steps = steps[valid]
    grad_norm = grad_norm[valid]
    lr = lr[valid]
    max_loss = max_loss[valid]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"AuroraGPT {model_name.upper()} Diagnostics  |  "
        f"{num_nodes} nodes ({num_nodes * 12} GPUs)  |  "
        f"step {int(steps[-1]):,}",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(steps, grad_norm, color=color, alpha=0.25, linewidth=0.5,
            rasterized=True)
    ax.plot(steps, smooth(grad_norm), color=color, linewidth=1.8, label="grad_norm (smoothed)")
    ax.set_ylabel("grad_norm")
    ax.set_title("Gradient Norm")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, lr, color="#FF9800", linewidth=1.8, label="lr")
    ax.set_ylabel("lr")
    ax.set_title("Learning Rate")
    ax.legend()

    ax = axes[2]
    ax.plot(steps, max_loss, color="#7B1FA2", alpha=0.25, linewidth=0.5,
            rasterized=True)
    ax.plot(steps, smooth(max_loss), color="#7B1FA2", linewidth=1.8, label="max_loss (smoothed)")
    ax.set_ylabel("global_max_loss")
    ax.set_xlabel("Training Step")
    ax.set_title("Per-Step Max Loss (across DP ranks)")
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig_both(fig, output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_tokens_vs_time(
    data: dict[str, np.ndarray],
    model_name: str,
    num_nodes: int,
    output_path: Path,
    prior_tokens: float = 0.0,
    token_target: float = OLMO_MIX_1124_TOKENS,
) -> Path:
    """Cumulative tokens vs wall-clock datetime.

    Sorts by timestamp (not step) so resumed runs read as a single
    monotonic curve. When restarts re-tokenize early steps after a
    checkpoint, those rows are filtered out via a running max so the
    cumulative curve never goes backwards.
    """
    color = MODEL_COLORS.get(model_name, "#1E88E5")
    ts = data["_timestamp"].astype(float)
    tokens = data["n_tokens_seen"].astype(float)

    valid = ~np.isnan(ts) & ~np.isnan(tokens)
    ts = ts[valid]
    tokens = tokens[valid]

    # Sort by wall-clock time so the curve reflects actual execution order.
    order = np.argsort(ts)
    ts = ts[order]
    tokens = tokens[order]

    # Restarts that wandb logged before the resume's `n_tokens_seen` caught
    # up to the previous best produce dips. Take the running max so the
    # curve stays monotonic non-decreasing.
    tokens = np.maximum.accumulate(tokens)
    # A stage-2 chain's logged `n_tokens_seen` ALSO restarts at 0: the trainer
    # zeroes it and only train_state persists it, while
    # --checkpoint.initial-load-path defaults to initial_load_model_only=True,
    # so the seed run's count is never restored. Shift past what the seed
    # checkpoint already consumed. Non-stage-2 chains pass 0 and are unchanged.
    tokens = tokens + prior_tokens

    times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times, tokens / 1e9, color=color, linewidth=1.8)

    final_tokens_b = tokens[-1] / 1e9
    # Was a hardcoded `4670`, a drifting duplicate of OLMO_MIX_1124_TOKENS that
    # was ALSO wrong for stage-2 chains: it divided an increment-frame numerator
    # by a cumulative-frame denominator (dolmino read "14.7%", correct in
    # neither frame). Both sides are now cumulative and per-chain.
    target_b = (prior_tokens + token_target) / 1e9
    pct = 100 * final_tokens_b / target_b

    ax.set_xlabel("Date")
    ax.set_ylabel("Tokens consumed (billions)")
    ax.set_title(
        f"AuroraGPT {model_name.upper()} — Tokens vs Wall Clock  |  "
        f"{num_nodes} nodes  |  "
        # The "of X target" label MUST be derived from target_b, not written
        # as a literal: a second hardcoded 4.67T survived the first fix here
        # and printed "77.2% of 4.67T" while dividing by 7.06T -- a percentage
        # and a denominator that disagreed, in the same sentence.
        f"{final_tokens_b:,.1f}B tokens ({pct:.1f}% of {target_b / 1000:.2f}T target)",
        fontsize=14,
        fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    fig.tight_layout()
    _savefig_both(fig, output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_overlay(
    series: list[dict],
    model_name: str,
    output_path: Path,
) -> Path:
    """Overlay multiple PRODUCTION_RUNS entries on a 3-panel loss/TPS/MFU
    dashboard.

    Each ``series`` entry is a dict with:
        - ``key``    : PRODUCTION_RUNS key (used in the legend)
        - ``data``   : already-fetched + concatenated W&B history
        - ``color``  : matplotlib color
        - ``alpha``  : line alpha (raw curve uses 0.4× this)

    The point is to make the v1 (bf16-tainted) vs v2 (fp32) contrast
    visually unmissable: the loss curves descend together but their
    *eval-time* behavior diverges because v1 has frozen RMSNorm
    weights. See docs/reference/guides/training-dtype-bf16-norm-freeze.md.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"AuroraGPT {model_name.upper()} — v1 (bf16-master) vs v2 (fp32-master)",
        fontsize=14,
        fontweight="bold",
    )

    last_steps = []
    for s in series:
        data = s["data"]
        steps = data["_step"].astype(float)
        loss = data["loss_metrics/global_avg_loss"].astype(float)
        tps = data["throughput(tps)"].astype(float)
        mfu = data["mfu(%)"].astype(float)
        valid = ~np.isnan(steps)
        steps = steps[valid]
        loss = loss[valid]
        tps = tps[valid]
        mfu = mfu[valid]
        if len(steps) == 0:
            continue
        last_steps.append(int(steps[-1]))

        color = s["color"]
        alpha = s["alpha"]
        label = s["key"]

        axes[0].plot(steps, loss, color=color, alpha=0.4 * alpha, linewidth=0.5,
                     rasterized=True)
        axes[0].plot(steps, smooth(loss), color=color, alpha=alpha, linewidth=1.8, label=label)

        axes[1].plot(steps, tps, color=color, alpha=0.4 * alpha, linewidth=0.5,
                     rasterized=True)
        axes[1].plot(steps, smooth(tps), color=color, alpha=alpha, linewidth=1.8, label=label)

        axes[2].plot(steps, mfu, color=color, alpha=0.4 * alpha, linewidth=0.5,
                     rasterized=True)
        axes[2].plot(steps, smooth(mfu), color=color, alpha=alpha, linewidth=1.8, label=label)

    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend(loc="upper right")
    axes[1].set_ylabel("Tokens/sec/GPU")
    axes[1].set_title("Throughput per GPU")
    axes[1].legend(loc="lower right")
    axes[2].set_ylabel("MFU (%)")
    axes[2].set_xlabel("Training Step")
    axes[2].set_title("Model FLOPs Utilization")
    axes[2].legend(loc="lower right")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig_both(fig, output_path, dpi=200)
    plt.close(fig)
    return output_path


# Color/alpha per PRODUCTION_RUNS key, derived from the shared palette so a
# chain renders identically here and in the combined overlay. Lower alpha for
# the v1 (bf16-tainted) entries so v2 reads as the "real" curve.
#
# Built from palette.CHAIN_COLORS rather than listed by hand: the old literal
# table omitted 20b_v2_256 and 2b_v2_512_lr3.22e-5 entirely, so those two fell
# through to a default and drew in whatever the model mid-tone happened to be.
OVERLAY_STYLE: dict[str, dict] = {
    _k: {"color": _c, "alpha": _pal_alpha(_k)}
    for _k, _c in _pal_chain_colors.items()
}


def overlay_keys_for_model(model: str) -> list[str]:
    """Return PRODUCTION_RUNS keys whose `model` field matches `model`."""
    return [k for k, v in PRODUCTION_RUNS.items() if v.get("model", k) == model]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "Run key from PRODUCTION_RUNS to plot (e.g. 2b_v1_256, "
            "2b_v2_256, 2b_v2_512, 20b_v1_256, 20b_v2_512), or omit "
            "to plot all."
        ),
    )
    parser.add_argument(
        "--overlay", type=str, default=None, choices=["2b", "20b"],
        help=(
            "Generate a v1-vs-v2 overlay dashboard for the given model, "
            "instead of (or in addition to) per-run dashboards. The "
            "figure goes to docs/production/agpt/<model>/figures/"
            "overlay_<model>_v1_vs_v2.svg."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: docs/production/agpt/<model>/n<num_nodes>/figures/ for per-trajectory dashboards; docs/production/agpt/<model>/figures/ for overlays)",
    )
    args = parser.parse_args()

    api = wandb.Api()

    if args.overlay is not None:
        keys_to_overlay = overlay_keys_for_model(args.overlay)
        if not keys_to_overlay:
            raise SystemExit(
                f"no PRODUCTION_RUNS entries found for model={args.overlay!r}"
            )
        # v1-vs-v2 overlay charts live in the historical archive.
        out_dir = args.output_dir or (
            DOCS_BASE / "production" / "agpt" / "historical" / "v1-bf16" / "figures"
        )
        series = []
        for key in keys_to_overlay:
            cfg = PRODUCTION_RUNS[key]
            print(f"\n=== Pulling {key} ({len(cfg['run_ids'])} runs) ===")
            data = concat_runs(api, cfg["run_ids"], cfg.get("olog_fallbacks"))
            print(f"  Concatenated: {len(data['_step'])} unique steps")
            if len(data["_step"]) == 0:
                print(f"  no data, skipping {key}")
                continue
            style = OVERLAY_STYLE.get(
                key, {"color": MODEL_COLORS.get(args.overlay, "#666"), "alpha": 1.0}
            )
            series.append({"key": key, "data": data, **style})
        if not series:
            raise SystemExit("no series with data — nothing to overlay")
        plot_overlay(
            series,
            args.overlay,
            out_dir / f"overlay_{args.overlay}_v1_vs_v2.svg",
        )
        return

    keys = [args.model] if args.model else list(PRODUCTION_RUNS)
    for key in keys:
        cfg = PRODUCTION_RUNS[key]
        # `model` (for color + figure title) defaults to the dict key.
        model_name = cfg.get("model", key)
        # Output dir is keyed on (model, node count) — per-trajectory
        # figures land at production/agpt/<model>/n<nodes>/figures/.
        # v1 (bf16-tainted) per-trajectory figures live in the
        # historical archive; the model-level overlay also goes there
        # (handled in the --overlay branch above).
        if "_v1_" in key:
            default_out = (
                DOCS_BASE
                / "production"
                / "agpt"
                / "historical"
                / "v1-bf16"
                / "figures"
            )
        else:
            default_out = (
                DOCS_BASE
                / "production"
                / "agpt"
                / model_name
                / f"n{cfg['num_nodes']}"
                / "figures"
            )
        out_dir = args.output_dir or default_out

        print(f"\n=== Pulling {key} ({len(cfg['run_ids'])} runs) ===")
        data = concat_runs(api, cfg["run_ids"], cfg.get("olog_fallbacks"))
        print(f"  Concatenated: {len(data['_step'])} unique steps")
        if len(data["_step"]) == 0:
            print(f"  no data, skipping {key}")
            continue

        plot_dashboard(
            data,
            model_name,
            cfg["num_nodes"],
            out_dir / f"production_{key}n.svg",
        )
        plot_diagnostics(
            data,
            model_name,
            cfg["num_nodes"],
            out_dir / f"training_diagnostics_{key}n.svg",
        )
        # PRODUCTION_RUNS is a legacy projection that drops prior_tokens /
        # token_target (and is asserted byte-for-byte in tests), so read the
        # full trajectory record for them.
        _traj = by_key(key)
        plot_tokens_vs_time(
            data,
            model_name,
            cfg["num_nodes"],
            out_dir / f"tokens_vs_time_{key}n.svg",
            prior_tokens=_traj.get("prior_tokens") or 0,
            token_target=_traj.get("token_target") or OLMO_MIX_1124_TOKENS,
        )


if __name__ == "__main__":
    main()
