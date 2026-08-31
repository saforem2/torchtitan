#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""muP coordinate check for the agpt model family.

Stage 1 + 2 of docs/experiments/mup/README.md. Trains several agpt models that
differ in WIDTH ONLY for a handful of steps and records, per module and per
step, the l1 activation coordinate ``x.abs().mean()``.

Under Maximal Update Parametrization those coordinates are width-INVARIANT.
Under the standard parametrization agpt uses today they GROW with width. This
harness exists to reproduce the SP signature first: a check that cannot detect
SP cannot validate muP either, so the SP baseline is the harness's own test.

What is held fixed across the sweep
-----------------------------------
n_layers, vocab_size, head_dim (= dim / n_heads, so n_heads scales with dim),
the FFN ratio hidden_dim/dim (computed directly -- NOT via
compute_ffn_hidden_dim, whose multiple_of rounding makes the ratio drift with
width), rope theta, seed, LR, optimizer, and the data batch.

Traps this harness is built around (all documented in the mup README)
--------------------------------------------------------------------
1. Top-level ``dim`` on an AgptModel.Config is DECORATIVE. Decoder.__init__
   builds from config.tok_embeddings / config.layers / config.norm /
   config.lm_head, each carrying its own baked dimensions
   (models/common/decoder.py:253-268). Setting ``dim`` alone builds N
   IDENTICAL models and would "confirm" muP while testing nothing. The width
   is threaded through ``_build_agpt_config`` (agpt/__init__.py:427) instead,
   and the driver ASSERTS that the parameter counts differ across widths.
2. torch.compile is fatal with hooks, not merely slow -- a ``.item()`` inside
   a compiled block raised InternalTorchDynamoError and killed a real job at
   0 steps. Always ``--compile.no-enable``; enforced here, not optional.
3. Activation checkpointing DOUBLE-FIRES forward hooks because AC replays the
   forward (3 steps gave nfire=6, with different values). AC is disabled here
   and the fire count is asserted against steps * gradient_accumulation_steps.

Each width runs in its OWN process (like scripts/oneoff/fp32_norms_ablation.py)
so no trainer, process-group, or RNG state leaks between arms.

Usage -- login node, CPU, no PBS allocation needed:

    module use /opt/aurora/26.181.0/modulefiles
    module load frameworks/2026.1.0
    source venvs/fw-2026.1-rc2-grain/bin/activate
    unset PYTHONPATH; export PYTHONPATH=$PWD
    TORCH_DEVICE=cpu RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
      MASTER_ADDR=127.0.0.1 MASTER_PORT=29733 \
      python3 -m torchtitan.experiments.ezpz.scripts.mup_coord_check \
        --widths 128,256,512,1024 --steps 8 --out /tmp/mup_coord

Outputs, under ``--out``:
    coord_w<dim>.json   one per width: full per-module per-step l1 series
    summary.json        cross-width slopes and the verdict
    coord_check.svg     the canonical coordinate-check figure (if matplotlib)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from typing import Any

# Widths for the default sweep. head_dim is held at --head-dim (64), so
# n_heads = dim // head_dim = 2, 4, 8, 16. Every rung divides exactly.
DEFAULT_WIDTHS = "128,256,512,1024"

# Anything at or above this |slope| of log2(l1) vs log2(width) is treated as a
# width-dependent coordinate. Chosen well above the run-to-run noise floor of
# a fixed-seed CPU run (which is 0 -- the runs are deterministic) but low
# enough to catch a mild drift. Reported alongside every raw number so the
# threshold never hides the measurement.
SLOPE_TOL = 0.05

# Minimum visible y-span (as a max/min ratio) for a coordinate panel. Without
# it matplotlib autoscales a dead-flat control to fill the axes and it reads
# as noise on par with a real divergence.
MIN_SPAN = 1.6

# tok_embeddings output is a width-independent control: agpt inits the
# embedding at std=1.0 regardless of dim (agpt/__init__.py:286), so its
# coordinates must stay flat under BOTH parametrizations. A sweep in which the
# control also moves means the harness itself is measuring something else.
CONTROL_MODULES = ("tok_embeddings",)


# --------------------------------------------------------------------------
# worker: one width, one process
# --------------------------------------------------------------------------


def _to_plain(t):
    """Local shard of a DTensor, or the tensor itself."""
    return t.to_local() if hasattr(t, "to_local") else t


def _l1(t) -> float:
    """The coordinate statistic: mean absolute value of an activation."""
    import torch

    with torch.no_grad():
        return float(_to_plain(t.detach()).float().abs().mean())


def _first_tensor(out):
    """Modules may return a tensor or a tuple; take the primary output."""
    import torch

    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            if isinstance(o, torch.Tensor):
                return o
    return None


def _probe_names(n_layers: int) -> list[str]:
    """Module FQNs to instrument. Identical strings at every width."""
    names = ["tok_embeddings"]
    for i in range(n_layers):
        names.append(f"layers.{i}.attention")
        names.append(f"layers.{i}.feed_forward")
        names.append(f"layers.{i}")
    names.append("norm")
    names.append("lm_head")
    return names


def run_worker(args: argparse.Namespace) -> int:
    import ezpz.distributed

    ezpz.distributed.setup_torch()
    os.environ.setdefault("RANK", str(ezpz.distributed.get_rank()))
    os.environ.setdefault("WORLD_SIZE", str(ezpz.distributed.get_world_size()))

    import torch

    from torchtitan.config import ConfigManager
    from torchtitan.components.optimizer import default_adamw
    from torchtitan.experiments.ezpz.agpt import _build_agpt_config
    from torchtitan.experiments.ezpz.logging import init_logger

    init_logger()

    dim = args.width
    if dim % args.head_dim != 0:
        raise ValueError(
            f"width {dim} is not a multiple of head_dim {args.head_dim}; the "
            "sweep must hold head_dim exactly constant"
        )
    n_heads = dim // args.head_dim
    # Compute hidden_dim from the ratio DIRECTLY. compute_ffn_hidden_dim's
    # multiple_of rounding is what makes the 20B flavor's ratio 2.80 instead
    # of its nominal 2.667, and a ratio that drifts with width is a second
    # variable in a one-variable experiment.
    hidden_dim = int(round(dim * args.ffn_ratio))
    if not math.isclose(hidden_dim / dim, args.ffn_ratio, rel_tol=1e-9):
        raise ValueError(
            f"ffn_ratio {args.ffn_ratio} is not exact at width {dim} "
            f"(hidden_dim would be {dim * args.ffn_ratio}); pick a ratio that "
            "divides every rung, e.g. 3.0 on power-of-two widths"
        )
    n_kv_heads = None if args.n_kv_heads <= 0 else args.n_kv_heads

    argv = [
        "--module",
        "ezpz.agpt",
        "--config",
        args.config,
        f"--training.steps={args.steps}",
        f"--training.max-context-length={args.seq_len}",
        f"--training.num-tokens-per-microbatch-per-dp-rank={args.seq_len}",
        # Non-negotiable: hooks under torch.compile raise
        # InternalTorchDynamoError, and a checkpoint load would abort before
        # step 1 on a model whose shapes no longer match what is on disk.
        "--compile.no-enable",
        "--checkpoint.no-enable",
        "--validator.no-enable",
        "--metrics.no-enable-wandb",
        f"--metrics.log-freq={max(args.steps, 1)}",
        "--debug.seed",
        str(args.seed),
        "--debug.deterministic",
        # Full LR from step 1. The registry default is warmup_steps=200, which
        # over a handful of steps leaves the LR near zero, the weights near
        # their init, and the coordinates trivially width-invariant -- a
        # false muP pass produced entirely by the schedule.
        f"--lr-scheduler.warmup-steps={args.warmup_steps}",
        *args.extra,
    ]

    config = ConfigManager().parse_args(argv)
    # muP changes BOTH halves and needs both to be measured together: the
    # d^-1 readout init does nothing on its own under Adam (which is
    # scale-invariant in the gradient), and the eta/m hidden LR group does
    # nothing without it. Running one without the other is a coordinate check
    # of neither parametrization.
    if args.mup:
        from torchtitan.experiments.ezpz.agpt.mup import default_mup_adamw

        config.optimizer = default_mup_adamw(
            lr=args.lr,
            dim=dim,
            base_dim=args.mup_base_dim,
            independent_weight_decay=args.mup_independent_wd,
        )
    else:
        config.optimizer = default_adamw(lr=args.lr)

    # THE WIDTH SWAP. Replace the whole model config, built by the one entry
    # point that threads dim into tok_embeddings / layers / norm / lm_head.
    build_kwargs = dict(
        dim=dim,
        n_layers=args.n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rope_theta=args.rope_theta,
        vocab_size=args.vocab_size,
        hidden_dim=hidden_dim,
        max_context_length=args.seq_len,
    )
    if args.mup:
        from torchtitan.experiments.ezpz.agpt.mup import build_mup_agpt_config

        # base_head_dim is pinned to the BASE rung's head_dim, not this rung's.
        # Defaulting it per-rung would make the attention scale reduce to
        # 1/sqrt(head_dim) at every width -- correct on a fixed-head_dim ladder
        # and silently wrong on one that grows head_dim, which is exactly the
        # ladder the attention term exists for.
        model_cfg = build_mup_agpt_config(
            base_dim=args.mup_base_dim,
            base_head_dim=args.head_dim,
            **build_kwargs,
        )
    else:
        model_cfg = _build_agpt_config(**build_kwargs)
    config.model_spec.model = model_cfg
    if hasattr(config.loss, "global_vocab_size"):
        config.loss.global_vocab_size = int(args.vocab_size)
    # AC replays the forward, so every hook would fire twice per step with
    # different values on the replay. Disabled rather than deduped: the check
    # is eager and cheap, and a dedupe would have to guess which fire is real.
    config.activation_checkpoint = None

    trainer = config.build()
    model = trainer.model_parts[0]

    n_params = sum(p.numel() for part in trainer.model_parts for p in part.parameters())

    # Verify the shapes actually moved, inside the worker as well as in the
    # driver -- the decorative-dim trap is silent, so it gets two gates.
    emb_shape = list(_to_plain(model.tok_embeddings.weight).shape)
    if emb_shape[-1] != dim:
        raise RuntimeError(
            f"embedding width is {emb_shape[-1]}, expected {dim}: the width "
            "did not reach the built model (decorative-dim trap)"
        )

    series: dict[str, list[float]] = {}
    nfire: dict[str, int] = {}
    probe = set(_probe_names(args.n_layers))
    handles = []
    hooked: list[str] = []

    def _make_hook(name: str):
        def _hook(_mod, _inp, out):
            t = _first_tensor(out)
            if t is None:
                return
            series.setdefault(name, []).append(_l1(t))
            nfire[name] = nfire.get(name, 0) + 1

        return _hook

    for name, mod in model.named_modules():
        if name in probe:
            handles.append(mod.register_forward_hook(_make_hook(name)))
            hooked.append(name)

    missing = sorted(probe - set(hooked))
    if missing:
        raise RuntimeError(
            f"probe modules not found in the built model: {missing}; the "
            "module tree changed and the harness is silently under-sampling"
        )

    # Hash the input tokens so the driver can PROVE every width saw the same
    # batch. Different data across arms would produce coordinate differences
    # that have nothing to do with width.
    batch_hashes: list[str] = []

    def _pre_hook(_mod, inp_args, inp_kwargs):
        tok = inp_args[0] if inp_args else inp_kwargs.get("tokens")
        if tok is not None:
            arr = _to_plain(tok.detach()).cpu().to(torch.int64).numpy().tobytes()
            batch_hashes.append(hashlib.sha256(arr).hexdigest()[:16])
        return None

    handles.append(model.register_forward_pre_hook(_pre_hook, with_kwargs=True))

    loss_curve: list[float] = []
    _orig_train_step = trainer.train_step

    def _recording_train_step(*a, **kw):
        out = _orig_train_step(*a, **kw)
        try:
            loss_curve.append(float(out))
        except (TypeError, ValueError):
            pass
        return out

    trainer.train_step = _recording_train_step

    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0

    for h in handles:
        h.remove()

    gas = int(getattr(trainer, "gradient_accumulation_steps", 1))
    expected_fires = args.steps * gas
    bad = {k: v for k, v in nfire.items() if v != expected_fires}
    if bad:
        raise RuntimeError(
            f"forward-hook fire count mismatch: expected {expected_fires} "
            f"(steps={args.steps} x gas={gas}) but got {bad}. Activation "
            "checkpointing replays the forward and doubles this; so does an "
            "enabled validator. Both must be off."
        )

    result = {
        "width": dim,
        "n_heads": n_heads,
        "head_dim": args.head_dim,
        "hidden_dim": hidden_dim,
        "ffn_ratio": hidden_dim / dim,
        "n_layers": args.n_layers,
        "n_kv_heads": n_kv_heads,
        "vocab_size": args.vocab_size,
        "seq_len": args.seq_len,
        "n_params": int(n_params),
        "embedding_shape": emb_shape,
        "steps": args.steps,
        "gradient_accumulation_steps": gas,
        "expected_fires": expected_fires,
        "seed": args.seed,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "config": args.config,
        "wall_seconds": wall,
        "losses": loss_curve,
        "batch_hashes": batch_hashes,
        "coords": {k: series[k] for k in sorted(series)},
        "torch": torch.__version__,
        "argv": argv,
    }

    rank0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if rank0:
        os.makedirs(os.path.dirname(os.path.abspath(args.worker_out)), exist_ok=True)
        with open(args.worker_out, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"[coord-check] width={dim} heads={n_heads} hidden={hidden_dim} "
            f"params={n_params/1e6:.2f}M modules={len(series)} "
            f"wall={wall:.1f}s -> {args.worker_out}",
            flush=True,
        )

    trainer.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return 0


# --------------------------------------------------------------------------
# driver: sweep, aggregate, verdict
# --------------------------------------------------------------------------


def _slope(widths: list[int], values: list[float]) -> float:
    """Least-squares slope of log2(value) vs log2(width).

    0 means width-invariant coordinates (what muP promises). Positive means
    they grow with width (the standard-parametrization signature).
    """
    xs = [math.log2(w) for w in widths]
    pos = [v for v in values if v > 0]
    if len(pos) != len(values):
        return float("nan")
    ys = [math.log2(v) for v in values]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def _aggregate(
    runs: list[dict[str, Any]], out_dir: str, *, mup: bool = False
) -> dict[str, Any]:
    runs = sorted(runs, key=lambda r: r["width"])
    widths = [r["width"] for r in runs]

    # GATE 1 -- the decorative-dim trap. If the widths built the same model,
    # every downstream number is meaningless and a flat result would read as a
    # muP pass. Fail loudly.
    counts = [r["n_params"] for r in runs]
    if len(set(counts)) != len(counts):
        raise RuntimeError(
            "parameter counts are NOT distinct across widths "
            f"({dict(zip(widths, counts))}). The width did not reach the "
            "built model -- this is the decorative-dim trap and the sweep is "
            "testing nothing."
        )

    # GATE 2 -- same data everywhere. Otherwise coordinate differences are
    # attributable to the batch, not the width.
    hashes = {r["width"]: r["batch_hashes"] for r in runs}
    ref = hashes[widths[0]]
    mismatched = [w for w in widths if hashes[w] != ref]
    if mismatched:
        raise RuntimeError(
            f"batch token hashes differ at widths {mismatched}; the arms did "
            "not see identical data and the comparison is invalid"
        )

    # GATE 3 -- invariants that define "width only".
    for field in ("head_dim", "n_layers", "vocab_size", "seq_len", "seed", "steps"):
        vals = {r[field] for r in runs}
        if len(vals) != 1:
            raise RuntimeError(f"{field} varies across the sweep: {vals}")
    ratios = {round(r["ffn_ratio"], 12) for r in runs}
    if len(ratios) != 1:
        raise RuntimeError(f"FFN ratio hidden_dim/dim varies across widths: {ratios}")

    steps = runs[0]["steps"]
    modules = sorted(set.intersection(*[set(r["coords"]) for r in runs]))

    per_module: dict[str, Any] = {}
    for m in modules:
        by_step = []
        for s in range(steps):
            vals = [r["coords"][m][s] for r in runs]
            by_step.append(
                {
                    "step": s + 1,
                    "values": vals,
                    "slope": _slope(widths, vals),
                    "ratio_max_over_min": (
                        vals[-1] / vals[0] if vals[0] > 0 else float("nan")
                    ),
                }
            )
        per_module[m] = {
            "control": m in CONTROL_MODULES,
            "first_step": by_step[0],
            "last_step": by_step[-1],
            "by_step": by_step,
        }

    probes = [m for m in modules if m not in CONTROL_MODULES]
    last_slopes = {m: per_module[m]["last_step"]["slope"] for m in probes}
    finite = {m: s for m, s in last_slopes.items() if not math.isnan(s)}
    worst = max(finite, key=lambda m: abs(finite[m])) if finite else None
    max_abs_slope = abs(finite[worst]) if worst else float("nan")

    ctrl_slopes = {
        m: per_module[m]["last_step"]["slope"] for m in modules if m in CONTROL_MODULES
    }

    detected = bool(worst) and max_abs_slope >= SLOPE_TOL
    summary = {
        "widths": widths,
        "n_params": counts,
        "steps": steps,
        "slope_tolerance": SLOPE_TOL,
        "max_abs_slope_last_step": max_abs_slope,
        "worst_module": worst,
        "control_slopes_last_step": ctrl_slopes,
        "width_dependence_detected": detected,
        "mup": bool(mup),
        # The SAME observation -- flat coordinates -- means opposite things
        # depending on which parametrization was built. Under SP, flat is a
        # broken harness. Under muP, flat IS the result. A single fixed string
        # told a correct muP run that it had failed.
        "verdict": (
            (
                "muP PASS -- coordinates are width-invariant within tolerance"
                if not detected
                else "muP FAIL -- coordinates still depend on width; the "
                "parametrization is incomplete or wrong"
            )
            if mup
            else (
                "SP DIVERGENCE REPRODUCED -- coordinates depend on width; the "
                "harness discriminates and can validate muP"
                if detected
                else "FLAT -- no width dependence detected. Under the CURRENT "
                "(standard) parametrization this is a HARNESS FAILURE, not a "
                "muP pass: investigate before trusting any muP result."
            )
        ),
        "runs": [
            {
                k: r[k]
                for k in (
                    "width",
                    "n_heads",
                    "hidden_dim",
                    "n_params",
                    "embedding_shape",
                    "wall_seconds",
                    "losses",
                )
            }
            for r in runs
        ],
        "modules": per_module,
    }

    path = os.path.join(out_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _print_table(summary: dict[str, Any]) -> None:
    widths = summary["widths"]
    steps = summary["steps"]
    print()
    print("=" * 78)
    # summary["mup"] already records which parametrization was built; the
    # label used to hardcode "standard", so a --mup run printed and SAVED a
    # figure claiming to be the SP baseline.
    _pname = "muP" if summary.get("mup") else "current (standard)"
    print(f"muP COORDINATE CHECK -- agpt, {_pname} parametrization")
    print("=" * 78)
    hdr = f"{'width':>8} {'heads':>6} {'hidden':>7} {'params':>12}  final_loss"
    print(hdr)
    for r in summary["runs"]:
        fl = r["losses"][-1] if r["losses"] else float("nan")
        print(
            f"{r['width']:>8} {r['n_heads']:>6} {r['hidden_dim']:>7} "
            f"{r['n_params']:>12,}  {fl:.5f}"
        )
    print()
    wcols = "".join(f"{('W=' + str(w)):>12}" for w in widths)
    print(f"l1 coordinate at step {steps} (last){wcols}{'slope':>9}")
    print("-" * (36 + 12 * len(widths) + 9))
    for m in sorted(summary["modules"]):
        info = summary["modules"][m]
        vals = info["last_step"]["values"]
        sl = info["last_step"]["slope"]
        tag = " [ctrl]" if info["control"] else ""
        cells = "".join(f"{v:>12.5f}" for v in vals)
        print(f"{(m + tag):<36}{cells}{sl:>9.3f}")
    print()
    print(f"slope = d log2(l1) / d log2(width).  0 = width-invariant (muP).")
    print(
        f"max |slope| over non-control modules at step {steps}: "
        f"{summary['max_abs_slope_last_step']:.3f} "
        f"({summary['worst_module']})  tol={summary['slope_tolerance']}"
    )
    print(f"VERDICT: {summary['verdict']}")
    print("=" * 78)


def _plot(summary: dict[str, Any], out_dir: str) -> str | None:
    """Canonical coordinate-check figure: l1 vs width, one line per step."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[coord-check] matplotlib unavailable ({exc}); skipping the plot")
        return None
    try:
        import ambivalent  # noqa: F401

        plt.style.use(ambivalent.STYLES["ambivalent"])
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Iosevka"] + list(
            plt.rcParams["font.sans-serif"]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[coord-check] ambivalent unavailable ({exc}); matplotlib defaults")

    widths = summary["widths"]
    steps = summary["steps"]
    mods = sorted(summary["modules"])
    ncol = 4
    nrow = (len(mods) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for idx, m in enumerate(mods):
        ax = axes[idx // ncol][idx % ncol]
        info = summary["modules"][m]
        for s in range(steps):
            vals = info["by_step"][s]["values"]
            ax.plot(
                widths,
                vals,
                marker="o",
                ms=3,
                lw=1.2,
                color=cmap(s / max(steps - 1, 1)),
                label=f"t={s + 1}" if idx == 0 else None,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels([str(w) for w in widths], fontsize=7)
        # Autoscale alone makes a FLAT panel look as dramatic as a diverging
        # one -- the width-independent tok_embeddings control varies by <1%
        # and still fills the axes. Floor the visible span at MIN_SPAN so
        # "flat" reads as flat and the eye is not misled by the y-axis.
        all_v = [v for s in range(steps) for v in info["by_step"][s]["values"] if v > 0]
        if all_v:
            lo, hi = min(all_v), max(all_v)
            if hi / lo < MIN_SPAN:
                mid = (lo * hi) ** 0.5
                ax.set_ylim(mid / MIN_SPAN**0.5, mid * MIN_SPAN**0.5)
        sl = info["last_step"]["slope"]
        flat = "  (flat)" if abs(sl) < SLOPE_TOL else ""
        ax.set_title(f"{m}\nslope={sl:.2f}{flat}", fontsize=7)
        ax.tick_params(labelsize=7)
    for idx in range(len(mods), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")
    fig.suptitle(
        f"agpt coordinate check -- "
        f"{'muP' if summary.get('mup') else 'standard'} parametrization\n"
        "l1 = |activation|.mean() vs width; flat lines = muP, rising = SP",
        fontsize=10,
    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, "coord_check.svg")
    fig.savefig(path)
    plt.close(fig)
    print(f"[coord-check] wrote {path}")
    return path


def run_driver(args: argparse.Namespace) -> int:
    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    if args.mup and args.mup_base_dim <= 0:
        # Base = the smallest rung, so it gets m=1 and is parametrization-
        # neutral. Resolved HERE, once, because a per-worker default would
        # make every rung its own base, every m equal 1, and the coordinate
        # check flat for a reason that has nothing to do with muP.
        args.mup_base_dim = min(widths)
        print(f"[coord-check] muP base_dim resolved to {args.mup_base_dim}")

    if len(widths) < 3:
        raise ValueError(
            f"a coordinate check needs at least 3 widths to fit a slope, got {widths}"
        )
    if args.steps < 5:
        raise ValueError(f"--steps must be >= 5 to see a trend, got {args.steps}")
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    base_port = int(os.environ.get("MASTER_PORT", "29733"))
    runs = []
    for i, w in enumerate(widths):
        path = os.path.join(out_dir, f"coord_w{w}.json")
        if args.reuse and os.path.exists(path):
            print(f"[coord-check] reusing {path}")
        else:
            env = dict(os.environ)
            # A fresh port per arm so a lingering socket from the previous
            # process cannot make the next one hang on rendezvous.
            env["MASTER_PORT"] = str(base_port + i)
            cmd = [
                sys.executable,
                "-m",
                "torchtitan.experiments.ezpz.scripts.mup_coord_check",
                "--_worker",
                "--width",
                str(w),
                "--worker-out",
                path,
                "--steps",
                str(args.steps),
                "--seq-len",
                str(args.seq_len),
                "--n-layers",
                str(args.n_layers),
                "--head-dim",
                str(args.head_dim),
                "--ffn-ratio",
                str(args.ffn_ratio),
                "--n-kv-heads",
                str(args.n_kv_heads),
                "--vocab-size",
                str(args.vocab_size),
                "--rope-theta",
                str(args.rope_theta),
                "--seed",
                str(args.seed),
                "--lr",
                str(args.lr),
                "--warmup-steps",
                str(args.warmup_steps),
                "--config",
                args.config,
            ]
            if args.mup:
                cmd += [
                    "--mup",
                    "--mup-base-dim",
                    str(args.mup_base_dim),
                ]
                if args.mup_independent_wd:
                    cmd.append("--mup-independent-wd")
            if args.extra:
                cmd += ["--extra", *args.extra]
            print(f"[coord-check] launching width={w} ...", flush=True)
            proc = subprocess.run(cmd, env=env)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"width={w} arm exited {proc.returncode}; see the traceback above"
                )
        if not os.path.exists(path):
            raise RuntimeError(
                f"width={w} arm exited 0 but wrote no {path} -- a silent no-op"
            )
        with open(path) as f:
            runs.append(json.load(f))

    summary = _aggregate(runs, out_dir, mup=args.mup)
    _print_table(summary)
    if not args.no_plot:
        _plot(summary, out_dir)
    print(f"[coord-check] summary -> {os.path.join(out_dir, 'summary.json')}")
    # Exit non-zero when the harness fails its own test, so a wrapper cannot
    # mistake "produced a flat figure" for "muP confirmed".
    #
    # The polarity DEPENDS on which parametrization was built, and 777181ba6
    # made the verdict string aware of that but left this return alone:
    #   SP  baseline: width dependence is EXPECTED, so detecting it is success.
    #   muP run:      width dependence is the FAILURE, so detecting it is a fail.
    # Without the branch, a passing muP run printed "muP PASS" and exited 2,
    # while a broken one exited 0 -- inverted for every rc-gating wrapper
    # (PBS, CI, `&&` chains, set -o pipefail).
    detected = summary["width_dependence_detected"]
    if args.mup:
        return 2 if detected else 0
    return 0 if detected else 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="muP coordinate check for agpt (stage 1+2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--widths", default=DEFAULT_WIDTHS, help="comma-separated dims")
    ap.add_argument("--steps", type=int, default=8, help="training steps per width")
    ap.add_argument("--out", default="/tmp/mup_coord", help="output directory")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument(
        "--head-dim",
        type=int,
        default=64,
        help="held constant; n_heads = width // head_dim",
    )
    ap.add_argument(
        "--ffn-ratio",
        type=float,
        default=3.0,
        help="hidden_dim / dim, held EXACT (not compute_ffn_hidden_dim)",
    )
    ap.add_argument(
        "--n-kv-heads",
        type=int,
        default=-1,
        help="<=0 means MHA (n_kv = n_heads), keeping the GQA ratio "
        "width-invariant; a fixed positive value makes it vary with width",
    )
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--rope-theta", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="1 = full LR from step 1; the registry default of 200 would keep "
        "the weights at init and fake a flat check",
    )
    ap.add_argument("--config", default="ezpz_agpt_debugmodel")
    ap.add_argument(
        "--mup",
        action="store_true",
        help=(
            "build every rung under muP (d^-1 readout init, muP attention "
            "scale, four LR groups with hidden at eta/m) instead of the "
            "current standard parametrization. Without this the check "
            "measures SP, which is what it was built to reproduce."
        ),
    )
    ap.add_argument(
        "--mup-base-dim",
        type=int,
        default=0,
        help=(
            "width eta is tuned at; m = dim / base_dim. Defaults to the "
            "SMALLEST swept width, so the base rung gets m=1 and is "
            "parametrization-neutral."
        ),
    )
    ap.add_argument(
        "--mup-independent-wd",
        action="store_true",
        help=(
            "decouple weight decay from the per-group lr. PyTorch AdamW "
            "decays by (1 - lr*wd), so scaling the hidden lr by 1/m also "
            "scales its decay by 1/m -- per the audit this is one of the "
            "three things documented to break muP transfer, so it is the "
            "first thing to try if the check fails."
        ),
    )
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--reuse", action="store_true", help="skip widths whose JSON already exists"
    )
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--width", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args._worker:
        return run_worker(args)
    return run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
