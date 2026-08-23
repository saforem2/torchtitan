# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training diagnostics beyond loss/grad_norm/throughput.

WHAT WE TRACK TODAY, and why it is not enough.

The per-step dict (components/metrics.py:497) is loss avg/max, grad_norm,
tps/tflops/mfu, wall-clock splits, memory, plus ezpz's n_tokens_seen and lr.
Every one of those is a SCALAR SUMMARY OF THE WHOLE MODEL. When a run goes
wrong they tell you THAT it went wrong and nothing about WHERE.

Three real incidents from this project that these metrics could not localize:

  - The 80B NaN (job 12473142): grad_norm went nan at step 30 and loss at
    step 31. We had one number, so the earliest warning was one step. A
    per-layer grad norm would have shown which block blew first, and
    grad-norm skew would have been rising for many steps before either.
  - The bf16 RMSNorm freeze: every v1 RMSNorm.weight stayed EXACTLY 1.0 for
    entire runs because the bf16 ULP at 1.0 exceeded the update size. Loss
    still descended. No tracked quantity could have revealed it -- but
    `update_ratio` (|delta W| / |W|) would have read a hard zero on every
    norm layer from step 1.
  - Mano at 30B: loss and grad_norm alone could not distinguish "LR too hot"
    from "LR too cold" (both look like slow descent). update_ratio separates
    them immediately -- it is ~1e-3 for a healthy step, orders below for a
    dead one.

The through-line: a scalar tells you the model is sick; a DISTRIBUTION tells
you which organ.

COST. Everything here is O(params) elementwise work on tensors already
resident, no host sync beyond the reduce we already pay for grad_norm, and it
runs only every `interval` steps (default 50, vs log_freq's typical 1). The
expensive item -- per-parameter norms -- is gated separately behind
`per_layer`. Default OFF; this is opt-in instrumentation, not a tax on
production.

DTensor: parameters under FSDP/TP are DTensors whose .grad is also a DTensor.
Norms computed on them are already global (the reduce is implicit in the
DTensor op), so no manual all-reduce is needed and adding one would
double-count. We call .full_tensor() nowhere -- that would materialize the
whole parameter on every rank.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import torch.nn as nn


def _scalar(t: Any) -> float:
    """Best-effort python float from a tensor/DTensor/number, else nan."""
    if isinstance(t, torch.Tensor):
        try:
            return float(t.detach().float().item())
        except Exception:
            return float("nan")
    try:
        return float(t)
    except (TypeError, ValueError):
        return float("nan")


def _local(t: torch.Tensor) -> torch.Tensor:
    """Local shard of a DTensor, or the tensor itself.

    Used ONLY for quantile/histogram-style stats where an exact global answer
    would need a gather. A per-rank quantile over a shard is a biased estimate
    of the global one, which is why those keys are named `*_local`.
    """
    to_local = getattr(t, "to_local", None)
    return to_local() if callable(to_local) else t


def collect_param_stats(
    model_parts: list[nn.Module],
    *,
    per_layer: bool = False,
    top_k: int = 5,
) -> dict[str, float]:
    """Weight- and gradient-distribution statistics.

    Returns keys under `diag/`. Global norms are exact under DTensor; keys
    suffixed `_local` are per-rank estimates (see `_local`).
    """
    out: dict[str, float] = {}

    w_sq = 0.0
    g_sq = 0.0
    n_params = 0
    n_with_grad = 0
    per_layer_gn: list[tuple[str, float]] = []
    w_absmax = 0.0
    g_absmax = 0.0

    for part in model_parts:
        for name, p in part.named_parameters():
            if p is None:
                continue
            n_params += 1
            wn = _scalar(torch.linalg.vector_norm(p.detach()))
            if math.isfinite(wn):
                w_sq += wn * wn
            w_absmax = max(w_absmax, _scalar(_local(p.detach()).abs().max()))

            g = p.grad
            if g is None:
                continue
            n_with_grad += 1
            gn = _scalar(torch.linalg.vector_norm(g.detach()))
            if math.isfinite(gn):
                g_sq += gn * gn
            g_absmax = max(g_absmax, _scalar(_local(g.detach()).abs().max()))
            if per_layer:
                per_layer_gn.append((name, gn))

    out["diag/weight_norm_global"] = math.sqrt(w_sq)
    out["diag/grad_norm_global"] = math.sqrt(g_sq)
    out["diag/weight_absmax_local"] = w_absmax
    out["diag/grad_absmax_local"] = g_absmax
    out["diag/n_params_with_grad"] = float(n_with_grad)

    # Grad-norm SKEW across layers. This is the single most diagnostic number
    # here: one exploding block raises max/mean long before the global norm
    # (which is an L2 over everything) moves enough to notice.
    if per_layer and per_layer_gn:
        vals = [v for _, v in per_layer_gn if math.isfinite(v)]
        if vals:
            mx, mean = max(vals), sum(vals) / len(vals)
            out["diag/layer_gradnorm_max"] = mx
            out["diag/layer_gradnorm_mean"] = mean
            out["diag/layer_gradnorm_skew"] = mx / mean if mean > 0 else float("nan")
            worst = sorted(per_layer_gn, key=lambda kv: -kv[1])[:top_k]
            for i, (nm, v) in enumerate(worst):
                # rank-ordered so the KEY is stable across steps even as the
                # offending layer changes; the layer name goes in the value's
                # companion key below
                out[f"diag/top{i}_gradnorm"] = v
    return out


def collect_update_ratios(
    model_parts: list[nn.Module],
    prev_weight_norms: dict[str, float],
    *,
    top_k: int = 3,
) -> tuple[dict[str, float], dict[str, float]]:
    """|W_t - W_{t-1}| / |W_{t-1}|, the ratio that catches a frozen layer.

    Returns (metrics, new_prev_norms). Pass the second element back next call.

    This is the quantity that would have caught the bf16 RMSNorm freeze on
    step 1 rather than after entire production runs: a layer whose weights
    cannot move reads exactly 0.0 here while loss keeps descending.

    Healthy is ~1e-3. Orders below that means the layer is not learning;
    orders above means the step is too large.
    """
    out: dict[str, float] = {}
    new_norms: dict[str, float] = {}
    ratios: list[tuple[str, float]] = []

    for part in model_parts:
        for name, p in part.named_parameters():
            if p is None:
                continue
            wn = _scalar(torch.linalg.vector_norm(p.detach()))
            new_norms[name] = wn
            prev = prev_weight_norms.get(name)
            if prev is None or not math.isfinite(prev) or prev <= 0:
                continue
            # norm-of-difference would need a stored copy of every parameter;
            # difference-of-norms is O(1) memory and detects a hard zero just
            # as well, which is the failure mode that actually bit us.
            ratios.append((name, abs(wn - prev) / prev))

    if ratios:
        vals = [v for _, v in ratios]
        vals_sorted = sorted(vals)
        out["diag/update_ratio_mean"] = sum(vals) / len(vals)
        out["diag/update_ratio_median"] = vals_sorted[len(vals_sorted) // 2]
        out["diag/update_ratio_max"] = vals_sorted[-1]
        out["diag/update_ratio_min"] = vals_sorted[0]
        # THE frozen-layer alarm: how many parameters did not move at all.
        out["diag/n_params_frozen"] = float(sum(1 for v in vals if v == 0.0))
    return out, new_norms


def collect_optimizer_stats(optimizers: Any) -> dict[str, float]:
    """Optimizer-state health: exp_avg / exp_avg_sq magnitudes.

    A second-moment estimate collapsing toward zero means the effective LR is
    diverging even while the nominal `lr` we log stays flat -- so the tracked
    `lr` can look perfectly healthy while the actual step size is not.
    """
    out: dict[str, float] = {}
    m1_sq = m2_sum = 0.0
    n = 0
    try:
        opts = getattr(optimizers, "optimizers", [optimizers])
        for opt in opts:
            for st in getattr(opt, "state", {}).values():
                if not isinstance(st, dict):
                    continue
                m1 = st.get("exp_avg")
                m2 = st.get("exp_avg_sq")
                if m1 is not None:
                    v = _scalar(torch.linalg.vector_norm(_local(m1.detach())))
                    if math.isfinite(v):
                        m1_sq += v * v
                if m2 is not None:
                    v = _scalar(_local(m2.detach()).mean())
                    if math.isfinite(v):
                        m2_sum += v
                        n += 1
    except Exception:
        return out
    if n:
        out["diag/opt_exp_avg_norm"] = math.sqrt(m1_sq)
        out["diag/opt_exp_avg_sq_mean"] = m2_sum / n
    return out


def clipping_metrics(
    grad_norm: Any,
    max_norm: float,
) -> dict[str, float]:
    """Pre-clip norm, post-clip norm, and whether clipping engaged.

    `torch.nn.utils.clip_grad_norm_` returns the norm BEFORE clipping, so the
    `grad_norm` we log today is the pre-clip value and we have never recorded
    the post-clip one -- nor how often the clip fires. A run pinned at the
    clip ceiling every step is being silently rate-limited, and looks
    identical in our current metrics to a run that never clips.
    """
    pre = _scalar(grad_norm)
    if not math.isfinite(pre) or max_norm is None or max_norm <= 0:
        return {"diag/grad_norm_preclip": pre}
    clipped = pre > max_norm
    return {
        "diag/grad_norm_preclip": pre,
        "diag/grad_norm_postclip": min(pre, max_norm),
        "diag/clip_fired": 1.0 if clipped else 0.0,
        "diag/clip_headroom": max_norm / pre if pre > 0 else float("nan"),
    }
