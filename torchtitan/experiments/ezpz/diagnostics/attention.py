# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention-score diagnostics, without materializing the score matrix.

WHY THIS IS NOT "attention entropy".

The obvious ask is entropy of the softmax attention distribution. We cannot
have it cheaply: `F.scaled_dot_product_attention` is FUSED -- the [B, N, L, L]
score matrix is never materialized, which is the entire point of the kernel.
Recomputing it costs O(L^2) memory per head; at seq=4096 with 48 heads that is
~800M entries per layer per step. That is not instrumentation, it is a second
forward pass.

So this measures the INPUTS to the softmax instead, which is where the
pathology we actually chase lives. The 80B work added `agpt_80b_softcap`
(Gemma-2 tanh score_mod, cap +/-30) and `agpt_80b_qknorm` precisely because
the QK logits were suspected of growing unbounded. A logit bound is a
statement about |q||k| -- and |q|, |k| we can measure for free, because they
are already in hand at the wrapper.

  qk_logit_bound_est = max|q| * max|k| * scale

is an upper bound on any single score. It is LOOSE (the true max needs the
dot product, and near-orthogonal q,k give far smaller scores), so read it as
a trend, not a value: if it climbs steadily while loss is flat, scores are
growing and softcap/qk-norm is the lever. That is exactly the signal the 80B
investigation lacked.

Cheap by construction: a couple of reductions over tensors already resident,
gated to fire every N steps on ONE layer, and OFF by default.
"""

from __future__ import annotations

import math
from typing import Any

import torch

# Module-level so the SDPA wrapper can be a pure function of its inputs --
# threading a config object through the local_map-wrapped forward would
# change its signature, and that signature is contract-checked under TP>1
# (see the _BLNH comment in agpt/__init__.py).
_ENABLED: bool = False
_EVERY: int = 50
_STEP: int = 0
_LAYER_TAG: str | None = None
_LATEST: dict[str, float] = {}


def configure(*, enabled: bool, every: int = 50) -> None:
    global _ENABLED, _EVERY
    _ENABLED = enabled
    _EVERY = max(1, every)


def set_step(step: int) -> None:
    """Called once per training step so the wrapper knows whether to sample."""
    global _STEP, _LAYER_TAG
    _STEP = step
    _LAYER_TAG = None  # re-arm: the next wrapper call this step is the sample


def should_sample() -> bool:
    return _ENABLED and (_STEP % _EVERY == 0)


def observe(q: torch.Tensor, k: torch.Tensor, scale: float | None) -> None:
    """Record QK statistics for ONE attention call per sampled step.

    Deliberately samples a single layer rather than averaging over all of
    them: the failure mode is one block's scores running away, and a mean over
    64 layers is exactly the kind of scalar summary that hid the 80B problem.
    First call per step wins, which is layer 0 -- stable across steps, so the
    series is comparable.
    """
    global _LAYER_TAG, _LATEST
    if not should_sample() or _LAYER_TAG is not None:
        return
    _LAYER_TAG = "layer0"
    try:
        with torch.no_grad():
            qd = q.detach()
            kd = k.detach()
            # .to_local() where present: an exact global max would need a
            # gather, and these are trend indicators. Named _local downstream.
            ql = qd.to_local() if hasattr(qd, "to_local") else qd
            kl = kd.to_local() if hasattr(kd, "to_local") else kd
            qmax = float(ql.abs().max().item())
            kmax = float(kl.abs().max().item())
            qrms = float(ql.float().pow(2).mean().sqrt().item())
            krms = float(kl.float().pow(2).mean().sqrt().item())
            sc = float(scale) if scale is not None else 1.0 / math.sqrt(qd.shape[-1])
            _LATEST = {
                "diag/qk_q_absmax_local": qmax,
                "diag/qk_k_absmax_local": kmax,
                "diag/qk_q_rms_local": qrms,
                "diag/qk_k_rms_local": krms,
                # loose upper bound on any single pre-softmax score
                "diag/qk_logit_bound_est": qmax * kmax * sc,
                # what a TYPICAL score looks like: rms(q).rms(k).sqrt(d).scale
                "diag/qk_logit_typical_est": qrms * krms * math.sqrt(qd.shape[-1]) * sc,
            }
    except Exception:
        _LATEST = {}


def drain() -> dict[str, float]:
    """Return and clear the latest sample."""
    global _LATEST
    out, _LATEST = _LATEST, {}
    return out


def activation_stats(out: torch.Tensor, tag: str = "attn_out") -> dict[str, float]:
    """Norm/absmax of an activation tensor. Catches drift before loss moves."""
    try:
        with torch.no_grad():
            o = out.detach()
            ol = o.to_local() if hasattr(o, "to_local") else o
            return {
                f"diag/{tag}_rms_local": float(ol.float().pow(2).mean().sqrt().item()),
                f"diag/{tag}_absmax_local": float(ol.abs().max().item()),
            }
    except Exception:
        return {}


def summarize(_: Any = None) -> dict[str, float]:
    return drain()
