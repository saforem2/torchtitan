"""Verify the diagnostics actually catch the failures they exist for.

Each case reconstructs a real incident from this project. CPU-only, no
distributed backend -- run:

  module load frameworks/2026.1.0 && source venvs/fw-2026.1-rc2/bin/activate
  python3 -m torchtitan.experiments.ezpz.tests.test_diagnostics
"""
import math

import torch
import torch.nn as nn

from torchtitan.experiments.ezpz.diagnostics import (
    clipping_metrics,
    collect_param_stats,
    collect_update_ratios,
)


def _model():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False))
    m(torch.ones(4, 8)).sum().backward()
    return m


def case_frozen_layer_detected():
    """The bf16 RMSNorm freeze: weights that cannot move must read 0.0."""
    m = _model()
    _, prev = collect_update_ratios([m], {})
    # second call with NOTHING changed -- every ratio must be exactly zero
    out, _ = collect_update_ratios([m], prev)
    assert out["diag/update_ratio_max"] == 0.0, out
    n_par = sum(1 for _ in m.parameters())
    assert out["diag/n_params_frozen"] == float(n_par), out
    return f"unchanged weights -> max ratio 0.0, {int(out['diag/n_params_frozen'])} frozen"


def case_moving_layer_not_flagged():
    """The critical negative: a training layer must NOT read as frozen."""
    m = _model()
    _, prev = collect_update_ratios([m], {})
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    out, _ = collect_update_ratios([m], prev)
    assert out["diag/update_ratio_max"] > 0.0, out
    assert out["diag/n_params_frozen"] == 0.0, out
    return f"moved weights -> max ratio {out['diag/update_ratio_max']:.2e}, 0 frozen"


def case_grad_skew_localizes_blowup():
    """The 80B NaN: one exploding layer must raise skew before the global norm."""
    m = _model()
    base = collect_param_stats([m], per_layer=True)
    with torch.no_grad():
        list(m.parameters())[0].grad.mul_(1000.0)
    blown = collect_param_stats([m], per_layer=True)
    assert blown["diag/layer_gradnorm_skew"] > base["diag/layer_gradnorm_skew"], (
        base, blown)
    return (f"skew {base['diag/layer_gradnorm_skew']:.2f} -> "
            f"{blown['diag/layer_gradnorm_skew']:.2f} when one layer blows")


def case_clip_fired_and_headroom():
    m_norm = 1.0
    hot = clipping_metrics(torch.tensor(5.0), m_norm)
    assert hot["diag/clip_fired"] == 1.0 and hot["diag/grad_norm_postclip"] == 1.0, hot
    cool = clipping_metrics(torch.tensor(0.25), m_norm)
    assert cool["diag/clip_fired"] == 0.0, cool
    assert abs(cool["diag/clip_headroom"] - 4.0) < 1e-6, cool
    return "pre 5.0 -> post 1.0 fired=1; pre 0.25 -> fired=0 headroom=4.0"


def case_nan_grad_does_not_crash():
    """Diagnostics must survive the exact state they are meant to report on."""
    m = _model()
    with torch.no_grad():
        list(m.parameters())[0].grad.fill_(float("nan"))
    out = collect_param_stats([m], per_layer=True)
    assert "diag/grad_norm_global" in out
    c = clipping_metrics(torch.tensor(float("nan")), 1.0)
    assert math.isnan(c["diag/grad_norm_preclip"]), c
    return "nan grads: no crash, nan surfaced rather than swallowed"


CASES = [
    case_frozen_layer_detected,
    case_moving_layer_not_flagged,
    case_grad_skew_localizes_blowup,
    case_clip_fired_and_headroom,
    case_nan_grad_does_not_crash,
]


def main() -> int:
    bad = 0
    for c in CASES:
        try:
            print(f"  PASS  {c.__name__}: {c()}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL  {c.__name__}: {e}")
    print(f"\n  {len(CASES) - bad}/{len(CASES)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
