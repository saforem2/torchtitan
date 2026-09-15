"""The non-finite capture must NAME the tensor, not just detect the overflow.

Regression test for a gap found 2026-09-06: `collect_param_stats(per_layer=True)`
filtered non-finite layer norms out of its per-layer stats, and emitted
`diag/topN_gradnorm` (a number) with no companion key for the layer NAME --
despite a comment claiming the name went "in the value's companion key below".

The effect: the capture in `trainer.py` fired, scanned for non-finite floats,
and found only global aggregates (`grad_absmax_local`, `top0_gradnorm`). Those
say something overflowed, which a non-finite grad_norm already established.
The site -- the entire point of the capture -- was unidentifiable.

The instrumentation had never fired in a real run, so this was never noticed.
Reading the code did not reveal it either; driving it with a poisoned gradient
did.
"""

import math

import torch
import torch.nn as nn

from torchtitan.experiments.ezpz import diagnostics as diag


class _Tiny(nn.Module):
    def __init__(self, n: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {str(i): nn.Linear(4, 4, bias=False) for i in range(n)}
        )


def _poisoned(layer: str = "2"):
    m = _Tiny()
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    m.layers[layer].weight.grad[0, 0] = float("inf")
    return m


def _named_from(stats):
    """Exactly what trainer.py's capture branch extracts."""
    return [
        f"{stats[k]} (gradnorm={stats.get(k[:-6], '?')})"
        for k in sorted(stats)
        if k.endswith("_layer")
        and isinstance(stats.get(k[:-6]), float)
        and not math.isfinite(stats[k[:-6]])
    ]


def test_capture_names_the_nonfinite_tensor():
    stats = diag.collect_param_stats([_poisoned("2")], per_layer=True)
    named = _named_from(stats)
    assert named, "capture found no NAMED tensor; only aggregates would be logged"
    assert "layers.2.weight" in named[0], named


def test_nonfinite_layers_are_counted():
    stats = diag.collect_param_stats([_poisoned("1")], per_layer=True)
    assert stats["diag/n_layers_nonfinite"] == 1.0


def test_max_mean_skew_stay_finite():
    """One inf must not saturate the summary stats -- they keep their
    resolution by excluding non-finite layers, which is exactly why the NAME
    has to be carried separately rather than recovered from them."""
    stats = diag.collect_param_stats([_poisoned("3")], per_layer=True)
    for k in ("diag/layer_gradnorm_max", "diag/layer_gradnorm_mean"):
        assert math.isfinite(stats[k]), (k, stats[k])


def test_clean_model_names_nothing():
    m = _Tiny()
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    stats = diag.collect_param_stats([m], per_layer=True)
    assert stats["diag/n_layers_nonfinite"] == 0.0
    assert not _named_from(stats)
