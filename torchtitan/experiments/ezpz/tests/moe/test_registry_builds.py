# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Every registered ezpz flavor is constructible.

The registries hold TWO kinds of value: 14 factory functions and 66 pre-built
Config objects. A naive sweep that calls ``fn()`` on all of them reports 66
failures with ``'Config' object is not callable`` -- which looks exactly like a
catastrophic regression and is purely an artifact of the check. That happened
while porting the sonic backend; the same numbers appeared on the pre-port
commit, which is the only reason it was not chased.

So: dispatch on the value type, and assert the split is what we expect. If a
future change turns a factory into an object (or vice versa) the counts move
and this test says so, instead of silently passing.
"""

from typing import Any, cast

import pytest

from torchtitan.experiments.ezpz.agpt import agpt_configs
from torchtitan.experiments.ezpz.moe import moe_configs


def test_stacked_ffn_preserves_legacy_rng_assignment():
    import torch

    from torchtitan.experiments.ezpz.moe import _dtensor_safe_fused_ffn_config

    gate_init = {"weight": lambda t: torch.nn.init.normal_(t, std=0.1)}
    up_init = {"weight": lambda t: torch.nn.init.normal_(t, std=0.2)}
    cfg = _dtensor_safe_fused_ffn_config(
        dim=7,
        hidden_dim=5,
        w1_param_init=gate_init,
        w2w3_param_init=up_init,
    )

    torch.manual_seed(42)
    legacy = torch.empty(5, 2, 7)
    gate_init["weight"](legacy[:, 0])
    up_init["weight"](legacy[:, 1])
    expected = legacy.transpose(0, 1).contiguous()

    torch.manual_seed(42)
    actual = torch.empty(2, 5, 7)
    assert cfg.w13.param_init is not None
    cfg.w13.param_init["weight"](actual)

    assert torch.equal(actual, expected)


def test_stacked_ffn_preserves_legacy_forward_and_backward_order():
    import torch
    import torch.nn.functional as F

    from torchtitan.experiments.ezpz.moe import _dtensor_safe_fused_ffn_config

    cfg = _dtensor_safe_fused_ffn_config(
        dim=7,
        hidden_dim=5,
        w1_param_init={"weight": torch.nn.init.normal_},
        w2w3_param_init={"weight": torch.nn.init.normal_},
    )
    linear = cfg.w13.build().to(dtype=torch.bfloat16)
    assert linear.legacy_interleaved_compute

    torch.manual_seed(123)
    stacked = torch.randn(2, 5, 7, dtype=torch.bfloat16)
    x = torch.randn(11, 7, dtype=torch.bfloat16, requires_grad=True)
    upstream_grad = torch.randn(11, 2, 5, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(stacked)

    actual = linear(x)
    actual.backward(upstream_grad)
    actual_x_grad = x.grad.detach().clone()
    actual_weight_grad = linear.weight.grad.detach().clone()

    legacy_weight = (
        stacked.transpose(0, 1).contiguous().flatten(0, 1).detach().requires_grad_()
    )
    legacy_x = x.detach().clone().requires_grad_()
    legacy_flat = F.linear(legacy_x, legacy_weight)
    expected = legacy_flat.unflatten(-1, (5, 2)).transpose(-2, -1)
    expected.backward(upstream_grad)

    assert torch.equal(actual, expected)
    assert torch.equal(actual_x_grad, legacy_x.grad)
    assert torch.equal(
        actual_weight_grad,
        legacy_weight.grad.unflatten(0, (5, 2)).transpose(0, 1),
    )


def _materialize(value):
    """Return the Config for a registry entry, factory or pre-built alike."""
    return value() if callable(value) else value


def _entries():
    for registry, label in ((moe_configs, "moe"), (agpt_configs, "agpt")):
        for name, value in registry.items():
            yield f"{label}:{name}", value


@pytest.mark.parametrize("name,value", list(_entries()))
def test_every_registered_flavor_materializes(name, value):
    cfg = _materialize(value)
    assert cfg is not None, f"{name} produced None"


def test_registry_shape_is_what_we_think():
    """Guard the assumption the sweep above depends on."""
    funcs = objs = 0
    for _, value in _entries():
        if callable(value):
            funcs += 1
        else:
            objs += 1
    total = funcs + objs
    assert total > 50, f"registries shrank unexpectedly: {total} entries"
    assert funcs > 0 and objs > 0, (
        f"expected BOTH factories and pre-built Configs, got "
        f"{funcs} callable / {objs} objects -- if this changed, the "
        f"_materialize dispatch above needs revisiting"
    )


def test_sonic_flavors_select_the_sonic_backend():
    """The port's own flavors actually wire aurora_full_sonic through."""
    from torchtitan.experiments.ezpz.moe.config_registry import (
        moe_debugmodel_sonic,
    )
    from torchtitan.experiments.ezpz.moe.routed_experts import EzpzRoutedExperts
    import torch

    cfg = moe_debugmodel_sonic()
    assert cfg.parallelism.expert_parallel_degree > 1, (
        "sonic owns its own expert-parallel all-to-all, so EP>1 is mandatory"
    )
    with torch.device("meta"):
        model = cfg.model.build()
    routed = [m for m in model.modules() if isinstance(m, EzpzRoutedExperts)]
    assert routed, "no EzpzRoutedExperts in the model -- the subclass is not wired"
    assert all(r._wants_routing() for r in routed), (
        "sonic model routed experts did not select the routing-aware path"
    )


def test_custom_token_dispatchers_implement_current_buffer_lifecycle():
    """Upstream RoutedExperts initializes every dispatcher unconditionally."""
    from torchtitan.experiments.ezpz.moe.token_dispatcher import (
        AllToAllTokenDispatcher,
        LocalTokenDispatcher,
    )

    for config in (
        LocalTokenDispatcher.Config(num_experts=4, top_k=2),
        AllToAllTokenDispatcher.Config(num_experts=4, top_k=2),
    ):
        dispatcher = config.build()
        assert dispatcher.init_buffer() is None


def test_routed_experts_wires_dispatcher_meshes(monkeypatch):
    """The recursive Module lifecycle must retain the old dispatcher handoff."""
    from torchtitan.experiments.ezpz.moe.routed_experts import EzpzRoutedExperts
    from torchtitan.models.common.moe import RoutedExperts

    class ParallelDims:
        def get_optional_mesh(self, name):
            return f"{name}-mesh"

    class Dispatcher:
        def wire_meshes(self, **kwargs):
            self.meshes = kwargs

    routed = object.__new__(EzpzRoutedExperts)
    routed.token_dispatcher = Dispatcher()
    monkeypatch.setattr(RoutedExperts, "_parallelize", lambda self, dims: None)

    routed._parallelize(cast(Any, ParallelDims()))

    assert routed.token_dispatcher.meshes == {
        "ep_mesh": "ep-mesh",
        "tp_mesh": "tp-mesh",
    }


def test_routed_experts_prefers_live_sparse_ep_mesh(monkeypatch):
    from torchtitan.experiments.ezpz.moe import routed_experts as routed_module

    class Dispatcher:
        ep_mesh = "stale-ep-mesh"

    routed = object.__new__(routed_module.EzpzRoutedExperts)
    routed.token_dispatcher = Dispatcher()
    monkeypatch.setattr(
        routed_module,
        "spmd_sparse_mesh",
        lambda: {"ep": "live-ep-mesh"},
    )

    assert routed._resolve_ep_mesh() == "live-ep-mesh"


def test_default_backend_does_not_take_the_routing_path():
    """The other five backends must be untouched by the sonic plumbing."""
    import torch

    from torchtitan.experiments.ezpz.moe.config_registry import moe_debugmodel
    from torchtitan.experiments.ezpz.moe.routed_experts import EzpzRoutedExperts

    cfg = moe_debugmodel()
    with torch.device("meta"):
        model = cfg.model.build()
    routed = [m for m in model.modules() if isinstance(m, EzpzRoutedExperts)]
    assert routed, "no EzpzRoutedExperts in the model"
    assert not any(r._wants_routing() for r in routed), (
        "the default backend must take the plain super() path"
    )
