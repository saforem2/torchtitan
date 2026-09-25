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

import pytest

from torchtitan.experiments.ezpz.agpt import agpt_configs
from torchtitan.experiments.ezpz.moe import moe_configs


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
