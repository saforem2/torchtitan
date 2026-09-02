import torch
import torch.nn.functional as F

from torchtitan.experiments.ezpz.moe.aurora import _shared_weights
from torchtitan.experiments.ezpz.moe.experts import EzpzGroupedExperts
from torchtitan.models.common.config_utils import make_ffn_config


def _constant(value):
    return lambda tensor: torch.nn.init.constant_(tensor, value)


def test_aurora_expert_layout_preserves_logical_initializers():
    experts = EzpzGroupedExperts.Config(
        dim=7,
        hidden_dim=11,
        num_experts=5,
        compute_backend="aurora_full_sonic",
        param_init={
            "w1_EFD": _constant(1),
            "w2_EDF": _constant(2),
            "w3_EFD": _constant(3),
        },
    ).build()
    experts.init_states(buffer_device=torch.device("cpu"))

    up, gate, down = experts.aurora_weights()
    assert up.shape == (5, 7, 11)
    assert gate.shape == (5, 7, 11)
    assert down.shape == (5, 11, 7)
    assert up.is_contiguous() and gate.is_contiguous() and down.is_contiguous()
    torch.testing.assert_close(gate, torch.ones_like(gate))
    torch.testing.assert_close(down, torch.full_like(down, 2))
    torch.testing.assert_close(up, torch.full_like(up, 3))


def test_aurora_shared_views_equal_feed_forward():
    torch.manual_seed(7)
    normal = {"weight": lambda tensor: torch.nn.init.normal_(tensor)}
    shared = make_ffn_config(
        dim=7,
        hidden_dim=22,
        w1_param_init=normal,
        w2w3_param_init=normal,
    ).build()
    shared.init_states(buffer_device=torch.device("cpu"))

    up, gate, down = _shared_weights(shared, expert_hidden_dim=11, dim=7)
    assert up.shape == gate.shape == (2, 7, 11)
    assert down.shape == (2, 11, 7)

    x = torch.randn(13, 7)
    actual = torch.zeros_like(x)
    for expert in range(2):
        actual += (F.silu(x @ gate[expert]) * (x @ up[expert])) @ down[expert]
    torch.testing.assert_close(actual, shared(x), rtol=1e-5, atol=1e-5)
