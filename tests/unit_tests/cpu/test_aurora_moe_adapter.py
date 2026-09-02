import sys
from types import ModuleType
from unittest.mock import Mock, patch

import torch

from torchtitan.experiments.ezpz.moe.aurora import AuroraMoE, AuroraRoutedExperts
from torchtitan.experiments.ezpz.moe.experts import EzpzGroupedExperts
from torchtitan.models.common.config_utils import make_ffn_config
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher


def _normal():
    return {"weight": torch.nn.init.normal_}


def _build_moe():
    dim, hidden_dim, num_experts = 4, 3, 2
    routed = AuroraRoutedExperts.Config(
        inner_experts=EzpzGroupedExperts.Config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            compute_backend="aurora_full_sonic",
            param_init={
                "w1_EFD": torch.nn.init.normal_,
                "w2_EDF": torch.nn.init.normal_,
                "w3_EFD": torch.nn.init.normal_,
            },
        ),
        token_dispatcher=LocalTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=1,
        ),
    )
    return AuroraMoE.Config(
        num_experts=num_experts,
        load_balance_coeff=None,
        router=TokenChoiceTopKRouter.Config(
            num_experts=num_experts,
            gate=Linear.Config(
                in_features=dim,
                out_features=num_experts,
                bias=False,
                param_init=_normal(),
            ),
            top_k=1,
            score_func="softmax",
        ),
        routed_experts=routed,
        shared_experts=make_ffn_config(
            dim=dim,
            hidden_dim=hidden_dim * 2,
            w1_param_init=_normal(),
            w2w3_param_init=_normal(),
        ),
    ).build()


def test_aurora_adapter_owns_routed_and_shared_forward():
    moe = _build_moe()
    moe.init_states(buffer_device=torch.device("cpu"))
    moe.routed_experts.ep_mesh = object()
    x = torch.randn(5, 4)
    expected = torch.randn_like(x)
    full_moe = Mock(return_value=expected)
    package = ModuleType("aurora_moe")
    package.__path__ = []
    runtime = ModuleType("aurora_moe.torchtitan_full")
    runtime.torchtitan_full_moe = full_moe

    with (
        patch.dict(
            sys.modules,
            {"aurora_moe": package, "aurora_moe.torchtitan_full": runtime},
        ),
        patch.object(
            moe.shared_experts,
            "forward",
            side_effect=AssertionError("shared branch ran twice"),
        ),
    ):
        torch.testing.assert_close(moe(x), expected)

    args = full_moe.call_args.args
    assert args[0] is x
    assert args[-1] is moe.routed_experts.ep_mesh
    assert args[3].shape == args[4].shape == (2, 4, 3)
    assert args[5].shape == (2, 3, 4)
    assert args[6].shape == args[7].shape == (2, 4, 3)
    assert args[8].shape == (2, 3, 4)
    assert full_moe.call_args.kwargs == {"backend": "sycl_sonic"}
