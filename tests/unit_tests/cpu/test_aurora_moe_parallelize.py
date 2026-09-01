from types import SimpleNamespace

import torch
import torch.distributed._composable.replicate_with_fsdp as replicate_api

from torchtitan.experiments.ezpz.moe.parallelize import apply_ep_replicate


class _LocalParameter:
    def to_local(self):
        return torch.empty(3, 4, 5)


class _Experts:
    def parameters(self):
        return iter((_LocalParameter(),))


class _Model:
    def __init__(self):
        self.routed = SimpleNamespace(inner_experts=_Experts())
        self.block = SimpleNamespace(
            moe_enabled=True,
            moe=SimpleNamespace(routed_experts=self.routed),
        )
        self.layers = {"0": self.block}
        self.tok_embeddings = None
        self.norm = None
        self.lm_head = None

    def modules(self):
        return ()


def test_ep_replicate_wraps_the_aurora_forward_owner(monkeypatch):
    calls = []
    monkeypatch.setattr(
        replicate_api,
        "replicate",
        lambda module, **kwargs: calls.append((module, kwargs)),
    )
    model = _Model()

    apply_ep_replicate(
        model,
        batch_mesh=object(),
        expert_dp_mesh=object(),
        expert_mp_mesh=object(),
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    assert calls[0][0] is model.routed
    assert all(module is not model.routed.inner_experts for module, _ in calls)
