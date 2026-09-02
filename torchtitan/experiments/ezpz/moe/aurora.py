"""Adapter from the ezpz MoE module graph to Aurora's optimized runtime."""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import MoE, RoutedExperts

from .experts import EzpzGroupedExperts


def _shared_weights(
    shared: FeedForward, expert_hidden_dim: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = (shared.w3.weight, shared.w1.weight, shared.w2.weight)
    if isinstance(weights[0], DTensor):
        weights = tuple(weight.to_local() for weight in weights)
    up_weight, gate_weight, down_weight = weights
    if up_weight.shape[0] % expert_hidden_dim:
        raise ValueError("Shared hidden dimension must be a multiple of expert hidden")
    count = up_weight.shape[0] // expert_hidden_dim
    up = up_weight.view(count, expert_hidden_dim, dim).transpose(-2, -1)
    gate = gate_weight.view(count, expert_hidden_dim, dim).transpose(-2, -1)
    down = down_weight.view(dim, count, expert_hidden_dim).permute(1, 2, 0)
    return up, gate, down


class AuroraRoutedExperts(RoutedExperts):
    includes_shared_experts = True

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        object.__setattr__(self, "_aurora_shared_experts", None)
        self.ep_mesh = None

    def attach_shared_experts(self, shared_experts: FeedForward) -> None:
        object.__setattr__(self, "_aurora_shared_experts", shared_experts)

    def parallelize(self, parallel_dims) -> None:
        super().parallelize(parallel_dims)
        self.ep_mesh = parallel_dims.get_optional_mesh("ep")

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        del num_local_tokens_per_expert_E
        if self.ep_mesh is None:
            raise ValueError("Aurora full MoE requires expert parallelism")
        if not isinstance(self.inner_experts, EzpzGroupedExperts):
            raise TypeError("Aurora full MoE requires EzpzGroupedExperts")
        shared = self._aurora_shared_experts
        if not isinstance(shared, FeedForward):
            raise TypeError("Aurora full MoE requires shared experts")

        from aurora_moe.torchtitan_full import torchtitan_full_moe

        up, gate, down = self.inner_experts.aurora_weights()
        shared_weights = _shared_weights(
            shared, self.inner_experts.hidden_dim, self.inner_experts.dim
        )
        backend = (
            "loop"
            if self.inner_experts.compute_backend == "aurora_full_loop"
            else "sycl_sonic"
        )
        return torchtitan_full_moe(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            up,
            gate,
            down,
            *shared_weights,
            self.ep_mesh,
            backend=backend,
        )


class AuroraMoE(MoE):
    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        if not isinstance(self.routed_experts, AuroraRoutedExperts):
            raise TypeError("AuroraMoE requires AuroraRoutedExperts")
        if not isinstance(self.shared_experts, FeedForward):
            raise TypeError("AuroraMoE requires shared experts")
        self.routed_experts.attach_shared_experts(self.shared_experts)
