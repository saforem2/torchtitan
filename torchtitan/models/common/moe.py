# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Partial

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear

from torchtitan.protocols.module import Module

from .token_dispatcher import _record_moe_fastpath, LocalTokenDispatcher


# NOTE: keeping this for-loop implementation for comparison
#       and readability, may remove later
def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor | list[int],
    w13: torch.Tensor | None = None,
    w2_t: torch.Tensor | None = None,
) -> torch.Tensor:
    # NOTE: this would incur a synchronization between device and host
    num_tokens_per_expert_list = (
        num_tokens_per_expert
        if isinstance(num_tokens_per_expert, list)
        else num_tokens_per_expert.tolist()
    )

    if (
        len(num_tokens_per_expert_list) > 0
        and all(
            count == num_tokens_per_expert_list[0]
            for count in num_tokens_per_expert_list
        )
        and num_tokens_per_expert_list[0] > 0
        and not torch.is_grad_enabled()
    ):
        _record_moe_fastpath("batched_no_grad_experts")
        tokens_per_expert = num_tokens_per_expert_list[0]
        expected_numel = (
            len(num_tokens_per_expert_list) * tokens_per_expert * x.shape[-1]
        )
        assert x.numel() == expected_numel
        x_grouped = x.reshape(
            len(num_tokens_per_expert_list), tokens_per_expert, x.shape[-1]
        )
        if w13 is None:
            w13 = torch.cat((w1, w3), dim=1)
        h13 = torch.bmm(x_grouped, w13.transpose(-2, -1))
        h1, h3 = h13.chunk(2, dim=-1)
        h = F.silu(h1) * h3
        if w2_t is None:
            w2_t = w2.transpose(-2, -1)
        return torch.bmm(h, w2_t).reshape(x.shape[0], -1)

    out_experts_splits = []
    offset = 0
    for expert_idx, count in enumerate(num_tokens_per_expert_list):
        x_expert = x[offset : offset + count]
        h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        # h shape (tokens_per_expert(varying), dim)
        out_experts_splits.append(h)
        offset += count
    return torch.cat(out_experts_splits, dim=0)


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    h = F.silu(
        torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
    )
    h = h * torch._grouped_mm(
        x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out


class GroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int
        use_grouped_mm: bool = True
        token_dispatcher: LocalTokenDispatcher.Config

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.w1 = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
        self.w3 = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.use_grouped_mm = config.use_grouped_mm
        self.token_dispatcher = config.token_dispatcher.build()
        self._w13_cache: torch.Tensor | None = None
        self._w13_cache_key: tuple | None = None
        self._w2_t_cache: torch.Tensor | None = None
        self._w2_t_cache_key: tuple | None = None

    def _get_cached_w13(
        self,
        w1: torch.Tensor,
        w3: torch.Tensor,
    ) -> torch.Tensor | None:
        if torch.is_grad_enabled():
            return None
        cache_key = (
            w1.untyped_storage().data_ptr(),
            w3.untyped_storage().data_ptr(),
            w1._version,
            w3._version,
            w1.shape,
            w3.shape,
            w1.dtype,
            w3.dtype,
            w1.device,
            w3.device,
        )
        if self._w13_cache is None or self._w13_cache_key != cache_key:
            _record_moe_fastpath("cached_w13_miss")
            self._w13_cache = torch.cat((w1, w3), dim=1)
            self._w13_cache_key = cache_key
        else:
            _record_moe_fastpath("cached_w13_hit")
        return self._w13_cache

    def _get_cached_w2_t(self, w2: torch.Tensor) -> torch.Tensor | None:
        if torch.is_grad_enabled():
            return None
        cache_key = (
            w2.untyped_storage().data_ptr(),
            w2._version,
            w2.shape,
            w2.dtype,
            w2.device,
        )
        if self._w2_t_cache is None or self._w2_t_cache_key != cache_key:
            _record_moe_fastpath("cached_w2_t_miss")
            self._w2_t_cache = w2.transpose(-2, -1).contiguous()
            self._w2_t_cache_key = cache_key
        else:
            _record_moe_fastpath("cached_w2_t_hit")
        return self._w2_t_cache

    def _experts_forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | list[int],
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine."""
        if isinstance(self.w1, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1 = self.w1.to_local()
            # pyrefly: ignore [missing-attribute]
            w2 = self.w2.to_local()
            # pyrefly: ignore [missing-attribute]
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        if self.use_grouped_mm:
            assert isinstance(num_tokens_per_expert, torch.Tensor)
            return _run_experts_grouped_mm(w1, w2, w3, x, num_tokens_per_expert)
        else:
            return _run_experts_for_loop(
                w1,
                w2,
                w3,
                x,
                num_tokens_per_expert,
                self._get_cached_w13(w1, w3),
                self._get_cached_w2_t(w2),
            )

    def forward(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add.

        shared_experts is passed to combine() where it overlaps with the async
        combine all-to-all (NCCL stream) or async DeepEP combine.
        """
        routed_input, num_tokens_local, metadata = self.token_dispatcher.dispatch(
            x, top_scores, selected_experts_indices, num_tokens_per_expert
        )
        num_tokens_per_expert_list = getattr(
            metadata, "num_tokens_per_expert_list", None
        )
        if not self.use_grouped_mm and num_tokens_per_expert_list is not None:
            num_tokens_for_experts = num_tokens_per_expert_list
        else:
            num_tokens_for_experts = num_tokens_local
        routed_output = self._experts_forward(routed_input, num_tokens_for_experts)
        return self.token_dispatcher.combine(routed_output, metadata, x, shared_experts)


class TokenChoiceTopKRouter(Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Optionally supports node-limited (group-limited) routing where experts are divided into groups
    (e.g., by node), and only num_limited_groups groups are considered before selecting top_k experts.
    This reduces cross-node communication in distributed settings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        gate: Linear.Config
        num_expert_groups: int | None = None  # must be a divisor of num_experts
        num_limited_groups: int | None = None
        top_k: int = 1
        score_func: Literal["softmax", "sigmoid"] = "sigmoid"
        route_norm: bool = False
        route_scale: float = 1.0
        _debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()
        self.num_experts = config.num_experts
        self.num_expert_groups = config.num_expert_groups
        self.num_limited_groups = config.num_limited_groups
        self.top_k = config.top_k
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self._debug_force_load_balance = config._debug_force_load_balance

    def _debug_force_load_balance_routing(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns (
            selected_experts_indices [N, K] LongTensor,
            top_scores [N, K] FloatTensor,
            num_tokens_per_expert [num_experts] LongTensor,
        ).
        """
        n_tokens = scores.size(0)
        n_assignments = n_tokens * self.top_k
        # Round-robin indices with exact balance.
        selected_experts_indices = (
            torch.arange(
                n_assignments, device=scores.device, dtype=torch.int64
            ).reshape(n_tokens, self.top_k)
            % self.num_experts
        )
        top_scores = scores.gather(dim=1, index=selected_experts_indices)  # [N,K]
        base_count = n_assignments // self.num_experts
        remainder = n_assignments % self.num_experts
        num_tokens_per_expert = torch.full(
            (self.num_experts,),
            base_count,
            device=scores.device,
            dtype=torch.int64,
        )
        if remainder > 0:
            num_tokens_per_expert[:remainder] += 1
        return selected_experts_indices, top_scores, num_tokens_per_expert

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice: torch.Tensor,
    ) -> torch.Tensor:
        """Select num_limited_groups groups based on group scores,
            and set expert scores in non-selected groups as -inf

        Args:
            scores_for_choice: Router scores with expert_bias (if any), shape (bs*slen, num_experts)

        Returns:
            scores_for_choice: shape (bs*slen, num_experts)
        """
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        assert self.num_expert_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")
        scores_grouped = scores_for_choice.view(
            -1, self.num_expert_groups, experts_per_group
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, False)  # False = selected groups (keep)
        # Mask out experts from non-selected groups
        scores_for_choice = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).view(-1, self.num_experts)

        return scores_for_choice

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        # Compute gate in float32 to help stability of expert load balancing.
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        # scored is already float32 from the autocast above.
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.score_func == "softmax":
            scores = F.softmax(scores, dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
                num_tokens_per_expert,
            ) = self._debug_force_load_balance_routing(scores)
        else:
            scores_for_choice = scores if expert_bias is None else scores + expert_bias
            # Apply node-limited routing if configured
            if self.num_expert_groups is not None:
                scores_for_choice = self._get_node_limited_routing_scores(
                    scores_for_choice
                )
            _, selected_experts_indices = torch.topk(
                scores_for_choice, k=self.top_k, dim=-1, sorted=False
            )

            # top scores shape (bs*slen, top_k)
            # NOTE: The expert_bias is only used for routing. The gating value
            #       top_scores is still derived from the original scores.
            top_scores = scores.gather(dim=1, index=selected_experts_indices)

            # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
            num_tokens_per_expert = torch.histc(
                selected_experts_indices.view(-1),
                bins=self.num_experts,
                min=0,
                max=self.num_experts,
            )

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        return top_scores, selected_experts_indices, num_tokens_per_expert


class MoE(Module):
    """Mixture of Experts layer.

    The forward pass proceeds as:
    1. Router computes expert assignments
    2. GroupedExperts.forward() handles:
       a. dispatch (TokenDispatcher) — reorder tokens by expert assignment.
          With EP, also performs all-to-all communication to send tokens
          to expert-owning ranks.
       b. expert computation
       c. combine (TokenDispatcher) — reverse the dispatch reordering.
          With EP, starts async communication (NCCL all-to-all or DeepEP
          combine), runs shared_experts in parallel, then forces sync
          (scatter_add for NCCL AllToAll, sync_combine for DeepEP) and
          produces final output.
          Without EP (LocalTokenDispatcher), no communication is needed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int = 8
        experts: GroupedExperts.Config
        router: TokenChoiceTopKRouter.Config
        load_balance_coeff: float | None = 1e-3
        shared_experts: FeedForward.Config | None = None

    def __init__(self, config: Config):
        super().__init__()

        num_experts = config.num_experts
        self.experts = config.experts.build()
        self.router = config.router.build()
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = config.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        # Convert DTensor to local tensor for MoE-internal computation.
        # grad_placements=(Partial(),) ensures x.grad is Partial on the tp_mesh
        # in backward, so gradient reduction (reduce-scatter from Partial to
        # Shard(1)) happens once at the MoE boundary rather than being
        # duplicated inside the MoE.
        #
        # Why grad(x) is Partial on the tp_mesh across all parallelism:
        # - TP only / TP+EP with ETP=TP: TP-sharded expert weights (Colwise on
        #   w1/w3, Rowwise on w2) produce Partial output gradients.
        # - TP+EP with ETP=1: each TP rank processes a disjoint token subset
        #   (via sequence-parallel token splitting in AllToAllTokenDispatcher),
        #   so grad(x) is non-zero only at each rank's token positions (Partial).
        #
        # This holds for all MoE components (router.gate, routed experts, shared
        # experts) and regardless of score_before_experts.
        if isinstance(x, DTensor):
            assert (
                x.device_mesh.ndim == 1
            ), f"Expected 1D mesh, got {x.device_mesh.ndim}D mesh"
            assert x.device_mesh.mesh_dim_names == (
                "tp",
            ), f"Expected TP mesh, got mesh_dim_names={x.device_mesh.mesh_dim_names}"
            x = x.to_local(grad_placements=(Partial(),))
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        # top_scores and selected_experts_indices shape (bs*slen, top_k)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
        ) = self.router(x, self.expert_bias)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        out = self.experts(
            x,
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
            shared_experts=self.shared_experts,
        )

        return out.reshape(bs, slen, dim)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        assert isinstance(buffer_device, torch.device)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
