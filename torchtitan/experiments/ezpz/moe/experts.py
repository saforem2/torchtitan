# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz expert compute backends for MoE.

Subclasses upstream `GroupedExperts` to add a `compute_backend` selector
without modifying core. Three backends are supported:

- ``"grouped_mm"`` (default): defer to upstream's ``torch._grouped_mm``
  path. Requires SM90+ on CUDA; on XPU there is no grouped-mm fallback.
- ``"for_loop"``: per-expert ``matmul`` loop. Slower but works on every
  device. Re-vendored from upstream's ``_run_experts_for_loop`` which
  was deleted in pytorch/torchtitan#3308.
- ``"batched_mm_padded"``: pads routed tokens to
  ``[num_experts, max_tokens_per_expert, dim]`` and uses ``torch.bmm``
  for the SwiGLU expert projections. Originally proposed in PR #10.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import GroupedExperts


ExpertComputeBackend = Literal["for_loop", "grouped_mm", "batched_mm_padded"]


def _empty_expert_output(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Zero-token expert call. Preserve a zero-gradient path through w1/w2/w3."""
    out = x.new_empty((0, w2.shape[1]))
    return out + (w1.sum() + w2.sum() + w3.sum() + x.sum()) * 0


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    if num_tokens_per_expert.numel() == 0:
        return _empty_expert_output(w1, w2, w3, x)

    # NOTE: this incurs a device-host sync.
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()

    x_splits = torch.split(
        x,
        split_size_or_sections=num_tokens_per_expert_list,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        out_experts_splits.append(h)
    return torch.cat(out_experts_splits, dim=0)


def _compute_expert_layout(
    counts: torch.Tensor,
    total_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map grouped routed tokens to padded expert rows.

    ``x`` is expected to be grouped by expert, with ``counts`` giving each
    expert's contiguous token count in the same order.
    """
    offsets = counts.cumsum(0) - counts
    expert_indices = torch.repeat_interleave(
        torch.arange(counts.numel(), device=device, dtype=torch.int64),
        counts,
    )
    token_indices_within_expert = torch.arange(
        total_tokens, device=device, dtype=torch.int64
    ) - torch.repeat_interleave(offsets, counts)
    return expert_indices, token_indices_within_expert


def _run_experts_batched_mm_padded(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    counts = num_tokens_per_expert.to(device=x.device, dtype=torch.int64)
    if counts.numel() == 0:
        return _empty_expert_output(w1, w2, w3, x)

    max_tokens = int(counts.max().item())
    if max_tokens == 0:
        return _empty_expert_output(w1, w2, w3, x)

    num_experts = counts.numel()
    total_tokens = x.shape[0]
    device = x.device

    expert_indices, token_indices_within_expert = _compute_expert_layout(
        counts, total_tokens, device
    )

    padded_x = x.new_zeros((num_experts, max_tokens, x.shape[-1]))
    padded_x[expert_indices, token_indices_within_expert] = x

    h = F.silu(torch.bmm(padded_x, w1.transpose(-2, -1)))
    h = h * torch.bmm(padded_x, w3.transpose(-2, -1))
    out_padded = torch.bmm(h, w2.transpose(-2, -1))

    return out_padded[expert_indices, token_indices_within_expert]


class EzpzGroupedExperts(GroupedExperts):
    """GroupedExperts variant that selects between expert compute backends.

    Defers to upstream's grouped-mm path by default. Set ``compute_backend``
    to ``"for_loop"`` on devices without grouped-mm support (e.g. XPU,
    pre-SM90 CUDA), or to ``"batched_mm_padded"`` for the bmm-based path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        compute_backend: ExpertComputeBackend = "grouped_mm"

    def __init__(self, config: Config):
        super().__init__(config)
        self.compute_backend: ExpertComputeBackend = config.compute_backend

    def _experts_forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if self.compute_backend == "grouped_mm":
            return super()._experts_forward(x, num_tokens_per_expert)

        if isinstance(self.w1, DTensor):
            w1 = self.w1.to_local()
            # pyrefly: ignore [missing-attribute]
            w2 = self.w2.to_local()
            # pyrefly: ignore [missing-attribute]
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        if self.compute_backend == "for_loop":
            return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)
        if self.compute_backend == "batched_mm_padded":
            return _run_experts_batched_mm_padded(
                w1, w2, w3, x, num_tokens_per_expert
            )
        raise ValueError(
            f"Unknown expert compute backend: {self.compute_backend!r}"
        )
