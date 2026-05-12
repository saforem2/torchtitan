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
  was deleted in pytorch/torchtitan#3308. Includes an equal-counts
  no-grad fast path that uses a single ``torch.bmm`` over a stacked
  ``w13`` and a transposed ``w2_t``.
- ``"batched_mm_padded"``: pads routed tokens to
  ``[num_experts, max_tokens_per_expert, dim]`` and uses ``torch.bmm``
  for the SwiGLU expert projections. Originally proposed in PR #10.
"""

import os
import threading
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import GroupedExperts


ExpertComputeBackend = Literal["for_loop", "grouped_mm", "batched_mm_padded"]


_FASTPATH_COUNTER_ENV = "EZPZ_MOE_FASTPATH_COUNTERS"
_fastpath_counters: dict[str, int] = {}
_fastpath_lock = threading.Lock()


def _record_moe_fastpath(name: str) -> None:
    """Bump a per-process counter for an expert fast-path event.

    No-op unless ``EZPZ_MOE_FASTPATH_COUNTERS=1`` is set; cheap when off.
    """
    if os.environ.get(_FASTPATH_COUNTER_ENV) != "1":
        return
    with _fastpath_lock:
        _fastpath_counters[name] = _fastpath_counters.get(name, 0) + 1


def get_moe_fastpath_counters() -> dict[str, int]:
    """Snapshot of fast-path event counters (for tests/instrumentation)."""
    with _fastpath_lock:
        return dict(_fastpath_counters)


def reset_moe_fastpath_counters() -> None:
    with _fastpath_lock:
        _fastpath_counters.clear()


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
    num_tokens_per_expert: torch.Tensor | list[int],
    w13: torch.Tensor | None = None,
    w2_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-expert matmul loop, with an equal-counts no-grad fast path.

    If every expert receives the same non-zero token count and we are
    outside an autograd region, fold the SwiGLU into a single ``torch.bmm``
    pair using a stacked ``w13 = cat(w1, w3)`` and a precomputed
    ``w2_t = w2.transpose(-2, -1)``. Caching the weight transforms across
    forward calls is the responsibility of the caller (see
    :class:`EzpzGroupedExperts`).
    """
    if isinstance(num_tokens_per_expert, list):
        num_tokens_per_expert_list = num_tokens_per_expert
    else:
        if num_tokens_per_expert.numel() == 0:
            return _empty_expert_output(w1, w2, w3, x)
        # NOTE: this incurs a device-host sync.
        num_tokens_per_expert_list = num_tokens_per_expert.tolist()

    if not num_tokens_per_expert_list:
        return _empty_expert_output(w1, w2, w3, x)

    # Equal-counts no-grad fast path: single bmm pair over [E, T, dim].
    first_count = num_tokens_per_expert_list[0]
    if (
        first_count > 0
        and all(c == first_count for c in num_tokens_per_expert_list)
        and not torch.is_grad_enabled()
    ):
        _record_moe_fastpath("batched_no_grad_experts")
        num_experts = len(num_tokens_per_expert_list)
        x_grouped = x.view(num_experts, first_count, x.shape[-1])
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
        out_experts_splits.append(h)
        offset += count
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

    Maintains version-keyed caches of the stacked ``w13 = cat(w1, w3)`` and
    transposed ``w2_t``; these get reused by the for-loop equal-counts
    no-grad fast path. Caches are bypassed when autograd is enabled and
    invalidated whenever any of the source weights change ``_version``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        compute_backend: ExpertComputeBackend = "grouped_mm"

    def __init__(self, config: Config):
        super().__init__(config)
        self.compute_backend: ExpertComputeBackend = config.compute_backend
        self._w13_cache: torch.Tensor | None = None
        self._w13_cache_key: tuple | None = None
        self._w2_t_cache: torch.Tensor | None = None
        self._w2_t_cache_key: tuple | None = None

    def _get_cached_w13(
        self, w1: torch.Tensor, w3: torch.Tensor
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
            return _run_experts_for_loop(
                w1,
                w2,
                w3,
                x,
                num_tokens_per_expert,
                self._get_cached_w13(w1, w3),
                self._get_cached_w2_t(w2),
            )
        if self.compute_backend == "batched_mm_padded":
            return _run_experts_batched_mm_padded(
                w1, w2, w3, x, num_tokens_per_expert
            )
        raise ValueError(
            f"Unknown expert compute backend: {self.compute_backend!r}"
        )
