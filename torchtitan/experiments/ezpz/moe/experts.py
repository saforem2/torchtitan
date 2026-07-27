# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz expert compute backends for MoE.

Subclasses upstream `GroupedExperts` to add a `compute_backend` selector
without modifying core. Three backends are supported here:

- ``"grouped_mm"`` (default): defer to upstream's ``torch._grouped_mm``
  path. Requires SM90+ on CUDA; on XPU there is no grouped-mm fallback.
- ``"for_loop"``: per-expert ``matmul`` loop. Slower but works on every
  device. Re-vendored from upstream's ``_run_experts_for_loop`` which
  was deleted in pytorch/torchtitan#3308.
- ``"bmm"``: batched ``torch.bmm`` over a padded ``(E, capacity, D)``
  buffer. On XPU ``torch.bmm`` lowers to a oneDNN batched matmul (a real
  grouped-GEMM equivalent), and unlike ``"for_loop"`` this path has
  static shapes (given a capacity) and is compile-friendly.
"""

from dataclasses import dataclass
import math
import os
from typing import Literal

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

# GroupedExperts comes straight from upstream — the local `.moe` copy
# was deleted because it was byte-identical to the upstream module. See
# moe/__init__.py for the same import-re-route.
from torchtitan.models.common.moe import GroupedExperts


ExpertComputeBackend = Literal["for_loop", "grouped_mm", "bmm"]


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}


def _empty_expert_output(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Zero-token expert call. Preserve a zero-gradient path through w1/w2/w3."""
    out = x.new_empty((0, w2.shape[1]))
    return out + (w1.sum() + w2.sum() + w3.sum() + x.sum()) * 0


@torch.compiler.disable
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

    if _env_flag_enabled("TT_MOE_EXPERT_PREALLOC_OUTPUT"):
        out = x.new_empty((x.shape[0], w2.shape[1]))
        offset = 0
        for expert_idx, num_tokens in enumerate(num_tokens_per_expert_list):
            if num_tokens == 0:
                continue
            x_expert = x[offset : offset + num_tokens]
            x_expert_bf16 = x_expert.bfloat16()
            h = F.silu(
                torch.matmul(
                    x_expert_bf16,
                    w1[expert_idx].bfloat16().transpose(-2, -1),
                )
            )
            gate = torch.matmul(
                x_expert_bf16,
                w3[expert_idx].bfloat16().transpose(-2, -1),
            )
            if _env_flag_enabled("TT_MOE_EXPERT_INPLACE_GATE_MUL"):
                h.mul_(gate)
            else:
                h = h * gate
            h = torch.matmul(h, w2[expert_idx].bfloat16().transpose(-2, -1))
            out[offset : offset + num_tokens].copy_(h.type_as(x))
            offset += num_tokens
        return out

    out_experts_splits = []
    offset = 0
    for expert_idx, num_tokens in enumerate(num_tokens_per_expert_list):
        if num_tokens == 0:
            continue
        x_expert = x[offset : offset + num_tokens]
        x_expert_bf16 = x_expert.bfloat16()
        h = F.silu(
            torch.matmul(
                x_expert_bf16,
                w1[expert_idx].bfloat16().transpose(-2, -1),
            )
        )
        gate = torch.matmul(
            x_expert_bf16,
            w3[expert_idx].bfloat16().transpose(-2, -1),
        )
        if _env_flag_enabled("TT_MOE_EXPERT_INPLACE_GATE_MUL"):
            h.mul_(gate)
        else:
            h = h * gate
        h = torch.matmul(h, w2[expert_idx].bfloat16().transpose(-2, -1))
        out_experts_splits.append(h.type_as(x))
        offset += num_tokens
    if len(out_experts_splits) == 0:
        return _empty_expert_output(w1, w2, w3, x)
    return torch.cat(out_experts_splits, dim=0)


def _run_experts_bmm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    capacity_factor: float,
) -> torch.Tensor:
    """Batched torch.bmm expert compute over a padded (E, capacity, D) buffer.

    Unlike `_run_experts_for_loop`, this function is NOT decorated with
    @torch.compiler.disable: given a fixed capacity, every op below has a
    static shape, so inductor can capture the whole thing. On XPU
    `torch.bmm` lowers to a oneDNN batched matmul -- a real grouped-GEMM
    equivalent -- unlike the serial per-expert `torch.matmul` loop.

    Tokens assigned to an expert beyond its capacity are dropped: their
    output rows are zero, matching standard capacity-limited MoE dispatch
    semantics. `x` is assumed sorted by expert, as required by the
    `EzpzGroupedExperts.forward` contract.
    """
    E = w1.shape[0]
    R = x.shape[0]
    D = x.shape[-1]
    if E == 0 or R == 0:
        return _empty_expert_output(w1, w2, w3, x)

    counts = num_tokens_per_expert

    # Max tokens any one expert may keep. A Python int derived from static
    # shapes (E, R) and the capacity_factor config value -- not from a
    # device-host sync on token counts. Given `cap`, every tensor built
    # below has a static shape.
    cap = max(1, math.ceil(R / E * capacity_factor))
    cap = min(cap, R)

    # offsets[e]: start row of expert e's token run within the sorted `x`.
    offsets = torch.cumsum(counts, dim=0) - counts

    # expert_ids[r] / pos_in_expert[r]: which expert row r of `x` belongs
    # to, and its 0-indexed position within that expert's run. Passing
    # output_size=R lets repeat_interleave skip the internal device-host
    # sync it would otherwise need to size its output from `counts`.
    expert_ids = torch.repeat_interleave(
        torch.arange(E, device=x.device), counts, output_size=R
    )
    pos_in_expert = torch.arange(R, device=x.device) - offsets[expert_ids]
    valid = pos_in_expert < cap

    # Dense (expert, capacity-slot) index for each of the R sorted tokens.
    # Valid indices are unique per token (pos_in_expert is unique within
    # an expert's run), so they never collide with each other. Tokens
    # past capacity are redirected to a shared scratch row at E * cap,
    # appended past the real (E, cap) grid and dropped/zeroed below --
    # colliding writes to that scratch row are harmless since it is
    # discarded.
    flat_idx = torch.where(
        valid,
        expert_ids * cap + pos_in_expert,
        torch.full_like(expert_ids, E * cap),
    )

    x_ECD_flat = x.new_zeros((E * cap + 1, D))
    x_ECD_flat.index_copy_(0, flat_idx, x)
    x_ECD = x_ECD_flat[: E * cap].view(E, cap, D)

    x_ECD_bf16 = x_ECD.bfloat16()
    w1_bf16 = w1.bfloat16()
    w2_bf16 = w2.bfloat16()
    w3_bf16 = w3.bfloat16()

    h = F.silu(torch.bmm(x_ECD_bf16, w1_bf16.transpose(-2, -1))) * torch.bmm(
        x_ECD_bf16, w3_bf16.transpose(-2, -1)
    )
    out_ECD = torch.bmm(h, w2_bf16.transpose(-2, -1))

    # Gather back to (R, D) input order. The appended zero row makes the
    # scratch index (E * cap) resolve to a zero output for dropped tokens.
    out_flat = out_ECD.reshape(E * cap, D)
    out_flat_padded = torch.cat([out_flat, out_flat.new_zeros((1, D))], dim=0)
    out_RD = out_flat_padded.index_select(0, flat_idx)
    return out_RD.to(x.dtype)


class EzpzGroupedExperts(GroupedExperts):
    """GroupedExperts variant that selects between expert compute backends.

    Defers to upstream's grouped-mm path by default. Set ``compute_backend``
    to ``"for_loop"`` or ``"bmm"`` on devices without grouped-mm support
    (e.g. XPU, pre-SM90 CUDA); ``"bmm"`` is the compile-friendly, batched
    alternative to the serial ``"for_loop"`` path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        compute_backend: ExpertComputeBackend = "grouped_mm"
        # Only used by the "bmm" backend: bounds the padded per-expert
        # capacity as `ceil(R / E * capacity_factor)`, where R is the
        # total routed token count and E the number of experts. Tokens
        # assigned to an expert beyond capacity are dropped (zeroed).
        capacity_factor: float = 1.25

    def __init__(self, config: Config):
        super().__init__(config)
        self.compute_backend: ExpertComputeBackend = config.compute_backend
        self.capacity_factor: float = config.capacity_factor

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: this method is intentionally NOT marked with
        # @torch.compiler.disable. The grouped_mm and bmm paths delegate to
        # compile-friendly implementations (upstream's `super().forward()`
        # and `_run_experts_bmm` respectively), and we want torch.compile
        # to see them. Only the for-loop path is opted out of compile, via
        # the module-level @torch.compiler.disable decorator on
        # `_run_experts_for_loop`.
        if self.compute_backend == "grouped_mm":
            return super().forward(x, num_tokens_per_expert)

        # Param names use Shazeer shape-suffix style post upstream PR #3425
        # (41st sync): w1_EFD, w2_EDF, w3_EFD.
        if isinstance(self.w1_EFD, DTensor):
            w1 = self.w1_EFD.to_local()
            # pyrefly: ignore [missing-attribute]
            w2 = self.w2_EDF.to_local()
            # pyrefly: ignore [missing-attribute]
            w3 = self.w3_EFD.to_local()
        else:
            w1 = self.w1_EFD
            w2 = self.w2_EDF
            w3 = self.w3_EFD

        if self.compute_backend == "for_loop":
            if _env_flag_enabled("TT_MOE_CHECKPOINT_EXPERTS"):
                return checkpoint(
                    _run_experts_for_loop,
                    w1,
                    w2,
                    w3,
                    x,
                    num_tokens_per_expert,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)
        if self.compute_backend == "bmm":
            return _run_experts_bmm(
                w1, w2, w3, x, num_tokens_per_expert, self.capacity_factor
            )
        raise ValueError(f"Unknown expert compute backend: {self.compute_backend!r}")
