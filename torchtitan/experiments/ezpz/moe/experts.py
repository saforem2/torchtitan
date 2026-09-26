# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz expert compute backends for MoE.

Provides alternate compute kernels for upstream ``GroupedLinear`` weights.
The backend selector lives on :class:`EzpzRoutedExperts`, matching the current
upstream ownership boundary. Current HEAD registers six backends:

- ``"grouped_mm"`` (default): defer to upstream's ``torch._grouped_mm``
  path. Requires SM90+ on CUDA; on XPU there is no grouped-mm fallback.
- ``"for_loop"``: portable per-expert reference implementation.
- ``"bmm"`` and ``"bmm_nodrop"``: padded batched-matmul variants, with and
  without capacity dropping respectively.
- ``"aurora_sycl"`` and ``"aurora_full_sonic"``: Intel XPU/SYCL backends.
  Sonic also owns expert-parallel dispatch/combine and therefore requires
  routing tensors and an EP process group supplied by ``EzpzRoutedExperts``.
"""

import math
import os
from typing import Literal

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

ExpertComputeBackend = Literal[
    "for_loop", "grouped_mm", "bmm", "bmm_nodrop", "aurora_sycl", "aurora_full_sonic"
]


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


def _run_experts_bmm_nodrop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Batched torch.bmm expert compute that drops NO tokens.

    Same batched-GEMM idea as `_run_experts_bmm`, but the padded buffer is
    sized to the largest actual per-expert count rather than to a capacity
    derived from `capacity_factor`. Nothing is dropped, so this is the
    numerically exact choice; the cost is a dynamic leading dimension
    (`max_tokens` moves with the routing each step), which makes it less
    friendly to torch.compile than the fixed-capacity `bmm` path.

    Ported from samuelwheeler/torchtitan feature/aurora-moe-training, where
    it is `_run_experts_batched_mm_padded`. Renamed here so the difference
    that matters -- drop vs no-drop -- is visible at the call site instead
    of buried in the implementation. Adapted to our post-#3425 parameter
    names (w1_EFD/w2_EDF/w3_EFD are unpacked by the caller).

    `x` is assumed sorted by expert, matching the
    `EzpzGroupedExperts.forward` contract.
    """
    counts = num_tokens_per_expert.to(device=x.device, dtype=torch.int64)
    if counts.numel() == 0:
        return _empty_expert_output(w1, w2, w3, x)

    max_tokens = int(counts.max().item())
    if max_tokens == 0:
        return _empty_expert_output(w1, w2, w3, x)

    num_experts = counts.numel()
    total_tokens = x.shape[0]
    device = x.device

    offsets = counts.cumsum(0) - counts
    expert_indices = torch.repeat_interleave(
        torch.arange(num_experts, device=device, dtype=torch.int64),
        counts,
    )
    token_indices_within_expert = torch.arange(
        total_tokens, device=device, dtype=torch.int64
    ) - torch.repeat_interleave(offsets, counts)

    padded_x = x.new_zeros((num_experts, max_tokens, x.shape[-1]))
    padded_x[expert_indices, token_indices_within_expert] = x

    # Compute in bf16 and return in the caller's dtype, matching
    # _run_experts_for_loop and _run_experts_bmm. Without this the GEMMs
    # raise "expected scalar type Float but found BFloat16" whenever the
    # parameters and the activations disagree -- which they do under
    # mixed precision, where params stay fp32 and x arrives bf16. The sww
    # original omitted the casts because on that branch the caller had
    # already aligned the dtypes.
    padded_x_bf16 = padded_x.bfloat16()
    w1_bf16 = w1.bfloat16()
    w2_bf16 = w2.bfloat16()
    w3_bf16 = w3.bfloat16()

    h = F.silu(torch.bmm(padded_x_bf16, w1_bf16.transpose(-2, -1)))
    h = h * torch.bmm(padded_x_bf16, w3_bf16.transpose(-2, -1))
    out_padded = torch.bmm(h, w2_bf16.transpose(-2, -1))

    return out_padded[expert_indices, token_indices_within_expert].type_as(x)


def _run_experts_aurora_sycl(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Exact compact expert GEMMs via the optional aurora-moe SYCL kernels.

        Routing and score application stay in the token dispatcher; this only
        replaces the expert GEMMs. Ported from samuelwheeler/torchtitan
        feature/aurora-moe-training, where it is the `aurora_sycl` backend.

        Uses the standard w1/w2/w3 parameters, so it is checkpoint-transparent:
        unlike that branch's scattermoe and aurora_full_* backends, selecting
        this one does not change any registered parameter name.

        The kernels are JIT-compiled by torch.utils.cpp_extension.load on first
        call, which needs oneMKL headers and a versioned
        libmkl_sycl_blas.so.* (found under $MKLROOT, default
        /opt/aurora/<ver>/oneapi/mkl/<ver>). Set AURORA_MOE_SYCL_BUILD_DIR to
        cache the build across jobs -- the first call is slow. Imported lazily
        so every other backend stays usable without aurora-moe installed.

    WORKS, BUT ONLY ON A COHERENT STACK. The venv, the compiler and the
        oneMKL must all come from the SAME Aurora release. Four measurements on
        real compute nodes, 2026-09-15/16:

          8829416  next-eval (TEST bkc). frameworks/2026.1.0 module python 3.12
                   + MKLROOT 26.181.0. Coherent -> WORKS.
          8829454  next-eval (TEST bkc). Our yeeted /tmp/.venv (py3.14,
                   2.13.0.dev20260428+xpu) + MKLROOT 26.26.0. Mismatched ->
                   UR_RESULT_ERROR_UNINITIALIZED (37).
          8829790  next-eval (TEST bkc). Same venv + MKLROOT 26.181.0. Still
                   mismatched, because that venv's torch is prod-era while the
                   image and compiler are test-era ->
                   oneapi::mkl::blas::gemm_bf16bf16bf16: unsupported device.
          8831582  debug (PROD bkc). Same venv + MKLROOT 26.26.0/2025.3 + icpx
                   2025.3.2. All three from one release -> WORKS.

        So the discriminator is coherence, not the venv and not the oneMKL
        version on its own. An earlier revision of this docstring said the
        backend "does not work under the production training venv"; that was
        measured only against test-bkc images and was wrong.

        Numerics on the coherent stack (8831582, vs the for_loop reference,
        bf16, 4 experts with one empty):

          rel_err 4.4e-03, 14 of 1280 elements (1.09%) outside atol/rtol 0.05,
          rowwise cosine min 0.999993

        That is bf16 rounding, not a kernel defect. A single bf16 matmul on this
        shape carries ~2e-03 and this is a three-matmul SwiGLU chain; the
        mismatched elements are large-magnitude ones where the absolute
        difference (max 8.0 against a reference max near 2000) is relatively
        small. A structurally wrong kernel would break the cosine, not a
        handful of elements. For contrast bmm_nodrop scored exactly 0.0 with 0
        mismatches in the same job on the same device.

        Historical note: aurora_moe previously hardcoded a
        libmkl_sycl_blas.so.5 check in three files (one_mkl_ops,
        one_mkl_grouped_gemm, one_mkl_exact_expert_gemm), so it refused the
        .so.6 that the 26.181.0 image ships. Fixed in 67d4f262f by accepting the
        installed versioned libmkl_sycl_blas.so.* soname.
    """
    try:
        from aurora_moe.torchtitan_experts import torchtitan_exact_experts
    except ImportError as error:
        raise ImportError(
            "the aurora_sycl expert backend requires the aurora_moe package "
            "(torchtitan/experiments/ezpz/vendor/aurora_moe_dropin/src) on PYTHONPATH"
        ) from error
    return torchtitan_exact_experts(w1, w2, w3, x, num_tokens_per_expert)


def _sonic_weight_layouts(
    w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert TorchTitan [E,F,D]/[E,D,F] weights to Sonic layouts."""
    return (
        w3.transpose(1, 2).contiguous(),  # up: [E, D, F]
        w1.transpose(1, 2).contiguous(),  # gate: [E, D, F]
        w2.transpose(1, 2).contiguous(),  # down: [E, F, D]
    )


def _run_experts_aurora_full_sonic(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    *,
    topk_scores: torch.Tensor | None,
    topk_indices: torch.Tensor | None,
    ep_mesh: object | None,
) -> torch.Tensor:
    """aurora_moe's expert-parallel Sonic backend.

    Unlike ``aurora_sycl`` -- a pure function over already-routed rows -- this
    one owns the dispatch: it takes the router's decision and performs its own
    expert-parallel all-to-all. That is why it needs ``topk_scores``,
    ``topk_indices`` and a mesh, none of which the
    ``forward(x, num_tokens_per_expert)`` contract carries.

    Feasibility was established before this was written (see
    ``docs/experiments/sonic-port-feasibility.md``):

    - EP>1 runs on torch 2.15 (job 8836014); the ``ur_die`` abort recorded on
      earlier stacks does not reproduce.
    - torchtitan's EP rank ordering matches aurora_moe's documented
      ``rank = dp_rank * EP + ep_rank`` at EP=2, 8 AND 12 (job 8836277), so no
      remap is needed. Had they disagreed the all-to-all would have crossed
      ranks silently -- wrong gradients, no error.

    ``mesh`` is aurora_moe's ``ParallelMesh``, NOT a torch ``DeviceMesh``: the
    kernel indexes ``mesh.group_size["ep_dispatch"]``. We build one from the
    EP process group torchtitan already made.
    """
    if topk_scores is None or topk_indices is None:
        raise ValueError(
            "the aurora_full_sonic expert backend needs the router's "
            "topk_scores/topk_indices. They are available in "
            "RoutedExperts.forward but core drops them before calling "
            "inner_experts (models/common/moe.py:163); use EzpzRoutedExperts, "
            "which forwards them."
        )
    if not isinstance(ep_mesh, DeviceMesh):
        raise ValueError("the aurora_full_sonic expert backend requires a real EP mesh")
    ep_size = ep_mesh.size()
    if ep_size <= 1:
        raise ValueError(
            "the aurora_full_sonic expert backend requires EP size greater than 1"
        )

    total_experts = int(num_tokens_per_expert.numel())
    if total_experts % ep_size:
        raise ValueError(
            f"global expert count {total_experts} is not divisible by EP size "
            f"{ep_size}; the Sonic kernel assumes an even per-rank split"
        )
    local_count = total_experts // ep_size
    weight_counts = (w1.shape[0], w2.shape[0], w3.shape[0])
    if weight_counts != (local_count,) * 3:
        raise ValueError(
            "local expert weight count must equal global expert count divided "
            f"by EP size ({local_count}), got {weight_counts}"
        )

    if topk_scores.dtype != torch.float32:
        raise ValueError(
            "topk_scores must have dtype float32 from the core router, "
            f"got {topk_scores.dtype}"
        )
    if topk_indices.dtype != torch.int64:
        raise ValueError(
            "topk_indices must have dtype int64 for aurora_full_sonic, "
            f"got {topk_indices.dtype}"
        )
    if num_tokens_per_expert.dtype != torch.int64:
        raise ValueError(
            "num_tokens_per_expert must have dtype int64, "
            f"got {num_tokens_per_expert.dtype}"
        )
    if topk_scores.device != x.device or topk_indices.device != x.device:
        raise ValueError("routing tensors must be on the same device as x")
    if num_tokens_per_expert.device != x.device:
        raise ValueError("num_tokens_per_expert must be on the same device as x")
    if (
        topk_scores.ndim != 2
        or topk_indices.shape != topk_scores.shape
        or topk_scores.shape[0] != x.shape[0]
    ):
        raise ValueError(
            "topk_scores and topk_indices must have matching [tokens, top_k] "
            "shapes whose token dimension matches x"
        )
    if topk_scores.shape[1] == 0:
        raise ValueError("routing tensors must have top_k greater than zero")
    if num_tokens_per_expert.ndim != 1:
        raise ValueError("num_tokens_per_expert must have shape [global_experts]")
    if bool((num_tokens_per_expert < 0).any()):
        raise ValueError("num_tokens_per_expert values must be non-negative")
    if int(num_tokens_per_expert.sum()) != topk_indices.numel():
        raise ValueError(
            "num_tokens_per_expert must sum to the number of routing entries"
        )
    if topk_indices.numel() and (
        bool((topk_indices < 0).any()) or bool((topk_indices >= total_experts).any())
    ):
        raise ValueError(f"topk_indices values must be in [0, {total_experts})")

    # sycl_sonic refuses to run without the compact alltoallv transport.
    if os.environ.get("AURORA_MOE_ALLTOALLV") != "1":
        raise RuntimeError(
            "the aurora_full_sonic backend requires AURORA_MOE_ALLTOALLV=1 "
            "(aurora_moe/_core.py:3727 raises otherwise)"
        )

    if x.dtype != torch.bfloat16:
        raise ValueError(
            "the aurora_full_sonic backend requires BF16 activations, got "
            f"{x.dtype}. Casting x here would hide a real dtype problem in the "
            "model, unlike the router scores, which are a routing decision."
        )

    topk_scores = topk_scores.to(torch.bfloat16)

    try:
        from aurora_moe._core import _routed_moe
        from aurora_moe.distributed import MoEProcessGroups, ParallelMesh
    except ImportError as error:
        raise ImportError(
            "the aurora_full_sonic expert backend requires the aurora_moe "
            "package (torchtitan/experiments/ezpz/vendor/aurora_moe_dropin/src) "
            "on PYTHONPATH"
        ) from error

    group = ep_mesh.get_group()
    mesh = ParallelMesh(MoEProcessGroups(ep_dispatch=group), x.device)

    # local_expert_ids only needs the right LENGTH: _routed_moe takes
    # len(local_expert_ids) as local_count, and the kernel then does
    #     dest      = topk_indices // local_count
    #     local_ids = topk_indices - dest * local_count
    # (_core.py:1682). So local_count must be experts PER RANK, and the kernel
    # shards the GLOBAL topk_indices itself. Passing the full expert count made
    # local_ids exceed the per-rank range and the kernel raised
    # "local_ids must be in [0, num_experts)" (job 8836331).
    local_expert_ids = list(range(local_count))
    # TorchTitan stores the projections as w1/w3=[E, F, D] and
    # w2=[E, D, F], while Sonic's ragged kernel expects
    # up/gate=[E, D, F] and down=[E, F, D]. The debug model has D == F,
    # which masked this contract mismatch; production 10B/2B does not.
    # Transpose as views and materialize contiguous layouts required by Sonic.
    up, gate, down = _sonic_weight_layouts(w1, w2, w3)
    return _routed_moe(
        x,
        topk_scores,
        topk_indices,
        local_expert_ids,
        mesh,
        up,
        gate,
        down,
        expert_backend="sycl_sonic",
    )
