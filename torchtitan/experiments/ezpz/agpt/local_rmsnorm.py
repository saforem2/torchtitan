# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local-shard RMSNorm for agpt QK-Norm under tensor parallelism.

Shape suffix legend (this file):
  B = batch, L = sequence length, N = num heads, H = per-head dim (head_dim).

Why this exists
---------------
agpt QK-Norm applies an ``RMSNorm(head_dim)`` to ``xq_BLNH`` / ``xk_BLNH``
(4-D, ``[B, L, N, H]``) *before* RoPE. Under tensor parallelism the heads
(dim 2) are sharded on the TP axis (``Shard(2)``), while the normalized dim
(head_dim, dim 3) and the norm ``weight`` (size head_dim) are both
unsharded / ``Replicate`` on TP.

With ``spmd_backend="default"`` (what the ezpz trainer runs -- see
``torchtitan/experiments/ezpz/trainer.py``), these activations are real
``torch.distributed.tensor.DTensor`` objects at runtime. DTensor's native
RMSNorm backward through a ``Shard(2)`` 4-D tensor loses its device/mesh when
the attention-block forward is RECOMPUTED during backward under
activation-checkpointing = full, at TP=4 (``RuntimeError: tensor does not have
a device`` + a DeviceMesh-in-saved-tensors assert on peer ranks). TP=2 happens
to avoid it; TP=4 hits it.

The fix
-------
Compute the RMSNorm on the LOCAL shard, bypassing DTensor's native RMSNorm
backward. This is NUMERICALLY BIT-IDENTICAL to the native path: the reduction
runs over head_dim, which is unsharded, and the ``weight`` is replicated, so
each rank already holds the complete row + the full weight -- local RMS ==
global RMS. Only the (crashing) DTensor autograd node is replaced; the arithmetic
is the same fused ``F.rms_norm`` the base module calls.

Gradient placement semantics (the load-bearing part)
----------------------------------------------------
- Activation ``x``: RMSNorm normalizes each head independently (reduction only
  over the unsharded head_dim), so ``dL/dx`` has the SAME placement as ``x``
  (``Shard(2)`` on TP, batch/seq shards on DP/CP). We pass
  ``grad_placements=x.placements`` explicitly so the reconstructed DTensor
  gradient keeps that sharding.
- Weight ``w`` (``Replicate`` on TP): each rank uses the full weight but only
  its local heads, so its local gradient is a PARTIAL sum over that rank's
  heads. The true gradient is the sum across TP ranks. Under the default
  backend the weight DTensor is on a TP-ONLY mesh (DP/CP out-of-band), so we
  pass ``grad_placements=[Partial()]`` (one entry, the TP axis) to
  ``w.to_local(...)``; the reconstructed weight gradient is then ``Partial``
  and FSDP2 all-reduces it to ``Replicate`` on the TP group. This is the SAME
  ``Partial -> Replicate`` reduction DTensor's native RMSNorm backward would
  have performed for the replicated weight; we just declare it explicitly
  because we bypassed that backward. The DP gradient reduction is FSDP's own
  reduce-scatter, out-of-band and independent of this call. See the inline
  note at the ``weight.to_local`` call for why this must not become multi-axis.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Partial, Shard

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common import RMSNorm


class LocalShardRMSNorm(RMSNorm):
    """RMSNorm computed on the local shard, for TP-sharded QK-Norm.

    Drop-in replacement for ``RMSNorm`` (same ``Config`` surface). Only the
    ``forward`` differs: when the input is a TP-sharded ``DTensor`` it runs the
    norm on the local shard and re-wraps the result, avoiding the native
    DTensor RMSNorm backward that crashes on a ``Shard(2)`` 4-D tensor under
    AC=full recompute at TP=4. For a plain tensor (TP=1) it is identical to the
    base module.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, DTensor):
            # Plain tensor. Under spmd_types this activation is a local tensor
            # carrying an SPMD type; run inside spmd.local() and re-anchor the
            # output type (heads on TP), mirroring local_qkv_head_split in
            # torchtitan/models/common/attention.py. spmd.local() and the
            # assert are no-ops at runtime when type checking is off, so the
            # default-backend TP=1 case is just the base RMSNorm forward.
            if get_spmd_backend() == "spmd_types":
                with spmd.local():
                    out = F.rms_norm(
                        x, self.normalized_shape, self.weight, self.eps
                    )
                    spmd.assert_type(
                        out, spmd.V, spmd.PartitionSpec("dp", "cp", "tp", None)
                    )
                return out
            return super().forward(x)

        # DTensor path (default backend, TP>1). The normalized
        # dims (the trailing len(normalized_shape) dims -- head_dim here) MUST
        # be unsharded for local RMS == global RMS; otherwise a rank would see
        # a partial row and silently compute a wrong norm. Validate explicitly.
        ndim = x.ndim
        first_norm_dim = ndim - len(self.normalized_shape)
        for placement in x.placements:
            if isinstance(placement, Shard):
                shard_dim = placement.dim if placement.dim >= 0 else ndim + placement.dim
                assert shard_dim < first_norm_dim, (
                    "LocalShardRMSNorm requires the normalized dims to be "
                    f"unsharded, but placement {placement} shards normalized "
                    f"dim {shard_dim} of a {ndim}-D input."
                )

        mesh = x.device_mesh
        placements = x.placements

        # Activation gradient keeps the input's own placement (norm is
        # elementwise along the sharded heads dim; reduction is over the
        # unsharded head_dim).
        x_local = x.to_local(grad_placements=placements)

        # Under TP the activation is a DTensor, so the weight was TP-distributed
        # (state_shardings) and FSDP-wrapped -> it must also be a DTensor. A
        # plain-tensor weight here would silently DROP the Partial-on-TP weight
        # gradient reduction (each rank sees only its local heads), producing
        # wrong grads. Fail loudly instead. (elementwise_affine is always True
        # for QK-Norm: param_init sets weight to ones_.)
        weight = self.weight
        assert isinstance(weight, DTensor), (
            "LocalShardRMSNorm: activation is a DTensor (TP on) but weight is "
            f"a plain {type(weight).__name__}; cannot reduce the Partial-on-TP "
            "weight gradient. Weight must be a DTensor under TP."
        )
        # IMPORTANT (backend-specific): under spmd_backend="default" the weight
        # DTensor lives on a TP-ONLY mesh -- the DP/CP axes are filtered out as
        # "out-of-band" (parallel_dims.py resolve_mesh -> ("tp","ep")), so
        # weight.placements == (Replicate(),), len == 1. Thus the list below is
        # exactly [Partial()] on the TP axis ONLY. Each TP rank's local dL/dw is
        # a partial sum over its disjoint local heads; declaring Partial-on-TP
        # makes FSDP2's _get_grad_inner_tensor all-reduce(SUM) it TP -> Replicate
        # (the reduction native RMSNorm backward would have done). The DP/FSDP
        # gradient reduction is handled ENTIRELY out-of-band by FSDP's own
        # reduce-scatter on the plain local grad -- it is NOT in weight.placements
        # and must NOT be declared here (doing so would double-count DP). Do not
        # "simplify" this to an explicit multi-axis list, and do not assume this
        # holds under full_dtensor (there the weight would be a genuine multi-axis
        # DTensor and this would need re-derivation).
        weight_local = weight.to_local(
            grad_placements=[Partial()] * len(weight.placements)
        )

        out_local = F.rms_norm(
            x_local, self.normalized_shape, weight_local, self.eps
        )

        # Re-anchor the output as a DTensor with the same placement as the
        # input; the output gradient (from the recomputed backward) flows back
        # through this boundary with matching placement.
        return DTensor.from_local(out_local, mesh, placements, run_check=False)
