# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Norms-only fp32 master weights (ablation arm B).

Why this exists
---------------
`docs/reference/guides/training-dtype-bf16-norm-freeze.md` established that with
`training.dtype = "bfloat16"` the master parameter copy is bf16, and every
`RMSNorm.weight` (init 1.0, bf16 ULP 7.8e-3) is frozen because the per-step
optimizer update (~1.6e-5) rounds to zero. The shipped fix was
`training.dtype = "float32"`, i.e. an fp32 master for EVERY parameter.

The open question that this module exists to answer empirically: was the
full-fp32 master over-broad? Non-norm parameters live at std ~0.005-0.02
where the bf16 ULP is ~3.8e-5, close enough to the update size that updates
do land. If a bf16 master is adequate for them, a NORMS-ONLY fp32 master
would recover the whole benefit at ~1/100th the extra master memory.

This module implements exactly that arm: bf16 master everywhere, fp32
master for the RMSNorm affine weights only.

Why it is not a one-line config change
--------------------------------------
FSDP2 requires a UNIFORM original parameter dtype within a single
`fully_shard` group:

    AssertionError: FSDP expects uniform original parameter dtype but got
    {torch.bfloat16, torch.float32}

(`torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py::_init_mp_dtypes`).

So mixing an fp32 `attention_norm.weight` into an otherwise-bf16 transformer
block group is rejected outright. Norms-only fp32 therefore requires
RE-GROUPING: the fp32 norm parameters must be pulled into their own
`fully_shard` group(s), separate from the bf16 bulk. `promote_norms_to_fp32`
does the dtype cast; `collect_norm_modules` supplies the module list that
`apply_fsdp` needs in order to shard them separately.

Cost of the regrouping (this is a real, reportable cost of arm B):
the norm parameters no longer ride along with the block they belong to, so
they get their own all-gather / reshard traffic on a separate schedule
instead of being bundled into the block's single flat parameter. Full-fp32
master (arm C) needs none of this -- the group structure is untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchtitan.tools.logging import logger

__all__ = ["collect_norm_modules", "promote_norms_to_fp32", "is_norm_module"]


def is_norm_module(module: nn.Module) -> bool:
    """True for the affine normalization layers whose weight inits at 1.0.

    Matches `nn.RMSNorm` (which `torchtitan.models.common.RMSNorm` and the
    ezpz `LocalShardRMSNorm` both subclass) plus `nn.LayerNorm`, and only
    when the layer actually owns an affine weight.
    """
    if not isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
        return False
    return getattr(module, "weight", None) is not None


def collect_norm_modules(model: nn.Module) -> list[nn.Module]:
    """Every affine norm submodule of `model`, in `named_modules` order."""
    return [m for _, m in model.named_modules() if is_norm_module(m)]


def promote_norms_to_fp32(model: nn.Module) -> list[nn.Module]:
    """Cast every affine norm weight to fp32 in place; return those modules.

    Must be called BEFORE `fully_shard`, because FSDP2 snapshots
    `orig_dtype` from the parameter at wrap time. Safe on meta device (the
    trainer builds the model under `torch.device("meta")`); `to_empty` and
    the subsequent `init_states` both preserve the parameter dtype.

    The returned modules must be handed to `apply_fsdp` so they are wrapped
    in their own group -- see the module docstring for why mixing them into
    a bf16 group raises.
    """
    norm_modules = collect_norm_modules(model)
    promoted = 0
    for module in norm_modules:
        weight = module.weight
        if weight.dtype == torch.float32:
            continue
        module.weight = nn.Parameter(
            weight.detach().to(torch.float32), requires_grad=weight.requires_grad
        )
        promoted += 1
    logger.info(
        "norms-only fp32 master: promoted %d/%d affine norm weights to float32",
        promoted,
        len(norm_modules),
    )
    return norm_modules
