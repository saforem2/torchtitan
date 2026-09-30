# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""XPU fallback for FSDP reduce-scatter copy-in's ``torch._chunk_cat``."""

from __future__ import annotations

import math
import os

import torch

from torchtitan.experiments.ezpz.logging import logger

_PATCHED_ATTR = "_ezpz_xpu_chunk_cat_patched"


def _import_fsdp_collectives():
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives

    return _fsdp_collectives


@torch.no_grad()
def chunk_cat_copy_in(
    tensors: list[torch.Tensor],
    out: torch.Tensor,
    world_size: int,
) -> None:
    """Copy dim-0 chunks into FSDP's padded rank-major output layout."""
    rank_rows = out.view(world_size, -1)
    rank_rows.zero_()
    offset = 0
    for tensor in tensors:
        rows_per_chunk = math.ceil(tensor.size(0) / world_size)
        chunk_numel = rows_per_chunk * tensor[0].numel()
        chunks = torch.chunk(tensor, world_size, dim=0)
        for rank, chunk in enumerate(chunks):
            flat = chunk.reshape(-1)
            rank_rows[rank, offset : offset + flat.numel()].copy_(flat)
        offset += chunk_numel
    if offset != rank_rows.size(1):
        raise RuntimeError(
            f"XPU chunk-cat copy filled {offset} values per rank, "
            f"expected {rank_rows.size(1)}"
        )


def maybe_install_xccl_chunk_cat_workaround() -> bool:
    """Replace FSDP's XPU ``_chunk_cat`` copy-in when explicitly requested."""
    if os.environ.get("EZPZ_XPU_CHUNK_CAT_FALLBACK") != "1":
        return False
    if not (torch.distributed.is_xccl_available() and torch.xpu.is_available()):
        return False

    _fsdp_collectives = _import_fsdp_collectives()

    if getattr(_fsdp_collectives, _PATCHED_ATTR, False):
        return True

    original = _fsdp_collectives.foreach_reduce_scatter_copy_in

    def _fallback(
        unsharded_grads: list[torch.Tensor],
        reduce_scatter_input: torch.Tensor,
        world_size: int,
    ) -> None:
        if reduce_scatter_input.device.type == "xpu":
            chunk_cat_copy_in(unsharded_grads, reduce_scatter_input, world_size)
        else:
            original(unsharded_grads, reduce_scatter_input, world_size)

    _fsdp_collectives.foreach_reduce_scatter_copy_in = _fallback
    setattr(_fsdp_collectives, f"{_PATCHED_ATTR}_fn", _fallback)
    setattr(_fsdp_collectives, _PATCHED_ATTR, True)
    logger.info("Installed XPU FSDP chunk-cat copy-in fallback")
    return True
