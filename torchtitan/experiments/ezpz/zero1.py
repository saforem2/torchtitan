# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ZeRO-1 optimizer sharding for the ezpz native-DDP path.

This reproduces the Megatron-DeepSpeed ``BF16_Optimizer`` communication
pattern on top of ordinary ``DistributedDataParallel``:

1. Parameters stay replicated, so forward and backward issue no per-layer
   parameter collectives.
2. DDP accumulates FP32 gradients locally across microbatches; the training
   engine disables synchronization for all but the final microbatch.
3. The final backward reduces DDP's gradient buckets once per update, with
   the optional BF16 compression hook, overlapped with that backward.
4. Every rank updates a contiguous ``1/world`` slice of one flat FP32
   parameter buffer.  Optimizer state exists only for that slice.
5. One ``all_gather_into_tensor`` redistributes the updated parameters.

The flat buffer is created before DDP wraps the model: every trainable
parameter is rebound to a view of it, so the all-gather writes the model
parameters directly.  The optimizer object configured by the recipe is kept
(the LR scheduler and checkpoint wrapper still reference it); only its
single parameter group is retargeted to the local shard.  This is exact for
elementwise optimizers such as ``SophiaG`` and ``AdamW``; optimizers that
couple elements of one tensor (Muon, Shampoo) are rejected.

``allgather_dtype="bfloat16"`` matches MDS more closely: the local shard is an
FP32 master copy, and every rank's model parameters hold the BF16-rounded
master values.  ``"float32"`` keeps the model parameters themselves as the
FP32 masters, which preserves the existing native-DDP numerics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn

__all__ = ["FlatParameterBuffer", "flatten_parameters_", "install_zero1"]

_ELEMENTWISE_OPTIMIZERS = ("SophiaG", "AdamW", "Adam", "SGD")


@dataclass
class FlatParameterBuffer:
    """Flat FP32 storage backing every trainable parameter of one module."""

    flat: torch.Tensor
    params: list[nn.Parameter]
    names: list[str]
    offsets: list[int]
    numel: int
    world_size: int
    rank: int
    shard_numel: int = field(init=False)

    def __post_init__(self) -> None:
        self.shard_numel = self.flat.numel() // self.world_size

    @property
    def shard_bounds(self) -> tuple[int, int]:
        start = self.rank * self.shard_numel
        return start, start + self.shard_numel

    def local_segments(self) -> list[tuple[nn.Parameter, int, int, int]]:
        """Return ``(param, param_start, param_end, shard_start)`` for this rank."""
        start, end = self.shard_bounds
        segments = []
        for param, offset in zip(self.params, self.offsets, strict=True):
            lo = max(start, offset)
            hi = min(end, offset + param.numel())
            if lo < hi:
                segments.append((param, lo - offset, hi - offset, lo - start))
        return segments


def flatten_parameters_(
    module: nn.Module, *, world_size: int, rank: int
) -> FlatParameterBuffer:
    """Rebind ``module``'s trainable parameters to views of one flat buffer.

    The buffer is padded to a multiple of ``world_size`` so that every rank
    owns an equal contiguous shard.  Call this before wrapping with DDP.
    """
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid rank {rank} for world size {world_size}")
    named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    if not named:
        raise ValueError("ZeRO-1 requires at least one trainable parameter")
    dtypes = {p.dtype for _, p in named}
    devices = {p.device for _, p in named}
    if dtypes != {torch.float32}:
        raise ValueError(f"ZeRO-1 requires FP32 parameters, got {sorted(map(str, dtypes))}")
    if len(devices) != 1:
        raise ValueError(f"ZeRO-1 requires one parameter device, got {devices}")
    for name, param in named:
        if not param.is_contiguous():
            raise ValueError(f"ZeRO-1 requires contiguous parameters: {name}")
        if hasattr(param, "to_local") or hasattr(param, "_local_tensor"):
            raise ValueError(f"ZeRO-1 requires plain tensors, got DTensor: {name}")
    total = sum(p.numel() for _, p in named)
    padded = -(-total // world_size) * world_size
    flat = torch.zeros(padded, dtype=torch.float32, device=next(iter(devices)))
    offsets = []
    offset = 0
    with torch.no_grad():
        for _, param in named:
            n = param.numel()
            view = flat[offset : offset + n]
            view.copy_(param.detach().reshape(-1))
            param.data = view.view(param.shape)
            offsets.append(offset)
            offset += n
    return FlatParameterBuffer(
        flat=flat,
        params=[p for _, p in named],
        names=[n for n, _ in named],
        offsets=offsets,
        numel=total,
        world_size=world_size,
        rank=rank,
    )


class _Zero1State:
    def __init__(
        self,
        buffer: FlatParameterBuffer,
        optimizer: torch.optim.Optimizer,
        process_group: dist.ProcessGroup | None,
        allgather_dtype: Literal["float32", "bfloat16"],
    ) -> None:
        self.buffer = buffer
        self.optimizer = optimizer
        self.process_group = process_group
        self.allgather_dtype = allgather_dtype
        start, end = buffer.shard_bounds
        self.segments = buffer.local_segments()
        self.covers_shard = sum(e - s for _, s, e, _ in self.segments) == end - start
        flat_shard = buffer.flat[start:end]
        if allgather_dtype == "float32":
            # The master is the model's own storage: no copy, no rounding.
            self.master = nn.Parameter(flat_shard, requires_grad=True)
            self.master_initialized = True
            self.gather_buffer = None
        elif allgather_dtype == "bfloat16":
            self.master = nn.Parameter(torch.empty_like(flat_shard), requires_grad=True)
            self.master_initialized = False
            self.gather_buffer = torch.empty(
                buffer.flat.numel(), dtype=torch.bfloat16, device=buffer.flat.device
            )
        else:
            raise ValueError(f"unsupported ZeRO-1 all-gather dtype {allgather_dtype}")
        self.shard_grad = torch.zeros_like(flat_shard)
        self.steps = 0
        self.allgathers = 0

    @torch.no_grad()
    def ensure_master(self) -> None:
        """Copy the (checkpoint-loaded) parameters into the FP32 master once.

        Deferred until first use because the trainer loads the initial model
        checkpoint after building the optimizer.  The BF16 mode then rounds
        the replicated model parameters, so compute uses BF16 weights from
        the first forward, as MDS does.
        """
        if self.master_initialized:
            return
        start, end = self.buffer.shard_bounds
        self.master.copy_(self.buffer.flat[start:end])
        assert self.gather_buffer is not None
        self.gather_buffer.copy_(self.buffer.flat)
        self.buffer.flat.copy_(self.gather_buffer)
        self.master_initialized = True

    @torch.no_grad()
    def load_shard_gradient(self) -> None:
        grad = self.master.grad
        # init_optim_state supplies its own zero gradient; respect it.
        if grad is not None and grad is not self.shard_grad:
            return
        if not self.covers_shard:
            self.shard_grad.zero_()
        for param, lo, hi, dst in self.segments:
            target = self.shard_grad[dst : dst + hi - lo]
            if param.grad is None:
                target.zero_()
            else:
                target.copy_(param.grad.reshape(-1)[lo:hi])
        self.master.grad = self.shard_grad

    @torch.no_grad()
    def allgather(self) -> None:
        flat = self.buffer.flat
        if self.allgather_dtype == "bfloat16":
            assert self.gather_buffer is not None
            send = self.master.detach().to(torch.bfloat16)
            if self.buffer.world_size > 1:
                dist.all_gather_into_tensor(
                    self.gather_buffer, send, group=self.process_group
                )
            else:
                self.gather_buffer.copy_(send)
            flat.copy_(self.gather_buffer)
        elif self.buffer.world_size > 1:
            # A separate send buffer avoids relying on backend support for an
            # input that aliases its own slot in the output.
            send = self.master.detach().clone()
            dist.all_gather_into_tensor(flat, send, group=self.process_group)
        self.allgathers += 1

    def zero_model_grads(self, set_to_none: bool = True) -> None:
        for param in self.buffer.params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.detach_()
                param.grad.zero_()

    def diagnostics(self) -> dict[str, object]:
        start, end = self.buffer.shard_bounds
        return {
            "rank": self.buffer.rank,
            "world_size": self.buffer.world_size,
            "numel": self.buffer.numel,
            "padded_numel": self.buffer.flat.numel(),
            "shard_start": start,
            "shard_end": end,
            "shard_segments": len(self.segments),
            "allgather_dtype": self.allgather_dtype,
            "optimizer_steps": self.steps,
            "allgathers": self.allgathers,
        }


def _parameter_ids(params: Iterable[torch.Tensor]) -> set[int]:
    return {id(p) for p in params}


def install_zero1(
    container: torch.optim.Optimizer,
    buffer: FlatParameterBuffer,
    *,
    process_group: dist.ProcessGroup | None,
    allgather_dtype: Literal["float32", "bfloat16"] = "float32",
    module: nn.Module | None = None,
) -> _Zero1State:
    """Retarget the container's single optimizer to this rank's flat shard.

    ``container`` is a TorchTitan ``OptimizersContainer`` holding exactly one
    optimizer with one parameter group covering ``buffer.params``.  When
    ``module`` is given and the BF16 all-gather is selected, a one-shot
    forward pre-hook initializes the FP32 master before the first forward.
    """
    optimizers = list(getattr(container, "optimizers", [container]))
    if len(optimizers) != 1:
        raise ValueError(f"ZeRO-1 requires one optimizer, got {len(optimizers)}")
    optimizer = optimizers[0]
    if type(optimizer).__name__ not in _ELEMENTWISE_OPTIMIZERS and not any(
        base.__name__ in _ELEMENTWISE_OPTIMIZERS for base in type(optimizer).__mro__
    ):
        raise ValueError(
            f"ZeRO-1 flat sharding requires an elementwise optimizer, got "
            f"{type(optimizer).__name__}"
        )
    if len(optimizer.param_groups) != 1:
        raise ValueError("ZeRO-1 requires a single optimizer parameter group")
    if optimizer.state:
        raise ValueError("ZeRO-1 must be installed before optimizer state exists")
    group = optimizer.param_groups[0]
    if _parameter_ids(group["params"]) != _parameter_ids(buffer.params):
        raise ValueError("optimizer parameters do not match the flat buffer")
    state = _Zero1State(buffer, optimizer, process_group, allgather_dtype)
    group["params"] = [state.master]
    # Rank-specific key: each rank checkpoints its own shard.  Resuming the
    # optimizer state therefore requires the same world size.
    group["param_names"] = [
        f"zero1.world{buffer.world_size}.rank{buffer.rank:05d}.flat_shard"
    ]

    def pre_step(_optimizer, _args, _kwargs):
        state.ensure_master()
        state.load_shard_gradient()
        return None

    def post_step(_optimizer, _args, _kwargs):
        state.steps += 1
        state.allgather()

    optimizer.register_step_pre_hook(pre_step)
    if callable(getattr(optimizer, "update_hessian", None)):
        # SophiaG's optional pre-step Hessian update reads ``param.grad``; a
        # container-level pre-hook runs before ``pre_step``, so stage the
        # shard gradient here as well.
        inner_update_hessian = optimizer.update_hessian

        def update_hessian(*args, **kwargs):
            state.ensure_master()
            state.load_shard_gradient()
            return inner_update_hessian(*args, **kwargs)

        optimizer.update_hessian = update_hessian  # type: ignore[method-assign]
    optimizer.register_step_post_hook(post_step)

    inner_zero_grad = optimizer.zero_grad

    def zero_grad(set_to_none: bool = True) -> None:
        state.zero_model_grads(set_to_none)
        inner_zero_grad(set_to_none=set_to_none)

    optimizer.zero_grad = zero_grad  # type: ignore[method-assign]

    if module is not None and not state.master_initialized:
        handle = None

        def first_forward(_module, _args):
            state.ensure_master()
            if handle is not None:
                handle.remove()

        handle = module.register_forward_pre_hook(first_forward)
    optimizer._ezpz_zero1 = state  # type: ignore[attr-defined]
    return state
