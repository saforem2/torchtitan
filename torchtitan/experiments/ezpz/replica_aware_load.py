# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DCP load that reads HSDP-replicated tensors once per shard group.

Under HSDP (``dp_replicate > 1``) every replica of a shard needs the same
checkpoint chunk, and ``dcp.load`` has each rank read its own copy. At 4B v2
512N (``dp_shard=12``, ``dp_replicate=512``) that reads the 56.85 GB of model
and optimizer state 512 times (~29 TB) and stalls past the idle watchdog; the
10B reads ~59 TB. An FSDP-only checkpoint (2B 512N) reads each byte once.

``ReplicaAwareLoadPlanner`` keeps the reads only on ranks whose
``dp_replicate`` coordinate is 0 and drops them elsewhere;
``broadcast_replicas`` then broadcasts those local shards along
``dp_replicate``. Every rank still joins DCP planning, so no subset process
group is needed. Tensors that are not DTensors replicated over
``dp_replicate`` load normally on every rank.

Use ``replica_aware_dcp_load`` rather than the planner alone: ``dcp.load``
calls ``load_state_dict`` on Stateful entries (the optimizer container) before
it returns, so the broadcast has to happen between the read and that call.
``ReplicaAwareCheckpointManager`` routes the native (non-HF) checkpoint load
through it.
"""

from __future__ import annotations

import dataclasses

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch.distributed.checkpoint.planner import LoadPlan
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.storage import StorageReader
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.checkpointer.base import EMA, MODEL
from torchtitan.experiments.ezpz.logging import logger
from torchtitan.tools import filesystem

_REPLICATE_AXIS = "dp_replicate"


def _replicated_over_axis(value: Any, axis: str) -> bool:
    """True for a DTensor replicated over a multi-rank ``axis`` of its mesh."""
    if not isinstance(value, DTensor):
        return False
    names = value.device_mesh.mesh_dim_names or ()
    if axis not in names:
        return False
    index = names.index(axis)
    if value.device_mesh.size(index) <= 1:
        return False
    return isinstance(value.placements[index], Replicate)


class ReplicaAwareLoadPlanner(DefaultLoadPlanner):
    """``DefaultLoadPlanner`` that reads ``dp_replicate``-replicated tensors once."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._replicated_fqns: list[str] = []

    def create_local_plan(self) -> LoadPlan:
        plan = super().create_local_plan()
        # Same list on every rank (sorted, rank-independent) so the broadcasts
        # in broadcast_replicas() line up across the replicate group.
        self._replicated_fqns = sorted(
            fqn
            for fqn, value in self.state_dict.items()
            if _replicated_over_axis(value, _REPLICATE_AXIS)
        )
        if not self._replicated_fqns:
            return plan
        sample = self.state_dict[self._replicated_fqns[0]]
        if sample.device_mesh.get_local_rank(_REPLICATE_AXIS) == 0:
            return plan
        skip = set(self._replicated_fqns)
        return dataclasses.replace(
            plan, items=[item for item in plan.items if item.dest_index.fqn not in skip]
        )

    def broadcast_replicas(self) -> None:
        """Broadcast the skipped local shards from ``dp_replicate`` rank 0."""
        if not self._replicated_fqns:
            return
        num_bytes = 0
        for fqn in self._replicated_fqns:
            value = self.state_dict[fqn]
            group = value.device_mesh.get_group(_REPLICATE_AXIS)
            local = value.to_local()
            buffer = local if local.is_contiguous() else local.contiguous()
            dist.broadcast(buffer, src=dist.get_global_rank(group, 0), group=group)
            if buffer is not local:
                local.copy_(buffer)
            num_bytes += buffer.numel() * buffer.element_size()
        if dist.get_rank() == 0:
            logger.info(
                "replica-aware load: %d tensors read once per %s group and "
                "broadcast (%.2f GB local on rank 0)",
                len(self._replicated_fqns),
                _REPLICATE_AXIS,
                num_bytes / 1e9,
            )
        torch.accelerator.synchronize()


def replica_aware_dcp_load(
    state_dict: dict[str, Any],
    *,
    storage_reader: StorageReader,
    allow_partial_load: bool = False,
) -> None:
    """``dcp.load`` that reads ``dp_replicate``-replicated tensors once.

    Mirrors ``dcp.load``'s Stateful handling so the broadcast lands before
    each Stateful's ``load_state_dict``. Updates ``state_dict`` in place.
    """
    stateful = {k: v for k, v in state_dict.items() if isinstance(v, Stateful)}
    plain = {k: (v.state_dict() if k in stateful else v) for k, v in state_dict.items()}
    planner = ReplicaAwareLoadPlanner(allow_partial_load=allow_partial_load)
    dcp.load(plain, storage_reader=storage_reader, planner=planner)
    planner.broadcast_replicas()
    for key, value in state_dict.items():
        if key in stateful:
            value.load_state_dict(plain[key])
        else:
            state_dict[key] = plain[key]


def _storage_reader(checkpoint_id: str) -> StorageReader:
    # Same choice as core's (private) _native_storage_reader.
    if filesystem.is_remote(checkpoint_id):
        from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader

        return FsspecReader(checkpoint_id)
    return dcp.FileSystemReader(checkpoint_id)


class ReplicaAwareCheckpointManager(CheckpointManager):
    """``CheckpointManager`` whose native checkpoint load is replica-aware.

    Only the non-HF branch of ``_load_checkpoint`` differs from core: it calls
    ``replica_aware_dcp_load`` where core calls ``dcp.load``. HF loads go to
    core unchanged.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        pass

    def _load_checkpoint(
        self,
        states: dict[str, Any],
        checkpoint_id: str,
        *,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        if from_hf:
            super()._load_checkpoint(
                states, checkpoint_id, from_hf=from_hf, from_quantized=from_quantized
            )
            return
        state_dict = self._flattened_model_states_sd(states)
        storage_reader = _storage_reader(checkpoint_id)
        if self.sd_adapter is None:
            replica_aware_dcp_load(state_dict, storage_reader=storage_reader)
        else:
            from torchtitan.components.checkpointer.legacy_native_dcp import (
                prepare_legacy_native_load,
            )

            checkpoint_metadata = storage_reader.read_metadata().state_dict_metadata
            model_keys = set(states[MODEL].state_dict()) if MODEL in states else set()
            state_dict, finish_load = prepare_legacy_native_load(
                state_dict, model_keys, checkpoint_metadata, self.sd_adapter
            )
            replica_aware_dcp_load(state_dict, storage_reader=storage_reader)
            state_dict = finish_load(state_dict)
        if MODEL in states:
            states[MODEL].load_state_dict(state_dict)
        # Same as core: reseed EMA from the loaded weights when EMA itself was
        # not restored (excluded, or a model-only load).
        if MODEL in states and EMA in self.states and EMA not in states:
            self.states[EMA].load_state_dict({})
