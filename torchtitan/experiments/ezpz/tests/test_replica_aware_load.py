# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replica-aware DCP load on a 2x2 HSDP mesh (dp_replicate=2, dp_shard=2).

Guards the invariant behind issue #114: ranks with a nonzero dp_replicate
coordinate must skip their read of HSDP-replicated tensors, and the broadcast
from replica 0 must still restore every value exactly, including DTensor state
inside a Stateful (the optimizer container shape) and plain non-DTensor
entries.  A regression either re-reads the checkpoint once per replica (the
6144-rank load stall) or silently leaves zeros on replica 1.
"""

import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.multiprocessing.spawn import spawn

from torchtitan.experiments.ezpz import replica_aware_load

WORLD = 4


class _OptimizerLike:
    """Stateful holding a DTensor, like the optimizer container."""

    def __init__(self, mesh):
        self.m = distribute_tensor(
            torch.arange(16.0).reshape(8, 2), mesh, [Replicate(), Shard(0)]
        )

    def state_dict(self):
        return {"m": self.m}

    def load_state_dict(self, sd):
        self.m.to_local().copy_(sd["m"].to_local())


def _worker(rank: int, rendezvous: str, ckpt: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD)
    try:
        mesh = init_device_mesh(
            "cpu", (2, 2), mesh_dim_names=("dp_replicate", "dp_shard")
        )
        torch.manual_seed(0)
        full = {f"w{i}": torch.randn(8, 4) for i in range(3)}
        sd = {
            k: distribute_tensor(v, mesh, [Replicate(), Shard(0)])
            for k, v in full.items()
        }
        opt = _OptimizerLike(mesh)
        sd["optimizer"] = opt
        sd["step"] = torch.tensor(7)
        dcp.save(sd, checkpoint_id=ckpt)

        for k in full:
            sd[k].to_local().zero_()
        opt.m.to_local().zero_()
        sd["step"] = torch.tensor(0)

        num_skipped = {}
        planner_cls = replica_aware_load.ReplicaAwareLoadPlanner
        create_local_plan = planner_cls.create_local_plan

        def counting_create_local_plan(self):
            base = replica_aware_load.DefaultLoadPlanner.create_local_plan(self)
            plan = create_local_plan(self)
            num_skipped["n"] = len(base.items) - len(plan.items)
            return plan

        planner_cls.create_local_plan = counting_create_local_plan
        try:
            replica_aware_load.replica_aware_dcp_load(
                sd, storage_reader=dcp.FileSystemReader(ckpt)
            )
        finally:
            planner_cls.create_local_plan = create_local_plan

        replica = mesh.get_local_rank("dp_replicate")
        # 3 model tensors + 1 optimizer tensor are replicated over dp_replicate.
        assert num_skipped["n"] == (0 if replica == 0 else 4)
        for k, v in full.items():
            torch.testing.assert_close(sd[k].full_tensor(), v, rtol=0, atol=0)
        torch.testing.assert_close(
            opt.m.full_tensor(), torch.arange(16.0).reshape(8, 2), rtol=0, atol=0
        )
        assert int(sd["step"]) == 7
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend is unavailable")
def test_replica_aware_load_reads_once_and_restores_exactly() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        spawn(
            _worker,
            args=(f"file://{tmpdir}/rendezvous", f"{tmpdir}/ckpt"),
            nprocs=WORLD,
            join=True,
        )
