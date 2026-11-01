# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AGPT FSDP placement when stacked-linear rows do not divide the shard degree.

AGPT-2B ``w13`` is ``[2, 11008, 2048]``. Core's ``Shard(1)`` override fails at
12/24 ranks (``11008 % 12 != 0``) because FSDP2 only pads dim 0. The ezpz
wrapper must fall back to ``Shard(0)`` for those params only, for FSDP and
HSDP meshes, and the fallback layout must round-trip through DCP.
"""

import os
import tempfile
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import DataParallelMeshDims
from torch.distributed.tensor import DTensor, Shard
from torch.multiprocessing.spawn import spawn

from torchtitan.experiments.ezpz.agpt import parallelize as agpt_parallelize
from torchtitan.models.common.linear import Linear


class _Block(nn.Module):
    def __init__(self, rows: int) -> None:
        super().__init__()
        self.w13 = Linear.Config(
            in_features=4, out_features=rows, num_linears=2, bias=True
        ).build()
        self.w2 = nn.Linear(rows, 4, bias=False)


class _Model(nn.Module):
    def __init__(self, rows: int) -> None:
        super().__init__()
        self.enable_weight_tying = False
        self.tok_embeddings = None
        self.norm = None
        self.lm_head = None
        self.layers = nn.ModuleDict({"0": _Block(rows)})


def _init(model: nn.Module, offset: float) -> nn.Module:
    """Deterministic, rank-independent values (Linear.build leaves them empty)."""
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            values = torch.arange(param.numel(), dtype=param.dtype) + 100 * i
            param.copy_(values.reshape(param.shape) + offset)
    return model


class _FakeMesh:
    def __init__(self, sizes: dict[str, int]) -> None:
        self._sizes = sizes
        self.ndim = len(sizes)

    def size(self, dim: int | None = None) -> int:
        values = list(self._sizes.values())
        if dim is None:
            out = 1
            for v in values:
                out *= v
            return out
        return values[dim]

    def __getitem__(self, name: str) -> "_FakeMesh":
        return _FakeMesh({name: self._sizes[name]})


def test_shard_degree_counts_shard_axes_only() -> None:
    mesh = cast(Any, _FakeMesh({"dp_replicate": 2, "dp_shard": 12}))
    assert agpt_parallelize.fsdp_shard_degree(mesh, None) == 12
    dims = DataParallelMeshDims(shard="dp_shard", replicate="dp_replicate")
    assert agpt_parallelize.fsdp_shard_degree(mesh, dims) == 12
    mesh = cast(Any, _FakeMesh({"dp_shard": 12, "cp": 2}))
    dims = DataParallelMeshDims(shard=("dp_shard", "cp"))
    assert agpt_parallelize.fsdp_shard_degree(mesh, dims) == 24
    assert (
        agpt_parallelize.fsdp_shard_degree(cast(Any, _FakeMesh({"x": 24})), None) == 24
    )


@pytest.mark.parametrize(
    ("rows", "degree", "expected"),
    [(11008, 16, Shard(1)), (11008, 12, None), (11008, 24, None), (24, 24, Shard(1))],
)
def test_placements_drop_only_uneven_rows(rows, degree, expected) -> None:
    block = _Block(rows)
    fallbacks: list = []
    placements = agpt_parallelize.divisible_linear_param_shard_placements(
        block, degree, fallbacks
    )
    assert placements.get(block.w13.weight) == expected
    assert placements.get(block.w13.bias) == expected
    # Unstacked linears never get an override (default Shard(0)).
    assert block.w2.weight not in placements
    assert len(fallbacks) == (0 if expected is not None else 2)


def _fsdp_with_fake_pg(mesh_shape, mesh_names, rows):
    from torch.testing._internal.distributed.fake_pg import FakeStore

    world = 1
    for s in mesh_shape:
        world *= s
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=world)
    try:
        mesh = init_device_mesh("cpu", mesh_shape, mesh_dim_names=mesh_names)
        with torch.device("meta"):
            model = _Model(rows)
        agpt_parallelize.apply_fsdp(
            model,
            mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
        )
        weight = model.layers["0"].w13.weight
        assert isinstance(weight, DTensor)
        return weight.placements[-1]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("mesh_shape", "mesh_names", "rows", "expected"),
    [
        ((24,), ("dp_shard",), 11008, Shard(0)),  # job 12481864 geometry
        ((16,), ("dp_shard",), 11008, Shard(1)),  # PPN=8 workaround
        ((2, 12), ("dp_replicate", "dp_shard"), 11008, Shard(0)),  # HSDP 2x12
        ((3, 8), ("dp_replicate", "dp_shard"), 11008, Shard(1)),  # HSDP 3x8
    ],
)
def test_fully_shard_accepts_agpt_2b_rows(mesh_shape, mesh_names, rows, expected):
    """Real ``fully_shard`` (fake PG) used to raise NotImplementedError here."""
    assert _fsdp_with_fake_pg(mesh_shape, mesh_names, rows) == expected


def _dcp_round_trip(rank: int, world: int, rendezvous: str, ckpt: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))

        def build(offset: float) -> _Model:
            # 4 rows over 3 ranks -> Shard(0) fallback
            model = _init(_Model(rows=4), offset)
            agpt_parallelize.apply_fsdp(
                model,
                mesh,
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                pp_enabled=False,
            )
            return model

        source = build(offset=0.0)
        weight = source.layers["0"].w13.weight
        assert weight.placements == (Shard(0),)
        # Shard(0) over a size-2 dim on 3 ranks leaves rank 2 empty.
        assert weight.to_local().shape[0] == (1 if rank < 2 else 0)
        dcp.save(source.state_dict(), checkpoint_id=ckpt)

        target = build(offset=0.5)
        state = target.state_dict()
        dcp.load(state, checkpoint_id=ckpt)
        target.load_state_dict(state)
        for (name, a), (_, b) in zip(
            source.state_dict().items(), target.state_dict().items()
        ):
            assert torch.equal(a.full_tensor(), b.full_tensor()), name
    finally:
        dist.destroy_process_group()


def _dcp_save_even(rank: int, world: int, rendezvous: str, ckpt: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        model = _init(_Model(rows=4), 0.0)
        agpt_parallelize.apply_fsdp(
            model,
            mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            pp_enabled=False,
        )
        assert model.layers["0"].w13.weight.placements == (Shard(1),)
        dcp.save(model.state_dict(), checkpoint_id=ckpt)
    finally:
        dist.destroy_process_group()


def _dcp_load_fallback(rank: int, world: int, rendezvous: str, ckpt: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        reference = _init(_Model(rows=4), 0.0)
        model = _init(_Model(rows=4), 0.5)
        agpt_parallelize.apply_fsdp(
            model,
            mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            pp_enabled=False,
        )
        assert model.layers["0"].w13.weight.placements == (Shard(0),)
        state = model.state_dict()
        dcp.load(state, checkpoint_id=ckpt)
        model.load_state_dict(state)
        for name, value in reference.state_dict().items():
            assert torch.equal(model.state_dict()[name].full_tensor(), value), name
    finally:
        dist.destroy_process_group()


def _spawn(fn, world: int, ckpt: str) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        rendezvous = f"file://{handle.name}"
    try:
        spawn(fn, args=(world, rendezvous, ckpt), nprocs=world, join=True)
    finally:
        if os.path.exists(handle.name):
            os.unlink(handle.name)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo unavailable")
def test_fallback_layout_round_trips_dcp() -> None:
    with tempfile.TemporaryDirectory() as ckpt:
        _spawn(_dcp_round_trip, 3, ckpt)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo unavailable")
def test_row_sharded_checkpoint_loads_into_fallback_layout() -> None:
    """A checkpoint saved with Shard(1) (2 ranks) loads into Shard(0) (3 ranks)."""
    with tempfile.TemporaryDirectory() as ckpt:
        _spawn(_dcp_save_even, 2, ckpt)
        _spawn(_dcp_load_fallback, 3, ckpt)
