# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import timedelta
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from torchtitan.components.checkpointer.legacy_native_dcp import (
    _empty_dtensor_from_metadata,
    LogicalOptimizerState,
    OptimizerFusedLayout,
)
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter


class _OptimizerState:
    def __init__(self, state):
        self.state = state
        self.loaded = None

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.loaded = state


def _check_native_logical_dtensor_roundtrip(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        build_config, max_context_length = llama3_configs["debugmodel"]
        config = build_config(attn_backend="flex", seq_len=max_context_length)
        adapter = Llama3StateDictAdapter(config, hf_assets_path=None)
        qkv_key = "layers.0.attention.qkv_linear.wqkv.weight"
        ffn_key = "layers.0.feed_forward.w13.weight"
        qkv = distribute_tensor(
            torch.arange(384 * 256, dtype=torch.float32).reshape(384, 256),
            mesh,
            (Shard(0),),
        )
        ffn = distribute_tensor(
            torch.arange(2 * 1024 * 256, dtype=torch.float32).reshape(2, 1024, 256),
            mesh,
            (Shard(0),),
        )
        native = {qkv_key: qkv, ffn_key: ffn}

        logical = adapter.native_fused_to_logical(native)
        with mock.patch(
            "torchtitan.protocols.state_dict_adapter.torch.cat",
            side_effect=AssertionError(
                "DTensor QKV restore must not materialize torch.cat"
            ),
        ):
            restored = adapter.native_logical_to_fused(logical)

        for key, expected in native.items():
            actual = restored[key]
            assert isinstance(actual, DTensor)
            assert actual.placements == expected.placements
            torch.testing.assert_close(
                actual.to_local(), expected.to_local(), rtol=0, atol=0
            )

        # Five KV groups across two ranks puts the Shard(0) boundary at row 20,
        # halfway through a packed 8-row QKV group. Exercise the destination
        # intersection-copy path rather than only group-aligned shards.
        uneven_key = "layers.0.attention.qkv_linear.wqkv.weight"
        uneven_global = torch.arange(40 * 16, dtype=torch.float32).reshape(40, 16)
        uneven_native = distribute_tensor(uneven_global, mesh, (Shard(0),))
        uneven_state = {uneven_key: uneven_native}
        adapter._split_qkv_linear(
            uneven_state,
            prefix="layers.0.attention.qkv_linear.",
            head_dim=2,
            heads_per_kv=2,
        )
        with mock.patch(
            "torchtitan.protocols.state_dict_adapter.torch.cat",
            side_effect=AssertionError(
                "DTensor QKV restore must not materialize torch.cat"
            ),
        ):
            adapter._merge_qkv_linear(
                uneven_state,
                prefix="layers.0.attention.qkv_linear.",
                head_dim=2,
                heads_per_kv=2,
            )
        uneven_restored = uneven_state[uneven_key]
        assert isinstance(uneven_restored, DTensor)
        assert uneven_restored.placements == uneven_native.placements
        torch.testing.assert_close(
            uneven_restored.to_local(), uneven_native.to_local(), rtol=0, atol=0
        )

        # 5 KV heads across two shards intentionally cuts through a packed
        # QKV group boundary. The optimizer migration must replicate before
        # reshaping, then restore the original Shard(0) placement.
        optimizer_qkv_global = torch.arange(40 * 16, dtype=torch.float32).reshape(
            40, 16
        )
        optimizer_qkv = distribute_tensor(
            optimizer_qkv_global,
            mesh,
            (Shard(0),),
        )
        optimizer_key = "state.layers.0.attention.qkv_linear.wqkv.weight.exp_avg"
        optimizer = _OptimizerState({optimizer_key: optimizer_qkv})
        checkpoint_metadata = {}
        for name, shape in (
            ("wq", (20, 16)),
            ("wk", (10, 16)),
            ("wv", (10, 16)),
        ):
            key = f"state.layers.0.attention.qkv_linear.{name}.weight.exp_avg"
            checkpoint_metadata[key] = TensorStorageMetadata(
                properties=TensorProperties(dtype=torch.float32),
                size=torch.Size(shape),
                chunks=[
                    ChunkStorageMetadata(
                        offsets=torch.Size((0, 0)), sizes=torch.Size(shape)
                    )
                ],
            )
        proxy = LogicalOptimizerState(
            optimizer,
            OptimizerFusedLayout(
                qkv={
                    "layers.0.attention.qkv_linear.": (
                        2,
                        2,
                        tuple(optimizer_qkv.shape),
                    )
                },
                stacked={},
            ),
            checkpoint_metadata,
        )

        logical_optimizer = proxy.state_dict()
        for name in ("wq", "wk", "wv"):
            logical = logical_optimizer[
                f"state.layers.0.attention.qkv_linear.{name}.weight.exp_avg"
            ]
            assert isinstance(logical, DTensor)
            assert logical.placements == optimizer_qkv.placements
        assert (
            sum(
                logical_optimizer[
                    f"state.layers.0.attention.qkv_linear.{name}.weight.exp_avg"
                ]
                .to_local()
                .numel()
                for name in ("wq", "wk", "wv")
            )
            == optimizer_qkv.to_local().numel()
        )

        packed = optimizer_qkv_global.reshape(5, 4, 2, 16)
        loaded_globals = {
            "wq": packed[:, :2].reshape(20, 16),
            "wk": packed[:, 2].reshape(10, 16),
            "wv": packed[:, 3].reshape(10, 16),
        }
        for name, expected in loaded_globals.items():
            key = f"state.layers.0.attention.qkv_linear.{name}.weight.exp_avg"
            loaded = distribute_tensor(expected, mesh, (Shard(0),))
            logical_optimizer[key].to_local().copy_(loaded.to_local())

        proxy.load_state_dict(logical_optimizer)

        actual = optimizer.loaded[optimizer_key]
        assert isinstance(actual, DTensor)
        assert actual.placements == optimizer_qkv.placements
        torch.testing.assert_close(
            actual.to_local(), optimizer_qkv.to_local(), rtol=0, atol=0
        )
    finally:
        dist.destroy_process_group()


def test_native_logical_dtensor_roundtrip(tmp_path: Path) -> None:
    mp.spawn(
        _check_native_logical_dtensor_roundtrip,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )


def _check_logical_placeholder_replaces_fused_strided_shard(
    rank: int, rendezvous: str
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        build_config, max_context_length = llama3_configs["debugmodel"]
        config = build_config(attn_backend="flex", seq_len=max_context_length)
        adapter = Llama3StateDictAdapter(config, hf_assets_path=None)
        fused = distribute_tensor(
            torch.arange(40 * 16, dtype=torch.float32).reshape(40, 16),
            mesh,
            (_StridedShard(0, split_factor=2), Shard(0)),
        )
        packed = fused.full_tensor().reshape(5, 4, 2, 16)
        logical_globals = {
            "wq": packed[:, :2].reshape(20, 16),
            "wk": packed[:, 2].reshape(10, 16),
            "wv": packed[:, 3].reshape(10, 16),
        }
        prefix = "layers.0.attention.qkv_linear."
        logical_state = {}
        for name, expected in logical_globals.items():
            metadata = TensorStorageMetadata(
                properties=TensorProperties(dtype=torch.float32),
                size=expected.shape,
                chunks=[
                    ChunkStorageMetadata(
                        offsets=torch.Size((0, 0)), sizes=expected.shape
                    )
                ],
            )
            placeholder = _empty_dtensor_from_metadata(fused, metadata)
            assert placeholder.placements == (Shard(0), Shard(0))
            loaded = distribute_tensor(expected, mesh, (Shard(0), Shard(0)))
            placeholder.to_local().copy_(loaded.to_local())
            logical_state[f"{prefix}{name}.weight"] = placeholder

        qkv_key = f"{prefix}wqkv.weight"
        adapter._qkv_linear_sharding[qkv_key] = (mesh, fused.placements)
        adapter._merge_qkv_linear(
            logical_state,
            prefix=prefix,
            head_dim=2,
            heads_per_kv=2,
        )

        restored = logical_state[qkv_key]
        assert restored.placements == fused.placements
        torch.testing.assert_close(
            restored.to_local(), fused.to_local(), rtol=0, atol=0
        )

        optimizer_key = f"state.{qkv_key}.exp_avg"
        optimizer = _OptimizerState({optimizer_key: fused})
        optimizer_metadata = {}
        for name, expected in logical_globals.items():
            key = f"state.{prefix}{name}.weight.exp_avg"
            optimizer_metadata[key] = TensorStorageMetadata(
                properties=TensorProperties(dtype=torch.float32),
                size=expected.shape,
                chunks=[
                    ChunkStorageMetadata(
                        offsets=torch.Size((0, 0)), sizes=expected.shape
                    )
                ],
            )
        proxy = LogicalOptimizerState(
            optimizer,
            OptimizerFusedLayout(
                qkv={prefix: (2, 2, tuple(fused.shape))},
                stacked={},
            ),
            optimizer_metadata,
        )
        logical_optimizer = proxy.state_dict()
        for name, expected in logical_globals.items():
            key = f"state.{prefix}{name}.weight.exp_avg"
            loaded = distribute_tensor(expected, mesh, (Shard(0), Shard(0)))
            logical_optimizer[key].to_local().copy_(loaded.to_local())
        with mock.patch(
            "torchtitan.components.checkpointer.legacy_native_dcp._fuse_qkv",
            side_effect=AssertionError(
                "optimizer QKV restore must not materialize global fusion"
            ),
        ):
            proxy.load_state_dict(logical_optimizer)
        optimizer_restored = optimizer.loaded[optimizer_key]
        assert optimizer_restored.placements == fused.placements
        torch.testing.assert_close(
            optimizer_restored.to_local(), fused.to_local(), rtol=0, atol=0
        )
    finally:
        dist.destroy_process_group()


def test_logical_placeholder_replaces_fused_strided_shard(tmp_path: Path) -> None:
    mp.spawn(
        _check_logical_placeholder_replaces_fused_strided_shard,
        args=(f"file://{tmp_path / 'strided-rendezvous'}",),
        nprocs=4,
        join=True,
    )
