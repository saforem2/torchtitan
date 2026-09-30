# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributed.checkpoint as dcp

from torchtitan.components.checkpointer.base import ModelWrapper
from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.checkpointer.legacy_native_dcp import (
    LogicalOptimizerState,
    OptimizerFusedLayout,
    prepare_legacy_native_load,
)
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter


class FakeOptimizerState:
    def __init__(self, state):
        self.current = state
        self.loaded = None

    def state_dict(self):
        return dict(self.current)

    def load_state_dict(self, state):
        self.loaded = state


def test_logical_optimizer_state_roundtrip() -> None:
    qkv = "layers.0.attention.qkv_linear."
    ffn = "layers.0.feed_forward."
    fused = {
        f"state.{qkv}wqkv.weight.exp_avg": torch.arange(48).reshape(16, 3),
        f"state.{qkv}wqkv.weight.hessian": torch.arange(48).reshape(16, 3) + 100,
        f"state.{qkv}wqkv.weight.step": torch.tensor([7.0]),
        f"param_groups.{qkv}wqkv.weight.lr": 1e-4,
        f"state.{ffn}w13.weight.exp_avg": torch.arange(24).reshape(2, 4, 3),
        f"state.{ffn}w13.weight.hessian": torch.arange(24).reshape(2, 4, 3) + 200,
        f"state.{ffn}w13.weight.step": torch.tensor([7.0]),
        f"param_groups.{ffn}w13.weight.weight_decay": 0.1,
    }
    optimizer = FakeOptimizerState(fused)
    layout = OptimizerFusedLayout(
        qkv={qkv: (2, 2, (16, 3))},
        stacked={ffn: (2, 4, 3)},
    )
    proxy = LogicalOptimizerState(optimizer, layout)

    logical = proxy.state_dict()

    assert f"state.{qkv}wqkv.weight.exp_avg" not in logical
    assert f"state.{qkv}wq.weight.exp_avg" in logical
    assert f"state.{qkv}wk.weight.exp_avg" in logical
    assert f"state.{qkv}wv.weight.exp_avg" in logical
    assert f"state.{ffn}w13.weight.exp_avg" not in logical
    assert f"state.{ffn}w1.weight.exp_avg" in logical
    assert f"state.{ffn}w3.weight.exp_avg" in logical
    assert logical[f"param_groups.{qkv}wk.weight.lr"] == 1e-4

    proxy.load_state_dict(logical)

    assert optimizer.loaded.keys() == fused.keys()
    for key, expected in fused.items():
        actual = optimizer.loaded[key]
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert actual == expected


def test_logical_optimizer_state_rejects_unsupported_tensor_shape() -> None:
    qkv = "layers.0.attention.qkv_linear."
    optimizer = FakeOptimizerState(
        {f"state.{qkv}wqkv.weight.coupled": torch.ones(2, 2)}
    )
    proxy = LogicalOptimizerState(
        optimizer,
        OptimizerFusedLayout(qkv={qkv: (2, 2, (16, 3))}, stacked={}),
    )

    with pytest.raises(ValueError, match="unsupported optimizer tensor shape"):
        proxy.state_dict()


def test_logical_optimizer_state_rejects_disagreeing_scalars() -> None:
    qkv = "layers.0.attention.qkv_linear."
    optimizer = FakeOptimizerState({})
    proxy = LogicalOptimizerState(
        optimizer,
        OptimizerFusedLayout(qkv={qkv: (2, 2, (16, 3))}, stacked={}),
    )
    logical = {
        f"state.{qkv}wq.weight.step": torch.tensor([1.0]),
        f"state.{qkv}wk.weight.step": torch.tensor([2.0]),
        f"state.{qkv}wv.weight.step": torch.tensor([1.0]),
    }

    with pytest.raises(ValueError, match="values disagree"):
        proxy.load_state_dict(logical)


def test_logical_optimizer_state_scalar_destinations_do_not_alias() -> None:
    qkv = "layers.0.attention.qkv_linear."
    optimizer = FakeOptimizerState(
        {f"state.{qkv}wqkv.weight.step": torch.tensor([7.0])}
    )
    proxy = LogicalOptimizerState(
        optimizer,
        OptimizerFusedLayout(qkv={qkv: (2, 2, (16, 3))}, stacked={}),
    )

    logical = proxy.state_dict()
    steps = [logical[f"state.{qkv}{name}.weight.step"] for name in ("wq", "wk", "wv")]

    assert len({step.data_ptr() for step in steps}) == 3
    steps[1].fill_(8.0)
    with pytest.raises(ValueError, match="values disagree"):
        proxy.load_state_dict(logical)


def test_logical_optimizer_state_preserves_current_only_defaults() -> None:
    qkv = "layers.0.attention.qkv_linear."
    fused_key = f"param_groups.{qkv}wqkv.weight.fused"
    foreach_key = f"param_groups.{qkv}wqkv.weight.foreach"
    lr_key = f"param_groups.{qkv}wqkv.weight.lr"
    optimizer = FakeOptimizerState(
        {fused_key: False, foreach_key: None, lr_key: 1e-4}
    )
    historical_keys = {
        f"param_groups.{qkv}{name}.weight.lr" for name in ("wq", "wk", "wv")
    }
    proxy = LogicalOptimizerState(
        optimizer,
        OptimizerFusedLayout(qkv={qkv: (2, 2, (16, 3))}, stacked={}),
        {key: object() for key in historical_keys},
    )

    logical = proxy.state_dict()

    assert not any(key.endswith(".fused") for key in logical)
    assert not any(key.endswith(".foreach") for key in logical)
    proxy.load_state_dict(logical)
    assert optimizer.loaded[fused_key] is False
    assert optimizer.loaded[foreach_key] is None
    assert optimizer.loaded[lr_key] == 1e-4


def test_prepare_legacy_native_load_preserves_auxiliary_state() -> None:
    build_config, max_context_length = llama3_configs["debugmodel"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)
    model = Llama3Model(config)
    model.init_states()
    adapter = Llama3StateDictAdapter(config, hf_assets_path=None)
    native = dict(model.state_dict())
    optimizer = FakeOptimizerState({})
    dataloader = object()
    train_state = object()
    native.update(
        optimizer=optimizer,
        dataloader=dataloader,
        train_state=train_state,
    )
    model_keys = set(model.state_dict())
    logical_keys = set(adapter.native_fused_to_logical(model.state_dict()))
    checkpoint_keys = logical_keys | {"optimizer", "dataloader", "train_state"}

    prepared, finish = prepare_legacy_native_load(
        native, model_keys, checkpoint_keys, adapter
    )

    assert "layers.0.attention.qkv_linear.wq.weight" in prepared
    assert "layers.0.attention.qkv_linear.wqkv.weight" not in prepared
    assert "layers.0.feed_forward.w1.weight" in prepared
    assert isinstance(prepared["optimizer"], LogicalOptimizerState)
    assert prepared["dataloader"] is dataloader
    assert prepared["train_state"] is train_state

    restored = finish(prepared)
    assert "layers.0.attention.qkv_linear.wqkv.weight" in restored
    assert "layers.0.attention.qkv_linear.wq.weight" not in restored
    assert "layers.0.feed_forward.w13.weight" in restored
    assert restored["optimizer"] is optimizer
    assert restored["dataloader"] is dataloader
    assert restored["train_state"] is train_state


def test_prepare_current_native_load_is_noop() -> None:
    build_config, max_context_length = llama3_configs["debugmodel"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)
    model = Llama3Model(config)
    model.init_states()
    adapter = Llama3StateDictAdapter(config, hf_assets_path=None)
    native = dict(model.state_dict())

    prepared, finish = prepare_legacy_native_load(
        native, set(native), set(native), adapter
    )

    assert prepared is native
    assert finish(prepared) is native


@pytest.mark.parametrize("split_family", ["qkv", "ffn"])
def test_prepare_transitional_native_load_migrates_families_independently(
    split_family,
) -> None:
    build_config, max_context_length = llama3_configs["debugmodel"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)
    model = Llama3Model(config)
    model.init_states()
    adapter = Llama3StateDictAdapter(config, hf_assets_path=None)
    native = dict(model.state_dict())
    logical = adapter.native_fused_to_logical(native)
    checkpoint_keys = set(native)
    if split_family == "qkv":
        checkpoint_keys -= {
            key for key in checkpoint_keys if ".qkv_linear.wqkv." in key
        }
        checkpoint_keys |= {
            key
            for key in logical
            if ".qkv_linear.wq." in key
            or ".qkv_linear.wk." in key
            or ".qkv_linear.wv." in key
        }
    else:
        checkpoint_keys -= {
            key for key in checkpoint_keys if ".feed_forward.w13." in key
        }
        checkpoint_keys |= {
            key
            for key in logical
            if ".feed_forward.w1." in key or ".feed_forward.w3." in key
        }

    prepared, finish = prepare_legacy_native_load(
        native, set(native), checkpoint_keys, adapter
    )

    if split_family == "qkv":
        assert "layers.0.attention.qkv_linear.wq.weight" in prepared
        assert "layers.0.feed_forward.w13.weight" in prepared
    else:
        assert "layers.0.attention.qkv_linear.wqkv.weight" in prepared
        assert "layers.0.feed_forward.w1.weight" in prepared
    restored = finish(prepared)
    assert restored.keys() == native.keys()


def test_prepare_native_load_rejects_mixed_keys_within_one_family() -> None:
    build_config, max_context_length = llama3_configs["debugmodel"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)
    model = Llama3Model(config)
    model.init_states()
    adapter = Llama3StateDictAdapter(config, hf_assets_path=None)
    native = dict(model.state_dict())
    checkpoint_keys = set(native)
    checkpoint_keys.add("layers.0.attention.qkv_linear.wq.weight")
    checkpoint_keys.add("layers.0.attention.qkv_linear.wk.weight")
    checkpoint_keys.add("layers.0.attention.qkv_linear.wv.weight")

    with pytest.raises(ValueError, match="mixes fused and logical QKV"):
        prepare_legacy_native_load(native, set(native), checkpoint_keys, adapter)


def test_prepare_native_load_allows_different_qkv_schema_per_layer(monkeypatch) -> None:
    build_config, max_context_length = llama3_configs["debugmodel"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)
    model = Llama3Model(config)
    model.init_states()
    adapter = Llama3StateDictAdapter(config, hf_assets_path=None)
    converted_keys = []
    convert = adapter.native_fused_to_logical

    def record_conversion(state_dict):
        converted_keys.extend(state_dict)
        return convert(state_dict)

    monkeypatch.setattr(adapter, "native_fused_to_logical", record_conversion)
    native = dict(model.state_dict())
    logical = adapter.native_fused_to_logical(native)
    converted_keys.clear()
    checkpoint_metadata = {key: object() for key in native}
    fused_key = "layers.0.attention.qkv_linear.wqkv.weight"
    checkpoint_metadata.pop(fused_key)
    checkpoint_metadata.update(
        {
            key: object()
            for key in logical
            if key.startswith("layers.0.attention.qkv_linear.w")
        }
    )

    prepared, finish = prepare_legacy_native_load(
        native, set(native), checkpoint_metadata, adapter
    )

    assert fused_key not in prepared
    assert "layers.0.attention.qkv_linear.wq.weight" in prepared
    assert "layers.1.attention.qkv_linear.wqkv.weight" in prepared
    assert "layers.1.attention.qkv_linear.wqkv.weight" not in converted_keys
    assert finish(prepared).keys() == native.keys()


def test_checkpoint_manager_loads_legacy_native_state(monkeypatch) -> None:
    build_config, max_context_length = llama3_configs["debugmodel"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)
    model = Llama3Model(config)
    model.init_states()
    wrapper = ModelWrapper(model)
    adapter = Llama3StateDictAdapter(config, hf_assets_path=None)
    optimizer = FakeOptimizerState({})
    train_state = object()
    logical_keys = set(adapter.native_fused_to_logical(wrapper.state_dict()))
    checkpoint_keys = logical_keys | {"optimizer", "train_state"}

    manager = object.__new__(CheckpointManager)
    manager.sd_adapter = adapter
    manager.states = {
        "model": wrapper,
        "optimizer": optimizer,
        "train_state": train_state,
    }

    class Metadata:
        state_dict_metadata = {key: object() for key in checkpoint_keys}

    class Reader:
        def __init__(self, path):
            self.path = path

        def read_metadata(self):
            return Metadata()

    def fake_load(state_dict, *, storage_reader=None, checkpoint_id=None):
        assert storage_reader is not None
        assert checkpoint_id is None
        assert "layers.0.attention.qkv_linear.wq.weight" in state_dict
        assert "layers.0.attention.qkv_linear.wqkv.weight" not in state_dict
        assert isinstance(state_dict["optimizer"], LogicalOptimizerState)
        state_dict["optimizer"].load_state_dict(state_dict["optimizer"].state_dict())

    monkeypatch.setattr(
        "torchtitan.components.checkpointer.dcp.dcp.FileSystemReader", Reader
    )
    monkeypatch.setattr("torchtitan.components.checkpointer.dcp.dcp.load", fake_load)

    manager._load_checkpoint(
        manager.states,
        "/checkpoint/step-1",
        from_hf=False,
        from_quantized=False,
    )

    current = wrapper.state_dict()
    assert "layers.0.attention.qkv_linear.wqkv.weight" in current
    assert "layers.0.attention.qkv_linear.wq.weight" not in current
    assert manager.states["optimizer"] is optimizer
    assert manager.states["train_state"] is train_state


def test_checkpoint_manager_preserves_remote_native_load(monkeypatch) -> None:
    manager = object.__new__(CheckpointManager)
    manager.sd_adapter = None
    manager.states = {"train_state": object()}

    def fake_load(state_dict, *, storage_reader=None, checkpoint_id=None):
        assert storage_reader is None
        assert checkpoint_id == "s3://bucket/checkpoints/step-1"

    monkeypatch.setattr(
        "torchtitan.components.checkpointer.dcp.filesystem.is_remote",
        lambda path: True,
    )
    monkeypatch.setattr("torchtitan.components.checkpointer.dcp.dcp.load", fake_load)

    manager._load_checkpoint(
        manager.states,
        "s3://bucket/checkpoints/step-1",
        from_hf=False,
        from_quantized=False,
    )


def test_checkpoint_manager_migrates_remote_historical_native_load(monkeypatch) -> None:
    build_config, max_context_length = llama3_configs["debugmodel"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)
    model = Llama3Model(config)
    model.init_states()
    wrapper = ModelWrapper(model)
    adapter = Llama3StateDictAdapter(config, hf_assets_path=None)
    logical_keys = set(adapter.native_fused_to_logical(wrapper.state_dict()))

    manager = object.__new__(CheckpointManager)
    manager.sd_adapter = adapter
    manager.states = {"model": wrapper}

    class Metadata:
        state_dict_metadata = {key: object() for key in logical_keys}

    class Reader:
        def read_metadata(self):
            return Metadata()

    reader = Reader()

    def fake_load(state_dict, *, storage_reader=None, checkpoint_id=None):
        assert storage_reader is reader
        assert checkpoint_id is None
        assert "layers.0.attention.qkv_linear.wq.weight" in state_dict

    monkeypatch.setattr(
        "torchtitan.components.checkpointer.dcp._native_storage_reader",
        lambda checkpoint_id: reader,
    )
    monkeypatch.setattr("torchtitan.components.checkpointer.dcp.dcp.load", fake_load)

    manager._load_checkpoint(
        manager.states,
        "s3://bucket/checkpoints/step-1",
        from_hf=False,
        from_quantized=False,
    )


def test_historical_dcp_roundtrip_saves_current_fused_schema(tmp_path) -> None:
    build_config, max_context_length = llama3_configs["debugmodel"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)
    source_model = Llama3Model(config)
    source_model.init_states()
    adapter = Llama3StateDictAdapter(config, hf_assets_path=None)
    logical_model = adapter.native_fused_to_logical(source_model.state_dict())

    qkv = "layers.0.attention.qkv_linear."
    ffn = "layers.0.feed_forward."
    qkv_weight = source_model.layers["0"].attention.qkv_linear.wqkv.weight
    ffn_weight = source_model.layers["0"].feed_forward.w13.weight
    fused_optimizer = {
        f"state.{qkv}wqkv.weight.exp_avg": torch.arange(qkv_weight.numel()).reshape(
            qkv_weight.shape
        ),
        f"state.{qkv}wqkv.weight.step": torch.tensor([9.0]),
        f"param_groups.{qkv}wqkv.weight.lr": 1e-4,
        f"state.{ffn}w13.weight.hessian": torch.arange(ffn_weight.numel()).reshape(
            ffn_weight.shape
        ),
        f"state.{ffn}w13.weight.step": torch.tensor([9.0]),
    }
    layout = OptimizerFusedLayout(
        qkv={
            qkv: (
                source_model.layers["0"].attention.qkv_linear.head_dim,
                source_model.layers["0"].attention.qkv_linear.heads_per_kv,
                tuple(qkv_weight.shape),
            )
        },
        stacked={ffn: tuple(ffn_weight.shape)},
    )
    source_optimizer = FakeOptimizerState(fused_optimizer)
    historical = {
        **logical_model,
        "optimizer": LogicalOptimizerState(source_optimizer, layout),
        "train_state": {"step": torch.tensor(9)},
    }
    source_dir = tmp_path / "historical"
    dcp.save(historical, checkpoint_id=source_dir)

    target_model = Llama3Model(config)
    target_model.init_states()
    target_wrapper = ModelWrapper(target_model)
    target_optimizer = FakeOptimizerState(fused_optimizer)
    target_train_state = {"step": torch.tensor(0)}
    manager = object.__new__(CheckpointManager)
    manager.sd_adapter = adapter
    manager.states = {
        "model": target_wrapper,
        "optimizer": target_optimizer,
        "train_state": target_train_state,
    }

    manager._load_checkpoint(
        manager.states,
        str(source_dir),
        from_hf=False,
        from_quantized=False,
    )

    assert target_optimizer.loaded.keys() == fused_optimizer.keys()
    assert target_train_state["step"].item() == 9

    current_dir = tmp_path / "current"
    dcp.save(manager._flattened_model_states_sd(), checkpoint_id=current_dir)
    keys = set(dcp.FileSystemReader(current_dir).read_metadata().state_dict_metadata)
    assert "layers.0.attention.qkv_linear.wqkv.weight" in keys
    assert "layers.0.attention.qkv_linear.wq.weight" not in keys
    assert "layers.0.feed_forward.w13.weight" in keys
    assert "layers.0.feed_forward.w1.weight" not in keys
