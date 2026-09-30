# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.components.checkpointer import ModelWrapper
from torchtitan.experiments.ezpz.eval.convert_to_hf import _checkpoint_load_state_dict
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter


def _state_and_adapter():
    build_config, max_context_length = llama3_configs["debugmodel"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)
    model = Llama3Model(config)
    model.init_states()
    return ModelWrapper(model)._get_state_dict(), Llama3StateDictAdapter(config, None)


def test_checkpoint_load_state_dict_keeps_current_fused_schema() -> None:
    state_dict, adapter = _state_and_adapter()

    destination = _checkpoint_load_state_dict(state_dict, adapter, set(state_dict))

    assert destination.keys() == state_dict.keys()
    assert "layers.0.attention.qkv_linear.wqkv.weight" in destination
    assert "layers.0.feed_forward.w13.weight" in destination


def test_checkpoint_load_state_dict_supports_historical_logical_schema() -> None:
    state_dict, adapter = _state_and_adapter()
    logical_state_dict = adapter._native_fused_linears_to_hf(state_dict)

    destination = _checkpoint_load_state_dict(
        state_dict, adapter, set(logical_state_dict)
    )

    assert destination.keys() == logical_state_dict.keys()
    assert "layers.0.attention.qkv_linear.wqkv.weight" not in destination
    assert "layers.0.attention.qkv_linear.wq.weight" in destination
    assert "layers.0.attention.qkv_linear.wk.weight" in destination
    assert "layers.0.attention.qkv_linear.wv.weight" in destination
    assert "layers.0.feed_forward.w13.weight" not in destination
    assert "layers.0.feed_forward.w1.weight" in destination
    assert "layers.0.feed_forward.w3.weight" in destination


def test_checkpoint_load_state_dict_rejects_incomplete_schema() -> None:
    state_dict, adapter = _state_and_adapter()

    with pytest.raises(RuntimeError, match="cannot represent current model state"):
        _checkpoint_load_state_dict(state_dict, adapter, {"tok_embeddings.weight"})
