# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Focused tests for the optional XPU and TorchStore compatibility hooks."""

from types import SimpleNamespace

import pytest


def test_torchstore_strategy_defaults_to_automatic_transport(monkeypatch):
    from torchtitan.torchstore_compat import torchstore_transport_from_env

    monkeypatch.delenv("TORCHTITAN_TORCHSTORE_TRANSPORT", raising=False)

    assert torchstore_transport_from_env() is None


def test_torchstore_strategy_rejects_unknown_transport(monkeypatch):
    from torchtitan.torchstore_compat import torchstore_transport_from_env

    monkeypatch.setenv("TORCHTITAN_TORCHSTORE_TRANSPORT", "bogus")

    with pytest.raises(ValueError, match="got 'bogus'"):
        torchstore_transport_from_env()


def test_xpu_flex_attention_uses_triton_backend():
    from torchtitan.experiments.ezpz.rl import xpu_overrides

    flex = object()
    triton_path = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"

    class FakeBackend:
        CUSTOM = object()
        FLEX_ATTENTION = flex
        TRITON_ATTN = SimpleNamespace(get_path=lambda: triton_path)

    class FakePlatform:
        @classmethod
        def get_attn_backend_cls(cls, selected_backend, *_args, **_kwargs):
            return f"original:{selected_backend!r}"

    assert (
        xpu_overrides._select_vllm_xpu_attention_backend(
            flex, FakeBackend, lambda *_args: "original"
        )
        == triton_path
    )
