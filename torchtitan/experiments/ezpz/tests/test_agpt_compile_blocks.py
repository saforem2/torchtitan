# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchtitan.experiments.ezpz.agpt.parallelize import (
    _maybe_compile_transformer_blocks,
)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({str(i): nn.Linear(4, 4) for i in range(2)})


def _compiled(module: nn.Module) -> bool:
    return getattr(module, "_compiled_call_impl", None) is not None


def test_compile_blocks_off_by_default(monkeypatch):
    monkeypatch.delenv("EZPZ_AGPT_COMPILE_BLOCKS", raising=False)
    model = _Model()
    _maybe_compile_transformer_blocks(model)
    assert not any(_compiled(block) for block in model.layers.values())


def test_compile_blocks_compiles_each_block(monkeypatch):
    monkeypatch.setenv("EZPZ_AGPT_COMPILE_BLOCKS", "default")
    model = _Model()
    x = torch.randn(3, 4)
    expected = model.layers["1"](model.layers["0"](x))
    _maybe_compile_transformer_blocks(model)
    assert all(_compiled(block) for block in model.layers.values())
    torch.testing.assert_close(model.layers["1"](model.layers["0"](x)), expected)
