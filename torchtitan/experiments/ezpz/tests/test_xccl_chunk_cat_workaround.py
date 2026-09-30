# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from torchtitan.experiments.ezpz.xccl_chunk_cat_workaround import (
    chunk_cat_copy_in,
    maybe_install_xccl_chunk_cat_workaround,
)


@pytest.mark.parametrize(
    "tensors,world_size",
    [
        (
            [
                torch.arange(12, dtype=torch.float32).reshape(3, 4),
                torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100,
            ],
            4,
        ),
        ([torch.arange(21).reshape(7, 3), torch.arange(20).reshape(5, 4)], 3),
        ([torch.arange(6).reshape(2, 3), torch.arange(4).reshape(1, 4)], 4),
        (
            [
                torch.arange(42).reshape(7, 6)[:, ::2],
                torch.arange(40).reshape(5, 8)[:, 1::2],
            ],
            3,
        ),
    ],
)
def test_chunk_cat_copy_in_matches_torch_chunk_cat_with_padding(
    tensors: list[torch.Tensor], world_size: int
) -> None:
    expected = torch._chunk_cat(tensors, dim=0, num_chunks=world_size)
    actual = torch.empty_like(expected)

    chunk_cat_copy_in(tensors, actual, world_size)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_chunk_cat_workaround_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("EZPZ_XPU_CHUNK_CAT_FALLBACK", raising=False)
    with patch.object(torch.xpu, "is_available", return_value=True):
        assert maybe_install_xccl_chunk_cat_workaround() is False


def test_chunk_cat_workaround_install_is_idempotent_and_delegates_cpu(
    monkeypatch,
) -> None:
    import torchtitan.experiments.ezpz.xccl_chunk_cat_workaround as workaround

    calls = []

    def original(grads, output, world_size) -> None:
        calls.append((grads, output, world_size))

    collectives = SimpleNamespace(foreach_reduce_scatter_copy_in=original)
    monkeypatch.setenv("EZPZ_XPU_CHUNK_CAT_FALLBACK", "1")
    monkeypatch.setattr(torch.distributed, "is_xccl_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(workaround, "_import_fsdp_collectives", lambda: collectives)

    assert maybe_install_xccl_chunk_cat_workaround() is True
    installed = collectives.foreach_reduce_scatter_copy_in
    assert installed is not original
    assert maybe_install_xccl_chunk_cat_workaround() is True
    assert collectives.foreach_reduce_scatter_copy_in is installed

    grads = [torch.ones(2, 2)]
    output = torch.empty(2, 2)
    installed(grads, output, 2)
    assert calls == [(grads, output, 2)]
