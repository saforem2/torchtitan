#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read or reset ``dataloader.consumed_samples`` in a DCP checkpoint.

    dcp_dataloader_counter.py show  <step-dir> [<step-dir> ...]
    dcp_dataloader_counter.py zero  <src-step-dir> <dst-step-dir>

``show`` prints the stored counter and every ``dataloader.*`` metadata key.
``zero`` copies a checkpoint with the counter set to 0, which reproduces a
checkpoint written before the counter was incremented. It loads the full
state on CPU, so it is meant for debug-size models.

Runs without a process group (``no_dist=True``). Do not launch under mpiexec.
"""
from __future__ import annotations

import sys

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

COUNTER_KEY = "dataloader.consumed_samples"


def show(step_dir: str) -> int:
    reader = FileSystemReader(step_dir)
    meta = reader.read_metadata()
    keys = sorted(k for k in meta.state_dict_metadata if k.startswith("dataloader"))
    state = {COUNTER_KEY: 0}
    dcp.load(state, storage_reader=reader, no_dist=True)
    print(f"{step_dir}: consumed_samples={state[COUNTER_KEY]} keys={keys}")
    return int(state[COUNTER_KEY])


def zero(src: str, dst: str) -> None:
    import os
    import tempfile

    import torch

    with tempfile.TemporaryDirectory() as tmp:
        flat = os.path.join(tmp, "full.pt")
        dcp_to_torch_save(src, flat)
        state = torch.load(flat, map_location="cpu", weights_only=False)
    loader = state.get("dataloader")
    assert isinstance(loader, dict) and "consumed_samples" in loader, sorted(state)
    print(f"{src}: consumed_samples {loader['consumed_samples']} -> 0")
    loader["consumed_samples"] = 0
    dcp.save(state, storage_writer=dcp.FileSystemWriter(dst), no_dist=True)
    assert show(dst) == 0


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "show":
        for d in argv[2:]:
            show(d)
        return 0
    if len(argv) == 4 and argv[1] == "zero":
        zero(argv[2], argv[3])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
