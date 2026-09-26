#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build one Monarch actor graph across two scheduler-launched hosts."""

from __future__ import annotations

import argparse
import os
import socket
from typing import Any

import torch.distributed as dist
from monarch._src.spmd.host_mesh import host_mesh_from_store
from monarch.actor import Actor, endpoint


class HostProbe(Actor):
    @endpoint
    async def identity(self) -> tuple[str, int]:
        return socket.gethostname(), os.getpid()


def _rank() -> int:
    for name in ("RANK", "PMI_RANK", "PALS_RANKID"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    raise RuntimeError("scheduler rank environment is missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--world-size", default=2, type=int)
    args = parser.parse_args()

    rank = _rank()
    store = dist.FileStore(args.store, args.world_size)
    hosts = host_mesh_from_store(
        store,
        monarch_port=args.port,
        name="sunspot_multihost_preflight",
        transport="tcp",
        rank=rank,
        local_rank=0,
        world_size=args.world_size,
        local_world_size=1,
    )

    if rank != 0:
        store.get("multihost_preflight_done")
        return

    assert hosts is not None
    try:
        hosts.initialized.get()
        actors = hosts.spawn_procs(per_host={"procs": 1}).spawn("probe", HostProbe)
        identities: Any = actors.identity.call().get()
        values = list(identities.values())
        print(f"ACTOR_IDENTITIES={values}", flush=True)
        distinct_hosts = sorted({hostname for hostname, _pid in values})
        if len(values) != args.world_size or len(distinct_hosts) != args.world_size:
            raise RuntimeError(
                f"expected {args.world_size} actors on distinct hosts, got {values}"
            )
        print(f"DISTINCT_HOSTS={distinct_hosts}", flush=True)
        print("MULTIHOST_ACTOR_PREFLIGHT_OK", flush=True)
    finally:
        try:
            hosts.shutdown().get()
        finally:
            store.set("multihost_preflight_done", b"1")


if __name__ == "__main__":
    main()
