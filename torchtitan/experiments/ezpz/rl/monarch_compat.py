# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Store-backed Monarch host-mesh rendezvous across Monarch versions.

``torchmonarch`` 0.6 exposed ``monarch._src.spmd.host_mesh.host_mesh_from_store``.
0.7 removed it; the public replacement is ``monarch.job.spmd.StoreJob``:
``StoreJob.from_store(...)`` takes the same keyword arguments, returns a job on
global rank 0 (``None`` elsewhere), and ``job.state()`` attaches to the workers.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any


def _single_local_ipv4(host: str) -> str | None:
    """Return one local IPv4 address for ``host`` if it has several, else None.

    Sirius and Polaris publish one A record per HSN NIC for each node's
    ``*.hsn`` name. Picks the lowest address this host can bind, so the
    choice is deterministic for the process that binds and the one that
    dials.
    """
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_INET)
    except OSError:
        return None
    addrs = sorted({info[4][0] for info in infos}, key=ipaddress.IPv4Address)
    if len(addrs) < 2:
        return None
    for addr in addrs:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((addr, 0))
            except OSError:
                continue
            return addr
    return None


def _pin_tcp_worker_address_to_ip() -> None:
    """Advertise a TCP worker by IP when its FQDN resolves to several addresses.

    ``StoreJob`` publishes ``tcp://<getfqdn()>:<port>``. The worker binds one
    resolution of that name and the attaching client may dial another, so on
    a multi-NIC node attach fails with "never connected" (Sirius 33504: the
    worker served 10.150.0.33, the client dialed 10.150.0.34). Hosts whose
    FQDN has one address keep the hostname form.
    """
    import monarch._src.spmd.host_mesh as spmd_host_mesh

    worker_address = spmd_host_mesh._worker_address
    if getattr(worker_address, "_ezpz_ip_pinned", False):
        return

    def pinned_worker_address(transport: str, **kwargs: Any) -> str:
        addr = worker_address(transport, **kwargs)
        if transport != "tcp":
            return addr
        ip = _single_local_ipv4(socket.getfqdn())
        if ip is None:
            return addr
        return f"tcp://{ip}:{kwargs['monarch_port']}"

    pinned_worker_address._ezpz_ip_pinned = True  # type: ignore[attr-defined]
    spmd_host_mesh._worker_address = pinned_worker_address


def host_mesh_from_store(store: Any, *, name: str, **kwargs: Any) -> Any | None:
    """Spawn/publish one Monarch worker per host and attach on global rank 0.

    Must be called collectively on every rank. Returns the ``HostMesh`` on
    global rank 0 and ``None`` on every other rank, as the 0.6 helper did.
    """
    import importlib.util

    # Fall back only when the 0.7 module is absent. If it exists but one of
    # its own imports fails (e.g. torchx, the torchmonarch[torchx] extra),
    # let that error surface instead of masking it with the 0.6 path.
    if importlib.util.find_spec("monarch.job.spmd") is not None:
        from monarch.job.spmd import StoreJob
    else:  # torchmonarch < 0.7
        from monarch._src.spmd.host_mesh import (
            host_mesh_from_store as _legacy_host_mesh_from_store,
        )

        return _legacy_host_mesh_from_store(store, name=name, **kwargs)

    _pin_tcp_worker_address_to_ip()
    job = StoreJob.from_store(store, name=name, **kwargs)
    if job is None:
        return None
    return getattr(job.state(), name)
