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

from typing import Any


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

    job = StoreJob.from_store(store, name=name, **kwargs)
    if job is None:
        return None
    return getattr(job.state(), name)
