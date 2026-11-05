# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Import shim for the FSDP mesh resolvers across the full_dtensor removal.

Upstream `601cf4d23` (#4217, 2026-08-19) deletes
`torchtitan/distributed/full_dtensor.py`. The two functions ezpz needs moved:

    resolve_fsdp_mesh         -> torchtitan/distributed/fsdp.py
    resolve_sparse_fsdp_mesh  -> torchtitan/distributed/fsdp.py

with their logic unchanged. A third, `validate_config`, was removed outright:
there is no definition anywhere in upstream/main, and the models that used to
call it (llama3, deepseek_v3) simply dropped the call.

This module resolves the two survivors from whichever location exists, so
`ezpz/{agpt,moe}/parallelize.py` import from here and keep working both
before and after the sync. Without it the 80th sync is an immediate
ImportError in both files.

`validate_config` is deliberately NOT re-exported or reimplemented. It was a
pre-parallelize sanity check on the config/model pair; upstream decided it
was not needed once `model.parallelize()` runs unconditionally under the SPMD
backends, and inventing a local replacement would mean maintaining a
divergent check that upstream has no counterpart for. Drop the call sites
instead.

Polaris fallback: on torch builds that lack the spmd_types FSDP consumer
(da19cbd78, plain->DTensor conversion), passing dp_mesh_dims crashes with
"all parameters must be DTensors". This wrapper detects that torch and
returns dp_mesh_dims=None so callers fall back to legacy plain-tensor FSDP.
"""

from __future__ import annotations

try:
    # Post-#4217 location. Try this first so that once the sync lands we are
    # on the upstream path with no fallback in play.
    from torchtitan.distributed.fsdp import (  # noqa: F401
        resolve_fsdp_mesh as _resolve_fsdp_mesh_upstream,
        resolve_sparse_fsdp_mesh as _resolve_sparse_fsdp_mesh_upstream,
    )
except ImportError:  # pragma: no cover - only on pre-#4217 trees
    from torchtitan.distributed.full_dtensor import (  # noqa: F401
        resolve_fsdp_mesh as _resolve_fsdp_mesh_upstream,
        resolve_sparse_fsdp_mesh as _resolve_sparse_fsdp_mesh_upstream,
    )


def _torch_has_spmd_consumer() -> bool:
    """Whether this torch can consume spmd_types annotations in FSDP."""
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

        return hasattr(FSDPParam, "_resolve_spmd_types_for_storage")
    except Exception:
        return False


def resolve_fsdp_mesh(parallel_dims, **kwargs):
    mesh, dims = _resolve_fsdp_mesh_upstream(parallel_dims, **kwargs)
    if dims is not None and not _torch_has_spmd_consumer():
        # Polaris 2.13 lacks da19cbd78 -> plain tensor ValueError if dims passed
        return mesh, None
    return mesh, dims


def resolve_sparse_fsdp_mesh(parallel_dims, **kwargs):
    mesh, dims = _resolve_sparse_fsdp_mesh_upstream(parallel_dims, **kwargs)
    if dims is not None and not _torch_has_spmd_consumer():
        return mesh, None
    return mesh, dims


__all__ = ["resolve_fsdp_mesh", "resolve_sparse_fsdp_mesh"]
