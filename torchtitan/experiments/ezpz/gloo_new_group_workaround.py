# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Workaround for new_group(backend="gloo") on an xccl-only XPU default PG.

Core torchtitan CheckpointManager.__init__ (components/checkpoint.py:515)
creates a CPU-side staging process group via::

    self.pg = dist.new_group(backend="gloo")

when async_mode in (ASYNC, ASYNC_WITH_PINNED_MEM). On XPU this raises::

    RuntimeError: No backend type associated with device type xpu

The cause is NOT that gloo is unbuildable. new_group builds the gloo group
fine, then hits an eager-connect optimization in
torch.distributed.distributed_c10d._new_group_with_tag (torch 2.13,
line ~5906)::

    if device_id and pg._get_backend(device_id).supports_splitting:
        ...

The new gloo group inherited device_id = xpu:N from the eager-bound xccl
default PG. pg._get_backend(xpu) on the gloo-only group has no xpu backend
registered, so C++ getBackend raises. Adding cpu:gloo to the *default*
PG (enable_cpu_backend=True) does NOT help: the gloo *subgroup* still
inherits bound_device_id=xpu and the same eager-connect fires (verified
empirically, job 8681375).

Fix (mirrors xccl_split_group_workaround.py): wrap dist.new_group so
that for backend="gloo" calls we temporarily clear the default PG's
bound_device_id. With device_id unset, the eager-connect guard at
line ~5906 is False, _get_backend(xpu) is skipped, and the gloo group
builds cleanly on cpu. xccl/xpu groups pass through untouched. Idempotent;
no-op off XPU. Remove when upstream makes the checkpointer gloo-subgroup
creation defensive, or binds gloo universally on XPU. See
docs/upstream-issues/checkpoint_async_gloo_on_xpu.md.
"""

import torch
import torch.distributed as dist

_PATCHED_ATTR = "_ezpz_gloo_new_group_patched"


def _should_patch() -> bool:
    try:
        return torch.xpu.is_available()
    except Exception:
        return False


def maybe_install_gloo_new_group_workaround() -> None:
    """Install the gloo new_group workaround if needed (idempotent, no-op off XPU)."""
    if not _should_patch():
        return
    if getattr(dist, _PATCHED_ATTR, False):
        return

    original_new_group = dist.new_group

    def _patched_new_group(*args, **kwargs):
        # backend is the 3rd positional arg or the 'backend' kwarg.
        backend = kwargs.get("backend", args[2] if len(args) > 2 else None)
        if backend is None or str(backend).lower() != "gloo":
            return original_new_group(*args, **kwargs)
        # gloo group: clear the default PG's bound_device_id (xpu) for the
        # duration of the call so the eager-connect _get_backend(xpu) guard
        # is skipped. Restore it afterward so xccl group creation is unaffected.
        try:
            default_pg = dist.distributed_c10d._get_default_group()
        except Exception:
            return original_new_group(*args, **kwargs)
        saved = getattr(default_pg, "bound_device_id", None)
        try:
            default_pg.bound_device_id = None
            return original_new_group(*args, **kwargs)
        finally:
            default_pg.bound_device_id = saved

    dist.new_group = _patched_new_group
    setattr(dist, _PATCHED_ATTR, True)
