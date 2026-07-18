# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Make torchtitan run on a local laptop (macOS / Apple Silicon) for dev.

Two independent problems are handled here:

1. ``get_peak_flops`` shells out to ``lspci`` (Linux-only) and only catches
   ``FileNotFoundError``; on macOS the missing binary surfaces as
   ``PermissionError`` (``execvp`` reports the first ``EACCES`` from a
   non-searchable PATH entry), which crashes trainer construction before the
   assume-A100 fallback. We replace it with a subprocess-free stub.

2. The active ``device_module`` (resolved by core via
   ``torch._utils._get_device_module``) does not satisfy the CUDA/XPU accelerator
   contract the trainer/metrics/distributed code calls into:

   - **MPS** omits ``set_device`` / ``current_device`` / ``is_initialized`` /
     ``get_device_properties`` / peak-memory stats / the stream API. More
     fundamentally, PyTorch has **no MPS distributed backend**: FSDP/DTensor
     route parameter init and collectives through ``c10d``, and
     ``c10d::broadcast_`` (and friends) raise ``NotImplementedError`` on MPS
     with no CPU fallback. So a real FSDP train step cannot run on MPS.

   - **CPU** has the ``gloo`` backend, so collectives work, and ops all run.
     Setting ``TORCH_DEVICE=cpu`` forces this path. ``torch.cpu`` still lacks a
     few of the accelerator-contract functions the code calls (``empty_cache``,
     ``get_device_name``, ``get_device_properties``, ``memory_stats``,
     ``reset_peak_memory_stats``), which we fill in.

``TORCH_DEVICE=cpu`` (honored here for *core* torchtitan, which otherwise ignores
it and keeps selecting MPS) is the only path that actually completes a step
locally; the MPS accelerator shim is best-effort and lets everything up to the
first collective run on the GPU.

Everything here is local-dev convenience; neither MPS nor single-process CPU is a
supported production backend. All patches are guarded to no-op on real
accelerators (CUDA/XPU), and only fill in functions that are absent.
"""

import contextlib
import os
from typing import Any

import torch

from torchtitan.tools.logging import logger

_A100_PEAK_BF16_FLOPS = 312e12


class _InertStream:
    """No-op stand-in for ``torch.cuda.Stream`` on single-queue local devices."""

    def __init__(self, device: Any = None, *_args: Any, **_kwargs: Any) -> None:
        self.device = device

    def synchronize(self) -> None:
        pass

    def wait_stream(self, _other: Any) -> None:
        pass

    def __enter__(self) -> "_InertStream":
        return self

    def __exit__(self, *_exc: Any) -> None:
        pass


def _system_ram_bytes() -> int:
    """Total physical RAM in bytes, or a 1 GiB floor if it cannot be queried."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, AttributeError, OSError):
        return 1 << 30


def _fill_missing(module: Any, functions: dict[str, Any]) -> list[str]:
    """Set each name on ``module`` only if it is absent. Returns names added."""
    added: list[str] = []
    for name, fn in functions.items():
        if not hasattr(module, name):
            setattr(module, name, fn)
            added.append(name)
    return added


def _accelerator_contract_stubs(device_label: str) -> dict[str, Any]:
    """Common device_module functions the trainer/metrics/DeviceMesh code calls.

    Values are correct for a single local device: index is always 0, there is one
    device, it is always "initialized", and per-device memory counters that the
    backend does not track report 0.
    """

    class _DeviceProperties:
        name = device_label
        # Read by MFU math as an SM/compute-unit count; unknown here, so 0.
        multi_processor_count = 0
        # Used by the memory monitor as a divisor (percent-of-capacity), so it
        # must be nonzero. Report total system RAM as the local "device"
        # capacity.
        total_memory = _system_ram_bytes()

    @contextlib.contextmanager
    def _stream_ctx(_stream_obj: Any = None):
        yield

    return {
        "set_device": lambda _device=0: None,
        "current_device": lambda: 0,
        "device_count": lambda: 1,
        "is_initialized": lambda: True,
        "get_device_name": lambda _device=0: device_label,
        "get_device_properties": lambda _device=0: _DeviceProperties(),
        "empty_cache": lambda: None,
        "reset_peak_memory_stats": lambda _device=0: None,
        "max_memory_allocated": lambda _device=0: 0,
        "memory_allocated": lambda _device=0: 0,
        "memory_reserved": lambda _device=0: 0,
        "memory_stats": lambda _device=0: {},
        "manual_seed_all": lambda seed: torch.manual_seed(seed),
        "synchronize": lambda _device=0: None,
        "Stream": _InertStream,
        "current_stream": lambda _device=0: _InertStream(),
        "stream": _stream_ctx,
    }


def maybe_patch_get_peak_flops() -> None:
    """Replace ``utils.get_peak_flops`` with a subprocess-free stub off-Linux.

    Only patches when there is no CUDA/XPU accelerator (i.e. a local laptop). On
    real hardware the upstream lspci/device-name detection is left untouched.
    """
    if torch.cuda.is_available() or (
        hasattr(torch, "xpu") and torch.xpu.is_available()
    ):
        return

    from torchtitan.tools import utils as _tt_utils

    if getattr(_tt_utils.get_peak_flops, "_ezpz_local_patched", False):
        return

    def _local_get_peak_flops(device_name: str) -> float:
        logger.warning(
            "get_peak_flops: no GPU/lspci on this host; using A100 fallback "
            "(%.0fe12). MFU is not meaningful for local CPU/MPS dev.",
            _A100_PEAK_BF16_FLOPS / 1e12,
        )
        return _A100_PEAK_BF16_FLOPS

    _local_get_peak_flops._ezpz_local_patched = True  # type: ignore[attr-defined]
    _tt_utils.get_peak_flops = _local_get_peak_flops


def _maybe_force_cpu_device() -> bool:
    """Point core torchtitan at CPU when ``TORCH_DEVICE=cpu``.

    Core resolves ``device_type`` / ``device_module`` once at import via
    ``torch._utils`` and ignores ``TORCH_DEVICE`` (that env is ezpz-only), so on a
    Mac it keeps selecting MPS -- which has no distributed backend. We rebind the
    module-level ``device_type`` / ``device_module`` on ``torchtitan.tools.utils``
    to CPU so the ``from ... import device_module, device_type`` bindings in
    metrics/trainer (not yet imported at ezpz import time) pick up CPU.

    Returns True if the CPU force path was taken.
    """
    if os.environ.get("TORCH_DEVICE", "").strip().lower().split(":", 1)[0] != "cpu":
        return False

    from torch._utils import _get_device_module

    from torchtitan.tools import utils as _tt_utils

    cpu_module = _get_device_module("cpu")
    added = _fill_missing(cpu_module, _accelerator_contract_stubs("cpu"))

    _tt_utils.device_type = "cpu"
    _tt_utils.device_module = cpu_module

    logger.info(
        "TORCH_DEVICE=cpu: forced core torchtitan device_type=cpu (gloo "
        "collectives). Patched torch.cpu contract: %s.",
        ", ".join(sorted(added)) or "none needed",
    )
    return True


def _maybe_install_mps_shim() -> None:
    """Best-effort accelerator shim for ``torch.mps`` (compute up to collectives).

    MPS cannot run FSDP collectives, so this does not enable a full train step; it
    exists so non-distributed local experimentation on MPS gets further before
    hitting the c10d wall. No-op when MPS is unavailable.
    """
    if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        return

    stubs = _accelerator_contract_stubs("mps")
    # Prefer real torch.mps memory introspection where it exists.
    stubs["memory_allocated"] = lambda _device=0: torch.mps.current_allocated_memory()
    stubs["synchronize"] = lambda _device=0: torch.mps.synchronize()

    added = _fill_missing(torch.mps, stubs)
    if added:
        logger.info(
            "Installed ezpz MPS accelerator shim on torch.mps (added: %s). "
            "Local dev only; MPS has no distributed backend, so FSDP steps still "
            "require TORCH_DEVICE=cpu.",
            ", ".join(sorted(added)),
        )


def maybe_install_local_device_compat() -> None:
    """Install all local-laptop compatibility patches. No-op on CUDA/XPU."""
    if torch.cuda.is_available() or (
        hasattr(torch, "xpu") and torch.xpu.is_available()
    ):
        return

    maybe_patch_get_peak_flops()
    if not _maybe_force_cpu_device():
        _maybe_install_mps_shim()
