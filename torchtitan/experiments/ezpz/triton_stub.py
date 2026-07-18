# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Workaround for the unconditional ``import triton`` in core torchtitan.

Core ``torchtitan/distributed/minimal_async_ep/kernels.py`` does a top-level
``import triton`` / ``import triton.language``. That module is pulled onto the
import path of *every* model through the chain::

    models/common/__init__ -> decoder.py -> minimal_async_ep/api.py -> kernels.py

triton ships no macOS wheels (any version), so on a Mac the import raises
``ModuleNotFoundError: No module named 'triton'``, which
``ConfigManager._load_config`` then reports (misleadingly) as
"Cannot import config_registry for module 'ezpz.agpt'".

The triton symbols in ``kernels.py`` are only exercised at *call* time inside
the host-side wrapper functions (``copy_full_counts_to_peers_kernel`` and
friends), which the MinimalAsyncEP token dispatcher invokes only when expert
parallelism is active. A dense CPU/MPS run never reaches them, so making the
*import* succeed is enough to run locally.

Why stub the ``kernels`` leaf module and not ``triton`` itself: torch's own
``torch.utils._triton.has_triton_package()`` does ``import triton`` to decide
whether ``torch._inductor`` should take its triton codegen path. A fake
``sys.modules["triton"]`` would flip that detection to ``True`` and then
``torch._inductor`` crashes on ``import triton.backends.compiler``. Stubbing
only ``minimal_async_ep.kernels`` leaves torch's triton detection correctly
reporting ``False``, so inductor takes its no-triton path unchanged.

The stub is a strict no-op when triton is importable (XPU ships
``pytorch-triton-xpu``, CUDA ships ``triton``), so this changes nothing on real
accelerator hardware. If a stubbed kernel is ever actually called, it raises a
clear ``RuntimeError`` rather than failing silently.
"""

import importlib.util
import sys
import types

from torchtitan.tools.logging import logger

_KERNELS_MODULE = "torchtitan.distributed.minimal_async_ep.kernels"


def _make_missing_kernel(kernel_name: str):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"MinimalAsyncEP triton kernel {kernel_name!r} was called, but triton "
            "is unavailable on this platform (no macOS wheel). This kernel is only "
            "reached under expert parallelism; run without EP, or on XPU/CUDA where "
            "triton is available."
        )

    return _raise


def maybe_install_minimal_async_ep_triton_stub() -> None:
    """Install a triton-free stub for ``minimal_async_ep.kernels`` if needed.

    No-op when triton is importable, or when the real kernels module has
    already been imported.
    """
    # triton present (XPU/CUDA): leave the real module import path alone.
    if importlib.util.find_spec("triton") is not None:
        return

    # Real kernels already imported (nothing to stub, would be too late anyway).
    if _KERNELS_MODULE in sys.modules:
        return

    def _stub_getattr(name: str):
        # PEP 562 module __getattr__: satisfies ``from kernels import <name>``
        # for every kernel host-wrapper name without enumerating them, and
        # returns a callable that raises only if the kernel is actually invoked.
        # Dunders (e.g. __path__, __all__) fall through to normal protocol.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _make_missing_kernel(name)

    stub = types.ModuleType(_KERNELS_MODULE)
    stub.__doc__ = "ezpz stub for minimal_async_ep.kernels (triton unavailable)."
    stub.__getattr__ = _stub_getattr
    sys.modules[_KERNELS_MODULE] = stub

    logger.info(
        "Installed ezpz triton-free stub for %s (triton not importable on this "
        "platform; MinimalAsyncEP kernels are call-time only and unused without EP).",
        _KERNELS_MODULE,
    )
