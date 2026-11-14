# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

# Processes that Monarch spawns for RL actors are not MPI ranks. On hosts named
# ``x3*`` (Polaris, Sirius) ``import ezpz`` runs ``from mpi4py import MPI``,
# whose ``MPI_Init`` aborts there (MPICH ``internal_Init_thread``, exit 15) on
# the PMI environment inherited from the launching rank. The RL provisioner
# sets this flag for those processes; it must apply before ``ezpz`` is imported.
if os.environ.get("EZPZ_NO_MPI_INIT") == "1":
    try:
        import mpi4py

        mpi4py.rc.initialize = False
        mpi4py.rc.finalize = False
    except ImportError:
        pass

# Install the triton-free stub for core ``minimal_async_ep.kernels`` before any
# ezpz submodule (agpt/moe/qwen3) pulls the core model import chain that ends in
# an unconditional ``import triton``. No-op when triton is importable (XPU/CUDA).
# See ``triton_stub.py`` for the full rationale.
from torchtitan.experiments.ezpz.triton_stub import (
    maybe_install_minimal_async_ep_triton_stub,
)

maybe_install_minimal_async_ep_triton_stub()

# Local laptop (macOS) dev support: force CPU when TORCH_DEVICE=cpu (core
# torchtitan otherwise keeps selecting MPS, which has no distributed backend),
# shim the accelerator device_module contract, and stop get_peak_flops from
# shelling out to lspci. No-op on CUDA/XPU. See ``local_device_compat.py``.
from torchtitan.experiments.ezpz.local_device_compat import (  # noqa: E402
    maybe_install_local_device_compat,
)

maybe_install_local_device_compat()

__all__ = []
