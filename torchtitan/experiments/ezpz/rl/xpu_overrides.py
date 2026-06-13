# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""XPU shims for upstream `torchtitan.experiments.rl/`.

Upstream `rl/` has two CUDA-specific touch points that need shimming
on Intel XPU (Aurora / Sunspot):

1. **`has_cuda_capability(9, 0)` checks** (referenced from
   `actors/generator.py:40,465` and `models/attention.py:20,70`).
   These gate FA3-vs-FA2 selection and a `block_size=256` override.
   XPU is "not Hopper" → take the FA2 / non-Hopper branch
   unconditionally. Monkey-patched in `torchtitan.tools.utils` so the
   real `attention.py` and `generator.py` see the patched version.

2. **`PerHostProvisioner`** in `train.py:45` — partitions GPUs across
   trainer/generator meshes by exporting `CUDA_VISIBLE_DEVICES`. The
   XPU equivalent for partitioning visibility is `ZE_AFFINITY_MASK`
   (Level Zero). Replaced wholesale in `ezpz/rl/train_upstream.py`.

We do NOT stub `models/attention.py` — vllm-xpu actually ships
`vllm.v1.attention.backends.flash_attn` (just like vllm-cuda does),
so the file imports cleanly. The custom `PyTorchVarlenAttentionBackend`
it registers is never selected by vllm-xpu (which uses
`AttentionBackendEnum.CUSTOM` only if explicitly chosen in EngineArgs),
so leaving the registration intact is harmless.

Used by `ezpz/rl/train_upstream.py` (the ezpz mirror of `rl/train.py`
that applies the patches before import-time triggers them).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import torch


logger = logging.getLogger(__name__)


def has_xpu_kernels(*_args, **_kwargs) -> bool:
    """Replacement for `has_cuda_capability(9, 0)` checks.

    Always returns False on XPU — semantically equivalent to "this is
    not a Hopper GPU", which is the branch that selects the FA2 code
    path (block_size=256). XPU has neither FA3 nor SM-version, so the
    non-Hopper path is the only sensible choice.

    Args are accepted and ignored so this can be a drop-in for any
    `has_cuda_capability(major, minor)` call site.
    """
    return False


class EzpzPerHostProvisioner:
    """XPU equivalent of upstream `PerHostProvisioner`.

    Upstream uses ``CUDA_VISIBLE_DEVICES`` to partition GPUs across
    trainer / generator meshes on a single host. The XPU equivalent
    for limiting Level Zero device visibility is ``ZE_AFFINITY_MASK``.

    On Aurora / Sunspot with the recommended FLAT device hierarchy
    (``ZE_FLAT_DEVICE_HIERARCHY=FLAT``), each PVC card exposes two
    tiles and a node has 12 tiles total. Each ``allocate(n)`` reserves
    the next *n* tile indices and returns a bootstrap callable that
    sets ``ZE_AFFINITY_MASK`` before torch.xpu initializes in the
    spawned process.
    """

    def __init__(self, total_gpus: int = 12):
        self.total_gpus = total_gpus
        self.next_gpu = 0

    @property
    def available(self) -> int:
        return self.total_gpus - self.next_gpu

    def allocate(self, num_gpus: int) -> Callable[[], None]:
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} "
                f"available (total={self.total_gpus}, allocated={self.next_gpu})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus
        # capture once for the closure
        local_size = num_gpus

        def _bootstrap():
            os.environ["ZE_AFFINITY_MASK"] = ",".join(str(g) for g in gpu_ids)
            # Inherited from upstream — eager torch import before
            # Monarch's pickle path can race.
            import torch  # noqa: F401

            # Monarch's spawn_procs forks fresh processes without the
            # PMI/PALS env that oneCCL needs to initialize its SYCL
            # queue per tile. Without these, every XCCL collective
            # fails the USM pointer check
            # ("ccl_check_usm_pointers: invalid usm pointer type").
            #
            # We don't know the actor's coordinate inside this
            # callable (Monarch doesn't pass it), but the actor will
            # set `LOCAL_RANK` from torch.distributed env when it
            # calls init_process_group. We can pre-populate the PALS
            # vars from the actor's MONARCH-side hostname env that
            # gets set when the proc mesh wires things up.
            #
            # Strategy: only set the env vars that don't depend on
            # rank (they're static for the host); the rank-specific
            # ones (PALS_RANKID, PALS_LOCAL_RANKID) get filled in by
            # the actor itself from its os.environ["LOCAL_RANK"]
            # once init_distributed has set that up.
            os.environ.setdefault("PALS_LOCAL_SIZE", str(local_size))
            os.environ.setdefault("PALS_NODEID", "0")
            os.environ.setdefault("PALS_DEPTH", "1")
            os.environ.setdefault("PALS_PMI", "pmix")

        return _bootstrap


def patch_init_distributed_for_xpu() -> None:
    """Inject PALS_LOCAL_RANKID/PALS_RANKID into env before init_process_group.

    oneCCL's XCCL backend reads these to set up per-tile SYCL queues.
    Monarch-spawned actors don't have them; torch.distributed sets
    LOCAL_RANK / RANK, so we can mirror those into the PALS vars right
    before the process group init.

    Patches `torchtitan.distributed.utils.init_distributed`.
    """
    if torch.cuda.is_available():
        return
    import torchtitan.distributed.utils as _dutils

    orig = _dutils.init_distributed
    if getattr(orig, "_xpu_patched", False):
        return

    def patched(*args, **kwargs):
        # Mirror LOCAL_RANK / RANK into PALS env so oneCCL sees them.
        lr = os.environ.get("LOCAL_RANK")
        r = os.environ.get("RANK")
        if lr is not None:
            os.environ.setdefault("PALS_LOCAL_RANKID", lr)
        if r is not None:
            os.environ.setdefault("PALS_RANKID", r)
        return orig(*args, **kwargs)

    patched._xpu_patched = True  # type: ignore[attr-defined]
    _dutils.init_distributed = patched
    logger.info(
        "Patched torchtitan.distributed.utils.init_distributed to inject PALS_* env vars for oneCCL XPU"
    )


def patch_has_cuda_capability_for_xpu() -> None:
    """Monkey-patch `torchtitan.tools.utils.has_cuda_capability`.

    Replaces it with `has_xpu_kernels` (always False) so the FA3-vs-FA2
    branches throughout `rl/` take the FA2 path on XPU. Call BEFORE
    importing anything from `torchtitan.experiments.rl`.

    Idempotent.
    """
    if torch.cuda.is_available():
        logger.info("CUDA available; not patching has_cuda_capability")
        return
    import torchtitan.tools.utils as _utils

    if getattr(_utils.has_cuda_capability, "_xpu_patched", False):
        return
    _utils.has_cuda_capability = has_xpu_kernels
    _utils.has_cuda_capability._xpu_patched = True  # type: ignore[attr-defined]
    logger.info("Patched torchtitan.tools.utils.has_cuda_capability for XPU")


def patch_dtensor_rng_broadcast_for_xpu() -> None:
    """Skip the world_size=1 RNG-state broadcast in DTensor.

    `torch.distributed.tensor._random.OffsetBasedRNGTracker.__init__`
    unconditionally calls `torch.distributed.broadcast(rng_state, 0)`
    to sync rank-0's RNG state across the mesh. On XPU, `rng_state`
    is a CPU ByteTensor moved to the XPU device, and oneCCL XCCL
    rejects the broadcast with "ccl_check_usm_pointers: invalid usm
    pointer type: unknown for device type: gpu" — the .to(xpu) path
    doesn't produce a SYCL-USM-device-allocated tensor.

    At world_size=1 the broadcast is a no-op, so we can simply skip
    it. At world_size>1 we still trip the bug; that needs an upstream
    fix in torch.xpu's allocator or in oneCCL's USM check.

    Idempotent.
    """
    if torch.cuda.is_available():
        return
    import torch.distributed as dist
    import torch.distributed.tensor._random as _dtrand

    orig_init = _dtrand.OffsetBasedRNGTracker.__init__
    if getattr(orig_init, "_xpu_patched", False):
        return

    def patched_init(self, device_mesh, run_state_sync=True):
        # If we're single-rank on this mesh, the broadcast is a no-op.
        # Force run_state_sync=False to skip the offending call.
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return orig_init(self, device_mesh, run_state_sync=False)
        # World > 1: same call path, will still trip the USM check.
        # TODO: replace rng_state.to(self._device) with a torch.xpu.empty()+copy_
        #       so the destination tensor is SYCL-USM-allocated.
        return orig_init(self, device_mesh, run_state_sync=run_state_sync)

    patched_init._xpu_patched = True  # type: ignore[attr-defined]
    _dtrand.OffsetBasedRNGTracker.__init__ = patched_init
    logger.info(
        "Patched OffsetBasedRNGTracker.__init__ to skip world_size=1 RNG broadcast on XPU"
    )


def apply_all_xpu_patches() -> None:
    """Apply every XPU compatibility patch before importing upstream rl/.

    Call this at the top of any entrypoint that pulls in
    `torchtitan.experiments.rl.*`. Currently:
      1. `has_cuda_capability` → always False on XPU.
      2. `OffsetBasedRNGTracker.__init__` → skip the broadcast at
         world_size=1.
      3. `init_distributed` → inject PALS_LOCAL_RANKID / PALS_RANKID
         env vars from torch's LOCAL_RANK / RANK, so oneCCL XCCL
         can set up its per-tile SYCL queue. (Without these, every
         XCCL collective fails the USM pointer check.)
    Provisioner replacement is done in the entrypoint itself.
    """
    patch_has_cuda_capability_for_xpu()
    patch_dtensor_rng_broadcast_for_xpu()
    patch_init_distributed_for_xpu()
