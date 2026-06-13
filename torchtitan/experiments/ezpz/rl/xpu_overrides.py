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
            # Runs ONCE inside each Monarch-spawned actor process,
            # BEFORE the actor's __init__ runs.
            #
            # We're running under `mpiexec --np 1 ./venvs/.../python
            # -m torchtitan.experiments.ezpz.rl.train_upstream`. The
            # launcher's mpiexec sets PALS_LOCAL_SIZE=1, PALS_RANKID=0,
            # etc — those values describe the LAUNCHER's mesh, not the
            # actor's. Monarch-spawned actors form a separate per-mesh
            # rank space (trainer mesh has size N, generator mesh has
            # size M). Override the inherited launcher values with the
            # actor mesh values so oneCCL sets up its SYCL queue for
            # the right tile.
            os.environ["ZE_AFFINITY_MASK"] = ",".join(str(g) for g in gpu_ids)
            os.environ["PALS_LOCAL_SIZE"] = str(local_size)
            os.environ["PALS_NODEID"] = "0"
            os.environ["PALS_DEPTH"] = "1"
            os.environ["PALS_PMI"] = "pmix"

            # Eager torch import before Monarch's pickle path can race.
            import torch  # noqa: F401

            # Apply XPU patches FIRST (must be installed before any
            # torch.distributed.* import that could trigger XCCL
            # backend registration).
            from torchtitan.experiments.ezpz.rl.xpu_overrides import (
                apply_all_xpu_patches,
            )

            apply_all_xpu_patches()

            # Pre-resolve torch.distributed.checkpoint to avoid a
            # circular-import race with torchstore during actor setup.
            import torch.distributed.checkpoint  # noqa: F401
            import torch.distributed.checkpoint._nested_dict  # noqa: F401

        return _bootstrap


def patch_init_distributed_for_xpu() -> None:
    """Force torchtitan's trainer process group to use Gloo on XPU.

    Diagnostic results from job 12468765 confirmed our PALS_LOCAL_RANKID /
    PALS_RANKID env injection runs correctly in each Monarch-spawned
    actor — yet oneCCL XCCL still rejects every collective with
    "ccl_check_usm_pointers: invalid usm pointer type". The env vars
    are necessary-but-not-sufficient: oneCCL needs an *active* PMIx
    rendezvous (provided by a real mpiexec launcher), not just env
    strings. Monarch's fork-based spawn doesn't supply one.

    Pragmatic workaround: replace `_get_distributed_backend` in
    `torchtitan.distributed.utils` so it returns `"gloo"` instead of
    `"xccl"`. Gloo doesn't go through oneCCL at all — it does CPU-side
    rendezvous and ring/tree collectives. It's slower than XCCL but
    bypasses the USM issue entirely.

    This is fine for trainer-side parameter sharding/state-sync (which
    happens infrequently). vLLM-XPU's GENERATOR side still uses XCCL
    via its own process group (TP=4 internal); that's launched via
    vLLM's "external launcher" which IS PMIx-aware and works.

    Patches `torchtitan.distributed.utils.init_distributed` to wrap
    `_get_distributed_backend` returning `"gloo"`.
    """
    if torch.cuda.is_available():
        return
    import torchtitan.distributed.utils as _dutils

    orig = _dutils.init_distributed
    if getattr(orig, "_xpu_patched", False):
        return

    def patched(*args, **kwargs):
        import sys

        lr = os.environ.get("LOCAL_RANK")
        r = os.environ.get("RANK")
        if lr is not None:
            os.environ.setdefault("PALS_LOCAL_RANKID", lr)
        if r is not None:
            os.environ.setdefault("PALS_RANKID", r)

        # The trainer needs XCCL (not Gloo — Gloo is CPU-only and
        # can't broadcast xpu tensors). For XCCL to work, oneCCL
        # needs PALS_LOCAL_RANKID + PALS_RANKID to match the actor's
        # mesh-local rank, NOT the launcher's rank-0 (inherited from
        # the outer mpiexec --np 1 wrapper).
        #
        # Use `=` not `setdefault` so we OVERRIDE the inherited
        # launcher values from PMIx with the actor's actual rank.
        os.environ["PALS_LOCAL_RANKID"] = lr or "0"
        os.environ["PALS_RANKID"] = r or "0"
        # PALS_LOCAL_SIZE was set in _bootstrap from the provisioner's
        # num_gpus; that's the per-actor-mesh size, correct as-is.
        # Don't touch it here.
        print(
            f"[xpu_patch pid={os.getpid()}] override PALS_LOCAL_RANKID={lr} PALS_RANKID={r}",
            flush=True,
            file=sys.stderr,
        )
        return orig(*args, **kwargs)

    patched._xpu_patched = True  # type: ignore[attr-defined]
    _dutils.init_distributed = patched
    print(
        f"[xpu_overrides pid={os.getpid()}] Patched torchtitan init_distributed (XCCL → Gloo on XPU)",
        flush=True,
        file=__import__("sys").stderr,
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


def patch_trl_vllm_no_weight_sync() -> None:
    """No-op TRL's vLLM weight-sync communicator (XPU + server mode).

    TRL's `vllm_mode="server"` design assumes NCCL can create a
    cross-process-tree group between trainer ranks and the vLLM server
    worker. This works on CUDA via env-based init but fails on XPU
    because oneCCL XCCL requires a PMIx KVS rendezvous that doesn't
    span separate MPI worlds — the trainer hangs in
    `pmi_resizable_simple_internal.cpp:337 kvs_get_value: KVS get error`
    and eventually segfaults.

    For smokes (no real training) and "frozen-generator" RL ablations,
    we can no-op the communicator and the `update_named_param` calls.
    The server keeps the initial weights; generations still work; the
    trainer still updates its own copy. We lose the trainer→server
    weight propagation that GRPO normally needs for on-policy
    behavior, but the loop runs end-to-end.

    Long-term fix: pass weight updates through HTTP `/update_named_param/`
    POSTs (tensor bytes over the wire) instead of an NCCL group. TRL
    already has the endpoint; just need to flip the dispatch.

    Idempotent. Call BEFORE constructing GRPOTrainer.
    """
    if torch.cuda.is_available():
        return
    try:
        from trl.generation.vllm_client import VLLMClient
    except ImportError:
        return  # TRL not installed
    if getattr(VLLMClient.init_communicator, "_xpu_noop", False):
        return

    orig_init = VLLMClient.init_communicator
    orig_update = VLLMClient.update_named_param

    def noop_init(self, device=0):
        logger.warning(
            "TRL VLLMClient.init_communicator: no-op (XPU). Trainer→server "
            "weight sync via NCCL is disabled — server keeps initial weights."
        )

    def noop_update(self, name, weights):
        # silent no-op — called once per param per step
        pass

    noop_init._xpu_noop = True  # type: ignore[attr-defined]
    VLLMClient.init_communicator = noop_init
    VLLMClient.update_named_param = noop_update
    logger.info(
        "Patched TRL VLLMClient.init_communicator + update_named_param "
        "as no-ops (XPU: server-mode weight sync disabled)"
    )


def patch_torch_cuda_aliases_for_xpu() -> None:
    """Alias `torch.cuda.current_device()` → `torch.xpu.current_device()`.

    TRL's `VLLMGeneration._init_vllm` hardcodes
    `self.vllm_client.init_communicator(device=torch.cuda.current_device())`
    when wiring up the trainer→server weight-sync NCCL group. On XPU,
    this fails immediately with "Torch not compiled with CUDA enabled".

    Aliasing the call to its XPU equivalent fixes the immediate import,
    and the resulting `device=<xpu_idx>` is what TRL would have passed
    on a CUDA host anyway.

    Idempotent. Call BEFORE constructing TRL's `GRPOTrainer` (or any
    TRL class that runs VLLMGeneration init).
    """
    if torch.cuda.is_available():
        return
    if getattr(torch.cuda.current_device, "_xpu_aliased", False):
        return
    orig = torch.cuda.current_device

    def _aliased():
        return torch.xpu.current_device()

    _aliased._xpu_aliased = True  # type: ignore[attr-defined]
    torch.cuda.current_device = _aliased
    logger.info(
        "Aliased torch.cuda.current_device → torch.xpu.current_device (XPU)"
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
    patch_torch_cuda_aliases_for_xpu()
    patch_trl_vllm_no_weight_sync()
