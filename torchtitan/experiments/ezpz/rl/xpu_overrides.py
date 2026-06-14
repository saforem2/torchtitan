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
            # CRITICAL: on XPU, each actor must get a SINGLE-TILE
            # ZE_AFFINITY_MASK (not the full mesh range). With
            # `ZE_AFFINITY_MASK=0,1` set on both trainer actors, both
            # processes see both tiles and oneCCL's USM check fails
            # because the SYCL contexts get confused about which tile
            # each tensor lives on.
            #
            # We don't have per-actor rank info inside this closure
            # (Monarch passes the full gpu_ids list to every actor).
            # But each actor's LOCAL_RANK is set by Monarch in env
            # BEFORE _bootstrap runs. Use it to pick a single tile
            # from gpu_ids.
            # Diagnostic dump: which env vars CAN we use to identify this actor?
            import sys as _sys

            _candidate_keys = [
                "LOCAL_RANK", "RANK", "WORLD_SIZE",
                "MONARCH_RANK", "MONARCH_LOCAL_RANK", "MONARCH_WORLD_SIZE",
                "HYPERACTOR_RANK", "HYPERACTOR_LOCAL_RANK",
                "PALS_LOCAL_RANKID", "PALS_RANKID",
                "PMI_RANK", "PMI_LOCAL_RANK",
                "PMIX_RANK", "PMIX_LOCAL_RANK",
            ]
            _hits = {k: os.environ[k] for k in _candidate_keys if k in os.environ}
            _monarch_kv = {
                k: os.environ[k]
                for k in os.environ
                if "MONARCH" in k.upper() or "HYPERACTOR" in k.upper()
            }
            print(
                f"[xpu_bootstrap pid={os.getpid()}] gpu_ids={gpu_ids} "
                f"candidate-env={_hits} monarch-env={_monarch_kv}",
                flush=True,
                file=_sys.stderr,
            )
            # Monarch encodes the actor's identity in HYPERACTOR_PROCESS_NAME
            # (as 'anon-N<...>'). Extract N as the per-actor rank within
            # this mesh and use it to pick a single tile from gpu_ids.
            #
            # Falls back to full-mesh ZE_AFFINITY_MASK (CUDA-style) if we
            # can't parse — e.g. when run outside Monarch.
            my_tile = None
            import re

            hpn = os.environ.get("HYPERACTOR_PROCESS_NAME", "")
            m = re.search(r"anon-(\d+)", hpn)
            if m:
                rank_in_mesh = int(m.group(1))
                if 0 <= rank_in_mesh < len(gpu_ids):
                    my_tile = gpu_ids[rank_in_mesh]
            if my_tile is not None:
                os.environ["ZE_AFFINITY_MASK"] = str(my_tile)
                # Each actor process now sees exactly ONE xpu tile,
                # re-indexed locally as xpu:0. Upstream rl/ does
                # `torch.xpu.set_device(int(os.environ['LOCAL_RANK']))`,
                # which would crash for rank_in_mesh>0 because xpu:N
                # is out of range. Override LOCAL_RANK to 0 so the
                # device lookup hits the (only) visible tile.
                #
                # Set PALS_LOCAL_RANKID to the REAL rank_in_mesh now
                # (before init_distributed's patched mirror would
                # otherwise copy from LOCAL_RANK=0 across all actors).
                os.environ["LOCAL_RANK"] = "0"
                os.environ["PALS_LOCAL_RANKID"] = str(rank_in_mesh)
                os.environ["PALS_RANKID"] = str(rank_in_mesh)
                print(
                    f"[xpu_bootstrap pid={os.getpid()}] HYPERACTOR_PROCESS_NAME={hpn!r} "
                    f"→ rank_in_mesh={rank_in_mesh} → ZE_AFFINITY_MASK={my_tile}, "
                    f"LOCAL_RANK=0, PALS_*_RANKID={rank_in_mesh}",
                    flush=True,
                    file=_sys.stderr,
                )
            else:
                os.environ["ZE_AFFINITY_MASK"] = ",".join(str(g) for g in gpu_ids)
                print(
                    f"[xpu_bootstrap pid={os.getpid()}] could not parse rank from "
                    f"HYPERACTOR_PROCESS_NAME={hpn!r}, falling back to "
                    f"ZE_AFFINITY_MASK={os.environ['ZE_AFFINITY_MASK']}",
                    flush=True,
                    file=_sys.stderr,
                )
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
    """Override per-rank PALS_*_RANKID right before init_process_group.

    Wraps `torchtitan.distributed.utils.init_distributed` and, just
    before the first XCCL collective fires, OVERRIDES the inherited
    PALS_LOCAL_RANKID / PALS_RANKID env vars with the actor's actual
    LOCAL_RANK / RANK. This matters when launching under
    `mpiexec --np 1` (the controller's launcher), which sets
    PALS_RANKID=0 on every child even though Monarch spawns N children
    with logical ranks 0..N-1.

    Pairs with `setup_oneccl_tcp_kvs_for_xpu()` which sets the static
    CCL_*/FI_* knobs in _bootstrap. Together they let oneCCL form an
    XCCL group across Monarch-spawned actors using TCP-KVS rendezvous
    (no PMIx parent needed).
    """
    if torch.cuda.is_available():
        return
    import torchtitan.distributed.utils as _dutils

    orig = _dutils.init_distributed
    if getattr(orig, "_xpu_patched", False):
        return

    def patched(*args, **kwargs):
        lr = os.environ.get("LOCAL_RANK")
        r = os.environ.get("RANK")
        if lr is not None:
            os.environ.setdefault("PALS_LOCAL_RANKID", lr)
        if r is not None:
            os.environ.setdefault("PALS_RANKID", r)

        # Override (not setdefault) so we replace the inherited
        # launcher's PALS_RANKID=0 with this actor's logical rank.
        os.environ["PALS_LOCAL_RANKID"] = lr or "0"
        os.environ["PALS_RANKID"] = r or "0"
        # PALS_LOCAL_SIZE was set in _bootstrap from the provisioner's
        # num_gpus; that's the per-actor-mesh size, correct as-is.
        ws = os.environ.get("WORLD_SIZE")
        print(
            f"[xpu_patch pid={os.getpid()}] env dump at init_distributed: "
            f"MASTER_ADDR={os.environ.get('MASTER_ADDR')!r} "
            f"MASTER_PORT={os.environ.get('MASTER_PORT')!r} "
            f"RANK={r!r} WORLD_SIZE={ws!r} LOCAL_RANK={lr!r}",
            flush=True,
            file=sys.stderr,
        )
        return orig(*args, **kwargs)

    patched._xpu_patched = True  # type: ignore[attr-defined]
    _dutils.init_distributed = patched
    print(
        f"[xpu_overrides pid={os.getpid()}] Patched torchtitan init_distributed (PALS_*_RANKID override)",
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
    """Fix DTensor's RNG-state broadcast for XPU.

    `torch.distributed.tensor._random.OffsetBasedRNGTracker.__init__`
    calls `torch.distributed.broadcast(rng_state, 0)` to sync rank-0's
    RNG state across the mesh. On XPU, `rng_state` is built via
    `device_handle.get_rng_state().to(self._device)` — a CPU ByteTensor
    moved to XPU via `.to()`. The resulting tensor's allocation is NOT
    SYCL-USM-device, so oneCCL XCCL rejects the broadcast with
    "ccl_check_usm_pointers: invalid usm pointer type: unknown".

    Two fixes:
      1. At world_size=1, skip the broadcast (no-op anyway).
      2. At world_size>1, replace `_get_device_state` to allocate via
         `torch.empty(..., device=...)` (which DOES go through the
         caching allocator → USM device memory) and copy_ into it.
    """
    if torch.cuda.is_available():
        return
    import torch.distributed as dist
    import torch.distributed.tensor._random as _dtrand

    orig_init = _dtrand.OffsetBasedRNGTracker.__init__
    orig_get_state = _dtrand.OffsetBasedRNGTracker._get_device_state
    if getattr(orig_init, "_xpu_patched", False):
        return

    def patched_get_device_state(self):
        # Always allocate a fresh device-side USM tensor and copy from
        # the CPU rng_state into it. The plain `.to(device)` path
        # produces a tensor whose allocation oneCCL can't classify.
        # `torch.empty(..., device=...)` goes through the caching
        # allocator → SYCL USM device memory → recognized by oneCCL.
        cpu_state = self._device_handle.get_rng_state()
        device_state = torch.empty_like(cpu_state, device=self._device)
        device_state.copy_(cpu_state)
        return device_state

    def patched_init(self, device_mesh, run_state_sync=True):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return orig_init(self, device_mesh, run_state_sync=False)
        return orig_init(self, device_mesh, run_state_sync=run_state_sync)

    patched_init._xpu_patched = True  # type: ignore[attr-defined]
    _dtrand.OffsetBasedRNGTracker.__init__ = patched_init
    _dtrand.OffsetBasedRNGTracker._get_device_state = patched_get_device_state
    logger.info(
        "Patched OffsetBasedRNGTracker for XPU: skip ws=1 broadcast + "
        "_get_device_state uses torch.empty(device=...) for USM allocation"
    )


def setup_oneccl_tcp_kvs_for_xpu() -> None:
    """Configure oneCCL to use TCP-KVS rendezvous (not PMI/MPI bootstrap).

    Discovery (2026-06-13 PM): a 2-process cross-tree XCCL broadcast on
    plain xpu tensors works when we set:

        CCL_PROCESS_LAUNCHER=none       (no MPI bootstrap expected)
        CCL_ATL_TRANSPORT=ofi           (OFI transport)
        FI_PROVIDER=tcp                 (TCP fabric, not Slingshot CXI)
        CCL_KVS_IP_PORT=127.0.0.1_PORT  (TCP-based KVS endpoint)
        unset CCL_OP_SYNC
        unset FI_CXI_*

    Without this, oneCCL hits `pmi_resizable_simple_internal.cpp:337
    kvs_get_value` timeouts and segfaults when two separate process
    trees (trainer + vllm-serve) try to form an XCCL group.

    Call from the actor `_bootstrap` BEFORE any torch.distributed XCCL
    init. The settings affect ALL XCCL groups in this process — including
    the trainer's intra-mesh group AND the trainer↔server weight-sync
    group. TCP-KVS works for both; the perf hit vs Slingshot CXI on
    intra-node collectives is acceptable for first-validation.

    Idempotent. Call from `_bootstrap` (which runs once per Monarch
    actor) or from the entrypoint just after `ezpz_setup_job`.
    """
    if torch.cuda.is_available():
        return
    os.environ["CCL_PROCESS_LAUNCHER"] = "none"
    os.environ["CCL_ATL_TRANSPORT"] = "ofi"
    os.environ["FI_PROVIDER"] = "tcp"
    # If the caller hasn't set CCL_KVS_IP_PORT, leave it alone — it
    # needs to be agreed upon by both ends of the cross-process group.
    # The launcher should set it before invoking python.
    os.environ.pop("CCL_OP_SYNC", None)
    for _k in (
        "FI_CXI_DEFAULT_CQ_SIZE",
        "FI_CXI_DEFAULT_TX_SIZE",
        "FI_CXI_OFLOW_BUF_COUNT",
        "FI_CXI_OFLOW_BUF_SIZE",
        "FI_CXI_RDZV_EAGER_SIZE",
        "FI_CXI_RDZV_THRESHOLD",
        "FI_CXI_REQ_BUF_MAX_CACHED",
        "FI_CXI_REQ_BUF_MIN_POSTED",
        "FI_CXI_REQ_BUF_SIZE",
        "FI_CXI_RX_MATCH_MODE",
        "FI_MR_CACHE_MAX_COUNT",
        "FI_MR_CACHE_MAX_SIZE",
        "FI_LOG_LEVEL",
        "FI_LOG_PROV",
        "FI_LOG_LOCATION",
    ):
        os.environ.pop(_k, None)
    logger.info(
        "Configured oneCCL for TCP-KVS rendezvous: CCL_PROCESS_LAUNCHER=none "
        "CCL_ATL_TRANSPORT=ofi FI_PROVIDER=tcp (CCL_KVS_IP_PORT=%s)",
        os.environ.get("CCL_KVS_IP_PORT", "<unset>"),
    )


def patch_torch_xpu_set_device_for_single_tile() -> None:
    """Pin all xpu device references in this process to `xpu:0`.

    When ZE_AFFINITY_MASK pins this process to a single tile, the
    only visible XPU device is `xpu:0` (re-indexed locally). But
    upstream `rl/actors/trainer.py:186` reads
    `int(os.environ['LOCAL_RANK'])` and uses it as an xpu device
    index. Monarch's lifecycle resets LOCAL_RANK between our
    `_bootstrap` override (LOCAL_RANK=0) and the trainer's
    `__init__` (LOCAL_RANK=1 for anon-1).

    Cleanest fix: replace `os.environ` with a wrapper that always
    returns `"0"` for `LOCAL_RANK` lookups, while passing every
    other key through unchanged. Then any code reading
    `os.environ['LOCAL_RANK']` gets the device index that actually
    works in this single-tile-masked process, regardless of who
    last wrote to the underlying env.

    The REAL per-actor rank is preserved in `PALS_*_RANKID` /
    `RANK`, which is what torch.distributed env-rendezvous uses.

    Also patches `torch.xpu.set_device` as belt-and-braces in case
    something computes the index from a different source.

    Idempotent. Call from _bootstrap BEFORE any torch.xpu use.
    """
    if torch.cuda.is_available():
        return
    if getattr(torch.xpu.set_device, "_xpu_pinned", False):
        return

    # 1. Override LOCAL_RANK reads via os.environ wrapper.
    # We can't subclass os._Environ trivially (it's bound to libc
    # in some Python builds), so we override the key directly and
    # use a sys.audit hook... actually simpler: just override the
    # __getitem__ via instance method swap.
    import os as _os
    _orig_getitem = _os.environ.__class__.__getitem__

    def _patched_getitem(self, key):
        if key == "LOCAL_RANK":
            return "0"
        return _orig_getitem(self, key)

    _os.environ.__class__.__getitem__ = _patched_getitem

    # Also override .get for consistency
    _orig_get = _os.environ.__class__.get

    def _patched_get(self, key, default=None):
        if key == "LOCAL_RANK":
            return "0"
        return _orig_get(self, key, default)

    _os.environ.__class__.get = _patched_get

    # 2. set_device — belt-and-braces
    orig_set = torch.xpu.set_device

    def pinned_set(device, /):
        return orig_set(0)

    pinned_set._xpu_pinned = True  # type: ignore[attr-defined]
    torch.xpu.set_device = pinned_set

    logger.info(
        "Patched os.environ['LOCAL_RANK']→'0' and torch.xpu.set_device→0 "
        "for single-tile actor"
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
    patch_torch_xpu_set_device_for_single_tile()
    setup_oneccl_tcp_kvs_for_xpu()
