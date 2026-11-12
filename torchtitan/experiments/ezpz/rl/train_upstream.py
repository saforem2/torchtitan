# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz mirror of `torchtitan.rl.train` for XPU.

Applies XPU compatibility patches (see `xpu_overrides`) BEFORE
importing anything from `torchtitan.rl`, then runs the upstream
`Controller` with one substitution:

  - `MonarchActorSystem` -> `EzpzMonarchActorSystem`, whose `provision`
    spawns the trainer/generator proc meshes with `EzpzPerHostProvisioner`
    (ZE_AFFINITY_MASK, LOCAL_RANK and PALS_* set before torch imports)
    instead of upstream's `PerHostProvisioner` (CUDA_VISIBLE_DEVICES).

Upstream moved process placement out of `rl/train.py` into the actor system
(`rl/actors/monarch/system.py`, `provisioning.py`); `Controller.setup_async()`
provisions through `Controller.Config.actor_system`.

Use it the same way as upstream:

    python -m torchtitan.experiments.ezpz.rl.train_upstream \
        --module torchtitan.experiments.ezpz.rl.alphabet_sort_agpt \
        --config rl_grpo_lora_agpt_2b_easy \
        --hf_assets_path <staged ckpt dir with the gemma chat_template>
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# XPU compatibility patches — MUST run before any rl/ import. Don't
# move these below the rl imports; that defeats the patching.
# ----------------------------------------------------------------------
import os

# vLLM's EngineCore subprocess inherits these. Same as bare smoke:
# ezpz_setup_env's CCL_*/FI_* overrides poison single-rank vLLM init.
for _var in (
    "CCL_OP_SYNC",
    "CCL_PROCESS_LAUNCHER",
    "CCL_ATL_TRANSPORT",
    "CCL_OFI_PROVIDER",
    "FI_PROVIDER",
    "FI_LOG_LEVEL",
    "FI_LOG_PROV",
    "FI_LOG_LOCATION",
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
):
    os.environ.pop(_var, None)

# Upstream's `train.py` sets this too. Harmless on XPU (torch.xpu
# ignores it), kept for parity.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from torchtitan.experiments.ezpz.rl.xpu_overrides import (
    apply_all_xpu_patches,
    EzpzPerHostProvisioner,
)

apply_all_xpu_patches()

# ----------------------------------------------------------------------
# Now safe to import upstream rl/. From here on, mirror train.py.
# ----------------------------------------------------------------------
import asyncio
import logging
import sys
from dataclasses import dataclass

from monarch.actor import ProcMesh, this_host

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.ezpz.config_manager import ConfigManager
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.rl.actors.monarch.provisioning import (
    _check_gpus_per_host,
    compute_generator_world_size,
    compute_trainer_world_size,
    HostMeshes,
    spawn_cpu_procs,
)
from torchtitan.rl.actors.monarch.system import MonarchActorSystem
from torchtitan.rl.actors.monarch.torch_elastic import setup_torch_elastic_env
from torchtitan.rl.actors.system import ActorSystem
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig

logger = logging.getLogger(__name__)

__all__ = [
    "compute_generator_world_size",
    "compute_trainer_world_size",
    "EzpzHostMeshController",
    "EzpzMonarchActorSystem",
    "HostMeshes",
    "spawn_proc_mesh",
    "use_ezpz_actor_system",
]


def _spawn_role_mesh(
    host_mesh,
    role_gpus_per_node: int,
    provisioner: EzpzPerHostProvisioner,
    *,
    extra_env: dict[str, str] | None = None,
) -> ProcMesh:
    """Spawn one role's proc mesh with ``role_gpus_per_node`` tiles on each host."""
    boot = provisioner.allocate(role_gpus_per_node)
    # `bootstrap_command=` injects env (ZE_AFFINITY_MASK, LOCAL_RANK, PALS_*)
    # into the actor's process BEFORE execve, so the eager `import torch` in
    # monarch's bootstrap sees the right tile visibility. `bootstrap=` runs
    # AFTER torch imports -- too late for the SYCL primary context.
    return host_mesh.spawn_procs(
        per_host={"gpus": role_gpus_per_node},
        bootstrap=boot,
        bootstrap_command=EzpzPerHostProvisioner.make_bootstrap_command_for_gpu_ids(
            boot.gpu_ids,  # type: ignore[attr-defined]
            extra_env=extra_env,
        ),
    )


def spawn_proc_mesh(
    trainer_world_size: int,
    per_generator_world_size: int,
    host_meshes: HostMeshes | None = None,
    *,
    num_generators: int = 1,
    generator_env: dict[str, str] | None = None,
) -> tuple[ProcMesh, list[ProcMesh]]:
    """XPU version of upstream `provisioning.spawn_proc_mesh`.

    Same placement rules as upstream (roles given the same host mesh object get
    non-overlapping tiles on it, trainer first); the provisioner is
    `EzpzPerHostProvisioner` (ZE_AFFINITY_MASK) instead of `PerHostProvisioner`
    (CUDA_VISIBLE_DEVICES).
    """
    total_gpus = trainer_world_size + num_generators * per_generator_world_size
    logger.info(
        f"{num_generators} generator(s) * {per_generator_world_size} GPUs + "
        f"{trainer_world_size} trainer GPUs = {total_gpus} total"
    )

    if host_meshes is None:
        # Single-node: partition tiles on this_host() via ZE_AFFINITY_MASK.
        # Sunspot / Aurora compute nodes have 12 tiles per host (6 PVCs x 2).
        host_mesh = this_host()
        provisioner = EzpzPerHostProvisioner(total_gpus=total_gpus)
        trainer_mesh = _spawn_role_mesh(host_mesh, trainer_world_size, provisioner)
        generator_meshes = [
            _spawn_role_mesh(
                host_mesh,
                per_generator_world_size,
                provisioner,
                extra_env=generator_env,
            )
            for _ in range(num_generators)
        ]
        return trainer_mesh, generator_meshes

    assert len(host_meshes.generators) == num_generators, (
        f"expected {num_generators} generator host mesh(es), "
        f"got {len(host_meshes.generators)}"
    )
    _check_gpus_per_host(host_meshes, trainer_world_size, per_generator_world_size)
    # Roles on the same mesh share one provisioner, which hands out the next
    # free tiles. Keyed by id(mesh) because a HostMesh can't be a dict key.
    provisioners: dict[int, EzpzPerHostProvisioner] = {}
    for host_mesh in [host_meshes.trainer, *host_meshes.generators]:
        provisioners.setdefault(
            id(host_mesh), EzpzPerHostProvisioner(total_gpus=host_meshes.gpus_per_node)
        )
    trainer_mesh = _spawn_role_mesh(
        host_meshes.trainer,
        trainer_world_size // len(host_meshes.trainer),
        provisioners[id(host_meshes.trainer)],
    )
    generator_meshes = [
        _spawn_role_mesh(
            generator_host_mesh,
            per_generator_world_size // len(generator_host_mesh),
            provisioners[id(generator_host_mesh)],
            extra_env=generator_env,
        )
        for generator_host_mesh in host_meshes.generators
    ]
    return trainer_mesh, generator_meshes


class EzpzMonarchActorSystem(MonarchActorSystem):
    """`MonarchActorSystem` that places the trainer and generators on XPU tiles.

    Only `provision` differs: it spawns the GPU proc meshes with ezpz's
    `spawn_proc_mesh`. Its body otherwise mirrors upstream
    `MonarchActorSystem.provision`; keep the two in step on upstream syncs.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MonarchActorSystem.Config):
        pass

    async def provision(
        self,
        *,
        trainer_parallelism: ParallelismConfig,
        generator_parallelism: InferenceParallelismConfig,
        num_generators: int,
    ) -> None:
        self._trainer_mesh, self._generator_meshes = spawn_proc_mesh(
            compute_trainer_world_size(trainer_parallelism),
            compute_generator_world_size(generator_parallelism),
            host_meshes=self._host_meshes,
            num_generators=num_generators,
        )
        # Router and batcher each get their own CPU proc, as upstream.
        self._router_mesh = spawn_cpu_procs(1)
        self._batcher_mesh = spawn_cpu_procs(1)

        await setup_torch_elastic_env(self._trainer_mesh)
        for generator_mesh in self._generator_meshes:
            await setup_torch_elastic_env(generator_mesh)


def use_ezpz_actor_system(config: Controller.Config) -> None:
    """Swap the default `MonarchActorSystem.Config` for `EzpzMonarchActorSystem`."""
    if isinstance(config.actor_system, EzpzMonarchActorSystem.Config):
        return
    if type(config.actor_system) is not MonarchActorSystem.Config:
        raise ValueError(
            "train_upstream places actors with EzpzMonarchActorSystem; got "
            f"actor_system={type(config.actor_system).__qualname__}"
        )
    config.actor_system = EzpzMonarchActorSystem.Config()


class EzpzHostMeshController(Controller):
    """`Controller` whose actor system runs on caller-provided host meshes.

    Upstream passes launcher-allocated hosts through
    `MonarchControllerWithLauncher._build_actor_system`; ezpz's multi-host
    entry point attaches its hosts itself (`host_mesh_from_store`) and hands
    them in here.
    """

    def __init__(self, config: Controller.Config, *, host_meshes: HostMeshes):
        use_ezpz_actor_system(config)
        super().__init__(config)
        self._host_meshes = host_meshes

    async def _build_actor_system(self) -> EzpzMonarchActorSystem:
        actor_system_config = self.config.actor_system
        assert isinstance(actor_system_config, EzpzMonarchActorSystem.Config)
        return actor_system_config.build(host_meshes=self._host_meshes)


async def main() -> ActorSystem | None:
    """Run RL training; return the actor system for the caller to shut down."""
    init_logger()
    os.environ["MONARCH_ACTOR_QUEUE_DISPATCH"] = "0"
    config = ConfigManager().parse_args()
    assert isinstance(config, Controller.Config)
    use_ezpz_actor_system(config)
    sl.init_structured_logger(
        source="rl_controller",
        output_dir=config.dump_folder,
        rank=0,
        enable=config.trainer.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    rl_trainer: Controller = config.build()
    try:
        await rl_trainer.setup_async()
        await rl_trainer.run()
    except KeyboardInterrupt:
        logger.info("Interrupted; attempting graceful shutdown...")
        raise
    except asyncio.CancelledError:
        logger.exception("RL controller cancelled; propagating failure after shutdown")
        raise
    finally:
        await rl_trainer.close()
    return rl_trainer.actor_system


if __name__ == "__main__":
    actor_system = asyncio.run(main())
    # Same exit path as upstream rl/train.py: shut down Monarch's client, flush,
    # and skip interpreter finalization (before CPython 3.14 it can abort with
    # exit 134 while Monarch's daemon threads are inside PyO3 calls).
    if actor_system is not None:
        actor_system.shutdown()
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
