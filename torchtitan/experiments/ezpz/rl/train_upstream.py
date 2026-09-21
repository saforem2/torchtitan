# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz mirror of `torchtitan.rl.train` for XPU.

Applies XPU compatibility patches (see `xpu_overrides`) BEFORE
importing anything from `torchtitan.rl`, then delegates
to the upstream training loop with a minimal substitution:

  - `PerHostProvisioner` → `EzpzPerHostProvisioner`
    (uses ZE_AFFINITY_MASK instead of CUDA_VISIBLE_DEVICES).

Use it the same way as upstream:

    python -m torchtitan.experiments.ezpz.rl.train_upstream \\
        --module rl --config rl_grpo_qwen3_0_6b_varlen \\
        --hf_assets_path torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B

The `--module rl` arg points the ConfigManager at the **upstream**
`config_registry.py`, which is exactly what we want — we're reusing
upstream's configs unchanged.
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
    EzpzPerHostProvisioner,
    apply_all_xpu_patches,
)

apply_all_xpu_patches()

# ----------------------------------------------------------------------
# Now safe to import upstream rl/. From here on, mirror train.py.
# ----------------------------------------------------------------------
import asyncio
import logging

from monarch.actor import ProcMesh, this_host

from torchtitan.config import ConfigManager
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.rl.controller import Controller

# Re-export upstream so future maintainers know what we depend on.
from torchtitan.rl.train import (
    HostMeshes,
    _compute_generator_world_size,
    _compute_trainer_world_size,
    breakable_cuda_graph_env,
)

logger = logging.getLogger(__name__)


def spawn_proc_mesh(
    trainer_world_size: int,
    per_generator_world_size: int,
    host_meshes: HostMeshes | None = None,
    *,
    num_generators: int = 1,
    generator_env: dict[str, str] | None = None,
) -> tuple[ProcMesh, list[ProcMesh]]:
    """XPU version of upstream `spawn_proc_mesh`.

    Identical to upstream except the provisioner is `EzpzPerHostProvisioner`
    (ZE_AFFINITY_MASK) instead of `PerHostProvisioner` (CUDA_VISIBLE_DEVICES).
    """
    total_generator_gpus = num_generators * per_generator_world_size
    total_gpus = trainer_world_size + total_generator_gpus
    logger.info(
        f"{num_generators} generator(s) * {per_generator_world_size} GPUs + "
        f"{trainer_world_size} trainer GPUs = {total_gpus} total"
    )

    if host_meshes is not None:
        trainer_host_mesh = host_meshes.trainer
        generator_host_meshes = host_meshes.generators
        gpus_per_node = host_meshes.gpus_per_node

        trainer_nodes = trainer_host_mesh.sizes["hosts"]
        assert len(generator_host_meshes) == num_generators
        assert trainer_world_size % trainer_nodes == 0, (
            f"trainer_world_size ({trainer_world_size}) must be "
            f"evenly divisible by trainer_nodes ({trainer_nodes})"
        )
        trainer_gpus_per_node = trainer_world_size // trainer_nodes

        trainer_provisioner = EzpzPerHostProvisioner(total_gpus=gpus_per_node)
        trainer_boot = trainer_provisioner.allocate(trainer_gpus_per_node)
        trainer_mesh = trainer_host_mesh.spawn_procs(
            per_host={"gpus": trainer_gpus_per_node},
            bootstrap=trainer_boot,
            bootstrap_command=EzpzPerHostProvisioner.make_bootstrap_command_for_gpu_ids(
                trainer_boot.gpu_ids  # type: ignore[attr-defined]
            ),
        )
        generator_meshes = []
        for generator_host_mesh in generator_host_meshes:
            generator_nodes = generator_host_mesh.sizes["hosts"]
            assert per_generator_world_size % generator_nodes == 0
            generator_gpus_per_node = per_generator_world_size // generator_nodes
            generator_provisioner = EzpzPerHostProvisioner(total_gpus=gpus_per_node)
            generator_boot = generator_provisioner.allocate(generator_gpus_per_node)
            generator_meshes.append(
                generator_host_mesh.spawn_procs(
                    per_host={"gpus": generator_gpus_per_node},
                    bootstrap=generator_boot,
                    bootstrap_command=EzpzPerHostProvisioner.make_bootstrap_command_for_gpu_ids(
                        generator_boot.gpu_ids,  # type: ignore[attr-defined]
                        extra_env=generator_env,
                    ),
                )
            )
    else:
        # Single-node: partition tiles on this_host() via ZE_AFFINITY_MASK.
        # Sunspot / Aurora compute nodes have 12 tiles per host (6 PVCs × 2).
        provisioner = EzpzPerHostProvisioner(total_gpus=total_gpus)
        # `bootstrap_command=` injects env (ZE_AFFINITY_MASK, LOCAL_RANK,
        # PALS_*) into the actor's process BEFORE execve, so the eager
        # `import torch` in monarch's bootstrap_main.py sees the right
        # tile visibility. `bootstrap=` runs AFTER torch imports — too
        # late for the SYCL primary context to pick up ZE_AFFINITY_MASK.
        trainer_boot = provisioner.allocate(trainer_world_size)
        generator_boots = [
            provisioner.allocate(per_generator_world_size)
            for _ in range(num_generators)
        ]
        trainer_mesh = this_host().spawn_procs(
            per_host={"gpus": trainer_world_size},
            bootstrap=trainer_boot,
            bootstrap_command=EzpzPerHostProvisioner.make_bootstrap_command_for_gpu_ids(
                trainer_boot.gpu_ids  # type: ignore[attr-defined]
            ),
        )
        generator_meshes = [
            this_host().spawn_procs(
                per_host={"gpus": per_generator_world_size},
                bootstrap=generator_boot,
                bootstrap_command=EzpzPerHostProvisioner.make_bootstrap_command_for_gpu_ids(
                    generator_boot.gpu_ids,  # type: ignore[attr-defined]
                    extra_env=generator_env,
                ),
            )
            for generator_boot in generator_boots
        ]

    return trainer_mesh, generator_meshes


async def main():
    init_logger()
    os.environ["MONARCH_ACTOR_QUEUE_DISPATCH"] = "0"
    config = ConfigManager().parse_args()
    assert isinstance(config, Controller.Config)
    sl.init_structured_logger(
        source="rl_controller",
        output_dir=config.dump_folder,
        rank=0,
        enable=config.trainer.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    rl_trainer: Controller = config.build()
    try:
        trainer_world_size = _compute_trainer_world_size(config.trainer.parallelism)
        per_generator_world_size = _compute_generator_world_size(
            config.generator.parallelism
        )
        trainer_mesh, generator_meshes = spawn_proc_mesh(
            trainer_world_size,
            per_generator_world_size,
            host_meshes=None,
            num_generators=config.num_generators,
            generator_env=breakable_cuda_graph_env(config.generator),
        )
        await rl_trainer.setup_async(
            trainer_mesh=trainer_mesh,
            generator_meshes=generator_meshes,
        )
        await rl_trainer.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted; attempting graceful shutdown...")
    finally:
        await rl_trainer.close()


if __name__ == "__main__":
    asyncio.run(main())
