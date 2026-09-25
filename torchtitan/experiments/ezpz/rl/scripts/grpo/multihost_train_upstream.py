#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the ezpz RL controller with trainer and generator on separate hosts."""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist
from monarch._src.spmd.host_mesh import host_mesh_from_store
from monarch.config import configure
from torchtitan.config import ConfigManager

# Importing this module applies the XPU compatibility patches before rl imports.
from torchtitan.experiments.ezpz.rl import train_upstream
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.rl.controller import Controller
from torchtitan.rl.train import (
    _compute_generator_world_size,
    _compute_trainer_world_size,
    HostMeshes,
)


def _rank() -> int:
    for name in ("RANK", "PMI_RANK", "PALS_RANKID"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    raise RuntimeError("scheduler rank environment is missing")


async def _run_controller(config: Controller.Config, hosts) -> None:
    trainer_world_size = _compute_trainer_world_size(config.trainer.parallelism)
    per_generator_world_size = _compute_generator_world_size(
        config.generator.parallelism
    )
    if config.num_generators != 1:
        raise ValueError("two-host validation requires exactly one generator")

    placement = HostMeshes(
        trainer=hosts.slice(hosts=slice(0, 1)),
        generators=[hosts.slice(hosts=slice(1, 2))],
        gpus_per_node=12,
    )
    print(
        "MULTIHOST_RL_PLACEMENT "
        f"trainer_host_index=0 generator_host_index=1 "
        f"trainer_world_size={trainer_world_size} "
        f"generator_world_size={per_generator_world_size}",
        flush=True,
    )

    controller: Controller = config.build()
    try:
        trainer_mesh, generator_meshes = train_upstream.spawn_proc_mesh(
            trainer_world_size,
            per_generator_world_size,
            host_meshes=placement,
            num_generators=config.num_generators,
            generator_env=None,
        )
        await controller.setup_async(
            trainer_mesh=trainer_mesh,
            generator_meshes=generator_meshes,
        )
        await controller.run()
    finally:
        await controller.close()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--multihost-store", type=Path, required=True)
    parser.add_argument("--multihost-port", type=int, required=True)
    parser.add_argument("--multihost-world-size", type=int, default=2)
    args, config_args = parser.parse_known_args()
    if args.multihost_world_size != 2:
        raise ValueError("this entry point requires exactly two scheduler hosts")

    configure(mesh_attach_config_timeout="120s")
    rank = _rank()
    store = dist.FileStore(str(args.multihost_store), args.multihost_world_size)
    # Non-controller ranks wait here for the complete RL lifecycle, including
    # vLLM startup, validation, weight transfers, optimizer steps, and shutdown.
    # FileStore defaults to five minutes, which can expire during a healthy run
    # and tear down rank 0's actor graph through the outer MPI launcher.
    store.set_timeout(timedelta(minutes=30))
    hosts = host_mesh_from_store(
        store,
        monarch_port=args.multihost_port,
        name="ezpz_rl_multihost",
        transport="tcp",
        rank=rank,
        local_rank=0,
        world_size=args.multihost_world_size,
        local_world_size=1,
    )

    if rank != 0:
        store.get("multihost_rl_done")
        return

    assert hosts is not None
    hosts.initialized.get()
    init_logger()
    os.environ["MONARCH_ACTOR_QUEUE_DISPATCH"] = "0"
    config = ConfigManager().parse_args(config_args)
    assert isinstance(config, Controller.Config)
    sl.init_structured_logger(
        source="rl_controller",
        output_dir=config.dump_folder,
        rank=0,
        enable=config.trainer.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    try:
        asyncio.run(_run_controller(config, hosts))
        print("MULTIHOST_TORCHSTORE_VLLM_OK", flush=True)
    finally:
        try:
            hosts.shutdown().get()
        finally:
            store.set("multihost_rl_done", b"1")


if __name__ == "__main__":
    main()
