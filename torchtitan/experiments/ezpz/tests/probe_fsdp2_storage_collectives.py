#!/usr/bin/env python3
"""Probe FSDP2 storage collectives at the canonical 30B embedding shape."""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torchtitan.distributed.fsdp import resolve_fsdp_mesh
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from torchtitan.experiments.ezpz.xccl_split_group_workaround import (
    maybe_install_xccl_split_group_workaround,
)


def _rank_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("PALS_RANKID", "0")))
    local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("PALS_LOCAL_RANKID", "0"))
    )
    world = int(os.environ.get("WORLD_SIZE", "0"))
    if world <= 1:
        raise RuntimeError(f"invalid distributed rank environment: world={world}")
    if not 0 <= rank < world:
        raise RuntimeError(f"invalid global rank: rank={rank}, world={world}")
    if not 0 <= local_rank < 12:
        raise RuntimeError(f"invalid Sunspot local rank: local_rank={local_rank}")
    pals_rank = os.environ.get("PALS_RANKID")
    pals_local_rank = os.environ.get("PALS_LOCAL_RANKID")
    if pals_rank is not None and int(pals_rank) != rank:
        raise RuntimeError(f"RANK={rank} disagrees with PALS_RANKID={pals_rank}")
    if pals_local_rank is not None and int(pals_local_rank) != local_rank:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} disagrees with PALS_LOCAL_RANKID={pals_local_rank}"
        )

    hostfile = os.environ.get("PBS_NODEFILE")
    master_addr = "127.0.0.1"
    if hostfile and os.path.exists(hostfile):
        with open(hostfile) as stream:
            master_addr = next((line.strip() for line in stream if line.strip()), "")
        if not master_addr:
            raise RuntimeError(f"PBS nodefile is empty: {hostfile}")
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", "29512")
    return rank, local_rank, world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-replicate", type=int, required=True)
    parser.add_argument("--dp-shard", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    rank, local_rank, world = _rank_env()
    if args.dp_replicate * args.dp_shard != world:
        raise ValueError(
            f"dp_replicate({args.dp_replicate}) * dp_shard({args.dp_shard}) "
            f"!= world({world})"
        )

    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl", rank=rank, world_size=world)
    print(f"PG_READY rank={rank} world={world} local_rank={local_rank}", flush=True)

    maybe_install_xccl_split_group_workaround()
    parallel_dims = ParallelDims(
        dp_replicate=args.dp_replicate,
        dp_shard=args.dp_shard,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=world,
        enable_sequence_parallel=False,
    )
    parallel_dims.build_mesh()
    mesh, mesh_dims = resolve_fsdp_mesh(parallel_dims)
    print(
        f"MESH_READY names={mesh.mesh_dim_names} "
        f"dp_replicate={args.dp_replicate} dp_shard={args.dp_shard} "
        f"mesh_dims={mesh_dims}",
        flush=True,
    )

    with torch.device("meta"):
        module = nn.Embedding(
            num_embeddings=100352,
            embedding_dim=6144,
            dtype=torch.float32,
        )
    annotate_replicated_parameters(module, parallel_dims)
    fully_shard(
        module,
        mesh=mesh,
        dp_mesh_dims=mesh_dims,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        ),
    )
    module.set_gradient_divide_factor(1.0)
    module.set_force_sum_reduction_for_comms(True)
    module.to_empty(device=torch.device(f"xpu:{local_rank}"))
    with torch.no_grad():
        module.weight.fill_(0.001)
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)
    print("FSDP_WRAPPED", flush=True)

    for step in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        parameter = next(module.parameters())
        before = parameter.to_local().detach().clone()
        rows_per_shard = 100352 // args.dp_shard
        shard_index = rank % args.dp_shard
        first_token = shard_index * rows_per_shard + step
        tokens = torch.tensor(
            [[first_token, first_token + 1]],
            dtype=torch.long,
            device=f"xpu:{local_rank}",
        )
        output = module(tokens)
        torch.xpu.synchronize()
        loss = output.float().sum()
        if not math.isfinite(float(loss)) or float(loss) == 0.0:
            raise RuntimeError(f"non-finite/zero loss at step {step}: {float(loss)}")
        print(f"FORWARD_OK step={step} loss={float(loss):.9g}", flush=True)
        loss.backward()
        grad = parameter.grad
        if grad is None or not bool(torch.isfinite(grad.to_local()).all()):
            raise RuntimeError(f"non-finite/missing gradient at step {step}")
        grad_sq = grad.to_local().float().square().sum()
        dist.all_reduce(grad_sq, group=parallel_dims.get_mesh("dp_shard").get_group())
        grad_norm = float(grad_sq.sqrt())
        if not math.isfinite(grad_norm) or grad_norm == 0.0:
            raise RuntimeError(f"non-finite/zero global gradient at step {step}: {grad_norm}")
        print(f"BACKWARD_OK step={step} grad_norm={grad_norm:.9g}", flush=True)
        optimizer.step()
        torch.xpu.synchronize()
        update_sq = (parameter.to_local().detach() - before).float().square().sum()
        dist.all_reduce(update_sq, group=parallel_dims.get_mesh("dp_shard").get_group())
        update_norm = float(update_sq.sqrt())
        if not math.isfinite(update_norm) or update_norm == 0.0:
            raise RuntimeError(
                f"non-finite/zero global parameter update at step {step}: {update_norm}"
            )
        print(f"UPDATE_OK step={step} update_norm={update_norm:.9g}", flush=True)
        dist.barrier(group=parallel_dims.get_mesh("dp_shard").get_group())
        if args.dp_replicate > 1:
            dist.barrier(group=parallel_dims.get_mesh("dp_replicate").get_group())

    if rank == 0:
        print(
            "FSDP2_XCCL_PROBE_PASS "
            f"dp_replicate={args.dp_replicate} dp_shard={args.dp_shard} "
            f"iterations={args.iterations}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
