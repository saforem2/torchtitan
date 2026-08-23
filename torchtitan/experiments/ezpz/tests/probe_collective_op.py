#!/usr/bin/env python3
"""Standalone repro: reduce_scatter_tensor SIGSEGVs on XPU (Sunspot, 2026-08-14).

Minimal, dependency-free (torch only -- no ezpz, no torchtitan). Segfaults with
12 ranks on a SINGLE node on a 1 KiB bf16 buffer, so no fabric is involved.

Run on one node:

    mpiexec -n 12 -ppn 12 python3 repro_reduce_scatter_segv.py

Expected on a healthy stack: five "OK" lines then ALL_PASSED.
Observed on Sunspot: "rank N died from signal 11 and dumped core" before the
first OK line prints; job exits 139.

Env: torch 2.13.0.dev20260519+xpu, python 3.14.2, XCCL backend,
oneAPI /opt/aurora/default/oneapi/ccl/latest, ZE_FLAT_DEVICE_HIERARCHY=FLAT.

Context: this is what blocks 80B training -- DTensor's redistribute calls
reduce_scatter_tensor, so any TP>1 config dies in the first forward pass.
See docs/guides/known-bugs/sunspot-reduce-scatter-segv-20260814.md
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def main() -> int:
    # PALS (the mpiexec on Sunspot/Aurora) sets PALS_*, NOT RANK/WORLD_SIZE and
    # NOT PMI_RANK/PMI_SIZE -- verified empirically in job 12473120. Reading the
    # wrong names silently yields world=1 on every process: each one then
    # rendezvouses alone, they all bind the same port, and the run dies with
    # EADDRINUSE having never executed a cross-rank collective (that is what
    # void-ran job 12473119). There is no PALS world-size var, so derive it from
    # local size x node count.
    rank = int(os.environ.get("RANK", os.environ.get("PALS_RANKID", 0)))
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("PALS_LOCAL_RANKID", 0)))
    world = os.environ.get("WORLD_SIZE")
    if world is None:
        lsize = int(os.environ.get("PALS_LOCAL_SIZE", 1))
        # PALS_NODEID is 0-based; +1 on the max is not visible per-rank, so use
        # the hostfile line count when present, else assume this node only.
        hf = os.environ.get("PBS_NODEFILE")
        nnodes = 1
        if hf and os.path.exists(hf):
            nnodes = len({ln.strip() for ln in open(hf) if ln.strip()})
        world = lsize * nnodes
    world = int(world)

    # MASTER_ADDR must be the SAME host on every rank (rank 0's). PMIX_HOSTNAME
    # is each rank's own node, so using it directly would shard the rendezvous
    # across nodes. Take the hostfile's first line; fall back to localhost for
    # the single-node case.
    hf0 = "127.0.0.1"
    _hf = os.environ.get("PBS_NODEFILE")
    if _hf and os.path.exists(_hf):
        _lines = [ln.strip() for ln in open(_hf) if ln.strip()]
        if _lines:
            hf0 = _lines[0]
    os.environ.setdefault("MASTER_ADDR", hf0)
    os.environ.setdefault("MASTER_PORT", "29511")
    if world <= 1:
        print(f"ABORT: world={world} -- rank env not detected, so this would "
              f"test NOTHING. Set WORLD_SIZE/RANK explicitly.", flush=True)
        return 2
    torch.xpu.set_device(local)
    dist.init_process_group(backend="xccl", rank=rank, world_size=world)

    dev = torch.device(f"xpu:{local}")
    if rank == 0:
        print(f"world={world} torch={torch.__version__} dev={dev}", flush=True)

    op = os.environ.get("EZPZ_PROBE_OP", "reduce_scatter")
    for mib in (0.001, 1, 16, 64, 144):
        nbytes = max(2 * world, int(mib * 2**20))
        n = nbytes // 2
        n -= n % world
        src_t = torch.ones(n, dtype=torch.bfloat16, device=dev)
        if op == "reduce_scatter":
            out = torch.empty(n // world, dtype=torch.bfloat16, device=dev)
            dist.reduce_scatter_tensor(out, src_t)
        elif op == "all_gather":
            out = torch.empty(n * world, dtype=torch.bfloat16, device=dev)
            dist.all_gather_into_tensor(out, src_t)
        elif op == "all_reduce":
            out = src_t
            dist.all_reduce(out)
        else:
            raise SystemExit(f"unknown EZPZ_PROBE_OP={op}")
        torch.xpu.synchronize()
        if rank == 0:
            print(f"  OK {op} {nbytes / 2**20:8.3f} MiB/rank n={n}", flush=True)

    dist.barrier()
    if rank == 0:
        print("ALL_PASSED", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
