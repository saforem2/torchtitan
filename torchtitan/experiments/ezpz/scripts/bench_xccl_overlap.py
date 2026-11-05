"""Measure whether XCCL collectives overlap with compute on XPU.

Each rank runs a bf16 matmul loop on the current stream while an
all-gather or reduce-scatter of FSDP-sized buffers runs on a side stream,
mirroring FSDP2's separate all-gather / reduce-scatter streams. For each
(collective, side-stream priority, enqueue order) case it reports

    compute   wall time of the matmul loop alone
    comm      wall time of the collective loop alone (side stream)
    both      wall time with both enqueued
    overlap   compute + comm - both   (0 = fully serialized)

plus a compute/compute control (a second matmul loop on the side stream),
which tells apart "comm cannot overlap" from "two queues cannot overlap".
With --trace-dir, rank 0 also records a torch profiler trace of one
concurrent run per case and reports device-kernel overlap from it.

oneCCL reads its environment once at communicator creation, so env knobs
are swept by launching this script once per configuration (see
bench_xccl_overlap_sunspot.sh). Launch with ezpz launch / mpiexec; rank
and size come from PALS/PMI variables, MASTER_ADDR/MASTER_PORT from env.
"""

from __future__ import annotations

import argparse
import faulthandler
import gzip
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist


def _env_int(*names: str, default: int | None = None) -> int:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    if default is None:
        raise RuntimeError(f"none of {names} set")
    return default


def _union(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def _length(intervals):
    return sum(end - start for start, end in intervals)


def _intersect(a, b):
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _is_comm(name: str) -> bool:
    lowered = name.lower()
    return "ccl" in lowered or "allgather" in lowered or "reduce_scatter" in lowered


def trace_overlap(path: Path) -> dict:
    """Kernel-level overlap (ms) between comm and compute device events."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    device = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    comm, comp = [], []
    barrier_n, barrier_us = 0, 0.0
    names = {}
    for e in device:
        name = e.get("name", "")
        span = (e["ts"], e["ts"] + e["dur"])
        if _is_comm(name):
            comm.append(span)
            if "barrier" in name.lower():
                barrier_n += 1
                barrier_us += e["dur"]
        else:
            comp.append(span)
        key = (name[:60], e.get("tid"))
        names[key] = names.get(key, 0.0) + e["dur"]
    cu, mu = _union(comp), _union(comm)
    top = sorted(names.items(), key=lambda kv: -kv[1])[:6]
    return {
        "trace_compute_ms": _length(cu) / 1e3,
        "trace_comm_ms": _length(mu) / 1e3,
        "trace_overlap_ms": _intersect(cu, mu) / 1e3,
        "trace_barrier_kernels": barrier_n,
        "trace_barrier_ms": barrier_us / 1e3,
        "trace_top": [f"{n}@tid{t}={d / 1e3:.1f}ms" for (n, t), d in top],
    }


class Bench:
    def __init__(self, args, rank: int, world: int, device: torch.device):
        self.args = args
        self.rank = rank
        self.world = world
        self.device = device
        n = args.mm_dim
        self.a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
        self.b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
        self.c = torch.empty(n, n, device=device, dtype=torch.bfloat16)
        self.a2 = torch.randn(n, n, device=device, dtype=torch.bfloat16)
        self.c2 = torch.empty(n, n, device=device, dtype=torch.bfloat16)
        rs_dtype = getattr(torch, args.rs_dtype)
        ag_elems = args.ag_mb * 2**20 // 2 // world * world
        rs_elems = args.rs_mb * 2**20 // rs_dtype.itemsize // world * world
        self.ag_out = torch.empty(ag_elems, device=device, dtype=torch.bfloat16)
        self.ag_in = torch.randn(ag_elems // world, device=device, dtype=torch.bfloat16)
        self.rs_in = torch.randn(rs_elems, device=device, dtype=rs_dtype)
        self.rs_out = torch.empty(rs_elems // world, device=device, dtype=rs_dtype)
        self.n_mm = args.mm_iters
        # The compute/compute control always uses a normal-priority stream:
        # torch.matmul on a priority=-1 XPU stream segfaults (torch 2.14, job
        # 12481898), and FSDP2 only runs copies and collectives there.
        self.cc_stream = torch.xpu.Stream(device=device, priority=0)

    def matmuls(self, n: int, side: bool = False) -> None:
        a, c = (self.a2, self.c2) if side else (self.a, self.c)
        for _ in range(n):
            torch.matmul(a, self.b, out=c)

    def comm(self, op: str, async_op: bool = False) -> None:
        # async_op=True runs on ProcessGroupXCCL's internal stream; wait()
        # then makes the current (side) stream wait on it, as FSDP2 does.
        works = []
        for _ in range(self.args.comm_iters):
            if op == "allgather":
                work = dist.all_gather_into_tensor(
                    self.ag_out, self.ag_in, async_op=async_op
                )
            else:
                work = dist.reduce_scatter_tensor(
                    self.rs_out, self.rs_in, async_op=async_op
                )
            if work is not None:
                works.append(work)
        for work in works:
            work.wait()

    def _sync(self) -> None:
        torch.xpu.synchronize(self.device)
        dist.barrier()
        torch.xpu.synchronize(self.device)

    def timed(self, fn) -> float:
        """Median wall time (ms) of fn() over --reps, synchronized on all ranks."""
        samples = []
        for _ in range(self.args.reps):
            self._sync()
            t0 = time.perf_counter()
            fn()
            torch.xpu.synchronize(self.device)
            samples.append((time.perf_counter() - t0) * 1e3)
        samples.sort()
        return samples[len(samples) // 2]

    def case(self, op, priority, order, async_op, trace_dir, tag):
        side = torch.xpu.Stream(device=self.device, priority=priority)
        main = torch.xpu.current_stream(self.device)

        def comm_only():
            with torch.xpu.stream(side):
                self.comm(op, async_op)

        def compute_only():
            self.matmuls(self.n_mm)

        def side_compute_only():
            with torch.xpu.stream(self.cc_stream):
                self.matmuls(self.n_mm, side=True)

        def both(other):
            def run():
                side.wait_stream(main)
                self.cc_stream.wait_stream(main)
                if order == "mm-first":
                    self.matmuls(self.n_mm)
                    other()
                else:
                    other()
                    self.matmuls(self.n_mm)

            return run

        def phase(name, fn):
            if self.rank == 0:
                print(f"  PHASE {name}", flush=True)
            return self.timed(fn)

        comp = phase("compute", compute_only)
        comm = phase("comm", comm_only)
        conc = phase("both", both(comm_only))
        comp2 = phase("cc_compute2", side_compute_only)
        conc_cc = phase("cc_both", both(side_compute_only))
        row = {
            "tag": tag,
            "op": op,
            "priority": priority,
            "order": order,
            "async_op": async_op,
            "compute_ms": comp,
            "comm_ms": comm,
            "both_ms": conc,
            "overlap_ms": comp + comm - conc,
            "overlap_frac": (comp + comm - conc) / max(min(comp, comm), 1e-9),
            "cc_compute2_ms": comp2,
            "cc_both_ms": conc_cc,
            "cc_overlap_ms": comp + comp2 - conc_cc,
        }
        self._both = both(comm_only)
        return row

    def trace(self, row, trace_dir, tag):
        """Profile one concurrent run (rank 0 records) and add kernel overlap."""
        op, priority, order, async_op = (
            row["op"],
            row["priority"],
            row["order"],
            row["async_op"],
        )
        both_comm = self._both
        if trace_dir is not None and str(int(async_op)) in self.args.trace_async.split(
            ","
        ):
            from torch.profiler import ProfilerActivity, profile

            self._sync()
            activities = [ProfilerActivity.CPU, ProfilerActivity.XPU]
            if self.rank == 0:
                with profile(activities=activities) as prof:
                    both_comm()
                    torch.xpu.synchronize(self.device)
                path = (
                    trace_dir
                    / f"{tag}-w{self.world}-{op}-p{priority}-{order}-a{int(async_op)}.json"
                )
                prof.export_chrome_trace(str(path))
                try:
                    row.update(trace_overlap(path))
                except (OSError, KeyError, ValueError) as exc:
                    row["trace_error"] = repr(exc)
            else:
                both_comm()
                torch.xpu.synchronize(self.device)
            self._sync()
        return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default=os.environ.get("BENCH_CFG", "default"))
    parser.add_argument("--ops", default="allgather,reduce_scatter")
    parser.add_argument("--priorities", default="0,-1")
    parser.add_argument("--orders", default="mm-first,comm-first")
    parser.add_argument("--async-ops", default="0,1", help="async_op values to sweep")
    parser.add_argument(
        "--ag-mb", type=int, default=128, help="all-gather OUTPUT size (bf16)"
    )
    parser.add_argument(
        "--rs-mb", type=int, default=128, help="reduce-scatter INPUT size"
    )
    parser.add_argument(
        "--rs-dtype", default="bfloat16", choices=["bfloat16", "float32"]
    )
    parser.add_argument("--comm-iters", type=int, default=16)
    parser.add_argument("--mm-dim", type=int, default=8192)
    parser.add_argument(
        "--mm-iters", type=int, default=0, help="0 = balance to comm time"
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--jsonl", default="")
    parser.add_argument(
        "--trace-async", default="0", help="async_op values that get a trace"
    )
    args = parser.parse_args()
    faulthandler.enable()

    rank = _env_int("RANK", "PALS_RANKID", "PMI_RANK", "PMIX_RANK")
    world = _env_int("WORLD_SIZE", "BENCH_WORLD", "PMI_SIZE", "PALS_SIZE")
    local = _env_int("LOCAL_RANK", "PALS_LOCAL_RANKID", "MPI_LOCALRANKID")
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))
    os.environ.setdefault("LOCAL_RANK", str(local))
    torch.xpu.set_device(local)
    device = torch.device("xpu", local)
    dist.init_process_group("xccl", rank=rank, world_size=world, device_id=device)
    bench = Bench(args, rank, world, device)

    ops = args.ops.split(",")
    # Warm up kernels, oneCCL communicators and algorithm caches.
    for _ in range(args.warmup):
        bench.matmuls(2)
        for op in ops:
            bench.comm(op)
    torch.xpu.synchronize(device)
    dist.barrier()

    if args.mm_iters == 0:
        one_mm = bench.timed(lambda: bench.matmuls(4)) / 4
        comm_ms = max(bench.timed(lambda op=op: bench.comm(op)) for op in ops)
        n = torch.tensor([max(1, round(comm_ms / one_mm))], device=device)
        dist.all_reduce(n, op=dist.ReduceOp.MAX)
        bench.n_mm = int(n.item())
        if rank == 0:
            print(
                f"BALANCE one_mm={one_mm:.2f}ms comm={comm_ms:.2f}ms mm_iters={bench.n_mm}"
            )

    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    if trace_dir is not None and rank == 0:
        trace_dir.mkdir(parents=True, exist_ok=True)
    env_keys = sorted(
        k
        for k in os.environ
        if k.startswith(
            ("CCL_", "UR_L0", "SYCL_PI", "ZEX_", "ZE_", "TORCH_XCCL", "TORCH_XPU")
        )
    )
    # Which oneCCL / UR / L0 the process really mapped (module LD paths can
    # shadow the venv's RUNPATH).
    with open("/proc/self/maps") as handle:
        libs = sorted(
            {
                line.split()[-1]
                for line in handle
                if any(
                    k in line
                    for k in ("libccl", "libur_adapter", "libze_", "libsycl.so")
                )
            }
        )
    if rank == 0:
        print("LIBS", " ".join(libs))
        print("ENV", " ".join(f"{k}={os.environ[k]}" for k in env_keys))
        print(
            f"CONFIG tag={args.tag} world={world} host={socket.gethostname()} "
            f"torch={torch.__version__} mm_dim={args.mm_dim} mm_iters={bench.n_mm} "
            f"ag_mb={args.ag_mb} rs_mb={args.rs_mb} rs_dtype={args.rs_dtype} comm_iters={args.comm_iters}"
        )
    rows = []
    for op in ops:
        for priority in (int(p) for p in args.priorities.split(",")):
            for order, async_op in (
                (o, bool(int(a)))
                for o in args.orders.split(",")
                for a in args.async_ops.split(",")
            ):
                if rank == 0:
                    print(
                        f"CASE_BEGIN {op} prio={priority} {order} async={int(async_op)}",
                        flush=True,
                    )
                row = bench.case(op, priority, order, async_op, trace_dir, args.tag)
                row["world"] = world
                rows.append(row)
                if rank == 0:
                    print(
                        f"RESULT tag={args.tag} world={world} op={op} prio={priority} order={order} async={int(async_op)} "
                        f"compute={row['compute_ms']:.1f} comm={row['comm_ms']:.1f} "
                        f"both={row['both_ms']:.1f} overlap={row['overlap_ms']:.1f} "
                        f"({row['overlap_frac']:.0%}) | cc: c2={row['cc_compute2_ms']:.1f} "
                        f"both={row['cc_both_ms']:.1f} ovl={row['cc_overlap_ms']:.1f}",
                        flush=True,
                    )
                    if args.jsonl:
                        with open(args.jsonl, "a") as handle:
                            env = {k: os.environ[k] for k in env_keys}
                            handle.write(
                                json.dumps({**row, "env": env, "libs": libs}) + "\n"
                            )
                bench.trace(row, trace_dir, args.tag)
                if rank == 0:
                    if "trace_overlap_ms" in row:
                        print(
                            f"TRACE tag={args.tag} world={world} op={op} prio={priority} order={order}"
                            f" comp={row['trace_compute_ms']:.1f} comm={row['trace_comm_ms']:.1f}"
                            f" ovl={row['trace_overlap_ms']:.1f} barriers={row['trace_barrier_kernels']}"
                            f"/{row['trace_barrier_ms']:.1f}ms",
                            flush=True,
                        )
                    if "trace_top" in row:
                        print("  TOP", "; ".join(row["trace_top"]), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
