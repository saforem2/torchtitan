#!/usr/bin/env python3
"""Summarize device kernel time in torch.profiler XPU traces.

Works on TT (``profiling/traces/iteration_N/rankR_trace.json.gz``) and MDS
(``tensorboard/*.pt.trace.json.gz``) traces, so the two frameworks can be
compared with the same categories. For each trace it prints one
``SUMMARY {json}`` line: device span, busy time (union of kernel intervals),
seconds per category and the top kernels by total time.

Times cover every kernel in the trace. Both launchers capture one update, so
the totals are per update.

Usage: summarize_xpu_trace.py TRACE [TRACE ...] [--top N]
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re

# First match wins; order matters (attention before generic patterns).
CATEGORIES = [
    ("attn_bwd", r"MhaBackward|flash.*b(ack)?w(ar)?d|fmha.*bwd|_bwd_kernel"),
    ("attn_bwd_aux", r"MhaDotDoO|MhdConvertDq|dot_do_o|convert_dq"),
    ("attn_fwd", r"XeFMHAFwd|MhaName|fmha|flash|attention"),
    ("gemm", r"gemm|matmul|xetla.*gemm|onednn|dnnl|jit:|ref:|brgemm|gen_gemm"),
    ("comm", r"oneccl|ccl_|allreduce|allgather|reduce_scatter|alltoall"),
    ("cast", r"_to_copy|CopyScalarFunc|convert|cast"),
    ("optimizer", r"MultiTensorApply|adam|fused_adam|multi_tensor"),
    ("norm", r"rms_norm|layer_norm|rmsnorm|norm"),
    ("loss", r"log_softmax|nll_loss|cross_entropy|softmax"),
    ("reduce", r"ReduceKernel|reduce|sum_kernel"),
    ("fill", r"FillFunctor|fill|zero"),
    ("elementwise", r"Elementwise|triton_poi|silu|swiglu|mul|add|rotary"),
]
COMPILED = [(n, re.compile(p, re.IGNORECASE)) for n, p in CATEGORIES]


def category(name: str) -> str:
    for cat, pat in COMPILED:
        if pat.search(name):
            return cat
    return "other"


def busy(intervals):
    total, end = 0.0, float("-inf")
    for s, e in sorted(intervals):
        if s >= end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def summarize(path: str, top: int) -> dict:
    with gzip.open(path) if path.endswith(".gz") else open(path) as f:
        data = json.load(f)
    kernels = [
        e
        for e in data["traceEvents"]
        if e.get("cat") == "kernel" and e.get("ph") == "X" and "dur" in e
    ]
    if not kernels:
        return {"trace": path, "error": "no kernel events"}
    by_cat = collections.defaultdict(float)
    by_name = collections.defaultdict(lambda: [0.0, 0])
    intervals = []
    for e in kernels:
        dur = float(e["dur"])
        by_cat[category(e["name"])] += dur
        rec = by_name[e["name"][:120]]
        rec[0] += dur
        rec[1] += 1
        intervals.append((float(e["ts"]), float(e["ts"]) + dur))
    start = min(s for s, _ in intervals)
    stop = max(e for _, e in intervals)
    us = 1e-6
    return {
        "trace": path,
        "kernels": len(kernels),
        "span_s": round((stop - start) * us, 4),
        "busy_s": round(busy(intervals) * us, 4),
        "kernel_sum_s": round(sum(by_cat.values()) * us, 4),
        "categories_s": {
            k: round(v * us, 4) for k, v in sorted(by_cat.items(), key=lambda x: -x[1])
        },
        "top": [
            {"name": n, "s": round(t * us, 4), "count": c}
            for n, (t, c) in sorted(by_name.items(), key=lambda x: -x[1][0])[:top]
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    for path in args.traces:
        print("SUMMARY " + json.dumps(summarize(path, args.top)), flush=True)


if __name__ == "__main__":
    main()
