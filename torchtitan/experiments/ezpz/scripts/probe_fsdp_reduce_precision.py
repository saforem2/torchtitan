#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure FSDP2 gradient-reduction precision on a real AGPT training batch.

Wraps ``torchtitan.experiments.ezpz.train``. The first
``TrainingEngine.forward_backward`` call is replayed under each reduction
variant with the same weights and microbatches, and the reduced sharded
gradients are compared. The configured path then runs once more, so the
optimizer step and the rest of training are unchanged.

Variants (all with BF16 parameters/compute, FP32 sharded gradients):

- ``fp32_rs_mb``: FP32 reduce-scatter per microbatch (TT default).
- ``fp32_rs_mb_repeat``: same again; run-to-run floor. Also records
  ``sum_i ||g_i||^2`` and ``||sum_i g_i||^2`` per FSDP group.
- ``bf16_rs_mb``: BF16 reduce-scatter per microbatch.
- ``defer_bf16``: BF16 local accumulation, one BF16 reduce-scatter.
- ``defer_fp32``: FP32 local accumulation, one FP32 reduce-scatter.
- ``mds_emul``: FP32 local accumulation, cast to BF16, one BF16
  reduce-scatter (the DeepSpeed BF16 wire contract, reduce-scatter form).
- ``fp64_rs_mb``: FP64 reduce-scatter per microbatch (reference).

Reduction dtype is patched on the live FSDP param groups the same way
``MixedPrecisionPolicy`` sets it: ``reduce_dtype=bfloat16`` with
``param_dtype=bfloat16`` clamps to ``None`` (reduce in the gradient dtype).

Environment: ``AGPT_REDUCE_PROBE_OUTPUT`` (required) is the output directory.
"""

from __future__ import annotations

import functools
import json
import math
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_api import ReduceScatter

CATEGORIES = ("embed", "attention", "mlp", "norm", "lm_head", "other")
# Sum fields, then max fields, per category.
SUM_FIELDS = (
    "diff_sq",
    "ref_sq",
    "var_sq",
    "dot",
    "numel",
    "ref_nonzero",
    "underflow_to_zero",
    "zero_to_nonzero",
    "differs",
    "sign_flip",
)
MAX_FIELDS = ("max_abs_diff", "max_abs_ref")
PHASES = (
    ("fp32_rs_mb", "fp32_rs_mb_repeat", "bf16_rs_mb", "defer_bf16"),
    ("defer_fp32", "mds_emul"),
    ("fp64_rs_mb",),
)
VARIANTS = {
    "fp32_rs_mb": dict(reduce=torch.float32, defer=False, wire=None),
    "fp32_rs_mb_repeat": dict(reduce=torch.float32, defer=False, wire=None, stats=True),
    "bf16_rs_mb": dict(reduce=None, defer=False, wire=None, stats=True),
    "defer_bf16": dict(reduce=None, defer=True, wire=None),
    "defer_fp32": dict(reduce=torch.float32, defer=True, wire=None),
    "mds_emul": dict(reduce=torch.float32, defer=True, wire=torch.bfloat16),
    "fp64_rs_mb": dict(reduce=torch.float64, defer=False, wire=None),
}
# (variant, reference) pairs reported after each phase that has both.
PAIRS = (
    ("fp32_rs_mb_repeat", "fp32_rs_mb"),
    ("bf16_rs_mb", "fp32_rs_mb"),
    ("defer_bf16", "fp32_rs_mb"),
    ("defer_bf16", "bf16_rs_mb"),
    ("defer_fp32", "fp32_rs_mb"),
    ("mds_emul", "fp32_rs_mb"),
    ("defer_bf16", "mds_emul"),
    ("fp32_rs_mb", "fp64_rs_mb"),
    ("defer_fp32", "fp64_rs_mb"),
    ("bf16_rs_mb", "fp64_rs_mb"),
    ("defer_bf16", "fp64_rs_mb"),
    ("mds_emul", "fp64_rs_mb"),
)


def category(name: str) -> str:
    if "tok_embeddings" in name:
        return "embed"
    if "lm_head" in name or name.startswith("output."):
        return "lm_head"
    if "norm" in name:
        return "norm"
    if ".attention." in name:
        return "attention"
    if "feed_forward" in name or ".mlp." in name:
        return "mlp"
    return "other"


class WireCastReduceScatter(ReduceScatter):
    """Reduce-scatter a higher-precision buffer through a lower-precision wire."""

    def __init__(self, inner, wire_dtype: torch.dtype):
        self.inner = inner
        self.wire_dtype = wire_dtype

    def allocate(self, size, *, dtype, device):
        return self.inner.allocate(size, dtype=dtype, device=device)

    def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
        wire_input = input_tensor.to(self.wire_dtype)
        wire_output = torch.empty(
            output_tensor.shape, dtype=self.wire_dtype, device=output_tensor.device
        )
        dist.reduce_scatter_tensor(wire_output, wire_input, op=op, group=group)
        output_tensor.copy_(wire_output)
        return None


class NormStatsReduceScatter(ReduceScatter):
    """Record local-input and reduced-output squared norms; values unchanged."""

    def __init__(self, inner, key: str, sink: dict):
        self.inner = inner
        self.key = key
        self.sink = sink

    def allocate(self, size, *, dtype, device):
        return self.inner.allocate(size, dtype=dtype, device=device)

    def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
        work = self.inner(
            output_tensor=output_tensor,
            input_tensor=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )
        if work is not None:
            work.wait()
        entry = self.sink.setdefault(self.key, [0, None, None])
        # Reduction kernels upcast internally; no full-size temporaries.
        local_sq = (
            torch.linalg.vector_norm(input_tensor, dtype=torch.float32)
            .double()
            .square()
        )
        reduced_sq = (
            torch.linalg.vector_norm(output_tensor, dtype=torch.float32)
            .double()
            .square()
        )
        entry[0] += 1
        entry[1] = local_sq if entry[1] is None else entry[1] + local_sq
        entry[2] = reduced_sq if entry[2] is None else entry[2] + reduced_sq
        return None


def fsdp_groups(model):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            yield from module._get_fsdp_state()._fsdp_param_groups


def group_key(group) -> str:
    fqn = getattr(group, "_module_fqn", "") or "<root>"
    if fqn.startswith("layers."):
        return "layers"
    return fqn


class Probe:
    def __init__(self, output: Path):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        from mpi4py import MPI

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world = self.comm.Get_size()
        self.result: dict = {
            "world_size": self.world,
            "torch": torch.__version__,
            "pbs_jobid": os.environ.get("PBS_JOBID"),
            "variants": {},
            "pairs": {},
            "kappa": {},
        }
        self.grads: dict[str, dict[str, torch.Tensor]] = {}

    def configure(self, engine, model, original_comm, original_run, name):
        spec = VARIANTS[name]
        self.stats_sink: dict = {}
        for group in fsdp_groups(model):
            group._reduce_dtype = spec["reduce"]
            for fsdp_param in group.fsdp_params:
                fsdp_param.reduce_dtype = spec["reduce"]
            comm = original_comm[id(group)]
            if spec["wire"] is not None:
                comm = WireCastReduceScatter(comm, spec["wire"])
            if spec.get("stats"):
                comm = NormStatsReduceScatter(comm, group_key(group), self.stats_sink)
            group._reduce_scatter_comm = comm
        engine._run_forward_backward = functools.partial(
            original_run.func,
            *original_run.args,
            **{**original_run.keywords, "defer_fsdp_gradient_reduction": spec["defer"]},
        )

    def snapshot(self, model) -> dict[str, torch.Tensor]:
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            local = (
                param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
            )
            grads[name] = local.detach().float().cpu().clone()
        return grads

    def compare(self, variant: str, reference: str) -> dict:
        sums = np.zeros((len(CATEGORIES), len(SUM_FIELDS)), dtype=np.float64)
        maxes = np.zeros((len(CATEGORIES), len(MAX_FIELDS)), dtype=np.float64)
        per_name = {}
        left, right = self.grads[variant], self.grads[reference]
        if set(left) != set(right):
            raise RuntimeError(
                f"{variant} and {reference} have different gradient sets"
            )
        for name in sorted(right):
            v, r = left[name].reshape(-1).double(), right[name].reshape(-1).double()
            d = v - r
            row = CATEGORIES.index(category(name))
            values = (
                float(d.square().sum()),
                float(r.square().sum()),
                float(v.square().sum()),
                float((v * r).sum()),
                float(r.numel()),
                float((r != 0).sum()),
                float(((r != 0) & (v == 0)).sum()),
                float(((r == 0) & (v != 0)).sum()),
                float((v != r).sum()),
                float((torch.sign(v) != torch.sign(r)).sum()),
            )
            sums[row] += values
            local_max = (
                float(d.abs().max()) if d.numel() else 0.0,
                float(r.abs().max()) if r.numel() else 0.0,
            )
            maxes[row] = np.maximum(maxes[row], local_max)
            per_name[name] = np.array(values[:2] + local_max)
        sums = self.allreduce(sums, "sum")
        maxes = self.allreduce(maxes, "max")
        gathered = self.comm.gather(per_name, root=0)
        report = {}
        for row, cat in enumerate(list(CATEGORIES) + ["total"]):
            if cat == "total":
                s, m = sums.sum(axis=0), maxes.max(axis=0)
            else:
                s, m = sums[row], maxes[row]
            if s[SUM_FIELDS.index("numel")] == 0:
                continue
            f = dict(zip(SUM_FIELDS, s.tolist()))
            f.update(zip(MAX_FIELDS, m.tolist()))
            ref_norm, var_norm = math.sqrt(f["ref_sq"]), math.sqrt(f["var_sq"])
            f["rel_l2"] = math.sqrt(f["diff_sq"]) / ref_norm if ref_norm else None
            f["norm_ratio_minus_1"] = var_norm / ref_norm - 1 if ref_norm else None
            f["cosine"] = (
                f["dot"] / (ref_norm * var_norm) if ref_norm and var_norm else None
            )
            f["underflow_fraction"] = (
                f["underflow_to_zero"] / f["ref_nonzero"] if f["ref_nonzero"] else None
            )
            f["differs_fraction"] = f["differs"] / f["numel"]
            f["sign_flip_fraction"] = f["sign_flip"] / f["numel"]
            f["ref_norm"] = ref_norm
            report[cat] = f
        if self.rank == 0:
            # Per-parameter relative L2 from all shards; keep the worst eight.
            totals: dict[str, np.ndarray] = {}
            for shard in gathered:
                for n, t in shard.items():
                    totals[n] = totals.get(n, 0) + t
            worst = sorted(
                (
                    (n, math.sqrt(t[0]) / math.sqrt(t[1]) if t[1] else 0.0)
                    for n, t in totals.items()
                ),
                key=lambda item: -item[1],
            )[:8]
            report["worst_parameters_rel_l2"] = worst
        return report

    def allreduce(self, array: np.ndarray, op: str) -> np.ndarray:
        from mpi4py import MPI

        array = np.ascontiguousarray(array, dtype=np.float64)
        out = np.empty_like(array)
        self.comm.Allreduce(array, out, op=MPI.SUM if op == "sum" else MPI.MAX)
        return out

    def write(self, tag: str) -> None:
        if self.rank == 0:
            path = self.output / f"reduce-precision-{tag}.json"
            path.write_text(json.dumps(self.result, indent=2, default=str) + "\n")
            print(f"REDUCE_PROBE_WRITE {path}", flush=True)
        self.comm.Barrier()

    def run(self, engine, original_forward_backward, kwargs) -> None:
        model = engine.model_parts[0]
        if len(engine.model_parts) != 1:
            raise RuntimeError("probe supports one model part")
        original_run = engine._run_forward_backward
        if not (
            isinstance(original_run, functools.partial)
            and original_run.func == engine._forward_backward_body
        ):
            raise RuntimeError(f"unexpected forward/backward runner: {original_run!r}")
        # Group dtypes are set at lazy init, normally by the first forward.
        model._get_fsdp_state()._lazy_init()
        groups = list(fsdp_groups(model))
        original_comm = {id(g): g._reduce_scatter_comm for g in groups}
        original_reduce = {
            id(g): (g._reduce_dtype, [p.reduce_dtype for p in g.fsdp_params])
            for g in groups
        }
        param_dtypes = {str(p.param_dtype) for g in groups for p in g.fsdp_params}
        orig_dtypes = {str(p.orig_dtype) for g in groups for p in g.fsdp_params}
        self.result["fsdp"] = {
            "groups": len(groups),
            "param_dtypes": sorted(param_dtypes),
            "orig_dtypes": sorted(orig_dtypes),
            "configured_reduce_dtypes": sorted(
                {str(v[0]) for v in original_reduce.values()}
            ),
            "gradient_divide_factor": sorted(
                {str(g.gradient_divide_factor) for g in groups}
            ),
            "force_sum_reduction_for_comms": sorted(
                {bool(g.force_sum_reduction_for_comms) for g in groups}
            ),
            "reduce_scatter_comm": sorted(
                {type(c).__name__ for c in original_comm.values()}
            ),
            "accumulation_steps": len(kwargs["microbatch_groups"]),
            "global_loss_token_counts": float(
                torch.as_tensor(kwargs["global_loss_token_counts"]).sum()
            ),
        }
        if param_dtypes != {"torch.bfloat16"}:
            raise RuntimeError(
                f"probe expects BF16 FSDP param_dtype, got {param_dtypes}"
            )
        ntokens_seen = engine.ntokens_seen
        torch.set_num_threads(8)
        try:
            for index, phase in enumerate(PHASES):
                for name in phase:
                    self.configure(engine, model, original_comm, original_run, name)
                    torch.xpu.synchronize()
                    torch.xpu.reset_peak_memory_stats()
                    started = time.perf_counter()
                    result = original_forward_backward(engine, **kwargs)
                    torch.xpu.synchronize()
                    elapsed = time.perf_counter() - started
                    loss = float(result.loss.detach().double().item())
                    self.grads[name] = self.snapshot(model)
                    peak = float(
                        self.allreduce(
                            np.array(
                                [torch.xpu.max_memory_allocated()], dtype=np.float64
                            ),
                            "max",
                        )[0]
                    )
                    losses = self.comm.gather(loss, root=0)
                    entry = {
                        "seconds_rank0": elapsed,
                        "peak_allocated_gib_max_rank": peak / 2**30,
                    }
                    if self.rank == 0:
                        entry["loss_rank0"] = loss
                        entry["loss_ranks_min_max"] = [min(losses), max(losses)]
                    if self.stats_sink:
                        for key, (calls, local_sq, reduced_sq) in sorted(
                            self.stats_sink.items()
                        ):
                            pair = np.array([float(local_sq), float(reduced_sq)])
                            pair = self.allreduce(pair, "sum")
                            self.result["kappa"].setdefault(name, {})[key] = {
                                "reduce_scatter_calls_per_rank": calls,
                                "sum_rank_local_sq": pair[0],
                                "reduced_sq": pair[1],
                                "kappa": pair[0] / pair[1] if pair[1] else None,
                            }
                    self.result["variants"][name] = entry
                    if self.rank == 0:
                        print(
                            f"REDUCE_PROBE_VARIANT {name} {json.dumps(entry)}",
                            flush=True,
                        )
                for variant, reference in PAIRS:
                    key = f"{variant}__vs__{reference}"
                    if (
                        key not in self.result["pairs"]
                        and variant in self.grads
                        and reference in self.grads
                    ):
                        self.result["pairs"][key] = self.compare(variant, reference)
                        if self.rank == 0:
                            total = self.result["pairs"][key]["total"]
                            print(
                                f"REDUCE_PROBE_PAIR {key} rel_l2={total['rel_l2']} "
                                f"max_abs_diff={total['max_abs_diff']} sign_flip={total['sign_flip_fraction']}",
                                flush=True,
                            )
                self.write(f"phase{index + 1}")
        finally:
            for group in groups:
                group._reduce_scatter_comm = original_comm[id(group)]
                group._reduce_dtype, dtypes = original_reduce[id(group)]
                for fsdp_param, dtype in zip(group.fsdp_params, dtypes):
                    fsdp_param.reduce_dtype = dtype
            engine._run_forward_backward = original_run
            engine.ntokens_seen = ntokens_seen
            self.grads.clear()


def install(output: Path) -> None:
    from torchtitan.training_engine import TrainingEngine

    original = TrainingEngine.forward_backward
    state = {"done": False}

    @functools.wraps(original)
    def probed_forward_backward(self, **kwargs):
        if not state["done"]:
            state["done"] = True
            Probe(output).run(self, original, kwargs)
        return original(self, **kwargs)

    TrainingEngine.forward_backward = probed_forward_backward


def main() -> None:
    output = Path(os.environ["AGPT_REDUCE_PROBE_OUTPUT"])
    install(output)
    sys.argv = ["torchtitan.experiments.ezpz.train", *sys.argv[1:]]
    runpy.run_module(
        "torchtitan.experiments.ezpz.train", run_name="__main__", alter_sys=True
    )


if __name__ == "__main__":
    main()
