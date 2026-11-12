#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare block-causal and default XPU attention gradients on real AGPT batches.

Wraps ``torchtitan.experiments.ezpz.train`` under native DDP. At each update
listed in ``AGPT_BLOCK_PROBE_UPDATES`` (default ``1,24``) the
``TrainingEngine.forward_backward`` call is replayed with the same weights and
microbatches under each attention variant, and the all-reduced gradients are
compared on rank 0. The configured path then runs once more, so the optimizer
step and the rest of training are unchanged.

Variants set ``causal_blocks`` on every ``XPUScaledDotProductAttention``:

- ``default``: torch-xpu flash attention with ``is_causal``.
- ``default_repeat``: same again; the run-to-run floor of the default path.
- ``blk1``: one causal flash call through the block-causal autograd.Function.
- ``blk4`` / ``blk8``: ``block_causal_flash_attention`` with 4 / 8 chunks.
- ``fwdnoise1e-3`` / ``fwdnoise4e-3``: default attention with its forward
  output multiplied by ``1 + sigma * N(0, 1)`` (gradient passes through).
- ``noise1e-3`` / ``noise4e-3``: default attention, with each element of the
  q/k/v gradients multiplied by ``1 + sigma * N(0, 1)``. A benign perturbation
  of known size, to show how much the backward amplifies one.

``torch._dynamo.reset()`` runs before every variant, so a compiled block
cannot reuse a graph traced under another ``causal_blocks`` value.

``AGPT_BLOCK_PROBE_DUMP=1`` (eager blocks only) also saves the inputs, layout
and output gradient of the first and last attention calls of the ``default``
variant on rank 0, for ``check_block_causal_compile.py --inputs``.

Environment: ``AGPT_BLOCK_PROBE_OUTPUT`` (required) is the output directory.
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

import torch
import torch.distributed as dist

CATEGORIES = ("embed", "attention", "mlp", "norm", "lm_head", "other")
DUMP = os.environ.get("AGPT_BLOCK_PROBE_DUMP", "0") == "1"
# name: (causal_blocks, relative q/k/v gradient noise)
VARIANTS = {
    "default": (0, 0.0),
    "default_repeat": (0, 0.0),
    # One causal flash call through the block-causal autograd.Function: the
    # same math as default, so it separates the wrapper from the chunking.
    "blk1": (1, 0.0),
    "blk4": (4, 0.0),
    "blk8": (8, 0.0),
    "noise1e-3": (0, 1e-3),
    "noise4e-3": (0, 4e-3),
    # Negative sigma: perturb the forward attention output instead,
    # out + sigma * N(0, 1) * out with the gradient passing straight through.
    "fwdnoise1e-3": (0, -1e-3),
    "fwdnoise4e-3": (0, -4e-3),
}
# AGPT_BLOCK_PROBE_VARIANTS (comma-separated) keeps a subset, in this order.
_KEEP = os.environ.get("AGPT_BLOCK_PROBE_VARIANTS")
if _KEEP:
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in _KEEP.split(",")}
PAIRS = (
    ("default_repeat", "default"),
    ("blk1", "default"),
    ("blk4", "blk1"),
    ("blk4", "default"),
    ("blk8", "default"),
    ("blk8", "blk4"),
    ("noise1e-3", "default"),
    ("noise4e-3", "default"),
    ("fwdnoise1e-3", "default"),
    ("fwdnoise4e-3", "default"),
)


class _GradNoise(torch.autograd.Function):
    """Identity forward; backward multiplies the gradient by 1 + sigma * N(0, 1)."""

    @staticmethod
    def forward(ctx, x, sigma: float):
        ctx.sigma = sigma
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        noise = torch.randn(grad.shape, dtype=torch.float32, device=grad.device)
        return (grad.float() * (1 + ctx.sigma * noise)).to(grad.dtype), None


def patch_attention_noise() -> None:
    from torchtitan.experiments.ezpz.agpt import XPUScaledDotProductAttention

    original = XPUScaledDotProductAttention._sdpa

    @functools.wraps(original)
    def _sdpa(self, q, k, v, **kwargs):
        sigma = getattr(self, "probe_grad_noise", 0.0)
        if sigma > 0:
            q, k, v = (_GradNoise.apply(t, sigma) for t in (q, k, v))
        if self.causal_blocks == 1 and kwargs.get("is_causal"):
            from torchtitan.experiments.ezpz.agpt.xpu_block_causal_attention import (
                block_causal_flash_attention,
            )

            out = block_causal_flash_attention(
                q, k, v, n_chunks=1, scale=kwargs.get("scale")
            )
        else:
            out = original(self, q, k, v, **kwargs)
        if sigma < 0:
            noise = torch.randn(out.shape, dtype=torch.float32, device=out.device)
            out = out + (out.float() * (-sigma) * noise).to(out.dtype).detach()
        path = getattr(self, "probe_dump_path", None)
        if path is not None:
            self.probe_dump_path = None
            dump_attention_call(path, q, k, v, out, kwargs)
        return out

    XPUScaledDotProductAttention._sdpa = _sdpa


def dump_attention_call(path: Path, q, k, v, out, kwargs) -> None:
    """Save one eager attention call's inputs, layout and output gradient."""
    meta = {
        name: {
            "dtype": str(t.dtype),
            "shape": list(t.shape),
            "stride": list(t.stride()),
            "contiguous": t.is_contiguous(),
            "requires_grad": t.requires_grad,
        }
        for name, t in (("q", q), ("k", k), ("v", v), ("out", out))
    }
    meta["autocast_xpu"] = torch.is_autocast_enabled("xpu")
    meta["autocast_dtype"] = str(torch.get_autocast_dtype("xpu"))
    meta["kwargs"] = {key: str(value) for key, value in kwargs.items()}
    data = {name: t.detach().cpu() for name, t in (("q", q), ("k", k), ("v", v))}
    data["meta"] = meta
    print(f"BLOCK_PROBE_DUMP_META {path.name} {json.dumps(meta)}", flush=True)

    def save(grad):
        data["do"] = grad.detach().cpu()
        meta["do"] = {"dtype": str(grad.dtype), "stride": list(grad.stride())}
        torch.save(data, path)
        print(f"BLOCK_PROBE_DUMP {path}", flush=True)

    out.register_hook(save)


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


def attention_modules(model):
    from torchtitan.experiments.ezpz.agpt import XPUScaledDotProductAttention

    found = [m for m in model.modules() if isinstance(m, XPUScaledDotProductAttention)]
    if not found:
        raise RuntimeError("no XPUScaledDotProductAttention modules in the model")
    return found


def snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().float().cpu().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def compare(left: dict, right: dict, device) -> dict:
    if set(left) != set(right):
        raise RuntimeError("variants have different gradient sets")
    acc = {
        c: {
            "diff_sq": 0.0,
            "ref_sq": 0.0,
            "var_sq": 0.0,
            "dot": 0.0,
            "numel": 0,
            "differs": 0,
            "sign_flip": 0,
            "max_abs_diff": 0.0,
            "max_abs_ref": 0.0,
        }
        for c in CATEGORIES
    }
    per_name = []
    for name in sorted(right):
        a = acc[category(name)]
        diff_sq = ref_sq = 0.0
        # FP64 in 32M-element chunks: the LM-head gradient alone is ~524M.
        for lv, rv in zip(
            left[name].reshape(-1).split(1 << 25),
            right[name].reshape(-1).split(1 << 25),
        ):
            v, r = lv.to(device).double(), rv.to(device).double()
            d = v - r
            diff_sq += float(d.square().sum())
            ref_sq += float(r.square().sum())
            a["var_sq"] += float(v.square().sum())
            a["dot"] += float((v * r).sum())
            a["differs"] += int((v != r).sum())
            a["sign_flip"] += int((torch.sign(v) != torch.sign(r)).sum())
            a["max_abs_diff"] = max(a["max_abs_diff"], float(d.abs().max()))
            a["max_abs_ref"] = max(a["max_abs_ref"], float(r.abs().max()))
        a["diff_sq"] += diff_sq
        a["ref_sq"] += ref_sq
        a["numel"] += right[name].numel()
        per_name.append((name, math.sqrt(diff_sq / ref_sq) if ref_sq else 0.0))
    total = {
        k: (
            max(x[k] for x in acc.values())
            if k.startswith("max")
            else sum(x[k] for x in acc.values())
        )
        for k in acc["embed"]
    }
    report = {}
    for cat, f in [*acc.items(), ("total", total)]:
        if not f["numel"]:
            continue
        ref, var = math.sqrt(f["ref_sq"]), math.sqrt(f["var_sq"])
        report[cat] = {
            **f,
            "rel_l2": math.sqrt(f["diff_sq"]) / ref if ref else None,
            "norm_ratio_minus_1": var / ref - 1 if ref else None,
            "cosine": f["dot"] / (ref * var) if ref and var else None,
            "differs_fraction": f["differs"] / f["numel"],
            "sign_flip_fraction": f["sign_flip"] / f["numel"],
        }
    report["worst_parameters_rel_l2"] = sorted(per_name, key=lambda x: -x[1])[:8]
    return report


class Probe:
    def __init__(self, output: Path):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.rank = dist.get_rank()
        self.result: dict = {
            "world_size": dist.get_world_size(),
            "torch": torch.__version__,
            "pbs_jobid": os.environ.get("PBS_JOBID"),
            "updates": {},
        }

    def run(self, engine, original_forward_backward, kwargs, update: int) -> None:
        if len(engine.model_parts) != 1:
            raise RuntimeError("probe supports one model part")
        wrapper = engine.model_parts[0]
        model = getattr(wrapper, "module", wrapper)
        modules = attention_modules(model)
        original = [m.causal_blocks for m in modules]
        ntokens_seen = engine.ntokens_seen
        grads, entry = (
            {},
            {"variants": {}, "pairs": {}, "attention_modules": len(modules)},
        )
        try:
            for name, (n, sigma) in VARIANTS.items():
                for m in modules:
                    m.causal_blocks = n
                    m.probe_grad_noise = sigma
                if name == "default" and self.rank == 0 and DUMP:
                    # Eager blocks only: torch.save cannot run in a traced graph.
                    for index in (0, len(modules) - 1):
                        modules[index].probe_dump_path = (
                            self.output / f"attention-u{update}-layer{index}.pt"
                        )
                torch._dynamo.reset()
                torch.xpu.synchronize()
                started = time.perf_counter()
                result = original_forward_backward(engine, **kwargs)
                torch.xpu.synchronize()
                elapsed = time.perf_counter() - started
                loss = torch.as_tensor(result.loss.detach(), dtype=torch.float64)
                losses = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
                dist.all_gather(losses, loss)
                if self.rank == 0:
                    grads[name] = snapshot(model)
                    entry["variants"][name] = {
                        "causal_blocks": n,
                        "grad_noise": sigma,
                        "seconds_rank0": elapsed,
                        "loss_rank0": float(loss),
                        "loss_mean": float(torch.stack(losses).mean()),
                    }
                    print(
                        f"BLOCK_PROBE_VARIANT update={update} {name} "
                        f"{json.dumps(entry['variants'][name])}",
                        flush=True,
                    )
                dist.barrier()
            if self.rank == 0:
                device = next(model.parameters()).device
                for variant, reference in PAIRS:
                    if variant not in grads or reference not in grads:
                        continue
                    report = compare(grads[variant], grads[reference], device)
                    entry["pairs"][f"{variant}__vs__{reference}"] = report
                    t = report["total"]
                    print(
                        f"BLOCK_PROBE_PAIR update={update} {variant}__vs__{reference} "
                        f"rel_l2={t['rel_l2']:.3e} cosine={t['cosine']:.9f} "
                        f"max_abs_diff={t['max_abs_diff']:.3e} differs={t['differs_fraction']:.4f}",
                        flush=True,
                    )
                self.result["updates"][str(update)] = entry
                path = self.output / "block-causal-grads.json"
                path.write_text(json.dumps(self.result, indent=2, default=str) + "\n")
                print(f"BLOCK_PROBE_WRITE {path}", flush=True)
            dist.barrier()
        finally:
            for m, n in zip(modules, original):
                m.causal_blocks = n
                m.probe_grad_noise = 0.0
                m.probe_dump_path = None
            torch._dynamo.reset()
            engine.ntokens_seen = ntokens_seen
            grads.clear()


def install(output: Path, updates: set[int]) -> None:
    from torchtitan.training_engine import TrainingEngine

    original = TrainingEngine.forward_backward
    state = {"calls": 0, "probe": None}

    @functools.wraps(original)
    def probed_forward_backward(self, **kwargs):
        state["calls"] += 1
        if state["calls"] in updates:
            if state["probe"] is None:
                state["probe"] = Probe(output)
            state["probe"].run(self, original, kwargs, state["calls"])
        return original(self, **kwargs)

    TrainingEngine.forward_backward = probed_forward_backward


def main() -> None:
    output = Path(os.environ["AGPT_BLOCK_PROBE_OUTPUT"])
    # Comma- or dash-separated: the matrix launcher's phase overrides cannot
    # carry commas.
    raw = os.environ.get("AGPT_BLOCK_PROBE_UPDATES", "1,24")
    updates = {int(x) for x in raw.replace("-", ",").split(",") if x}
    patch_attention_noise()
    install(output, updates)
    sys.argv = ["torchtitan.experiments.ezpz.train", *sys.argv[1:]]
    runpy.run_module(
        "torchtitan.experiments.ezpz.train", run_name="__main__", alter_sys=True
    )


if __name__ == "__main__":
    main()
