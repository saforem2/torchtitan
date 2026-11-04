#!/usr/bin/env python3
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
- ``blk4`` / ``blk8``: ``block_causal_flash_attention`` with 4 / 8 chunks.

``torch._dynamo.reset()`` runs before every variant, so a compiled block
cannot reuse a graph traced under another ``causal_blocks`` value.

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
VARIANTS = {"default": 0, "default_repeat": 0, "blk4": 4, "blk8": 8}
PAIRS = (
    ("default_repeat", "default"),
    ("blk4", "default"),
    ("blk8", "default"),
    ("blk8", "blk4"),
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
            for name, n in VARIANTS.items():
                for m in modules:
                    m.causal_blocks = n
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
    updates = {
        int(x) for x in os.environ.get("AGPT_BLOCK_PROBE_UPDATES", "1,24").split(",")
    }
    install(output, updates)
    sys.argv = ["torchtitan.experiments.ezpz.train", *sys.argv[1:]]
    runpy.run_module(
        "torchtitan.experiments.ezpz.train", run_name="__main__", alter_sys=True
    )


if __name__ == "__main__":
    main()
