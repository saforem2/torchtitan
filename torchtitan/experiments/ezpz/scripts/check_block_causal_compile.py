#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check block-causal attention gradients eager vs under torch.compile on one XPU tile.

Shape B=1, S=8192, Hq=16, Hkv=4, D=128, bf16, the AGPT-2B attention call.
Inputs are built in [B, S, H, D] and transposed to [B, H, S, D], as
``XPUScaledDotProductAttention`` passes them. Each case prints one
``RESULT {json}`` line with out/dq/dk/dv relative L2 error against an FP32
math reference computed in query chunks (no [S, S] matrix for all heads).

Cases: default SDPA eager (twice) and compiled, block-causal (4, 8 chunks)
eager and compiled with ``fullgraph=True``. Each case after the first also
reports its relative L2 difference from the first default eager run.

``--inputs FILE ...`` instead replays attention calls saved by
``probe_block_causal_grads.py`` (``AGPT_BLOCK_PROBE_DUMP=1``): the recorded
q/k/v values, strides and dtypes, output gradient, scale and autocast state.
"""

from __future__ import annotations

import argparse
import contextlib
import json

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torchtitan.experiments.ezpz.agpt.xpu_block_causal_attention import (
    block_causal_flash_attention,
)

B, S, HQ, HKV, D = 1, 8192, 16, 4, 128
SCALE = D**-0.5
CTX = contextlib.nullcontext
BACKENDS = [SDPBackend.OVERRIDEABLE, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]


def reference(q, k, v, do):
    """FP32 causal attention and its gradients, 1024 query rows at a time."""
    q, k, v = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    S = q.shape[-2]
    kk = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    vv = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    outs = []
    for start in range(0, S, 1024):
        rows = slice(start, start + 1024)
        s = (q[:, :, rows] @ kk[:, :, : start + 1024].transpose(-1, -2)) * SCALE
        mask = torch.ones(1024, start + 1024, dtype=torch.bool, device=q.device).tril(
            start
        )
        p = s.masked_fill(~mask, float("-inf")).softmax(-1)
        o = p @ vv[:, :, : start + 1024]
        o.backward(do[:, :, rows].float())
        outs.append(o.detach())
    return [torch.cat(outs, dim=2), q.grad, k.grad, v.grad]


def default_attention(q, k, v):
    with sdpa_kernel(BACKENDS):
        return F.scaled_dot_product_attention(
            q, k, v, scale=SCALE, is_causal=True, enable_gqa=True
        )


def rel(got, ref):
    return {
        n: float((a - b).norm() / b.norm())
        for n, a, b in zip(("out", "dq", "dk", "dv"), got, ref)
    }


def leaf(t):
    """Copy keeping the strides, including a non-dense view of a fused projection."""
    x = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
    return x.copy_(t.detach()).requires_grad_(True)


def run(name, fn, base, ref, default=None):
    q, k, v = (leaf(t) for t in base[:3])
    try:
        with CTX():
            out = fn(q, k, v)
        out.backward(base[3])
        torch.xpu.synchronize()
        got = [out.detach().float(), q.grad.float(), k.grad.float(), v.grad.float()]
        rec = {"vs_fp32": rel(got, ref)}
        if default is not None:
            rec["vs_default_eager"] = rel(got, default)
    except Exception as e:  # noqa: BLE001
        got, rec = (
            None,
            {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"},
        )
    print("RESULT " + json.dumps({"case": name, **rec}), flush=True)
    torch.xpu.empty_cache()
    return got


def replay(paths):
    """Default vs block-causal attention on recorded training calls."""
    global SCALE, CTX
    dev = torch.device("xpu")
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        meta = data["meta"]
        print(f"INPUTS {path} {json.dumps(meta)}", flush=True)
        base = []
        for name in ("q", "k", "v"):
            m = meta[name]
            t = torch.empty_strided(
                m["shape"], m["stride"], dtype=data[name].dtype, device=dev
            )
            base.append(t.copy_(data[name]))
        base.append(data["do"].to(dev))
        scale = meta["kwargs"].get("scale")
        SCALE = (
            float(scale) if scale not in (None, "None") else base[0].shape[-1] ** -0.5
        )
        CTX = (
            (lambda: torch.autocast("xpu", dtype=torch.bfloat16))
            if meta["autocast_xpu"]
            else contextlib.nullcontext
        )
        ref = reference(*base[:3], base[3])
        tag = path.rsplit("/", 1)[-1]
        default = run(f"{tag}:default", default_attention, base, ref)
        run(f"{tag}:default_repeat", default_attention, base, ref, default)
        for n in (4, 8):

            def blk(q, k, v, n=n, scale=SCALE):
                return block_causal_flash_attention(q, k, v, n_chunks=n, scale=scale)

            run(f"{tag}:blk{n}", blk, base, ref, default)
            # Same values, dense BSHD layout and bf16: isolates layout/dtype.
            dense = [
                t.detach()
                .to(torch.bfloat16)
                .transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
                for t in base[:3]
            ] + [base[3]]
            run(f"{tag}:blk{n}_dense_bf16", blk, dense, ref, default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*")
    args = parser.parse_args()
    if args.inputs:
        replay(args.inputs)
        return
    torch.manual_seed(0)
    dev = torch.device("xpu")

    def bshd(h):
        return torch.randn(B, S, h, D, device=dev, dtype=torch.bfloat16).transpose(1, 2)

    base = [bshd(HQ), bshd(HKV), bshd(HKV), bshd(HQ)]
    ref = reference(*base[:3], base[3])
    print(f"torch={torch.__version__} dev={torch.xpu.get_device_name()}", flush=True)
    default = run("default_eager", default_attention, base, ref)
    cases = {"default_eager_repeat": default_attention}
    cases["default_compiled"] = torch.compile(default_attention, fullgraph=True)
    for n in (4, 8):

        def blk(q, k, v, n=n):
            return block_causal_flash_attention(q, k, v, n_chunks=n, scale=SCALE)

        cases[f"blk{n}_eager"] = blk
        cases[f"blk{n}_compiled"] = torch.compile(blk, fullgraph=True)
    for name, fn in cases.items():
        run(name, fn, base, ref, default)


if __name__ == "__main__":
    main()
