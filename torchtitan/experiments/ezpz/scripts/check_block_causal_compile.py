#!/usr/bin/env python3
"""Check block-causal attention gradients eager vs under torch.compile on one XPU tile.

Shape B=1, S=8192, Hq=16, Hkv=4, D=128, bf16, the AGPT-2B attention call.
Inputs are built in [B, S, H, D] and transposed to [B, H, S, D], as
``XPUScaledDotProductAttention`` passes them. Each case prints one
``RESULT {json}`` line with out/dq/dk/dv relative L2 error against an FP32
math reference computed in query chunks (no [S, S] matrix for all heads).

Cases: default SDPA eager and compiled, block-causal (4, 8 chunks) eager and
compiled with ``fullgraph=True``.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchtitan.experiments.ezpz.agpt.xpu_block_causal_attention import (
    block_causal_flash_attention,
)

B, S, HQ, HKV, D = 1, 8192, 16, 4, 128
SCALE = D**-0.5
BACKENDS = [SDPBackend.OVERRIDEABLE, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]


def reference(q, k, v, do):
    """FP32 causal attention and its gradients, 1024 query rows at a time."""
    q, k, v = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    kk = k.repeat_interleave(HQ // HKV, dim=1)
    vv = v.repeat_interleave(HQ // HKV, dim=1)
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


def run(name, fn, base, ref):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in base[:3])
    try:
        out = fn(q, k, v)
        out.backward(base[3])
        torch.xpu.synchronize()
        got = [out.detach().float(), q.grad.float(), k.grad.float(), v.grad.float()]
        rec = {
            n: float((a - b).norm() / b.norm())
            for n, a, b in zip(("out", "dq", "dk", "dv"), got, ref)
        }
    except Exception as e:  # noqa: BLE001
        rec = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"}
    print("RESULT " + json.dumps({"case": name, **rec}), flush=True)
    torch.xpu.empty_cache()


def main():
    torch.manual_seed(0)
    dev = torch.device("xpu")

    def bshd(h):
        return torch.randn(B, S, h, D, device=dev, dtype=torch.bfloat16).transpose(1, 2)

    base = [bshd(HQ), bshd(HKV), bshd(HKV), bshd(HQ)]
    ref = reference(*base[:3], base[3])
    print(f"torch={torch.__version__} dev={torch.xpu.get_device_name()}", flush=True)
    cases = {"default_eager": default_attention}
    cases["default_compiled"] = torch.compile(default_attention, fullgraph=True)
    for n in (4, 8):

        def blk(q, k, v, n=n):
            return block_causal_flash_attention(q, k, v, n_chunks=n, scale=SCALE)

        cases[f"blk{n}_eager"] = blk
        cases[f"blk{n}_compiled"] = torch.compile(blk, fullgraph=True)
    for name, fn in cases.items():
        run(name, fn, base, ref)


if __name__ == "__main__":
    main()
