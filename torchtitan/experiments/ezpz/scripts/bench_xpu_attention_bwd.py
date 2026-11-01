#!/usr/bin/env python3
"""Microbenchmark XPU attention fwd+bwd at the AuroraGPT-2B shape.

Default shape: B=1, S=8192, Hq=16, Hkv=4, D=128, bf16, causal, dropout 0,
scale 1/sqrt(128) -- exactly what TT and MDS pass to SDPA on Sunspot.

Each case runs in its own try/except and prints one ``RESULT {json}`` line:
ms per fwd, ms per bwd, TFLOP/s (causal FLOPs = half the dense count), the
dominant device kernels seen by torch.profiler for one fwd+bwd, and the max
abs diff of out/dq/dk/dv vs an fp32 math reference.

Runs in either venv. ``--ipex`` imports intel_extension_for_pytorch first
(the MDS path: DeepSpeed's FlashAttentionBuilder imports IPEX and calls
plain F.scaled_dot_product_attention).

Example (one tile)::

    ZE_AFFINITY_MASK=0 python bench_xpu_attention_bwd.py --tag tt214
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback

import torch
import torch.nn.functional as F


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run")
    p.add_argument("--ipex", action="store_true")
    p.add_argument("--seq", type=int, default=8192)
    p.add_argument("--hq", type=int, default=16)
    p.add_argument("--hkv", type=int, default=4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--cases", default="all", help="comma list of case names")
    p.add_argument("--no-ref", action="store_true")
    p.add_argument("--no-profile", action="store_true")
    return p.parse_args()


ARGS = parse()
if ARGS.ipex:
    import intel_extension_for_pytorch  # noqa: F401

from torch.nn.attention import sdpa_kernel, SDPBackend  # noqa: E402

DEV = torch.device("xpu")
DT = torch.bfloat16
B, S, HQ, HKV, D = ARGS.batch, ARGS.seq, ARGS.hq, ARGS.hkv, ARGS.dim
SCALE = 1.0 / math.sqrt(D)
REP = HQ // HKV


def flops(causal: bool) -> tuple[float, float]:
    fwd = 4.0 * B * HQ * S * S * D
    if causal:
        fwd /= 2
    return fwd, 2.5 * fwd


def make_base(seed: int = 0):
    """q/k/v allocated BSHD-contiguous (what the QKV projection produces)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, S, HQ, D, generator=g).to(DEV, DT)
    k = torch.randn(B, S, HKV, D, generator=g).to(DEV, DT)
    v = torch.randn(B, S, HKV, D, generator=g).to(DEV, DT)
    do = torch.randn(B, S, HQ, D, generator=g).to(DEV, DT)
    return q, k, v, do


BASE = make_base()


def leaves():
    return [t.detach().clone().requires_grad_(True) for t in BASE[:3]]


def prep(q, k, v, layout: str, gqa: str):
    """Return BHSD q, k, v as the case wants them (autograd-tracked)."""
    if gqa == "expand":
        # Megatron repeat_kv: [b,s,nkv,d] -> [b,s,nkv*rep,d] (head-major groups)
        k = k[:, :, :, None, :].expand(B, S, HKV, REP, D).reshape(B, S, HQ, D)
        v = v[:, :, :, None, :].expand(B, S, HKV, REP, D).reshape(B, S, HQ, D)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    if layout == "bhsd":
        q, k, v = (t.contiguous() for t in (q, k, v))
    return q, k, v


_MASK = None


def causal_mask():
    global _MASK
    if _MASK is None:
        _MASK = torch.ones(S, S, dtype=torch.bool, device=DEV).tril()
    return _MASK


def sdpa_case(backend, layout="bshd", gqa="native", causal=True, mask=False):
    def run(q, k, v):
        qq, kk, vv = prep(q, k, v, layout, gqa)
        kw = dict(scale=SCALE, dropout_p=0.0)
        if mask:
            kw["attn_mask"] = causal_mask()
        else:
            kw["is_causal"] = causal
        if gqa == "native":
            kw["enable_gqa"] = True
        ctx = sdpa_kernel([backend]) if backend is not None else _null()
        with ctx:
            o = F.scaled_dot_product_attention(qq, kk, vv, **kw)
        return o.transpose(1, 2)

    return run


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_FLEX = {}


def flex_case(gqa="native"):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    if "fn" not in _FLEX:
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=False)

        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        _FLEX["bm"] = create_block_mask(causal, None, None, S, S, device=DEV)

    def run(q, k, v):
        qq, kk, vv = prep(q, k, v, "bshd", gqa)
        o = _FLEX["fn"](
            qq, kk, vv, block_mask=_FLEX["bm"], scale=SCALE,
            enable_gqa=(gqa == "native"),
        )
        return o.transpose(1, 2)

    return run


def build_cases():
    M, FL, EF, OV = (
        SDPBackend.MATH,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.OVERRIDEABLE,
    )
    c = {}
    c["default_bshd_gqa"] = (sdpa_case(None), True)
    c["default_bshd_expand"] = (sdpa_case(None, gqa="expand"), True)
    for name, be in (("flash", FL), ("eff", EF), ("ovr", OV)):
        c[f"{name}_bshd_gqa"] = (sdpa_case(be), True)
        c[f"{name}_bshd_expand"] = (sdpa_case(be, gqa="expand"), True)
        c[f"{name}_bhsd_gqa"] = (sdpa_case(be, layout="bhsd"), True)
        c[f"{name}_bhsd_expand"] = (sdpa_case(be, layout="bhsd", gqa="expand"), True)
        c[f"{name}_bshd_expand_noncausal"] = (
            sdpa_case(be, gqa="expand", causal=False), False)
        c[f"{name}_bshd_expand_mask"] = (
            sdpa_case(be, gqa="expand", mask=True), True)
    c["math_bshd_expand"] = (sdpa_case(M, gqa="expand"), True)
    c["flex_gqa"] = (flex_case(), True)
    c["flex_expand"] = (flex_case(gqa="expand"), True)
    return c


def reference(causal: bool):
    q, k, v = (t.detach().float().requires_grad_(True) for t in BASE[:3])
    qq, kk, vv = prep(q, k, v, "bhsd", "expand")
    with sdpa_kernel([SDPBackend.MATH]):
        o = F.scaled_dot_product_attention(qq, kk, vv, scale=SCALE, is_causal=causal)
    o = o.transpose(1, 2)
    o.backward(BASE[3].float())
    r = [t.detach() for t in (o, q.grad, k.grad, v.grad)]
    del q, k, v, qq, kk, vv, o
    torch.xpu.empty_cache()
    return r


def top_kernels(fn):
    from torch.profiler import ProfilerActivity, profile

    q, k, v = leaves()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        o = fn(q, k, v)
        o.backward(BASE[3])
        torch.xpu.synchronize()
    out = []
    for e in prof.key_averages():
        t = getattr(e, "device_time_total", None)
        if t is None:
            t = getattr(e, "xpu_time_total", 0)
        if t and e.device_type.name in ("XPU", "CUDA", "PrivateUse1"):
            out.append((t, e.key))
    out.sort(reverse=True)
    return [f"{t/1e3:.2f}ms {k[:110]}" for t, k in out[:4]]


def bench(name, fn, causal, refs):
    rec = {"tag": ARGS.tag, "case": name, "torch": torch.__version__,
           "ipex": ARGS.ipex, "shape": [B, S, HQ, HKV, D], "causal": causal}
    try:
        tf, tb = [], []
        for i in range(ARGS.warmup + ARGS.iters):
            q, k, v = leaves()
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            o = fn(q, k, v)
            torch.xpu.synchronize()
            t1 = time.perf_counter()
            o.backward(BASE[3])
            torch.xpu.synchronize()
            t2 = time.perf_counter()
            if i >= ARGS.warmup:
                tf.append((t1 - t0) * 1e3)
                tb.append((t2 - t1) * 1e3)
        tf.sort(), tb.sort()
        mf, mb = tf[len(tf) // 2], tb[len(tb) // 2]
        ff, fb = flops(causal)
        rec.update(fwd_ms=round(mf, 3), bwd_ms=round(mb, 3),
                   fwd_min_ms=round(tf[0], 3), bwd_min_ms=round(tb[0], 3),
                   fwd_tflops=round(ff / mf / 1e9, 1),
                   bwd_tflops=round(fb / mb / 1e9, 1))
        if refs is not None and causal in refs:
            r = refs[causal]
            got = [o.detach().float(), q.grad.float(), k.grad.float(), v.grad.float()]
            rec["maxabs"] = {n: float((a - b).abs().max()) for n, a, b in
                             zip(("out", "dq", "dk", "dv"), got, r)}
            rec["refmax"] = {n: float(b.abs().max()) for n, b in
                             zip(("out", "dq", "dk", "dv"), r)}
        if not ARGS.no_profile:
            rec["kernels"] = top_kernels(fn)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
        traceback.print_exc(limit=2, file=sys.stderr)
    torch.xpu.empty_cache()
    print("RESULT " + json.dumps(rec), flush=True)


def main():
    print(f"torch={torch.__version__} ipex={ARGS.ipex} dev={torch.xpu.get_device_name()}",
          flush=True)
    cases = build_cases()
    want = list(cases) if ARGS.cases == "all" else ARGS.cases.split(",")
    refs = None
    if not ARGS.no_ref:
        refs = {True: reference(True)}
        if any(not cases[w][1] for w in want if w in cases):
            refs[False] = reference(False)
    for w in want:
        if w not in cases:
            print(f"RESULT {json.dumps({'tag': ARGS.tag, 'case': w, 'error': 'unknown'})}")
            continue
        fn, causal = cases[w]
        bench(w, fn, causal, refs)


if __name__ == "__main__":
    main()
