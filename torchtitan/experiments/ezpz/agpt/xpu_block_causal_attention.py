# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block-causal flash attention for XPU that skips the masked upper triangle.

Why: the torch-xpu sycl-tla flash attention BACKWARD kernel
(``cute::MhaBackward<FAKernel<bf16,128,128,128,32,4,4,4,true>>``, torch 2.14)
does not skip fully masked key tiles under ``is_causal=True``. Measured at the
AGPT-2B shape (B=1, S=8192, Hq=16, Hkv=4, D=128, bf16, one Max 1550 tile):
causal bwd 24.7 ms vs non-causal bwd 25.0 ms -- the same work. IPEX xetla
(the MDS path) skips them: causal 10.2 ms vs non-causal 19.0 ms.

What: split the sequence into ``n`` chunks of ``c = S/n`` and compute only
the lower-triangle chunk pairs ``(i, j <= i)``. Pairs on the same block
diagonal ``d = i - j`` share a shape, so each diagonal is ONE flash call with
its ``n - d`` blocks folded into the batch dim (a view for BSHD-contiguous
input, no copy). Diagonal ``d = 0`` is square, so ``is_causal`` is legal
there; ``d > 0`` is dense. That is ``n`` kernel calls doing
``n(n+1)/2`` of the ``n^2`` block products (8 chunks: 56%).

Forward merges the per-diagonal outputs with their log-sum-exp
(global LSE first, then one fused multiply-add per diagonal). Backward uses
the ring/context-parallel identity: each diagonal's flash backward is given
the GLOBAL output and GLOBAL LSE, which makes its dQ/dK/dV contribution exact;
contributions are summed in fp32.

Only the ATen flash ops are used (``_scaled_dot_product_flash_attention`` and
its ``_backward``), so this needs no IPEX and no core torchtitan change.
"""

from __future__ import annotations

import torch

_aten = torch.ops.aten

# Base of the kernel's returned log-sum-exp. Natural log is verified for the
# torch 2.14 xpu sycl-tla kernel (base 2 is wrong by ~0.1 abs in out).
LSE_BASE2 = False


def _logaddexp(a, b):
    return torch.logaddexp2(a, b) if LSE_BASE2 else torch.logaddexp(a, b)


def _exp(x):
    return torch.exp2(x) if LSE_BASE2 else torch.exp(x)


def _fold(x, start: int, nblk: int, c: int):
    """[B, H, S, D] rows [start*c, (start+nblk)*c) -> [B*nblk, H, c, D]."""
    B, H, _, D = x.shape
    t = x.transpose(1, 2)[:, start * c : (start + nblk) * c]
    return t.reshape(B * nblk, c, H, D).transpose(1, 2)


def _unfold(x, B: int):
    """[B*nblk, H, c, D] -> [B, H, nblk*c, D]."""
    Bn, H, c, D = x.shape
    nblk = Bn // B
    return x.transpose(1, 2).reshape(B, nblk * c, H, D).transpose(1, 2)


def _fold_lse(lse, start: int, nblk: int, c: int):
    """[B, H, S] -> [B*nblk, H, c] (contiguous)."""
    B, H, _ = lse.shape
    t = lse[:, :, start * c : (start + nblk) * c].reshape(B, H, nblk, c)
    return t.permute(0, 2, 1, 3).reshape(B * nblk, H, c).contiguous()


def _unfold_lse(lse, B: int):
    Bn, H, c = lse.shape
    nblk = Bn // B
    return lse.reshape(B, nblk, H, c).permute(0, 2, 1, 3).reshape(B, H, nblk * c)


class _BlockCausalFlash(torch.autograd.Function):
    # All big accumulators live in [B, S, H, D] ("BSHD") so that the row
    # ranges each diagonal touches are contiguous slices: one fused kernel per
    # diagonal instead of strided BHSD updates and per-diagonal copies.
    @staticmethod
    def forward(ctx, q, k, v, n: int, scale: float):
        # q: [B, Hq, S, D]; k, v: [B, Hkv, S, D] (GQA handled by the kernel).
        B, H, S, D = q.shape
        c = S // n
        outs, lses, rng = [], [], []
        for d in range(n):
            nblk = n - d
            res = _aten._scaled_dot_product_flash_attention(
                _fold(q, d, nblk, c),
                _fold(k, 0, nblk, c),
                _fold(v, 0, nblk, c),
                0.0,
                d == 0,
                False,
                scale=scale,
            )
            rng.append((res[6], res[7]))
            outs.append(res[0])  # [B*nblk, H, c, D]
            lses.append(_unfold_lse(res[1], B))  # [B, H, nblk*c] fp32
        # Global LSE first (small tensors), then each diagonal's output is
        # added exactly once with weight exp(lse_d - lse).
        lse = lses[0].clone()
        for d in range(1, n):
            rows = slice(d * c, S)
            lse[:, :, rows] = _logaddexp(lse[:, :, rows], lses[d])
        acc = torch.empty(B, S, H, D, dtype=torch.float32, device=q.device)
        for d in range(n):
            rows = slice(d * c, S)
            w = _exp(lses[d] - lse[:, :, rows]).transpose(1, 2).unsqueeze(-1)
            o = _unfold(outs[d], B).transpose(1, 2)  # [B, rows, H, D]
            if d == 0:
                torch.mul(o, w, out=acc)
            else:
                acc[:, rows].addcmul_(o, w)
        out = acc.to(q.dtype)  # [B, S, H, D] contiguous
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.n, ctx.scale, ctx.rng = n, scale, rng
        return out.transpose(1, 2)

    @staticmethod
    def backward(ctx, do):
        q, k, v, out_bshd, lse = ctx.saved_tensors
        n, scale = ctx.n, ctx.scale
        B, _, S, _ = q.shape
        c = S // n
        out = out_bshd.transpose(1, 2)
        do = do.transpose(1, 2).contiguous().transpose(1, 2)
        acc = [
            torch.empty(
                t.shape[0],
                S,
                t.shape[1],
                t.shape[3],
                dtype=torch.float32,
                device=t.device,
            )
            for t in (q, k, v)
        ]
        for d in range(n):
            nblk = n - d
            seed, off = ctx.rng[d]
            grads = _aten._scaled_dot_product_flash_attention_backward(
                _fold(do, d, nblk, c),
                _fold(q, d, nblk, c),
                _fold(k, 0, nblk, c),
                _fold(v, 0, nblk, c),
                _fold(out, d, nblk, c),
                _fold_lse(lse, d, nblk, c),
                None,
                None,
                c,
                c,
                0.0,
                d == 0,
                seed,
                off,
                scale=scale,
            )
            spans = (slice(d * c, S), slice(0, nblk * c), slice(0, nblk * c))
            for a, g, rows in zip(acc, grads, spans):
                g = _unfold(g, B).transpose(1, 2)
                if d == 0:
                    a.copy_(g)
                else:
                    a[:, rows].add_(g)
        dq, dk, dv = (a.to(t.dtype).transpose(1, 2) for a, t in zip(acc, (q, k, v)))
        return dq, dk, dv, None, None


def block_causal_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    n_chunks: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal attention on [B, H, S, D] tensors via ``n_chunks`` block diagonals.

    ``n_chunks <= 1`` is one causal flash call through the same path.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    S = q.shape[-2]
    n = max(1, int(n_chunks))
    if S % n != 0 or k.shape[-2] != S:
        raise ValueError(
            f"block-causal attention needs seq {S} divisible by n_chunks "
            f"{n} and seqlen_q == seqlen_k (got k seq {k.shape[-2]})"
        )
    return _BlockCausalFlash.apply(q, k, v, n, scale)
