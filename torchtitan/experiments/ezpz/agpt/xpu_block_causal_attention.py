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

Forward merges the per-diagonal outputs with their log-sum-exp. Backward uses
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


def _merge(o_acc, lse_acc, o, lse):
    if LSE_BASE2:
        new = torch.logaddexp2(lse_acc, lse)
        a, b = torch.exp2(lse_acc - new), torch.exp2(lse - new)
    else:
        new = torch.logaddexp(lse_acc, lse)
        a, b = torch.exp(lse_acc - new), torch.exp(lse - new)
    return o_acc * a.unsqueeze(-1) + o * b.unsqueeze(-1), new


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
    @staticmethod
    def forward(ctx, q, k, v, n: int, scale: float):
        # q: [B, Hq, S, D]; k, v: [B, Hkv, S, D] (GQA handled by the kernel).
        B, _, S, _ = q.shape
        c = S // n
        o_acc = lse_acc = None
        rng = []
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
            o = _unfold(res[0], B).float()
            lse = _unfold_lse(res[1], B)
            if d == 0:
                o_acc, lse_acc = o, lse.clone()
            else:
                rows = slice(d * c, S)
                o_acc[:, :, rows], lse_acc[:, :, rows] = _merge(
                    o_acc[:, :, rows], lse_acc[:, :, rows], o, lse
                )
        out = o_acc.to(q.dtype)
        ctx.save_for_backward(q, k, v, out, lse_acc)
        ctx.n, ctx.scale, ctx.rng = n, scale, rng
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        n, scale = ctx.n, ctx.scale
        B, _, S, _ = q.shape
        c = S // n
        dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
        dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
        for d in range(n):
            nblk = n - d
            seed, off = ctx.rng[d]
            gq, gk, gv = _aten._scaled_dot_product_flash_attention_backward(
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
            dq[:, :, d * c :] += _unfold(gq, B)
            dk[:, :, : nblk * c] += _unfold(gk, B)
            dv[:, :, : nblk * c] += _unfold(gv, B)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


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
