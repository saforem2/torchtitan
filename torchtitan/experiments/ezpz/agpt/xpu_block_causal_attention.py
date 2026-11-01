"""Block-causal flash attention for XPU that skips the masked upper triangle.

Why: the torch-xpu sycl-tla flash attention BACKWARD kernel
(``cute::MhaBackward<FAKernel<bf16,128,128,128,32,4,4,4,true>>``, torch 2.14)
does not skip fully masked key tiles under ``is_causal=True``. Measured at the
AGPT-2B shape (B=1, S=8192, Hq=16, Hkv=4, D=128, bf16, one Max 1550 tile):
causal bwd 24.6 ms vs non-causal bwd 25.0 ms -- the same work. IPEX xetla
(the MDS path) skips them and runs the causal bwd in ~13.7 ms.

What: split the sequence into ``n`` chunks and only run the (i, j <= i) chunk
pairs -- ``n(n+1)/2`` of ``n^2`` blocks. Diagonal blocks use ``is_causal``
(square, so the XPU flash restriction ``seqlen_q == seqlen_k`` holds);
below-diagonal blocks are dense. Forward merges the per-block outputs with
their log-sum-exp. Backward is the ring/context-parallel attention identity:
each block's flash backward is called with the GLOBAL output and GLOBAL LSE,
which makes its dQ/dK/dV contribution exact; contributions are summed in fp32.

Only the ATen flash ops are used (``_scaled_dot_product_flash_attention`` and
its ``_backward``), so this needs no IPEX and no core torchtitan change.
"""

from __future__ import annotations

import torch

_aten = torch.ops.aten

# Base of the kernel's returned log-sum-exp. Natural log is what the CUDA
# flash kernels return; flip only if the numerics check says otherwise.
LSE_BASE2 = False


def _merge(o_acc, lse_acc, o, lse):
    if LSE_BASE2:
        new = torch.logaddexp2(lse_acc, lse)
        a, b = torch.exp2(lse_acc - new), torch.exp2(lse - new)
    else:
        new = torch.logaddexp(lse_acc, lse)
        a, b = torch.exp(lse_acc - new), torch.exp(lse - new)
    return o_acc * a.unsqueeze(-1) + o * b.unsqueeze(-1), new


def _fwd(q, k, v, causal: bool, scale: float):
    out = _aten._scaled_dot_product_flash_attention(
        q, k, v, 0.0, causal, False, scale=scale
    )
    # (out, logsumexp, cum_q, cum_k, max_q, max_k, philox_seed, philox_offset, mask)
    return out[0], out[1], out[6], out[7]


def _bwd(do, q, k, v, o, lse, causal, scale, seed, offset):
    sq, sk = q.shape[-2], k.shape[-2]
    return _aten._scaled_dot_product_flash_attention_backward(
        do, q, k, v, o, lse, None, None, sq, sk, 0.0, causal, seed, offset,
        scale=scale,
    )


class _BlockCausalFlash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, n_chunks: int, scale: float):
        # q: [B, Hq, S, D]; k, v: [B, Hkv, S, D] (GQA handled by the kernel).
        S = q.shape[-2]
        c = S // n_chunks
        outs, lses, rng = [], [], {}
        for i in range(n_chunks):
            qi = q[:, :, i * c:(i + 1) * c]
            o_acc = lse_acc = None
            for j in range(i + 1):
                kj = k[:, :, j * c:(j + 1) * c]
                vj = v[:, :, j * c:(j + 1) * c]
                o, lse, seed, off = _fwd(qi, kj, vj, i == j, scale)
                rng[(i, j)] = (seed, off)
                o = o.float()
                if o_acc is None:
                    o_acc, lse_acc = o, lse
                else:
                    o_acc, lse_acc = _merge(o_acc, lse_acc, o, lse)
            outs.append(o_acc.to(q.dtype))
            lses.append(lse_acc)
        out = torch.cat(outs, dim=-2)
        lse = torch.cat(lses, dim=-1)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.n_chunks, ctx.scale, ctx.rng = n_chunks, scale, rng
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        n, scale = ctx.n_chunks, ctx.scale
        S = q.shape[-2]
        c = S // n
        do = do.contiguous() if do.stride(-1) != 1 else do
        dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
        dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
        for i in range(n):
            si = slice(i * c, (i + 1) * c)
            qi, oi, doi = q[:, :, si], out[:, :, si], do[:, :, si]
            lsei = lse[:, :, si].contiguous()
            for j in range(i + 1):
                sj = slice(j * c, (j + 1) * c)
                seed, off = ctx.rng[(i, j)]
                gq, gk, gv = _bwd(
                    doi, qi, k[:, :, sj], v[:, :, sj], oi, lsei, i == j, scale,
                    seed, off,
                )
                dq[:, :, si] += gq.float()
                dk[:, :, sj] += gk.float()
                dv[:, :, sj] += gv.float()
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def block_causal_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    n_chunks: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal attention on [B, H, S, D] tensors via ``n_chunks`` square blocks.

    ``n_chunks <= 1`` falls back to one causal flash call.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    S = q.shape[-2]
    if n_chunks <= 1:
        return _BlockCausalFlash.apply(q, k, v, 1, scale)
    if S % n_chunks != 0 or k.shape[-2] != S:
        raise ValueError(
            f"block-causal attention needs seq {S} divisible by n_chunks "
            f"{n_chunks} and seqlen_q == seqlen_k (got k seq {k.shape[-2]})"
        )
    return _BlockCausalFlash.apply(q, k, v, n_chunks, scale)
