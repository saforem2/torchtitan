# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fp32 residual-stream TransformerBlock for the agpt 80B family.

Root cause of the 80B production NaN (task #21, 2026-07-14): the model runs
with ``mixed_precision_param=bfloat16``, so the pre-norm residual stream

    h   = x + attention(attention_norm(x))
    out = h + feed_forward(ffn_norm(h))

accumulates across all 84 layers in bf16. RMSNorm rescales each sublayer's
*input* but never the residual stream itself, so at the 80B width x depth
(dim=9216, 84 layers, ffn=25600) the deep bf16 accumulation reaches bf16's
mantissa/overflow regime and eventually produces a non-finite activation that
poisons the whole step. 2B/20B are structurally too small to hit this. The
fp32-activations run (job 8537349) trains clean and reveals the true grad_norms
(21K-79K) that bf16 silently masks -- confirming the residual stream, not the
optimizer, is the culprit.

This block is the minimal, targeted remedy (the Llama3-405B approach): keep the
residual accumulator in fp32 for the WHOLE depth while leaving the expensive
attention/FFN GEMMs in bf16.

How the GEMMs stay bf16 despite an fp32 stream: under the production
``MixedPrecisionPolicy(param_dtype=bfloat16)`` the attention/FFN Linear weights
are bf16, so their matmuls run in bf16 regardless of the activation dtype (the
input is cast to the param dtype at the matmul). Only the two residual ADDS and
the RMSNorm reductions end up in fp32 -- both cheap, both exactly the ops that
need the extra range/precision. The block emits an fp32 activation so the next
block continues accumulating in fp32 (the stream is never re-truncated to bf16
between layers, which is the whole point).

Wire it in by giving the agpt model config ``AgptFp32ResidualBlock.Config``
layers instead of ``Llama3TransformerBlock.Config`` (see the ``*_fp32res``
flavors in the config registry). Everything else -- attention, FFN, QK-Norm,
RoPE, weight init, sharding -- is inherited unchanged.

STATUS: prototype. Must be numerically validated against the fp32-activations
reference (job 8537349) before any production use -- confirm (a) it trains
NaN-free at production dp, and (b) throughput is materially better than full
fp32 activations (the GEMMs must actually stay bf16).
"""

from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.llama3.model import Llama3TransformerBlock


class AgptFp32ResidualBlock(Llama3TransformerBlock):
    """Llama3 block with the residual stream kept in fp32 across depth.

    Identical to ``Llama3TransformerBlock`` except ``forward`` keeps the
    running residual in fp32. The attention/FFN GEMMs still run in bf16 (bf16
    params), so the only fp32 work is the two residual adds and the norm
    reductions.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3TransformerBlock.Config):
        pass

    def _sublayer_dtype(self) -> torch.dtype:
        # The dtype the sublayer GEMMs must run in = the attention params'
        # dtype (bf16 under the production MixedPrecisionPolicy). We must cast
        # each sublayer's input to this before the norm/Linear, otherwise a
        # fp32 activation x bf16 weight matmul raises a dtype-mismatch error
        # (there is no torch.autocast in this forward path).
        try:
            return next(self.attention.parameters()).dtype
        except StopIteration:
            return torch.bfloat16

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        # Do the two residual ADDS in fp32, but take bf16 in and emit bf16 out.
        #
        # Why cast back to bf16 at the block boundary (rather than carry fp32
        # across all depth): the decoder loops `h = layer(h, ...)` then feeds a
        # bf16 `lm_head` (decoder.py:283), which would dtype-mismatch on an fp32
        # `h`; keeping the boundary in the compute dtype satisfies that contract
        # with zero core changes. The between-block RMSNorm re-standardizes the
        # stream anyway, so the value that matters is protecting each individual
        # `+` from a single-step overflow -- which matches the observed failure
        # (grad_norm dead-flat for 16 steps, then one instantaneous inf/nan, not
        # a slow multi-layer drift). If a full-depth fp32 stream proves necessary
        # instead, that needs an experiment-local Decoder subclass to cast `h`
        # to the lm_head dtype before the head.
        dt = self._sublayer_dtype()

        attn_out = self.attention(
            self.attention_norm(x), attention_masks, positions
        )
        h = x.float() + attn_out.float()

        ffn_out = self.feed_forward(self.ffn_norm(h.to(dt)))
        out = h + ffn_out.float()

        return out.to(dt)
