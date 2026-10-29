# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat
from torch import nn

from torchtitan.experiments.ezpz.logging import logger, warn_once

from torchtitan.models.common.attention import (
    FlexAttentionMetadata,
    GQAttention,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.flops import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.models.common.linear import maybe_gather_tp_input
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe.token_dispatcher import (
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
)
from torchtitan.models.deepseek_v3.model import (
    DeepSeekMLAAttention,
    DeepSeekV3TransformerBlock,
)
from torchtitan.tools.utils import has_cuda_capability


def _sdpa_config_types() -> tuple[type, ...]:
    """SDPA inner-attention config classes (ezpz-owned; imported lazily)."""
    from torchtitan.experiments.ezpz.agpt import ScaledDotProductInnerAttention

    return (ScaledDotProductInnerAttention.Config,)


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}


def _maybe_release_device_cache_between_attention_and_moe(x: torch.Tensor) -> None:
    if not _env_flag_enabled("TT_MOE_EMPTY_CACHE_BETWEEN_ATTN_MOE"):
        return
    if x.device.type == "xpu":
        torch.xpu.synchronize(x.device)
        torch.xpu.empty_cache()
    elif x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
        torch.cuda.empty_cache()


class Attention(DeepSeekMLAAttention):
    """Multi-head latent attention (MLA) with the XPU SDPA value-padding fix.

    Upstream's :class:`DeepSeekMLAAttention` is reused for structure, config
    and sharding. Only ``forward`` differs: the XPU SDPA math backend returns
    the Q/K head dimension instead of V's when they differ, so V is padded up to
    the Q/K head dimension and the output sliced back (see the NOTE below).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekMLAAttention.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
    ):
        # Mirrors DeepSeekMLAAttention.forward (flat [T, D] token layout, remat
        # annotations, TP input gather) plus the XPU value padding below.
        x = maybe_gather_tp_input(self, x)

        num_tokens = x.shape[0]

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_a(x)
            remat.recompute_needs_tensor(q)
            q = self.wq_b(self.q_norm(q))

        # spmd.local(): the unflatten splits a TP-sharded feature axis into
        # (heads, head_dim). Under TP the -1 resolves to the LOCAL head count,
        # so this must be treated as a local-shard op, not a global one.
        with spmd.local():
            q = q.view(num_tokens, -1, self.qk_head_dim)

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Key-value projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # rope, the query concat, and kv_norm read the query and wkv_a outputs
        # with bare ops.
        remat.recompute_needs_tensor(q_nope, q_pe, kv, k_pe)
        # unsqueeze(1), not (2): k_pe is [T, rope_dim] here (rank 2), so the
        # head axis is inserted at 1.
        q_pe, k_pe = self.rope(q_pe, k_pe.unsqueeze(1), positions)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.wkv_b(self.kv_norm(kv))

        with spmd.local():
            kv = kv.view(num_tokens, -1, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            # The key concat reads the wkv_b projection output with bare ops.
            remat.recompute_needs_tensor(k_nope)
            # expand to k_nope's LOCAL head count, not self.n_heads: under TP
            # each rank holds n_heads/tp_degree heads, so the global count
            # would over-expand and only fail at tp>1.
            k = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1)], dim=-1)

        # NOTE: The XPU SDPA backend on Aurora doesn't properly handle
        # different head dimensions for Q/K vs V.
        # On Intel XPU (Aurora), F.scaled_dot_product_attention returns output
        # with Q/K's head dimension instead of V's.
        # Standard CUDA backends correctly return (B, H, L, Ev) when E != Ev,
        # but the XPU MATH backend does not.
        pad_v = self.qk_head_dim != self.v_head_dim
        if pad_v:
            v = F.pad(v, (0, self.qk_head_dim - self.v_head_dim))

        output = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(q, k, v, attention_metadata=attention_metadata, scale=self.softmax_scale)

        if pad_v:
            output = output[..., : self.v_head_dim]

        # The copy below reads the inner_attention output with bare ops.
        remat.recompute_needs_tensor(output)
        output = output.contiguous().view(num_tokens, -1)
        return self.wo(output)


class moeTransformerBlock(DeepSeekV3TransformerBlock):  # noqa: N801
    """DeepSeek-style block (``feed_forward`` is dense or ``MoE``) plus ezpz's
    optional device-cache release between attention and the MoE.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3TransformerBlock.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ):
        attn_out = self.attention(self.attention_norm(x), attention_metadata, positions)
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(attn_out)
        x = x + attn_out
        _maybe_release_device_cache_between_attention_and_moe(x)
        if isinstance(self.feed_forward, MoE):
            ffn_out = self.feed_forward(
                self.ffn_norm(x),
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            ffn_out = self.feed_forward(self.ffn_norm(x))
        # Trailing add, always saved: it saves nothing for backward, so replay
        # skips it and its inputs need no persisting.
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(x, ffn_out)


class moeModel(Decoder):  # noqa: N801
    """
    moe Transformer model with attention and feed-forward layers.
    """

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)

    def parallelize(self, **kwargs):
        from .parallelize import parallelize_moe

        parallelism_context = kwargs["parallelism_context"]
        with parallelism_context.activate_spmd():
            return parallelize_moe(self, **kwargs)

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2048
        vocab_size: int = 102400
        local_compile_regions: list[str] = field(
            default_factory=lambda: [
                "loss",
                "fused_binary_activation",
                "fp32_to_bf16_split",
                "moe_aux_loss",
            ]
        )

        def __post_init__(self) -> None:
            # Flavors are prebuilt templates; give every config the sharding
            # declarations for the default parallelism (no SP, no EP) so a
            # config used without a trainer (tests, conversion) is complete.
            # ``update_from_config`` re-applies them for the final parallelism.
            from torchtitan.experiments.ezpz.moe.sharding import set_moe_sharding_config

            set_moe_sharding_config(self, enable_sp=False, enable_ep=False)

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            """Apply the trainer's parallelism/debug settings to this config.

            Upstream deleted ``Decoder.Config.update_from_config``: sharding is
            built into the model config and TP/EP divisibility is checked in
            ``Decoder.__init__`` / ``validate_model_training_config``. What
            remains here is ezpz-specific: the expert-backend fallback, the deepep/hybridep EP guard, the CP
            attention restriction, and re-applying sharding for the final
            parallelism (ezpz flavors are prebuilt templates).
            """
            parallelism = config.parallelism

            # for_loop fallback when CUDA SM90+ grouped_mm is unavailable
            # (notably on XPU). Upstream routed experts don't know about
            # compute_backend, so this stays.
            for _, moe, _, _ in self.traverse(MoE.Config):
                experts_cfg = moe.routed_experts
                if getattr(
                    experts_cfg, "compute_backend", "grouped_mm"
                ) == "grouped_mm" and not has_cuda_capability(9, 0):
                    # warn_once collapses the per-layer repetition (this loop
                    # fires once per MoE layer).
                    warn_once(
                        logger,
                        "torch._grouped_mm requires SM90+ CUDA; falling "
                        "back to for_loop expert backend.",
                    )
                    experts_cfg.compute_backend = "for_loop"
                # Detect deepep/hybridep configs by the dispatcher Config
                # class. Keep the EP=1 guard so a misconfigured deepep/hybridep
                # user gets a clear error before model init.
                if isinstance(
                    experts_cfg.token_dispatcher,
                    (DeepEPTokenDispatcher.Config, HybridEPTokenDispatcher.Config),
                ):
                    dispatcher_name = type(
                        experts_cfg.token_dispatcher
                    ).__qualname__.split(".")[0]
                    if parallelism.expert_parallel_degree == 1:
                        raise ValueError(
                            f"{dispatcher_name} requires expert "
                            "parallelism (expert_parallel_degree > 1)."
                        )

            if parallelism.context_parallel_degree > 1 and not isinstance(
                self.layers[0].attention.inner_attention,
                _sdpa_config_types(),
            ):
                raise NotImplementedError(
                    "Context Parallel for MoE only supports "
                    "ScaledDotProductInnerAttention. Got "
                    f"{type(self.layers[0].attention.inner_attention).__name__}."
                )

            # Re-apply the sharding declarations for the final parallelism so
            # Module.parallelize() can distribute params/activations.
            from torchtitan.experiments.ezpz.moe.sharding import set_moe_sharding_config

            set_moe_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            # Upstream #4300 removed get_moe_model_nparams_and_flops. Its
            # replacement returns (nparams, ACTIVE nparams) and no longer
            # computes flops, so the attention term is summed here and the
            # 6*N matmul term applied explicitly. Mirrors
            # models/deepseek_v3/model.py:get_deepseek_v3_nparams_and_flops,
            # minus its MTP lm_head term -- this decoder has no mtp_layers.
            #
            # Per-layer rather than layers[0] x depth: the old helper took a
            # single head count for the whole model, which is only correct
            # when every layer's attention is identical. Summing per layer
            # costs nothing and does not silently mismeasure a mixed stack.
            nparams, active_nparams = get_nparams_and_active_nparams(model)

            attention_op_flops = 0
            for layer in self.layers:
                attention = layer.attention
                if isinstance(attention, Attention.Config):
                    qk_head_dim = (
                        attention.qk_nope_head_dim + attention.qk_rope_head_dim
                    )
                    v_head_dim = attention.v_head_dim
                elif isinstance(attention, GQAttention.Config):
                    qk_head_dim = attention.head_dim or (
                        attention.dim // attention.n_heads
                    )
                    v_head_dim = qk_head_dim
                else:
                    raise TypeError(
                        "MoE FLOP accounting requires MLA Attention.Config or "
                        f"GQAttention.Config; got {type(attention).__name__}"
                    )
                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=qk_head_dim,
                    v_head_dim=v_head_dim,
                    seq_len=seq_len,
                )

            return nparams, 6 * active_nparams + attention_op_flops


# Imported after moeModel is defined because the adapter annotates its config.
from .state_dict_adapter import moeStateDictAdapter  # noqa: E402

moeModel.state_dict_adapter_cls = moeStateDictAdapter
