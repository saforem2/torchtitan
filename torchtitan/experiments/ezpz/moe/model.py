# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import dataclasses
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    ScaledDotProductAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.models.common.token_dispatcher import (
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
)
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module
from torchtitan.experiments.ezpz.logging import warn_once
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


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


class Attention(BaseAttention):
    """
    Multi-head latent attention (MLA) module.

    This is moe-specific and NOT shared with other models.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        dim: int
        wq: Linear.Config | None = None
        wq_a: Linear.Config | None = None
        wq_b: Linear.Config | None = None
        wkv_a: Linear.Config
        wkv_b: Linear.Config
        wo: Linear.Config
        q_lora_rank: int = 0
        kv_lora_rank: int = 512
        q_norm: RMSNorm.Config
        kv_norm: RMSNorm.Config
        qk_nope_head_dim: int = 128
        qk_rope_head_dim: int = 64
        v_head_dim: int = 128
        rope: RoPE.Config
        inner_attention: Module.Config
        mask_type: str = "causal"
        mscale: float = 1.0

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        if self.q_lora_rank == 0:
            assert config.wq is not None, "wq is required when q_lora_rank == 0"
            self.wq = config.wq.build()
        else:
            assert (
                config.wq_a is not None and config.wq_b is not None
            ), "wq_a and wq_b are required when q_lora_rank > 0"
            self.wq_a = config.wq_a.build()
            self.q_norm = config.q_norm.build()
            self.wq_b = config.wq_b.build()

        # TODO(fegin): revisit
        # https://github.com/pytorch/torchtitan/pull/2785#discussion_r3034078575
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.wo = config.wo.build()
        self.softmax_scale = self.qk_head_dim**-0.5

        if config.rope.max_context_length > config.rope.original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.inner_attention = config.inner_attention.build()
        self.rope = config.rope.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        # Flat token layout (upstream #4121): x is [T, D], NOT [B, L, D]. The
        # loader folds [B, L] -> [T] unconditionally, so every reshape below
        # works on a token count rather than a (batch, seq) pair. Ported from
        # torchtitan/models/deepseek_v3/model.py, which is what this MLA forked
        # from -- the axis shifts here are subtle enough that inferring them
        # from a traceback gets two of them wrong.
        num_tokens = x.shape[0]

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_a(x)
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

        # PR #3458 (RoPE refactor): rope module now owns its cache and
        # rotates q+k in one call; freqs_cis no longer threaded through
        # forward.
        # unsqueeze(1), not (2): k_pe is [T, rope_dim] here (rank 2), so the
        # head axis is inserted at 1. It was 2 when this tensor was 3D.
        q_pe, k_pe = self.rope(q_pe, k_pe.unsqueeze(1), positions)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.wkv_b(self.kv_norm(kv))

        with spmd.local():
            kv = kv.view(num_tokens, -1, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
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

        output = self.inner_attention(
            q, k, v, attention_masks=attention_masks, scale=self.softmax_scale
        )

        if pad_v:
            output = output[..., : self.v_head_dim]

        output = output.contiguous()
        output = output.view(num_tokens, -1)
        return self.wo(output)


class moeTransformerBlock(TransformerBlock):  # noqa: N801
    """
    moe Transformer block with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        x = x + self.attention(self.attention_norm(x), attention_masks, positions)
        _maybe_release_device_cache_between_attention_and_moe(x)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x


class moeModel(Decoder):  # noqa: N801
    """
    moe Transformer model with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2048
        vocab_size: int = 102400

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            # Run Decoder.Config's validation + MoE/TP/EP checks first.
            # After PR #3395 (42nd sync) base validation moved from
            # per-model overrides into the shared Decoder.Config helper.
            # After PR #3458 (47th sync) per-layer rope sync is no longer
            # needed: each Attention.Config carries its own RoPE.Config
            # instance instead of inheriting fields from self.rope.
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism
            debug = config.debug

            # for_loop fallback when CUDA SM90+ grouped_mm is unavailable
            # (notably on XPU). Upstream Decoder.Config doesn't know about
            # compute_backend, so this stays.
            for layer_cfg in self.layers:
                if layer_cfg.moe is not None:
                    experts_cfg = layer_cfg.moe.routed_experts.inner_experts
                    if getattr(
                        experts_cfg, "compute_backend", "grouped_mm"
                    ) == "grouped_mm" and not has_cuda_capability(9, 0):
                        # warn_once collapses the per-layer repetition
                        # (this loop fires once per MoE layer → 26 dup
                        # lines on a 26-layer model otherwise).
                        warn_once(
                            logger,
                            "torch._grouped_mm requires SM90+ CUDA; falling "
                            "back to for_loop expert backend.",
                        )
                        experts_cfg.compute_backend = "for_loop"
                    layer_cfg.moe.router._debug_force_load_balance = (
                        debug.moe_force_load_balance
                    )
                    if hasattr(
                        layer_cfg.moe.routed_experts.token_dispatcher, "force_load_balance"
                    ):
                        layer_cfg.moe.routed_experts.token_dispatcher.force_load_balance = (
                            debug.moe_force_load_balance
                        )
                    # Detect deepep/hybridep configs by the dispatcher
                    # Config class, not by a `comm_backend` attribute that
                    # doesn't exist on the dispatcher Config. The old
                    # `getattr(..., "comm_backend", "standard")` lookup
                    # always returned "standard" (the function-arg name
                    # was never stored on the resulting Config), so the
                    # downstream EP=1 guard was dead code.
                    #
                    # We also dropped the `MoE → DeepEPMoE.Config` swap
                    # that used to live here — `DeepEPMoE` no longer
                    # exists upstream (the dispatcher classes now own
                    # the comm-backend-specific logic via their own
                    # `dispatch` / `combine` implementations). Keep
                    # the EP=1 guard so a misconfigured deepep/hybridep
                    # user gets a clear error before model init.
                    if isinstance(
                        layer_cfg.moe.routed_experts.token_dispatcher,
                        (
                            DeepEPTokenDispatcher.Config,
                            HybridEPTokenDispatcher.Config,
                        ),
                    ):
                        dispatcher_name = type(
                            layer_cfg.moe.routed_experts.token_dispatcher
                        ).__qualname__.split(".")[0]
                        if parallelism.expert_parallel_degree == 1:
                            raise ValueError(
                                f"{dispatcher_name} requires expert "
                                "parallelism (expert_parallel_degree > 1)."
                            )

            if parallelism.context_parallel_degree > 1 and not isinstance(
                self.layers[0].attention.inner_attention,
                ScaledDotProductAttention.Config,
            ):
                raise NotImplementedError(
                    "Context Parallel for MoE only supports "
                    "ScaledDotProductAttention. Got "
                    f"{type(self.layers[0].attention.inner_attention).__name__}."
                )

            # Fill ShardingConfig on every sub-module config so
            # Module.parallelize(parallel_dims) can distribute params/activations.
            # Post upstream PR #3386 (37th sync), MoE submodules (router gate,
            # shared experts, routed experts) are populated via upstream's
            # set_moe_sharding_config helper inside our sharding.py.
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
                assert isinstance(attention, Attention.Config)
                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=(
                        attention.qk_nope_head_dim + attention.qk_rope_head_dim
                    ),
                    v_head_dim=attention.v_head_dim,
                    seq_len=seq_len,
                )

            return nparams, 6 * active_nparams + attention_op_flops
