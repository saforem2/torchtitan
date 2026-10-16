# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based SPMD sharding for the ezpz/moe (DeepSeek-V3-derived) model.

Upstream builds sharding at config-construction time
(``torchtitan.models.deepseek_v3.flavors`` / ``config_utils.make_*``). ezpz
flavors are prebuilt templates and the CLI may change
``parallelism.enable_sequence_parallel`` / ``expert_parallel_degree``
afterwards, so :func:`set_moe_sharding_config` re-applies the same
declarations to an existing ``moeModel.Config`` (called from
``moeModel.Config.__post_init__`` with the template defaults and again from
``update_from_config`` with the final parallelism).

Layers may use MLA attention (``moe.model.Attention``) or the fused-QKV
``GQAttention`` shared with dense AGPT. Dense feed-forwards (and the MoE shared
experts) keep the historical interleaved ``[2F, D]`` ``w13`` storage; see
``moe.__init__._LegacyInterleavedColumnParallelLinear``.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.experiments.ezpz.agpt.sharding import set_gqa_attention_config_sharding
from torchtitan.experiments.ezpz.moe.model import Attention
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    feed_forward_sharding_config,
    inner_attention_local_spmd_sharding_config,
    lm_head_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    replicated_input_colwise_sharding_config,
    replicated_state_sharding_config,
    residual_activation_placement,
    rope_cache_sharding_config,
    rowwise_sharding_config,
    stacked_colwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.linear import Linear, RowParallelLinear
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe.sharding import (
    moe_sharding_config,
    routed_expert_linear_sharding_config,
    routed_experts_sharding_config,
    router_gate_sharding_config,
    router_sharding_config,
    shared_expert_rowwise_sharding_config,
    shared_experts_output_placement,
    shared_experts_sharding_config,
)
from torchtitan.models.deepseek_v3.sharding import mla_attention_sharding_config

if TYPE_CHECKING:
    from torchtitan.experiments.ezpz.moe.model import moeModel, moeTransformerBlock


def set_moe_sharding_config(
    config: "moeModel.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all moe sub-configs (dense + MoE).

    Declarations are populated unconditionally; the SPMD mesh active at
    ``Module.parallelize()`` time decides which axes apply.
    """
    config.tok_embeddings.sharding_config = token_embedding_sharding_config(
        enable_sp=enable_sp
    )
    config.norm.sharding_config = pre_lm_head_norm_sharding_config(enable_sp=enable_sp)
    config.lm_head.sharding_config = lm_head_sharding_config()
    for layer_cfg in config.layers:
        _set_moe_layer_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)


def _set_moe_layer_sharding(
    layer_cfg: "moeTransformerBlock.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Set sharding on one moe transformer layer."""
    attention = layer_cfg.attention
    if not isinstance(attention, (Attention.Config, GQAttention.Config)):
        raise TypeError(
            "MoE layers require MLA Attention.Config or GQAttention.Config; "
            f"got {type(attention).__name__}"
        )

    norm = norm_sharding_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    x_layout = residual_activation_placement(enable_sp=enable_sp)

    if isinstance(attention, GQAttention.Config):
        # AGPT-MoE uses the same fused GQA block as dense AGPT.
        set_gqa_attention_config_sharding(attention, enable_sp=enable_sp)
    else:
        _set_mla_attention_sharding(attention, enable_sp=enable_sp)

    feed_forward = layer_cfg.feed_forward
    if isinstance(feed_forward, MoE.Config):
        _set_moe_block_sharding(feed_forward, enable_sp=enable_sp, enable_ep=enable_ep)
    else:
        assert isinstance(feed_forward, DenseFeedForward.Config)
        feed_forward.sharding_config = feed_forward_sharding_config(enable_sp=enable_sp)
        _set_dense_ffn_projections(feed_forward, x_layout=x_layout)


def _set_mla_attention_sharding(attention: Attention.Config, *, enable_sp: bool) -> None:
    """MLA: low-rank projections replicated, up-projections column-parallel.

    Mirrors ``deepseek_v3.flavors.make_mla_attention_config``: the MLA module
    gathers its input once (``maybe_gather_tp_input``), so the projections
    declare a replicated input layout; ``wo`` returns to the residual layout.
    """
    x_layout = residual_activation_placement(enable_sp=enable_sp)
    attention.sharding_config = mla_attention_sharding_config(enable_sp=enable_sp)
    attention.inner_attention.sharding_config = (
        inner_attention_local_spmd_sharding_config()
    )
    attention.rope.sharding_config = rope_cache_sharding_config()
    attention.wkv_a.sharding_config = replicated_state_sharding_config("weight")
    attention.kv_norm.sharding_config = replicated_state_sharding_config("weight")
    attention.wkv_b.sharding_config = replicated_input_colwise_sharding_config()
    attention.wo.sharding_config = rowwise_sharding_config(output_layout=x_layout)
    if attention.q_lora_rank == 0:
        assert attention.wq is not None
        attention.wq.sharding_config = replicated_input_colwise_sharding_config()
    else:
        assert attention.wq_a is not None and attention.wq_b is not None
        attention.wq_a.sharding_config = replicated_state_sharding_config("weight")
        attention.q_norm.sharding_config = replicated_state_sharding_config("weight")
        attention.wq_b.sharding_config = replicated_input_colwise_sharding_config()


def _set_dense_ffn_projections(ffn: DenseFeedForward.Config, *, x_layout) -> None:
    """Gate/up (``w13``) column-parallel and down (``w2``) row-parallel."""
    ffn.w13.sharding_config = stacked_colwise_sharding_config(input_layout=x_layout)
    ffn.w2.sharding_config = rowwise_sharding_config(output_layout=x_layout)
    _restore_legacy_w13_placement(ffn)


def _restore_legacy_w13_placement(ffn: DenseFeedForward.Config) -> None:
    """Keep rows sharded on dim 0 for the historical interleaved ``[2F, D]`` w13.

    Upstream's stacked helper shards ``[2, F, D]`` on dimension 1; for the
    historical 2-D parameter the corresponding output-feature axis is
    dimension 0. The activation contract installed above is kept; only the
    parameter placement is restored so every FSDP/TP rank owns rows.
    """
    if not type(ffn.w13).__qualname__.startswith("_LegacyInterleavedColumnParallelLinear."):
        return
    sharding = ffn.w13.sharding_config
    assert sharding is not None
    sharding.state_shardings["weight"] = dense_param_placement(tp=spmd.S(0))
    sharding.state_shardings["bias"] = dense_param_placement(tp=spmd.S(0))


def _set_moe_block_sharding(
    moe: MoE.Config, *, enable_sp: bool, enable_ep: bool
) -> None:
    """Router, routed experts, shared experts and the MoE boundary."""
    x_layout = residual_activation_placement(enable_sp=enable_sp)
    moe.sharding_config = moe_sharding_config(enable_sp=enable_sp)
    moe.router.sharding_config = router_sharding_config()
    moe.router.gate.sharding_config = router_gate_sharding_config()
    routed = moe.routed_experts
    routed.sharding_config = routed_experts_sharding_config()
    routed.w13.sharding_config = routed_expert_linear_sharding_config(enable_ep=enable_ep)
    routed.w2.sharding_config = routed_expert_linear_sharding_config(enable_ep=enable_ep)

    shared = moe.shared_experts
    if shared is None:
        return
    shared.sharding_config = shared_experts_sharding_config(enable_sp=enable_sp)
    shared.w13.sharding_config = stacked_colwise_sharding_config(input_layout=x_layout)
    # With SP the shared down projection reduce-scatters to the SP layout
    # (RowParallelLinear); without SP a plain Linear keeps its TP-Partial
    # output so the MoE reduces routed + shared once.
    w2 = shared.w2
    w2_cls = RowParallelLinear if enable_sp else Linear
    if type(w2) is not w2_cls.Config:
        shared.w2 = w2_cls.Config(
            in_features=w2.in_features,
            out_features=w2.out_features,
            num_linears=w2.num_linears,
            bias=w2.bias,
            param_init=w2.param_init,
            sharding_config=w2.sharding_config,
        )
    shared.w2.sharding_config = shared_expert_rowwise_sharding_config(
        output_layout=shared_experts_output_placement(enable_sp=enable_sp)
    )
    _restore_legacy_w13_placement(shared)
