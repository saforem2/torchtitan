# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based SPMD sharding for the agpt (Llama3-derived) model.

Upstream builds sharding at config-construction time: ``make_gqa_config`` /
``make_ffn_config`` and ``torchtitan.models.llama3.flavors`` bake an
``enable_sp`` flag into every sub-config. ezpz flavors are prebuilt
templates (``agpt_configs``) and the CLI may change
``parallelism.enable_sequence_parallel`` afterwards, so this module re-applies
the same sharding declarations to an existing ``AgptModel.Config`` once the
final parallelism config is known (``AgptModel.Config.update_from_config``).

It mirrors the sharding that ``torchtitan.models.llama3.flavors`` produces,
with one deliberate difference: the optional QK-Norm sub-module is a
``LocalShardRMSNorm`` that handles its own DTensor boundary, so it only
declares its replicated weight (upstream's ``head_norm_sharding_config``
declares activation layouts the module would then double-handle).
"""

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    colwise_sharding_config,
    dense_param_placement,
    feed_forward_sharding_config,
    gqa_attention_sharding_config,
    inner_attention_local_spmd_sharding_config,
    lm_head_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    residual_activation_placement,
    rope_cache_sharding_config,
    rowwise_sharding_config,
    stacked_colwise_sharding_config,
    token_embedding_sharding_config,
)

if TYPE_CHECKING:
    from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock


def set_agpt_sharding_config(
    config: "Llama3Model.Config",
    *,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_config`` on all agpt sub-configs.

    Specs are populated unconditionally; the SPMD mesh active at
    ``Module.parallelize()`` time decides which declarations apply.
    """
    config.tok_embeddings.sharding_config = token_embedding_sharding_config(
        enable_sp=enable_sp
    )
    config.norm.sharding_config = pre_lm_head_norm_sharding_config(enable_sp=enable_sp)
    config.lm_head.sharding_config = lm_head_sharding_config()
    for layer_cfg in config.layers:
        _set_agpt_layer_sharding(layer_cfg, enable_sp=enable_sp)


def set_gqa_attention_config_sharding(attention, *, enable_sp: bool) -> None:
    """Sharding for a fused-QKV ``GQAttention.Config`` (with optional QK-Norm).

    Shared by agpt and the AGPT-MoE flavors, which both use the same GQA block.
    """
    x_layout = residual_activation_placement(enable_sp=enable_sp)
    attention.sharding_config = gqa_attention_sharding_config(enable_sp=enable_sp)
    attention.qkv_linear.wqkv.sharding_config = colwise_sharding_config(
        input_layout=x_layout
    )
    attention.wo.sharding_config = rowwise_sharding_config(output_layout=x_layout)
    # Static local-SPMD boundary on the inner attention (matches in_dst
    # shardings by positional-arg NAME: q_THK / k_THK / v_THV). Every inner
    # attention type, including SoftcappedFlexAttention, goes through this.
    attention.inner_attention.sharding_config = (
        inner_attention_local_spmd_sharding_config()
    )
    if attention.rope is not None:
        attention.rope.sharding_config = rope_cache_sharding_config()

    qk_norm = getattr(attention, "qk_norm", None)
    if qk_norm is not None:
        # QK-Norm uses LocalShardRMSNorm (agpt/local_rmsnorm.py): a drop-in
        # RMSNorm subclass whose forward runs the per-head norm on the local TP
        # shard and re-anchors its own DTensor output boundary, so this config
        # only declares the replicated weight (spmd.R, == I after layout
        # resolution). Declaring the full S(2) activation boundary here was a
        # no-op for the TP>1 backward crash: matched src/dst placements insert
        # no redistribute autograd node.
        qk_norm.sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        )


def _set_agpt_layer_sharding(
    layer_cfg: "Llama3TransformerBlock.Config",
    *,
    enable_sp: bool,
) -> None:
    """Set sharding on one agpt transformer layer (with optional QK-Norm)."""
    norm = norm_sharding_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    x_layout = residual_activation_placement(enable_sp=enable_sp)

    set_gqa_attention_config_sharding(layer_cfg.attention, enable_sp=enable_sp)

    ffn = layer_cfg.feed_forward
    assert ffn is not None
    ffn.sharding_config = feed_forward_sharding_config(enable_sp=enable_sp)
    ffn.w13.sharding_config = stacked_colwise_sharding_config(input_layout=x_layout)
    ffn.w2.sharding_config = rowwise_sharding_config(output_layout=x_layout)
