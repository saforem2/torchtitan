# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based DTensor sharding for the agpt (Llama3-derived) model.

Mirrors `torchtitan.models.llama3.sharding` but additionally fills in
sharding for the optional QK-Norm RMSNorm sub-module that agpt configs
may include. Upstream's `set_gqa_attention_sharding` does not know about
qk_norm.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock


def set_agpt_sharding_config(
    config: "Llama3Model.Config",
    *,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_config`` on all agpt sub-configs.

    Same plan as `set_llama3_sharding_config` plus QK-Norm sharding when
    present. Specs are populated unconditionally; the runtime mesh
    determines which declarations apply.
    """
    # 57th sync: upstream PR #3694 removed the loss_parallel kwarg.
    # TP-on now always implies LP-on via tp_gather_logits=False; logits
    # come out vocab-sharded and cross_entropy_loss handles the
    # all-reduce via _LossParallelCrossEntropy autograd.
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_agpt_layer_sharding(layer_cfg, enable_sp=enable_sp)


def _set_agpt_layer_sharding(
    layer_cfg: "Llama3TransformerBlock.Config",
    *,
    enable_sp: bool,
) -> None:
    """Set sharding on one agpt transformer layer (with optional QK-Norm)."""
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm

    set_gqa_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)
    # Static LocalMapConfig on the inner-attention config (upstream #2986
    # replaced runtime DTensor detection in `LocalMapInnerAttention` with
    # this config-driven approach). All inner attention types — including
    # SoftcappedFlexAttention — go through this same path.
    set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)

    qk_norm = getattr(layer_cfg.attention, "qk_norm", None)
    if qk_norm is not None:
        # QK-Norm uses LocalShardRMSNorm (agpt/local_rmsnorm.py): a drop-in
        # RMSNorm subclass whose forward runs the per-head norm on the local TP
        # shard and re-anchors its own DTensor output boundary. So this config
        # only needs to declare the replicated weight; the module handles input
        # unwrap / output re-wrap and the gradient placements itself. The prior
        # attempt to fix the TP>1 backward crash by declaring the full S(2)
        # in/out activation boundary here (mirroring qwen3) was a NO-OP: matched
        # src/dst placements insert no redistribute autograd node, so DTensor's
        # native (crashing) RMSNorm backward was unchanged. weight -> spmd.R
        # (== I after layout resolution).
        qk_norm.sharding_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        )

    assert layer_cfg.feed_forward is not None
    # Upstream PR #3501 (SpmdLayout for NamedPlacement) renamed
    # set_dense_ffn_sharding's `attn_x_placement: Placement` arg to
    # `attn_x_layout: SpmdLayout`. Build via the dense_*_placement
    # helpers (same pattern as llama3/sharding.py).
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    set_dense_ffn_sharding(
        layer_cfg.feed_forward,
        attn_x_layout=attn_x_layout,
        enable_sp=enable_sp,
    )
