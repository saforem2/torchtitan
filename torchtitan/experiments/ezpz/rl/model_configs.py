# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AuroraGPT RL ``(trainer, generator)`` model configs.

Upstream split the single RL ``Controller.Config.model`` into a trainer model
(built for the trainer's ``ParallelismConfig``) and a generator model (built
for ``InferenceParallelismConfig.to_training()``), both from one
``build_model_config`` call (``torchtitan_recipes.rl.model_configs``). The
AuroraGPT overlays use the ezpz ``agpt`` model, optionally with LoRA on the
fused attention projections.
"""

from typing import Any, cast

from torchtitan_recipes.rl.model_configs import build_rl_model_configs

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LoRATransform, transform_model_config_
from torchtitan.experiments.ezpz.agpt import build_model_config
from torchtitan.models.common.decoder import Decoder
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig


def agpt_rl_model_configs(
    *,
    trainer_parallelism: ParallelismConfig,
    generator_parallelism: InferenceParallelismConfig,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
) -> tuple[Decoder.Config, Decoder.Config]:
    """``ezpz.agpt`` ``"2b-rl"`` for RL: fp32 lm_head logits + LoRA (wqkv/wo).

    ``build_rl_model_configs`` applies the fp32 lm_head conversion (RL
    logprob/KL math needs fp32 logits); LoRA then freezes every other
    parameter and adds adapters on ``wqkv`` and ``wo``. ``freeze_moe_routing``
    is off: the dense AuroraGPT model has no router.
    """
    trainer_model, generator_model = build_rl_model_configs(
        build_model_config,
        "2b-rl",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
        freeze_moe_routing=False,
    )
    lora = LoRATransform(
        rank=lora_rank,
        alpha=lora_alpha,
        target_modules=["wqkv", "wo"],
    )
    return (
        cast(Any, transform_model_config_(trainer_model, [lora])),
        cast(Any, transform_model_config_(generator_model, [lora])),
    )
