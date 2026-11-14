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

from torchtitan.config import PrecisionConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LoRATransform, transform_model_config_
from torchtitan.experiments.ezpz.agpt import build_model_config
from torchtitan.models.common.decoder import Decoder
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig

AGPT_RL_PRECISION = PrecisionConfig(dtype="float32", param_dtype="float32")
"""Precision of every AuroraGPT RL overlay: fp32 master weights and fp32 compute.

AuroraGPT-2B generates incoherently through vLLM in bf16 (the 256k vocab and wide
FFN turn rounding into greedy-argmax flips), so the generator must run in fp32.
Before #5776 that was ``VLLMGenerator.Config.model_dtype="float32"`` while the
trainer kept ``mixed_precision_param="bfloat16"``. The generator now mirrors the
trainer's ``precision.param_dtype``: the trainer sends its weights in that dtype
(``Trainer._transfer_dtype``) and the generator builds its model in it, and
#5894 stores frozen (LoRA base) weights in it. ``param_dtype="float32"`` is
the only way to keep fp32 generation and fp32 base weights; the trainer forward,
which FSDP previously cast to bf16, now also computes in fp32.
"""


def disable_flex_max_autotune() -> None:
    """Rebuild ``FlexInnerAttention``'s compiled kernel without max-autotune.

    Backward autotuning exceeds XPU register limits (``OUT_OF_RESOURCES``).
    Must run before the model is compiled. Mirrors upstream's compile call
    (``attention.py``: ``fullgraph=True``, ``recompile_limit=16``) so only the
    two tuning options change.
    """
    from torch.nn.attention.flex_attention import flex_attention

    from torchtitan.distributed.local_compile import compile as local_compile
    from torchtitan.models.common.attention import FlexInnerAttention

    FlexInnerAttention.inductor_configs = {
        **FlexInnerAttention.inductor_configs,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
    }
    FlexInnerAttention._compiled_flex_attn = local_compile(
        flex_attention,
        fullgraph=True,
        options=FlexInnerAttention.inductor_configs,
        recompile_limit=16,
    )


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
    parameter and adds adapters on ``wqkv`` and ``wo``. ``disable_moe_load_balancing``
    is off: the dense AuroraGPT model has no router.
    """
    trainer_model, generator_model = build_rl_model_configs(
        build_model_config,
        "2b-rl",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
        disable_moe_load_balancing=False,
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
