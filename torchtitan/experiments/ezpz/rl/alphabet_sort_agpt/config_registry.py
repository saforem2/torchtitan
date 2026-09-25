# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GRPO+LoRA-on-XPU config for AuroraGPT-2B (the SFT checkpoint-900 deliverable).

Thin ezpz overlay: imports the UPSTREAM `torchtitan.rl` engine and
backs it with OUR `ezpz.agpt` model (`model_registry("2b-rl", converters=...)`).
Nothing in `experiments/rl/` or core is modified. `ezpz.agpt.model_registry`
returns the model config consumed directly by the RL controller.

Invoke via the arbitrary-dotted-path `--module` branch of the ConfigManager:

    python -m torchtitan.experiments.ezpz.rl.train_upstream \\
        --module torchtitan.experiments.ezpz.rl.alphabet_sort_agpt \\
        --config rl_grpo_lora_agpt_2b_easy \\
        --hf_assets_path <staged ckpt-900 dir with gemma chat_template>

Winning settings from the 3-way training study (docs/production/rl/
grpo-lora-agpt2b-repro.md) are baked in:
  - fp32 generation (`generator.model_dtype="float32"`): THE fix -- bf16 corrupts
    agpt-2b generation through vLLM (large vocab/ffn -> rounding flips greedy argmax).
  - linear reward (`RewardAlphabetSort.Config(similarity_power=1)`): the stock `**4`
    curve starves GRPO of gradient; linear gives partial-credit variance that trains.
  - one-shot format example (overlay `few_shot_env.py`) so the model emits the
    `<alphabetical_sorted>` block often enough for dense signal.
  - `rl_grpo_lora_agpt_2b_easy()` uses the easy task (1 turn, <=3 names) + lr 2e-5 --
    the config that produced the rising reward curve (0.167 -> 0.26).

agpt "2b-rl" uses FlexInnerAttention (agpt/__init__.py:449); the flex-disable monkeypatch
below caps its autotune. The generator uses vLLM own attention regardless.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from renderers import DefaultRendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper

# 79th sync: upstream #4172 deleted components/lr_scheduler.py (it had become
# a re-export shim when the optimizer components were grouped into a package
# by #4140). LRSchedulersContainer now lives in components.optimizer.
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.renderer import ExtraStopTokensRendererConfig, from_renderers
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.config.transform import (
    LinearLoRAHandler,
    LoRATransform,
    transform_model_config_,
)
from torchtitan.config.transform.cast_linear import LMHeadCastConverter
from torchtitan.experiments.ezpz.agpt import model_registry as agpt_model_registry
from torchtitan.experiments.ezpz.rl.alphabet_sort_agpt.few_shot_env import (
    AgptFewShotAlphabetSortEnv,
)
from torchtitan.experiments.ezpz.rl.alphabet_sort_agpt.shaped_reward import (
    ShapedRewardAlphabetSort,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.examples.alphabet_sort.data import AlphabetSortDataset
from torchtitan.rl.examples.alphabet_sort.env import AlphabetSortEnv
from torchtitan.rl.examples.alphabet_sort.rubric import RewardAlphabetSort
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import GRPOLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rubric import Rubric
from torchtitan.rl.trainer import Trainer


def _agpt_rl_model_config(*, lora_rank: int = 8, lora_alpha: float = 16.0):
    """`ezpz.agpt.model_registry("2b-rl")` for RL: LoRA (wqkv/wo) + fp32 lm_head.

    Converter order: LoRA first (wraps the base linears in frozen+adapter form),
    then the lm_head fp32 cast (RL logprob/KL math needs fp32 logits).
    """
    model_config = agpt_model_registry(
        "2b-rl",
        converters=[LMHeadCastConverter.Config()],
    )
    model_config = cast(
        Any,
        transform_model_config_(
            model_config,
            [
                LoRATransform(
                    handlers=(LinearLoRAHandler(),),
                    rank=lora_rank,
                    alpha=lora_alpha,
                    target_modules=["wqkv", "wo"],
                )
            ],
        ),
    )
    return model_config


def _agpt_rollouter(
    *,
    max_turns: int,
    max_names_per_turn: int,
    shaped: bool = False,
    few_shot: bool = True,
) -> Rollouter.Config:
    """Configure alphabet-sort rollout prompts and reward shaping."""
    worker = RolloutWorker.Config(
        rubric=Rubric.Config(
            reward_fns=[
                ShapedRewardAlphabetSort.Config(weight=1.0)
                if shaped
                else RewardAlphabetSort.Config(weight=1.0, similarity_power=1)
            ]
        ),
        message_env=(
            AgptFewShotAlphabetSortEnv.Config()
            if few_shot
            else AlphabetSortEnv.Config()
        ),
    )
    return Rollouter.Config(
        train_dataset=AlphabetSortDataset.Config(
            seed=42, max_turns=max_turns, max_names_per_turn=max_names_per_turn
        ),
        validation_dataset=AlphabetSortDataset.Config(
            seed=99, max_turns=max_turns, max_names_per_turn=max_names_per_turn
        ),
        worker=worker,
    )


def _agpt_grpo_config(
    *,
    num_training_steps: int,
    num_groups_per_train_step: int,
    max_turns: int,
    max_names_per_turn: int,
    lr: float,
    lora_rank: int = 8,
    clip_eps: float = 0.2,
    shaped_reward: bool = False,
) -> Controller.Config:
    # Disable FlexInnerAttention max_autotune on XPU (backward autotuning exceeds XPU
    # register limits -> OUT_OF_RESOURCES). Matches the working fork run. Must run
    # before the model is compiled; recompile the cached flex kernel.
    from torch.nn.attention.flex_attention import flex_attention

    from torchtitan.models.common.attention import FlexInnerAttention

    FlexInnerAttention.inductor_configs = {
        **FlexInnerAttention.inductor_configs,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
    }
    FlexInnerAttention._compiled_flex_attn = torch.compile(
        flex_attention, options=FlexInnerAttention.inductor_configs
    )
    model_config = _agpt_rl_model_config(
        lora_rank=lora_rank, lora_alpha=2.0 * lora_rank
    )
    return Controller.Config(
        model=model_config,
        # Overridden on the CLI with --hf_assets_path=<staged ckpt-900 dir>.
        hf_assets_path=(
            "outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf"
        ),
        async_loop=AsyncLoopConfig(
            num_training_steps=num_training_steps,
            num_prompts_per_train_step=num_groups_per_train_step,
            num_samples_per_prompt=8,
            validation=ValidationConfig(num_samples=20),
        ),
        compile=CompileConfig(backend="aot_eager"),
        rollouter=_agpt_rollouter(
            max_turns=max_turns,
            max_names_per_turn=max_names_per_turn,
            shaped=shaped_reward,
        ),
        # AuroraGPT uses the original Gemma chat template with
        # <start_of_turn>/<end_of_turn>. Gemma4RendererConfig expects the newer
        # <|turn> vocabulary, so use the checkpoint's own Jinja template.
        renderer=ExtraStopTokensRendererConfig(
            renderer=from_renderers(DefaultRendererConfig()),
            extra_stop_token_ids=(1, 107),
        ),
        metrics=MetricsProcessor.Config(enable_wandb=False),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=lr),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear"
            ),
            training=TrainingConfig(
                dtype="float32",
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * 2048,
                max_context_length=2048,
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1, tensor_parallel_degree=1
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=20,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    clip_eps=clip_eps,
                    global_vocab_size=decoder_vocab_size(model_config),
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            # fp32 generation -- THE fix for coherent agpt-2b output on XPU.
            model_dtype="float32",
            gpu_memory_limit=0.70,
            cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1, tensor_parallel_degree=1
            ),
            checkpointer=None,
            sampling=SamplingConfig(temperature=0.8, top_p=0.95, max_tokens=700),
        ),
    )


def rl_grpo_lora_agpt_2b() -> Controller.Config:
    """Default agpt-2b GRPO+LoRA smoke (stock task difficulty, short run).

    Also pass ``--async-loop.training-sample-builder.no-drop-zero-std-reward-groups``
    on the CLI so all-zero-reward cold-start groups still assemble a batch.
    """
    return _agpt_grpo_config(
        num_training_steps=3,
        num_groups_per_train_step=4,
        max_turns=3,
        max_names_per_turn=5,
        lr=2e-6,
    )


def rl_grpo_lora_agpt_2b_easy() -> Controller.Config:
    """Easy-task variant (1 turn, <=3 names) + lr 2e-5 -- the config that produced
    a rising reward curve (0.167 -> 0.26) in the 3-way study."""
    return _agpt_grpo_config(
        num_training_steps=100,
        num_groups_per_train_step=8,
        max_turns=1,
        max_names_per_turn=3,
        lr=2e-5,
    )


def rl_sync_agpt_2b_clean() -> Controller.Config:
    """One-step clean-prompt config for TorchStore synchronization validation."""
    config = _agpt_grpo_config(
        num_training_steps=1,
        num_groups_per_train_step=2,
        max_turns=1,
        max_names_per_turn=3,
        lr=2e-6,
    )
    config.rollouter = _agpt_rollouter(
        max_turns=1,
        max_names_per_turn=3,
        few_shot=False,
    )
    config.async_loop.validation = ValidationConfig(num_samples=8)
    return config


def rl_grpo_agpt_2b_bounded_v3() -> Controller.Config:
    """Three-step GRPO canary for the clean/adversarial-gated robust-v3 policy.

    Keep the experiment deliberately small while sampling a harder one-turn
    distribution (up to five names) than the direct release gate. The launcher
    supplies the exact robust-v3 HF artifact and retains all raw rollouts.
    """
    config = _agpt_grpo_config(
        num_training_steps=3,
        num_groups_per_train_step=4,
        max_turns=1,
        max_names_per_turn=5,
        lr=2e-6,
    )
    config.async_loop.validation = ValidationConfig(num_samples=8)
    assert config.trainer.checkpointer is not None
    config.trainer.checkpointer.interval = 1
    config.generator.sampling.max_tokens = 128
    return config


# --- "beat v5" sweep (2026-07-19): all on the easy task (the winner), each
# --- combining/extending the levers v5/v6 left on the table. ---


def rl_grpo_lora_agpt_2b_w1() -> Controller.Config:
    """w1 = v5 easy task + v6's higher LR (5e-5). The untested v5xv6 combo:
    easy-task dense signal + stronger updates. Most likely to beat v5."""
    return _agpt_grpo_config(
        num_training_steps=100,
        num_groups_per_train_step=8,
        max_turns=1,
        max_names_per_turn=3,
        lr=5e-5,
    )


def rl_grpo_lora_agpt_2b_w2() -> Controller.Config:
    """w2 = easy task + bigger LoRA (rank 32). More adapter capacity to learn the
    sort itself, not just the format. lr 2e-5 (v5's stable LR)."""
    return _agpt_grpo_config(
        num_training_steps=100,
        num_groups_per_train_step=8,
        max_turns=1,
        max_names_per_turn=3,
        lr=2e-5,
        lora_rank=32,
    )


def rl_grpo_lora_agpt_2b_w3() -> Controller.Config:
    """w3 = easy task + lr 5e-5 + more groups/step (16) for a larger, less-noisy
    effective batch (v6 at 5e-5 was noisy; more groups should smooth it)."""
    return _agpt_grpo_config(
        num_training_steps=100,
        num_groups_per_train_step=16,
        max_turns=1,
        max_names_per_turn=3,
        lr=5e-5,
    )


def rl_grpo_lora_agpt_2b_shaped() -> Controller.Config:
    """Ceiling-attack: componentized SHAPED reward (format+completeness+order)
    replacing the saturating char-ratio, + the sweep's best levers (LoRA rank 32,
    lr 5e-5). Hypothesis: the ~0.25 mean-reward ceiling is the reward SHAPE, not
    capacity; an order-specific additive reward gives GRPO a gradient toward fully
    correct sorts. Compare against v5/w2 (same easy task)."""
    return _agpt_grpo_config(
        num_training_steps=100,
        num_groups_per_train_step=8,
        max_turns=1,
        max_names_per_turn=3,
        lr=5e-5,
        lora_rank=32,
        shaped_reward=True,
    )
