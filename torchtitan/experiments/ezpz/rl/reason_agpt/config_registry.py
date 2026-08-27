# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GRPO+LoRA-on-XPU config for AuroraGPT-2B GSM8K chain-of-thought (Stage 2 of the
CoT plan, docs/live/chains/rl/plans/cot.md).

Thin ezpz overlay: imports the UPSTREAM ``torchtitan.experiments.rl`` engine and
backs it with OUR ``ezpz.agpt`` model (``model_registry("2b-rl", converters=...)``).
Nothing in ``experiments/rl/`` or core is modified. Sibling of
``alphabet_sort_agpt/`` -- same structure and validated XPU knobs, but the task is
GSM8K CoT: the model reasons inside ``<think></think>`` and answers inside
``<answer>\\boxed{}</answer>``, scored by a componentized reward.

Invoke via the arbitrary-dotted-path ``--module`` branch of the ConfigManager:

    python -m torchtitan.experiments.ezpz.rl.train_upstream \\
        --module torchtitan.experiments.ezpz.rl.reason_agpt \\
        --config rl_grpo_lora_agpt_2b_gsm8k_easy \\
        --hf_assets_path <staged Stage-1 cold-start ckpt dir w/ gemma chat_template>

Validated knobs baked in (from the alphabet_sort_agpt 3-way study + the CoT plan's
"copy the XPU knobs verbatim" list):
  - fp32 generation (``generator.model_dtype="float32"``): THE fix -- agpt-2b
    through vLLM in bf16 emits gibberish (large vocab/ffn -> rounding flips greedy
    argmax).
  - LoRA (``target_modules=["wqkv","wo"]``, ``alpha=2*rank``) + ``LMHeadCastConverter``
    fp32 head (RL logprob/KL math needs fp32 logits).
  - the ``2b-rl`` fused-QKV flavor (vocab 256000).
  - FlexAttention ``max_autotune=False`` (XPU OUT_OF_RESOURCES on the backward).
  - dense componentized reward (0.05 format + 0.05 extractable + 0.20 dense
    closeness + 0.70 exact-match) -- the ceiling-attack lesson applied to GSM8K.
  - difficulty curriculum via the dataset's ``max_steps`` (easy subset -> more
    MIXED groups -> denser GRPO gradient).
  - ``group_size=4`` (num_generations 4; the CoT plan's Path B knob) with GRPO
    std-normalized advantage.

EOS / stop handling: the controller fills ``sampling.stop_token_ids`` from
``renderer.get_stop_token_ids()`` (controller.py:370), which for the gemma/"auto"
renderer resolves the AuroraGPT-2B stop set (eos ``</s>``=1 and
``<end_of_turn>``=107 -- the gemma turn boundary). Generation therefore stops at
the end-of-turn token AND the eos token, so a ``<think>``/``<answer>`` completion
terminates cleanly. Nothing to override here; ``enable_thinking`` is left off
(the gemma/"auto" renderer has no reasoning channel -- reasoning is prompt-driven
and regex-scored, per the CoT plan).

Also pass ``--async-loop.training-sample-builder.no-drop-zero-std-reward-groups``
on the CLI so all-zero-reward cold-start groups still assemble a batch.
"""

from __future__ import annotations

import torch

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.lora import LoRAConverter
# 79th sync: upstream #4172 deleted components/lr_scheduler.py (it had become
# a re-export shim when the optimizer components were grouped into a package
# by #4140). LRSchedulersContainer now lives in components.optimizer.
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.ezpz.agpt import model_registry as agpt_model_registry
from torchtitan.experiments.ezpz.rl.reason_agpt.data import GSM8KReasonDataset
from torchtitan.experiments.ezpz.rl.reason_agpt.env import GSM8KReasonEnv
from torchtitan.experiments.ezpz.rl.reason_agpt.reward import (
    AnswerCloseReward,
    AnswerCorrectReward,
    AnswerExtractableReward,
    ThinkFormatReward,
)
from torchtitan.experiments.ezpz.rl.reason_agpt.rollouter import GSM8KReasonRollouter
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.models.cast_linear import LMHeadCastConverter
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.rubrics import Rubric


def _agpt_rl_model_spec(*, lora_rank: int = 8, lora_alpha: float = 16.0):
    """``ezpz.agpt.model_registry("2b-rl")`` for RL: LoRA (wqkv/wo) + fp32 lm_head.

    Converter order: LoRA first (wraps the base linears in frozen+adapter form),
    then the lm_head fp32 cast (RL logprob/KL math needs fp32 logits).
    """
    return agpt_model_registry(
        "2b-rl",
        converters=[
            LoRAConverter.Config(
                rank=lora_rank, alpha=lora_alpha, target_modules=["wqkv", "wo"]
            ),
            LMHeadCastConverter.Config(),
        ],
    )


def _gsm8k_rollouter(
    *, max_steps: int, num_samples: int = 0
) -> GSM8KReasonRollouter.Config:
    """Upstream-style rollouter with: our one-shot CoT env, the four-component
    dense reward, GRPO std-normalized advantage, and the given difficulty band."""
    return GSM8KReasonRollouter.Config(
        train_dataset=GSM8KReasonDataset.Config(
            seed=42, split="train", max_steps=max_steps, num_samples=num_samples
        ),
        validation_dataset=GSM8KReasonDataset.Config(
            seed=99, split="test", max_steps=0, shuffle=False
        ),
        rubric=Rubric.Config(
            reward_fns=[
                ThinkFormatReward.Config(weight=0.20),
                AnswerExtractableReward.Config(weight=0.05),
                AnswerCloseReward.Config(weight=0.20),
                AnswerCorrectReward.Config(weight=0.55),
            ],
            truncation_reward=0.0,
        ),
        message_env=GSM8KReasonEnv.Config(),
        token_env=TokenEnv.Config(max_rollout_tokens=2048, max_num_turns=1),
        # Standard GRPO (std-normalized advantage) -- the CoT plan's Path B knob;
        # the dense closeness reward supplies the within-group std to normalize by.
        advantage=AdvantageEstimator.Config(should_std_normalize=True),
    )


def _agpt_grpo_config(
    *,
    num_training_steps: int,
    num_groups_per_train_step: int,
    group_size: int,
    max_steps: int,
    lr: float,
    lora_rank: int = 8,
    clip_eps: float = 0.2,
    ckpt_interval: int = 20,
    max_tokens: int = 700,
    num_samples: int = 0,
) -> Controller.Config:
    # Disable FlexAttention max_autotune on XPU (backward autotuning exceeds XPU
    # register limits -> OUT_OF_RESOURCES). Matches the working alphabet_sort_agpt
    # run. Must run before the model is compiled; recompile the cached flex kernel.
    from torch.nn.attention.flex_attention import flex_attention
    from torchtitan.models.common.attention import FlexAttention

    FlexAttention.inductor_configs = {
        **FlexAttention.inductor_configs,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
    }
    FlexAttention._compiled_flex_attn = torch.compile(
        flex_attention, options=FlexAttention.inductor_configs
    )
    return Controller.Config(
        model_spec=_agpt_rl_model_spec(
            lora_rank=lora_rank, lora_alpha=2.0 * lora_rank
        ),
        # Overridden on the CLI with --hf_assets_path=<staged Stage-1 ckpt dir>.
        # Default points at the Stage-1 cold-start CoT-SFT checkpoint (cot.md).
        hf_assets_path="outputs/sft/agpt2b-gsm8k-r1cot-8n/checkpoint-16-hf",
        async_loop=AsyncLoopConfig(
            num_training_steps=num_training_steps,
            num_prompts_per_train_step=num_groups_per_train_step,
            num_samples_per_prompt=group_size,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=_gsm8k_rollouter(max_steps=max_steps, num_samples=num_samples),
        # name="auto": resolve gemma/llama tokenizer from hf_assets_path. The staged
        # ckpt dir must carry a chat_template. enable_thinking=False: the gemma/auto
        # renderer has no reasoning channel; reasoning is prompt-driven + regex-scored.
        renderer=RendererConfig(name="auto", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=False),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=lr),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear"
            ),
            training=TrainingConfig(dtype="float32"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1, tensor_parallel_degree=1
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=ckpt_interval,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(clip_eps=clip_eps),
        ),
        generator=VLLMGenerator.Config(
            # fp32 generation -- THE fix for coherent agpt-2b output on XPU.
            model_dtype="float32",
            gpu_memory_limit=0.70,
            cudagraph=VLLMCudagraphConfig(enable=False),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1, tensor_parallel_degree=1
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            # max_tokens raised well above the alphabet_sort default so a full CoT
            # (<think> reasoning + <answer>) is not truncated to zero reward.
            sampling=SamplingConfig(
                temperature=0.8, top_p=0.95, max_tokens=max_tokens
            ),
        ),
    )


def rl_grpo_lora_agpt_2b_gsm8k() -> Controller.Config:
    """Default agpt-2b GSM8K CoT GRPO+LoRA smoke (short run, full difficulty).

    Also pass ``--async-loop.training-sample-builder.no-drop-zero-std-reward-groups``
    on the CLI so all-zero-reward cold-start groups still assemble a batch.
    """
    return _agpt_grpo_config(
        num_training_steps=3,
        num_groups_per_train_step=4,
        group_size=4,
        max_steps=0,
        lr=2e-6,
    )


def rl_grpo_lora_agpt_2b_gsm8k_long() -> Controller.Config:
    """Longer easy-subset run (400 steps) -- same validated config as the _easy
    smoke that trained clean to 100 steps (reward 0.118 -> 0.217), just 4x the
    steps to see how far CoT accuracy climbs. group_size=4, easy curriculum."""
    return _agpt_grpo_config(
        num_training_steps=400,
        num_groups_per_train_step=8,
        group_size=4,
        max_steps=3,
        lr=2e-5,
    )


def rl_grpo_lora_agpt_2b_gsm8k_easy() -> Controller.Config:
    """Easy-subset variant (<=3 calculator steps) + lr 2e-5 -- the CoT-plan
    recommendation: a difficulty band where more groups start MIXED so GRPO has a
    dense gradient. group_size=4 (num_generations 4)."""
    return _agpt_grpo_config(
        num_training_steps=100,
        num_groups_per_train_step=8,
        group_size=4,
        max_steps=3,
        lr=2e-5,
    )


def rl_grpo_lora_agpt_2b_gsm8k_b2long() -> Controller.Config:
    """Longer B2 run (100 steps) -- the smoke's proven-safe recipe scaled up.

    Same knobs as rl_grpo_lora_agpt_2b_gsm8k_b2smoke (lr=2e-6, easy curriculum,
    group_size=4) which lifted held-out validation reward 0.382 -> 0.425 and
    ThinkFormat 0.950 -> 1.000 over 20 steps with NO envelope decay. Deliberately
    keeps lr=2e-6 (not the hotter _easy/_long default 2e-5) for an UNATTENDED run:
    GRPOLoss has no KL anchor, so the base prior holding format is the only
    restoring force -- a conservative lr keeps it. ckpt every 25 so a walltime
    kill still leaves resumable + evaluatable checkpoints. Guardrail: format must
    stay >= 0.9; re-eval a real merged checkpoint on the shared 200-problem GSM8K
    metric, do not trust the reward curve."""
    return _agpt_grpo_config(
        num_training_steps=100,
        num_groups_per_train_step=8,
        group_size=4,
        max_steps=3,
        lr=2e-6,
        ckpt_interval=25,
    )


def rl_grpo_lora_agpt_2b_gsm8k_b2smoke() -> Controller.Config:
    """B2 pre-flight smoke on the STRONG cold-start base (checkpoint-93-hf,
    format 0.985, acc 0.205). Audit-recommended SAFE settings for a base we do
    NOT want to destabilize: easy curriculum (more MIXED groups -> dense
    gradient), lr=2e-6 (10x below the _easy default 2e-5, which was validated
    only on the old weak base and has no KL anchor to fall back on), 20 training
    steps, ckpt every 10 (so a walltime kill still leaves a resumable ckpt --
    the launcher wraps python in `timeout`, and last-step force-save only fires
    on clean completion). Pass CKPT=<checkpoint-93-hf> on the CLI. Success =
    mean reward rises AND format_hit_rate stays >= 0.9 (the hard tripwire: with
    only clip_eps and no KL/ref anchor, the B2 base prior holding the envelope
    is the sole restoring force)."""
    return _agpt_grpo_config(
        num_training_steps=20,
        num_groups_per_train_step=8,
        group_size=4,
        max_steps=3,
        lr=2e-6,
        ckpt_interval=10,
    )
