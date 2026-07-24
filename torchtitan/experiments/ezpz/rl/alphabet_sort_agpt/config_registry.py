# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GRPO+LoRA-on-XPU config for AuroraGPT-2B (the SFT checkpoint-900 deliverable).

Thin ezpz overlay: imports the UPSTREAM `torchtitan.experiments.rl` engine and
backs it with OUR `ezpz.agpt` model (`model_registry("2b-rl", converters=...)`).
Nothing in `experiments/rl/` or core is modified. `ezpz.agpt.model_registry`
returns a `FaultTolerantModelSpec`, which the RL engine consumes as a plain
`ModelSpec` (it reads only common fields).

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

agpt "2b-rl" uses FlexAttention (agpt/__init__.py:449); the flex-disable monkeypatch
below caps its autotune. The generator uses vLLM own attention regardless.
"""

from __future__ import annotations

import torch

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lora import LoRAConverter
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.ezpz.agpt import model_registry as agpt_model_registry
from torchtitan.experiments.ezpz.rl.alphabet_sort_agpt.few_shot_env import (
    AgptFewShotAlphabetSortEnv,
)
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
from torchtitan.experiments.rl.examples.alphabet_sort.data import AlphabetSortDataset
from torchtitan.experiments.rl.examples.alphabet_sort.rollouter import (
    AlphabetSortRollouter,
)
from torchtitan.experiments.rl.examples.alphabet_sort.rubric import RewardAlphabetSort
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.models.cast_linear import LMHeadCastConverter
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rubrics import Rubric
from torchtitan.experiments.ezpz.rl.alphabet_sort_agpt.shaped_reward import (
    ShapedRewardAlphabetSort,
)


def _agpt_rl_model_spec(*, lora_rank: int = 8, lora_alpha: float = 16.0):
    """`ezpz.agpt.model_registry("2b-rl")` for RL: LoRA (wqkv/wo) + fp32 lm_head.

    Converter order: LoRA first (wraps the base linears in frozen+adapter form),
    then the lm_head fp32 cast (RL logprob/KL math needs fp32 logits).
    """
    return agpt_model_registry(
        "2b-rl",
        converters=[
            LoRAConverter.Config(rank=lora_rank, alpha=lora_alpha, target_modules=["wqkv", "wo"]),
            LMHeadCastConverter.Config(),
        ],
    )


def _agpt_rollouter(
    *, max_turns: int, max_names_per_turn: int, shaped: bool = False
) -> AlphabetSortRollouter.Config:
    """Upstream AlphabetSortRollouter with: our few-shot env, a linear reward
    (similarity_power=1), and the given task difficulty."""
    return AlphabetSortRollouter.Config(
        train_dataset=AlphabetSortDataset.Config(
            seed=42, max_turns=max_turns, max_names_per_turn=max_names_per_turn
        ),
        validation_dataset=AlphabetSortDataset.Config(
            seed=99, max_turns=max_turns, max_names_per_turn=max_names_per_turn
        ),
        rubric=Rubric.Config(
            reward_fns=[
                ShapedRewardAlphabetSort.Config(weight=1.0)
                if shaped
                else RewardAlphabetSort.Config(weight=1.0, similarity_power=1)
            ]
        ),
        message_env=AgptFewShotAlphabetSortEnv.Config(),
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
    # Disable FlexAttention max_autotune on XPU (backward autotuning exceeds XPU
    # register limits -> OUT_OF_RESOURCES). Matches the working fork run. Must run
    # before the model is compiled; recompile the cached flex kernel.
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
        model_spec=_agpt_rl_model_spec(lora_rank=lora_rank, lora_alpha=2.0 * lora_rank),
        # Overridden on the CLI with --hf_assets_path=<staged ckpt-900 dir>.
        hf_assets_path=(
            "outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf"
        ),
        async_loop=AsyncLoopConfig(
            num_training_steps=num_training_steps,
            num_prompts_per_train_step=num_groups_per_train_step,
            num_samples_per_prompt=8,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=_agpt_rollouter(
            max_turns=max_turns,
            max_names_per_turn=max_names_per_turn,
            shaped=shaped_reward,
        ),
        # name="auto": resolve gemma/llama tokenizer from hf_assets_path. The staged
        # ckpt dir must carry a chat_template (SFT gemma template injected at staging
        # time -- scripts/stage_agpt2b.sh).
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
                interval=20,
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
