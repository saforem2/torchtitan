# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Alphabet-sort GRPO recipes for the Aurora XPU trial.

This module used to be a verbatim copy of Guoqiong's
``rl-xpu-aurora-torch214`` ``torchtitan_recipes/rl/alphabet_sort.py``, pinned to
that branch's RL API (``VLLMGenerator``, ``torchtitan.rl.losses``,
``torchtitan.rl.rubric``, ``LinearLoRAHandler``, ...). Upstream has since rebuilt
the RL stack and ships its own copy of the recipe in-tree, so the copy no longer
imported.

Now it re-exports the upstream recipes (``torchtitan_recipes.rl.alphabet_sort``)
and adds only what the trial added on top: the multi-host Qwen3-0.6B layouts

    rl_grpo_qwen3_0_6b_{full,lora}_<hosts>n_ppn<gpus>[_rep<R>|_gtp<T>]_<steps>steps

built on the current API. Model configs bake in their parallelism (each role's
sharding config is derived from it), so a layout rebuilds both model configs
rather than editing ``parallelism`` on a finished config.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, cast

from torchtitan_recipes.rl.alphabet_sort import *  # noqa: F401,F403
from torchtitan_recipes.rl.alphabet_sort import (
    _qwen3_rl_model_configs,
    rl_grpo_qwen3_0_6b_flex,
)

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LoRATransform, transform_model_config_
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig

# Qwen3-0.6B's stacked FFN weight has dim 1 = 3072; FSDP must divide it.
_QWEN3_0_6B_FFN_DIM1 = 3072
_SEQ_LEN = 2048  # rl_grpo_qwen3_0_6b_flex's sequence length


def _qwen3_0_6b_host_layout(
    num_hosts: int, gpus_per_host: int
) -> tuple[int, int, int]:
    """``(dp_replicate, dp_shard, num_generator_gpus)`` for ``num_hosts`` hosts.

    One host splits its GPUs between trainer and generators; more hosts train on
    half of them with one dp_replicate group per trainer host, so the per-layer
    all-gather stays on a host. When FSDP cannot shard the FFN weight that way,
    the smallest dp_replicate that can is used.
    """
    if num_hosts == 1:
        trainer = gpus_per_host // 2
        generators = gpus_per_host - gpus_per_host // 2
        replicas = 1
    else:
        trainer = num_hosts // 2 * gpus_per_host
        generators = (num_hosts - num_hosts // 2) * gpus_per_host
        replicas = num_hosts // 2
    for rep in (replicas, *range(1, trainer + 1)):
        shard = trainer // rep
        if (
            trainer % rep == 0
            and shard <= gpus_per_host
            and (shard <= 2 or _QWEN3_0_6B_FFN_DIM1 % shard == 0)
        ):
            return rep, shard, generators
    raise ValueError(f"no trainer mesh for {num_hosts} hosts x {gpus_per_host} GPUs")


def _rl_grpo_qwen3_0_6b_hosts(
    arm: str,
    num_hosts: int,
    gpus_per_host: int,
    steps: int,
    *,
    dp_replicate: int | None = None,
    generator_tp: int = 1,
) -> Controller.Config:
    """``rl_grpo_qwen3_0_6b_flex`` laid out on ``num_hosts`` x ``gpus_per_host``.

    Trainer TP=1; loads the HF weights but saves no checkpoint; no validation.
    ``arm="lora"`` trains rank-32 adapters on ``wqkv`` and ``wo`` (generators load
    the base checkpoint and weight syncs carry only the adapters, upstream #5844).
    ``dp_replicate`` overrides the per-host replicas, so one model copy is sharded
    across hosts. ``generator_tp`` spreads each vLLM generator over that many GPUs.
    """
    if arm not in ("full", "lora"):
        raise ValueError(f"arm must be 'full' or 'lora', got {arm!r}")
    rep, shard, generator_gpus = _qwen3_0_6b_host_layout(num_hosts, gpus_per_host)
    if dp_replicate is not None:
        if rep * shard % dp_replicate != 0:
            raise ValueError(
                f"dp_replicate={dp_replicate} does not divide {rep * shard} trainer GPUs"
            )
        rep, shard = dp_replicate, rep * shard // dp_replicate
    if generator_gpus % generator_tp != 0:
        raise ValueError(
            f"generator_tp={generator_tp} does not divide {generator_gpus} generator GPUs"
        )

    trainer_parallelism = ParallelismConfig(
        data_parallel_replicate_degree=rep,
        data_parallel_shard_degree=shard,
        tensor_parallel_degree=1,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=generator_tp,
    )
    trainer_model, generator_model = _qwen3_rl_model_configs(
        "0.6B",
        seq_len=_SEQ_LEN,
        attn_backend="flex",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    if arm == "lora":
        lora = LoRATransform(rank=32, alpha=64.0, target_modules=["wqkv", "wo"])
        trainer_model = cast(Any, transform_model_config_(trainer_model, [lora]))
        generator_model = cast(Any, transform_model_config_(generator_model, [lora]))

    config = rl_grpo_qwen3_0_6b_flex()
    config.num_generators = generator_gpus // generator_tp
    config.async_loop.num_training_steps = steps
    config.async_loop.validation = dataclasses.replace(
        config.async_loop.validation, steps=0, validate_at_step_0=False
    )
    config.trainer.model = trainer_model
    config.trainer.parallelism = trainer_parallelism
    assert config.trainer.checkpointer is not None
    config.trainer.checkpointer.load_only = True
    assert isinstance(config.trainer.loss, ChunkedLossWrapper.Config)
    config.trainer.loss.loss_fn.global_vocab_size = decoder_vocab_size(trainer_model)
    config.generator.model = generator_model
    config.generator.parallelism = generator_parallelism
    return config


# rl_grpo_qwen3_0_6b_{full,lora}_<hosts>n_ppn<gpus>_<steps>steps, e.g.
# rl_grpo_qwen3_0_6b_lora_8n_ppn4_10steps, for multi-host runs and the scaling
# matrix; on 4+ hosts also ..._rep<R>_<steps>steps (one model copy sharded across
# hosts) and, where the generator GPUs divide, ..._gtp<T>_<steps>steps.
for _arm in ("full", "lora"):
    for _hosts in (1, 2, 4, 8, 16):
        for _gpus in (1, 2, 4, 6, 8, 10, 12):
            for _steps in (2, 10, 200):
                if _hosts == 1 and _gpus < 2:
                    continue
                _stem = f"rl_grpo_qwen3_0_6b_{_arm}_{_hosts}n_ppn{_gpus}"
                globals()[f"{_stem}_{_steps}steps"] = functools.partial(
                    _rl_grpo_qwen3_0_6b_hosts, _arm, _hosts, _gpus, _steps
                )
                for _rep in (1, 2):
                    if _hosts > 2 and _hosts // 2 * _gpus % _rep == 0:
                        globals()[f"{_stem}_rep{_rep}_{_steps}steps"] = (
                            functools.partial(
                                _rl_grpo_qwen3_0_6b_hosts,
                                _arm,
                                _hosts,
                                _gpus,
                                _steps,
                                dp_replicate=_rep,
                            )
                        )
                _generator_gpus = _gpus - _gpus // 2 if _hosts == 1 else _gpus
                for _gtp in (2, 4):
                    if _generator_gpus % _gtp == 0:
                        globals()[f"{_stem}_gtp{_gtp}_{_steps}steps"] = (
                            functools.partial(
                                _rl_grpo_qwen3_0_6b_hosts,
                                _arm,
                                _hosts,
                                _gpus,
                                _steps,
                                generator_tp=_gtp,
                            )
                        )

del _arm, _hosts, _gpus, _steps, _stem, _rep, _gtp, _generator_gpus
