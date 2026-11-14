# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""XPU settings for the upstream RL generator.

This module used to hold ``EzpzVLLMGenerator``, a placeholder subclass of the
old ``torchtitan.rl.generator.generator.VLLMGenerator`` with no overrides (only
TODOs); no config ever used it. Upstream then split that class into
``Generator`` (endpoints, engine loop, weight prefetch) and a
``GenerationEngine`` selected by ``Generator.Config.engine``
(``VLLMEngine.Config`` by default). Every XPU difference the old TODO list named
now has a home that needs no generator subclass:

1. ``has_cuda_capability(9, 0)`` in ``VLLMEngine._build_engine_args`` is False on
   XPU, so the engine already takes the ``block_size=256`` branch
   (``xpu_overrides.patch_has_cuda_capability_for_xpu`` makes it explicit).
2. No CUDA-graph capture on vLLM-XPU: ``xpu_vllm_engine_config()`` below sets
   ``VLLMCudaGraphConfig(mode="NONE")``, which the engine turns into
   ``enforce_eager=True``.
3. Attention backend: the engine asks vLLM for ``FLEX_ATTENTION`` or ``CUSTOM``
   (``vllm_attention_backend``); ``xpu_overrides.patch_vllm_xpu_attention_backend``
   maps those to vLLM-XPU implementations.
4. ``set_batch_invariance`` only runs with ``debug.batch_invariant``, which the
   ezpz overlays leave off.
5. TorchStore transport: ``rl/actors/monarch/system.py``
   ``_torchstore_strategy_from_env()`` (``TORCHTITAN_TORCHSTORE_TRANSPORT``) plus
   ``xpu_overrides.patch_torchstore_network_availability_for_xpu``.
6. vLLM worker: ``rl/generator/vllm/worker.py`` subclasses vLLM's ``XPUWorker`` on
   XPU and keeps its native device setup and model runner.

Generator dtype: ``VLLMGenerator.Config.model_dtype`` is gone; the generator
computes in ``Controller.Config.precision.param_dtype``. See
``rl/model_configs.py`` ``AGPT_RL_PRECISION``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.rl.generator.vllm.engine import VLLMEngine


def xpu_vllm_engine_config(**kwargs: Any) -> VLLMEngine.Config:
    """``VLLMEngine.Config`` for vLLM-XPU: eager execution, no CUDA-graph capture.

    ``kwargs`` go to ``VLLMEngine.Config`` (e.g. ``watermark``,
    ``extra_engine_args``). Imports vLLM, so call it from a config factory, not at
    module import.
    """
    from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
    from torchtitan.rl.generator.vllm.engine import VLLMEngine

    if "cuda_graph" in kwargs:
        raise ValueError("xpu_vllm_engine_config always disables CUDA graphs")
    return VLLMEngine.Config(cuda_graph=VLLMCudaGraphConfig(mode="NONE"), **kwargs)
