# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz-side settings for the upstream RL actors on XPU.

The upstream generator needs no ezpz subclass on XPU (see ``ezpz_generator``);
the trainer/generator process placement lives in ``rl/train_upstream.py``
(``EzpzMonarchActorSystem``).
"""

from .ezpz_generator import xpu_vllm_engine_config

__all__ = ["xpu_vllm_engine_config"]
