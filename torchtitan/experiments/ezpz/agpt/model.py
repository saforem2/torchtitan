# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""agpt model class.

agpt reuses Llama3's architecture verbatim but needs a sharding hook that
knows about the optional QK-Norm sub-module. We subclass Llama3Model to
override the sharding-config setter; everything else (forward, init,
weight-tying, etc.) is inherited.
"""

from dataclasses import dataclass

from torchtitan.experiments.torchft.diloco import fragment_llm
from torchtitan.models.llama3.model import Llama3Model

from .state_dict_adapter import AgptStateDictAdapter


class AgptModel(Llama3Model):
    """Llama3 with agpt's QK-Norm-aware sharding setter."""

    state_dict_adapter_cls = AgptStateDictAdapter
    _fragment = staticmethod(fragment_llm)

    def parallelize(self, **kwargs):
        from .parallelize import parallelize_llama

        parallelism_context = kwargs["parallelism_context"]
        with parallelism_context.activate_spmd():
            return parallelize_llama(self, **kwargs)

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        def set_sharding_(self, parallelism) -> None:
            from torchtitan.experiments.ezpz.agpt.sharding import (
                set_agpt_sharding_config,
            )

            set_agpt_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
            )
