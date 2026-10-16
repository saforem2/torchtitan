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
    """Llama3 with agpt's QK-Norm-aware sharding and ezpz's parallelize."""

    state_dict_adapter_cls = AgptStateDictAdapter
    _fragment = staticmethod(fragment_llm)

    def parallelize(self, **kwargs):
        """Apply the ordered ezpz parallelization lifecycle.

        Mirrors ``BaseModel.parallelize`` (local compile -> SPMD sharding ->
        activation remat -> FSDP) with ezpz's FSDP grouping, native-DDP skip and
        optional fp32 norm groups in ``parallelize_llama``.
        """
        from .parallelize import parallelize_llama

        parallelism_context = kwargs["parallelism_context"]
        with parallelism_context.activate_spmd():
            return parallelize_llama(self, **kwargs)

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            """Re-apply sequence-parallel sharding for the final parallelism.

            Upstream bakes sharding into the model config at construction
            time (``build_model_config(parallelism=...)``); ezpz flavors are
            prebuilt templates, so the CLI-resolved ``enable_sequence_parallel``
            is applied here, including QK-Norm sharding.
            """
            from torchtitan.experiments.ezpz.agpt.sharding import (
                set_agpt_sharding_config,
            )

            set_agpt_sharding_config(
                self,
                enable_sp=config.parallelism.enable_sequence_parallel,
            )
