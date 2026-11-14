# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter

if TYPE_CHECKING:
    from .model import moeModel


class moeStateDictAdapter(DeepSeekV3StateDictAdapter):  # noqa: N801
    """StateDictAdapter for the ezpz moe model.

    The moe model is DeepSeek-V3's MLA + MoE stack without MTP, so it uses
    DeepSeek-V3's HF mapping verbatim. This used to be a copy of that adapter;
    the copy had drifted (it mapped HF ``lm_head.weight`` to the nonexistent
    ``output.weight`` and still named MoE layers ``.moe.``, which upstream
    renamed to ``.feed_forward.`` in #5345). ``DeepSeekV3StateDictAdapter``
    already guards its MTP handling with ``getattr(model_config, "mtp_layers")``.
    """

    def __init__(
        self,
        model_config: moeModel.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)  # type: ignore[arg-type]
        # ezpz's MLA keeps wkv_b as a direct child of the attention module
        # (moe/model.py); upstream maps kv_b_proj to inner_attention.wkv_b (#4760).
        self.from_hf_map = {
            hf_key: native_key.replace(
                ".attention.inner_attention.wkv_b.", ".attention.wkv_b."
            )
            for hf_key, native_key in self.from_hf_map.items()
        }
