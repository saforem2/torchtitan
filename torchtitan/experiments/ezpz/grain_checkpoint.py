# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint compatibility for Grain-backed Hugging Face streams."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict_saver import _stateful_to_state_dict

from torchtitan.components.checkpointer.base import MODEL
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager


logger = logging.getLogger(__name__)

_OPTIONAL_HF_PREVIOUS_STATE_SUFFIX = ".previous_state"


def _optional_hf_previous_state_missing_keys(
    state_dict: dict[str, Any], checkpoint_keys: set[str]
) -> set[str]:
    """Validate and return nullable HF iterator keys absent from a checkpoint.

    Hugging Face ``RebatchedArrowExamplesIterable`` initializes
    ``previous_state`` to ``None``. DCP omits ``None`` leaves, but a fresh Grain
    graph still advertises the key while loading, causing strict DCP planning to
    fail. We allow partial loading only after proving that *every* missing live
    key is one of those nullable dataloader leaves.
    """
    materialized = _stateful_to_state_dict(state_dict)
    flattened, _ = flatten_state_dict(materialized)
    missing = set(flattened) - checkpoint_keys
    invalid = {
        key
        for key in missing
        if not (
            key.startswith("dataloader.")
            and key.endswith(_OPTIONAL_HF_PREVIOUS_STATE_SUFFIX)
            and flattened[key] is None
        )
    }
    if invalid:
        sample = ", ".join(sorted(invalid)[:5])
        raise RuntimeError(
            "checkpoint is missing required live state keys; refusing partial "
            f"load ({len(invalid)} missing, sample: {sample})"
        )
    return missing


class GrainStreamingCheckpointManager(TorchFTCheckpointManager):
    """DCP manager with narrow compatibility for omitted nullable HF state."""

    @dataclass(kw_only=True, slots=True)
    class Config(TorchFTCheckpointManager.Config):
        pass

    def _load_checkpoint(
        self,
        states: dict[str, Any],
        checkpoint_id: str,
        *,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        if from_hf:
            super()._load_checkpoint(
                states,
                checkpoint_id,
                from_hf=from_hf,
                from_quantized=from_quantized,
            )
            return

        state_dict = self._flattened_model_states_sd(states)
        reader = FileSystemReader(checkpoint_id)
        checkpoint_keys = set(reader.read_metadata().state_dict_metadata)
        optional_missing = _optional_hf_previous_state_missing_keys(
            state_dict, checkpoint_keys
        )
        if optional_missing:
            logger.warning(
                "Loading checkpoint with %d omitted nullable Hugging Face "
                "iterator previous_state leaves; all other live state keys "
                "were verified present.",
                len(optional_missing),
            )
        dcp.load(
            state_dict,
            storage_reader=reader,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )

        # Match the base DCP manager's flattened-model restore behavior.
        if MODEL in states:
            states[MODEL].load_state_dict(state_dict)

        if self.enable_ft_dataloader_checkpoints:
            load_step = self._parse_step(checkpoint_id.rsplit("/", 1)[-1])
            if load_step is not None:
                self._ft_load(load_step)
