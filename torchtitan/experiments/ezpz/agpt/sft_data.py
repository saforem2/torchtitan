# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Streaming Grain recipe for AuroraGPT instruction tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    DatasetMixConfig,
    SampleProcessor,
    SingleDatasetConfig,
    TextSequence,
    WeightedDataset,
)
from torchtitan.components.data.sources import HuggingFaceStreamingSource
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import IGNORE_INDEX


_START_OF_TURN = "<start_of_turn>"
_END_OF_TURN = "<end_of_turn>\n"


def _tulu_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    return sample["messages"]


def _openmath_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": sample["problem"]},
        {"role": "assistant", "content": sample["generated_solution"]},
    ]


def _ultrachat_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    return sample["messages"]


def _has_usable_assistant_turn(sample: dict[str, Any], *, source: str) -> bool:
    if source == "openmath":
        return bool(str(sample.get("problem", "")).strip()) and bool(
            str(sample.get("generated_solution", "")).strip()
        )
    messages = sample.get("messages") or []
    return any(
        message.get("role") == "assistant"
        and bool(str(message.get("content", "")).strip())
        for message in messages
    )


class AgptChatProcessor(SampleProcessor):
    """Tokenize AGPT chat rows lazily and supervise assistant content only.

    Serialization is byte-identical to the established TRL Gemma template:
    no BOS/EOS injection, ``<start_of_turn>``/``<end_of_turn>`` boundaries,
    system messages mapped to the user role, and leading newlines stripped at
    every content boundary. Assistant role headers are masked; assistant body,
    end-of-turn, and trailing newline tokens are supervised.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        source: str

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        self._tokenizer = context.tokenizer
        self._max_context_length = context.max_context_length
        try:
            self._messages_fn = {
                "tulu": _tulu_messages,
                "openmath": _openmath_messages,
                "ultrachat": _ultrachat_messages,
            }[config.source]
        except KeyError as exc:
            raise ValueError(f"unknown AGPT SFT source: {config.source!r}") from exc

    def _append_segment(
        self,
        text: str,
        *,
        supervise: bool,
        full_text: str,
        full_tokens: list[int],
        token_mask: list[bool],
    ) -> tuple[str, list[int], list[bool]]:
        next_text = full_text + text
        next_tokens = self._tokenizer.encode(
            next_text, add_bos=False, add_eos=False
        )
        if next_tokens[: len(full_tokens)] != full_tokens:
            raise ValueError(
                "AGPT chat segment changed an earlier token boundary; refusing "
                "to construct an ambiguous assistant-only loss mask"
            )
        token_mask.extend([supervise] * (len(next_tokens) - len(full_tokens)))
        return next_text, next_tokens, token_mask

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TextSequence | None:
        del rng
        messages = self._messages_fn(sample)
        if not messages or messages[-1].get("role") != "assistant":
            return None

        full_text = ""
        full_tokens: list[int] = []
        token_mask: list[bool] = []
        for message in messages:
            role = message.get("role")
            content = str(message.get("content", "")).lstrip("\n")
            if role in {"system", "user"}:
                segment = f"{_START_OF_TURN}user\n{content}{_END_OF_TURN}"
                full_text, full_tokens, token_mask = self._append_segment(
                    segment,
                    supervise=False,
                    full_text=full_text,
                    full_tokens=full_tokens,
                    token_mask=token_mask,
                )
            elif role == "assistant":
                full_text, full_tokens, token_mask = self._append_segment(
                    f"{_START_OF_TURN}model\n",
                    supervise=False,
                    full_text=full_text,
                    full_tokens=full_tokens,
                    token_mask=token_mask,
                )
                full_text, full_tokens, token_mask = self._append_segment(
                    f"{content}{_END_OF_TURN}",
                    supervise=True,
                    full_text=full_text,
                    full_tokens=full_tokens,
                    token_mask=token_mask,
                )
            else:
                raise ValueError(f"unsupported chat role: {role!r}")

        if len(full_tokens) < 2:
            return None
        labels = np.asarray(full_tokens[1:], dtype=np.int64)
        shifted_mask = np.asarray(token_mask[1:], dtype=np.bool_)
        labels[~shifted_mask] = IGNORE_INDEX
        if not shifted_mask.any():
            return None
        return TextSequence(
            input_ids=np.asarray(full_tokens[:-1], dtype=np.int64),
            labels=labels,
        )


def _source(
    *,
    path: str,
    split: str,
    source: str,
    name: str | None = None,
) -> SingleDatasetConfig:
    return SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path=path,
            name=name,
            split=split,
        ),
        pre_filters=(lambda sample, source=source: _has_usable_assistant_turn(sample, source=source),),
        processor=AgptChatProcessor.Config(source=source),
        post_filters=(lambda sequence: sequence is not None,),
    )


def tulu_math_uc_streaming_dataset() -> ConcatThenSplitPackingConfig:
    """65% Tulu-3, 15% OpenMathInstruct-2, 20% UltraChat, packed online."""
    mixture = DatasetMixConfig(
        datasets=(
            WeightedDataset(
                dataset=_source(
                    path="allenai/tulu-3-sft-mixture",
                    split="train",
                    source="tulu",
                ),
                weight=0.65,
            ),
            WeightedDataset(
                dataset=_source(
                    path="nvidia/OpenMathInstruct-2",
                    split="train",
                    source="openmath",
                ),
                weight=0.15,
            ),
            WeightedDataset(
                dataset=_source(
                    path="HuggingFaceH4/ultrachat_200k",
                    split="train_sft",
                    source="ultrachat",
                ),
                weight=0.20,
            ),
        )
    )
    return ConcatThenSplitPackingConfig(dataset=mixture)
