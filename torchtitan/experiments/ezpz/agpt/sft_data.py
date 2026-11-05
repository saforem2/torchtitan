# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Streaming Grain recipe for AuroraGPT instruction tuning.

Supports both general SFT (tulu + openmath + ultrachat) and reasoning
CoT SFT (gsm8k-r1cot + OpenR1-Math + OpenThoughts) through the same
Grain streaming pipeline — no TRL, no pre-tokenization.
"""

from __future__ import annotations

import os
import re
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

# ── CoT envelope (byte-identical to rl/datasets_sft.py + eval) ──────────
_OPENR1_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_OPENR1_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_OPENR1_SUFFIX = (
    "\nReason step by step inside <think></think>, then give the final "
    "answer inside <answer>\\boxed{}</answer>."
)
# Verified against HF API: median OpenR1 trace is ~13k chars (n=75 sample),
# only 6.7% fall under 4000. The 1200 inherited from TRL makes _openr1_messages
# return [] for every row, so _cot_ok drops 100% before tokenization. Use the
# same generous ceiling as OpenThoughts (16k) — the B3 verbosity concern was
# about a 2B *generating* run-ons, not about training on frontier traces.
OPENR1_MAX_THINK_CHARS = int(os.environ.get("OPENR1_MAX_THINK_CHARS", "16000"))
DISTILL_MAX_THINK_CHARS = int(os.environ.get("DISTILL_MAX_THINK_CHARS", "16000"))
_OT_THOUGHT_RE = re.compile(
    r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL
)
_GSM8K_CALC_RE = re.compile(r"<<[^>]*>>")


def _tulu_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    return sample["messages"]


def _gsm8k_r1cot_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    answer = sample.get("answer", "")
    if "####" in answer:
        cot, final = answer.split("####", 1)
    else:
        cot, final = answer, ""
    cot = _GSM8K_CALC_RE.sub("", cot).strip()
    final = final.strip().replace(",", "")
    if not cot or not final:
        return []
    completion = f"<think>{cot}</think>\n<answer>\\boxed{{{final}}}</answer>"
    return [
        {"role": "user", "content": sample["question"] + _OPENR1_SUFFIX},
        {"role": "assistant", "content": completion},
    ]


def _openr1_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    gens = sample.get("generations") or []
    correct = sample.get("correctness_math_verify") or []
    complete = sample.get("is_reasoning_complete") or []
    for i, gen in enumerate(gens):
        correct_i = correct[i] if i < len(correct) else False
        complete_i = complete[i] if i < len(complete) else False
        if not correct_i or not complete_i:
            continue
        if not gen:
            continue
        tm = _OPENR1_THINK_RE.search(gen)
        if not tm:
            continue
        trace = tm.group(1).strip()
        if len(trace) > OPENR1_MAX_THINK_CHARS or not trace:
            continue
        boxes = _OPENR1_BOXED_RE.findall(gen)
        if not boxes:
            continue
        answer = boxes[-1].strip()
        if not answer:
            continue
        completion = f"<think>{trace}</think>\n<answer>\\boxed{{{answer}}}</answer>"
        return [
            {"role": "user", "content": sample["problem"] + _OPENR1_SUFFIX},
            {"role": "assistant", "content": completion},
        ]
    return []


def _openthoughts_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    question = ""
    for turn in sample.get("conversations") or []:
        if isinstance(turn, dict) and turn.get("from") in ("user", "human"):
            question = str(turn.get("value", ""))
            break
    if not question:
        return []
    assistant = ""
    for turn in sample.get("conversations") or []:
        if isinstance(turn, dict) and turn.get("from") == "assistant":
            assistant = str(turn.get("value", ""))
            break
    if not assistant:
        return []
    tm = _OT_THOUGHT_RE.search(assistant)
    trace = tm.group(1).strip() if tm else ""
    if not trace or len(trace) > DISTILL_MAX_THINK_CHARS:
        return []
    boxes = _OPENR1_BOXED_RE.findall(assistant)
    if not boxes:
        return []
    answer = boxes[-1].strip()
    if not answer:
        return []
    completion = f"<think>{trace}</think>\n<answer>\\boxed{{{answer}}}</answer>"
    return [
        {"role": "user", "content": question + _OPENR1_SUFFIX},
        {"role": "assistant", "content": completion},
    ]


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
    # CoT datasets arrive as raw HF rows (question/answer, problem/generations,
    # conversations) -- not preformatted messages. Their validity (including
    # verified + boxed checks) is enforced by _cot_ok; this filter just
    # ensures the raw fields are non-empty so we don't waste cycles.
    if source == "gsm8k_r1cot":
        return bool(str(sample.get("question", "")).strip()) and bool(
            str(sample.get("answer", "")).strip()
        )
    if source == "openr1":
        return bool(str(sample.get("problem", "")).strip()) and bool(
            sample.get("generations")
        )
    if source == "openthoughts":
        convs = sample.get("conversations") or []
        return bool(convs)
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
                "gsm8k_r1cot": _gsm8k_r1cot_messages,
                "openr1": _openr1_messages,
                "openthoughts": _openthoughts_messages,
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
        next_tokens = self._tokenizer.encode(next_text, add_bos=False, add_eos=False)
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
        pre_filters=(
            lambda sample, source=source: _has_usable_assistant_turn(
                sample, source=source
            ),
        ),
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


# ── GSM8K decontamination (Grain pre_filter, lazy) ──────────────────────
_GSM8K_DETECTOR = None


def _gsm8k_decontam_ok(sample: dict[str, Any], *, source: str) -> bool:
    global _GSM8K_DETECTOR
    if _GSM8K_DETECTOR is None:
        try:
            from torchtitan.experiments.ezpz.rl.decontam_traces import (
                GSM8KDecontaminator,
                load_gsm8k_test_questions,
            )

            _GSM8K_DETECTOR = GSM8KDecontaminator(load_gsm8k_test_questions())
        except Exception:
            return True
    q = ""
    if source == "openr1":
        q = str(sample.get("problem", ""))
    elif source == "openthoughts":
        for turn in sample.get("conversations") or []:
            if isinstance(turn, dict) and turn.get("from") in ("user", "human"):
                q = str(turn.get("value", ""))
                break
    elif source == "gsm8k_r1cot":
        q = str(sample.get("question", ""))
    if not q:
        return True
    contaminated, _ = _GSM8K_DETECTOR.is_contaminated(q)
    return not contaminated


def _cot_source(
    *,
    path: str,
    split: str,
    source: str,
    name: str | None = None,
) -> SingleDatasetConfig:
    def _cot_ok(sample: dict[str, Any], source: str = source) -> bool:
        if not _gsm8k_decontam_ok(sample, source=source):
            return False
        msgs_fn = {
            "gsm8k_r1cot": _gsm8k_r1cot_messages,
            "openr1": _openr1_messages,
            "openthoughts": _openthoughts_messages,
        }.get(source)
        if msgs_fn is not None:
            msgs = msgs_fn(sample)
            if not msgs or msgs[-1].get("role") != "assistant":
                return False
        return True

    return SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(path=path, name=name, split=split),
        pre_filters=(
            lambda s, source=source: _has_usable_assistant_turn(s, source=source),
            _cot_ok,
        ),
        processor=AgptChatProcessor.Config(source=source),
        post_filters=(lambda sequence: sequence is not None,),
    )


def gsm8k_r1cot_streaming_dataset() -> ConcatThenSplitPackingConfig:
    """GSM8K alone wrapped in <think>/<answer>\\boxed{}. No TRL."""
    return ConcatThenSplitPackingConfig(
        dataset=DatasetMixConfig(
            datasets=(
                WeightedDataset(
                    dataset=_cot_source(
                        path="openai/gsm8k",
                        split="train",
                        source="gsm8k_r1cot",
                        name="main",
                    ),
                    weight=1.0,
                ),
            )
        )
    )


def distill_cot_mix_streaming_dataset(
    weights: tuple[float, float] = (0.5, 0.5),
) -> ConcatThenSplitPackingConfig:
    """50/50 OpenThoughts-114k + OpenR1-Math-220k CoT, both decontaminated."""
    if len(weights) != 2:
        raise ValueError(
            f"distill_cot_mix weights must be (w_ot, w_openr1); got {weights!r}"
        )
    return ConcatThenSplitPackingConfig(
        dataset=DatasetMixConfig(
            datasets=(
                WeightedDataset(
                    dataset=_cot_source(
                        path="open-thoughts/OpenThoughts-114k",
                        split="train",
                        source="openthoughts",
                        name="default",
                    ),
                    weight=weights[0],
                ),
                WeightedDataset(
                    dataset=_cot_source(
                        path="open-r1/OpenR1-Math-220k",
                        split="train",
                        source="openr1",
                        name="default",
                    ),
                    weight=weights[1],
                ),
            )
        )
    )


def gsm8k_plus_distill_cot_streaming_dataset(
    stage_ratio: float = 0.5,
) -> ConcatThenSplitPackingConfig:
    """GSM8K-R1CoT + distill breadth (all 3 CoT sources)."""
    return ConcatThenSplitPackingConfig(
        dataset=DatasetMixConfig(
            datasets=(
                WeightedDataset(
                    dataset=_cot_source(
                        path="openai/gsm8k",
                        split="train",
                        source="gsm8k_r1cot",
                        name="main",
                    ),
                    weight=stage_ratio,
                ),
                WeightedDataset(
                    dataset=_cot_source(
                        path="open-thoughts/OpenThoughts-114k",
                        split="train",
                        source="openthoughts",
                        name="default",
                    ),
                    weight=(1 - stage_ratio) * 0.5,
                ),
                WeightedDataset(
                    dataset=_cot_source(
                        path="open-r1/OpenR1-Math-220k",
                        split="train",
                        source="openr1",
                        name="default",
                    ),
                    weight=(1 - stage_ratio) * 0.5,
                ),
            )
        )
    )
