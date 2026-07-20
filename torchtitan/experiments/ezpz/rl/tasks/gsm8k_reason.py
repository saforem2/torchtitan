# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# gsm8k_reason: GRPO task for Stage 2 of the CoT plan
# (docs/production/rl/plans/cot.md). Trains the Stage-1 cold-start checkpoint to
# reason BETTER: reward keys on the <answer> span (never the reasoning text),
# and is COMPONENTIZED per the ceiling-attack result -- separate additive reward
# funcs (each shows up in reward_breakdown) rather than one binary exact-match,
# so partial/mis-answered rollouts still get a format+extractability gradient
# instead of a saturating all-or-nothing signal.

from __future__ import annotations

import re

from datasets import Dataset

from torchtitan.experiments.ezpz.rl.tasks import RLTask, register_task
from torchtitan.experiments.ezpz.rl.tasks.common import get_completion_text

# same prompt suffix as gsm8k-r1cot SFT + the Stage 0 eval (train/eval/RL match).
_PROMPT_SUFFIX = (
    "\nReason step by step inside <think></think>, then give the final "
    "answer inside <answer>\\boxed{}</answer>."
)

_FORMAT_RE = re.compile(
    r"<think>.*?</think>\s*<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_GOLD_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def _norm_num(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    m = _NUM_RE.search(s)
    if not m:
        return None
    n = m.group(0).replace(",", "")
    try:
        f = float(n)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def _answer_span(text: str) -> tuple[bool, str | None]:
    """(format_ok, extracted_answer). Reads only the <answer> span after
    </think>; no last-number fallback (would grab CoT scratch numbers)."""
    m = _FORMAT_RE.search(text)
    if m:
        span = m.group(1)
        boxed = _BOXED_RE.search(span)
        ans = _norm_num(boxed.group(1)) if boxed else _norm_num(span)
        return (ans is not None), ans
    stripped = _THINK_RE.sub("", text)
    boxed = _BOXED_RE.search(stripped)
    return False, (_norm_num(boxed.group(1)) if boxed else None)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_dataset(num_samples: int = 0, split: str = "train", seed: int = 42) -> Dataset:
    """GSM8K prompts (with the <think>/<answer> instruction) + gold answers.

    ``num_samples`` > 0 truncates (after shuffle) for a smoke; 0 uses the full
    split. Gold answer parsed from the '#### N' marker.
    """
    from datasets import load_dataset

    raw = load_dataset("openai/gsm8k", "main", split=split)
    raw = raw.shuffle(seed=seed)
    if num_samples and num_samples < len(raw):
        raw = raw.select(range(num_samples))

    def _format(ex):
        m = _GOLD_RE.search(ex["answer"])
        return {
            "prompt": [{"role": "user", "content": ex["question"] + _PROMPT_SUFFIX}],
            "answer": _norm_num(m.group(1)) if m else "",
        }

    return raw.map(_format, remove_columns=raw.column_names)


# ---------------------------------------------------------------------------
# Reward functions (componentized -- each logged separately in reward_breakdown)
# ---------------------------------------------------------------------------


def think_format_reward(completions, **kwargs) -> list[float]:
    """0.2 if a well-formed <think>...</think><answer>...</answer> envelope with a
    non-empty answer span was emitted. Dense signal toward the FORMAT even when
    the final answer is wrong (anti-saturation)."""
    out = []
    for completion in completions:
        text = get_completion_text(completion)
        out.append(0.2 if _FORMAT_RE.search(text) else 0.0)
    return out


def answer_extractable_reward(completions, **kwargs) -> list[float]:
    """0.1 if a parseable numeric answer can be extracted from the <answer>
    span. Rewards committing to a concrete answer (separate from correctness)."""
    out = []
    for completion in completions:
        _, ans = _answer_span(get_completion_text(completion))
        out.append(0.1 if ans is not None else 0.0)
    return out


def answer_correct_reward(completions, answer, **kwargs) -> list[float]:
    """0.7 if the extracted <answer> matches the gold answer. The dominant
    (correctness) component; GSM8K answers are integers so exact-match is the
    right signal, but it is only ~70% of the total so partial rollouts still
    climb via the format/extractable components."""
    out = []
    for completion, gold in zip(completions, answer):
        _, ans = _answer_span(get_completion_text(completion))
        out.append(0.7 if (ans is not None and gold and ans == str(gold)) else 0.0)
    return out


register_task(
    RLTask(
        name="gsm8k_reason",
        build_dataset=build_dataset,
        reward_funcs=[
            think_format_reward,
            answer_extractable_reward,
            answer_correct_reward,
        ],
        description=(
            "GSM8K chain-of-thought RLVR (Stage 2). Componentized reward: "
            "0.2 <think>/<answer> format + 0.1 extractable + 0.7 correct. "
            "For the Stage-1 cold-start checkpoint."
        ),
    )
)
