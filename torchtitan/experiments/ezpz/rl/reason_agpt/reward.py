# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Componentized GSM8K chain-of-thought reward for the Monarch RL engine.

Ports the four additive reward functions from the TRL task
``rl/tasks/gsm8k_reason.py`` to upstream ``RewardFn`` subclasses so each shows up
SEPARATELY in ``reward_breakdown`` (keyed by class name). Packing them into one
fn would collapse the per-component metrics -- exactly what the CoT plan warns
against for Path B (docs/production/rl/plans/cot.md, "On Path B (Monarch overlay)
specifically: these must be separate RewardFn classes in the Rubric list").

The reward keys ONLY on the ``<answer>`` span after ``</think>`` (never the
reasoning text) and has NO last-number fallback -- a bare last number would grab
CoT scratch numbers. Components (weights carried over verbatim from the TRL task):

  ThinkFormatReward     (0.05): a well-formed <think>...</think><answer>...</answer>
                               envelope was emitted. Guard: pinned at max once the
                               cold-start model is in-envelope, so it only signals
                               a REGRESSION (inert for the GRPO gradient otherwise,
                               since a constant cancels in group-normalized
                               advantage).
  AnswerExtractableReward (0.05): a parseable numeric answer can be pulled from the
                               <answer> span. Also a regression guard.
  AnswerCloseReward     (0.20): DENSE partial credit for an extractable-but-WRONG
                               answer, from relative error to gold. The KEY
                               variance source -- gives all-wrong groups (~75% at
                               cold start) a gradient. Zeroed on an exact match so
                               it is strictly a consolation term.
  AnswerCorrectReward   (0.70): dominant exact-match RLVR signal.

Weights are ABSOLUTE (a well-formed, correct rollout scores 0.05+0.05+0.20*0+0.70
= 0.80; a perfect near-miss tops out at 0.05+0.05+0.20 = 0.30). Note the upstream
``Rubric`` normalizes each fn's ``weight`` to sum to 1.0 across the reward-fn
list, so we bake the intended ratio into the ``weight`` fields directly (0.05 /
0.05 / 0.20 / 0.70, summing to 1.0) and have each fn return a plain [0, 1] score.
That reproduces the TRL task's effective per-component contribution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from torchtitan.experiments.rl.rollout import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn

# --- <think>/<answer> parsing (ported verbatim from rl/tasks/gsm8k_reason.py) ---

_FORMAT_RE = re.compile(
    r"<think>.*?</think>\s*<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


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
    # No envelope -> NOT format_ok AND no answer. (Previously accepted a bare
    # \boxed{} here, which let extractable/close/correct fire without the
    # <think>/<answer> envelope -- the reward-hack path that decayed format
    # 0.955->0.635. A well-formed answer MUST come through the envelope branch.)
    return False, None


def _relative_closeness(pred: str | None, gold: str) -> float:
    """Dense [0, 1] partial credit for an extractable-but-WRONG numeric answer,
    from relative error to gold. The GSM8K analog of the ceiling-attack dense
    'order fraction': gives within-group variance among all-wrong rollouts (which
    otherwise share a constant 0 correctness -> zero GRPO advantage). NOT awarded
    on an exact match (that is AnswerCorrectReward's job), so partial credit never
    exceeds correct credit. Relative (not absolute) error keeps it scale-invariant
    across GSM8K's wide answer range."""
    if pred is None or not gold:
        return 0.0
    try:
        p = float(pred)
        g = float(str(gold))
    except ValueError:
        return 0.0
    if p == g:
        return 0.0
    return max(0.0, 1.0 - abs(p - g) / max(1.0, abs(g)))


def _completion_text(rollout: Rollout) -> str:
    """The last turn's assistant content (the CoT rollouts here are single-turn).

    The env ends the rollout on the first completion, so the final turn carries
    the model's whole <think>/<answer> response. Falls back to '' if a rollout
    errored before producing any turn."""
    if not rollout.turns:
        return ""
    message = rollout.turns[-1].completion_message
    if not message:
        return ""
    content = message.get("content") or ""
    # vLLM DefaultRenderer splits a <think>...</think> prefix out of content
    # into reasoning_content (tags stripped). The model DID emit the envelope
    # (raw-generation eval scores format ~0.985), so reconstruct the full
    # completion the policy actually sampled -- else the reward sees only the
    # <answer> span and scores every in-envelope rollout 0 (zero gradient).
    reasoning = message.get("reasoning_content") or ""
    if reasoning and "<think>" not in content:
        return "<think>" + reasoning + "</think>" + content
    return content


# --- Reward functions (each logged separately in reward_breakdown) -------------


class ThinkFormatReward(RewardFn):
    """Guard: 1.0 if a well-formed <think>...</think><answer>...</answer> envelope
    was emitted, else 0.0. Weighted low (0.05); pinned at max once the model is
    in-envelope, so it only signals a REGRESSION."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        return 1.0 if _FORMAT_RE.search(_completion_text(rollout)) else 0.0


class AnswerExtractableReward(RewardFn):
    """Guard: 1.0 if a parseable numeric answer can be extracted from the <answer>
    span, else 0.0. Weighted low (0.05); a regression guard, not a live signal."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        format_ok, ans = _answer_span(_completion_text(rollout))
        return 1.0 if (format_ok and ans is not None) else 0.0


class AnswerCloseReward(RewardFn):
    """Dense partial credit (relative closeness to gold) for a near-miss answer --
    the key gradient source for all-wrong groups at cold start. Zeroed on an exact
    match so it is strictly a consolation gradient; weighted (0.20) below
    AnswerCorrectReward so it can never dominate.

    ``env_input`` is the ``GSM8KReasonSample`` (has ``.answer``, the gold)."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        format_ok, ans = _answer_span(_completion_text(rollout))
        if not format_ok:
            return 0.0
        return _relative_closeness(ans, env_input.answer)


class AnswerCorrectReward(RewardFn):
    """Dominant exact-match RLVR signal: 1.0 if the extracted <answer> equals the
    gold answer, else 0.0. GSM8K answers are integers so exact match is the right
    correctness signal; AnswerCloseReward supplies the partial-credit variance that
    binary correctness lacks at cold start. Weighted 0.70."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        format_ok, ans = _answer_span(_completion_text(rollout))
        if not format_ok:
            return 0.0
        gold = env_input.answer
        return 1.0 if (ans is not None and gold and ans == str(gold)) else 0.0


__all__ = [
    "ThinkFormatReward",
    "AnswerExtractableReward",
    "AnswerCloseReward",
    "AnswerCorrectReward",
]
