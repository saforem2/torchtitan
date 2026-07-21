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
    import os

    from datasets import load_dataset

    raw = load_dataset("openai/gsm8k", "main", split=split)

    # Difficulty curriculum: GSM8K has no difficulty label, but the number of
    # calculator annotations `<<...>>` in the solution is a clean reasoning-step
    # proxy. GSM8K_MAX_STEPS=N keeps only problems with <=N steps -- an easier
    # subset raises the per-prompt solve rate, so more GRPO groups start MIXED
    # (>=1 correct, not all correct) and produce gradient. At ~15% base accuracy
    # ~75% of full-difficulty groups are all-wrong (zero advantage); an easy
    # subset is the single biggest lever on that (per the diagnosis subagents).
    max_steps = int(os.environ.get("GSM8K_MAX_STEPS", "0"))  # 0 = no filter
    if max_steps > 0:
        step_re = re.compile(r"<<[^>]*>>")

        def _easy(ex):
            return len(step_re.findall(ex["answer"])) <= max_steps

        n_before = len(raw)
        raw = raw.filter(_easy)
        print(
            "gsm8k_reason: GSM8K_MAX_STEPS=%d -> kept %d/%d problems (<=%d steps)"
            % (max_steps, len(raw), n_before, max_steps)
        )

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


# Reward weights. GRPO computes advantages by normalizing the SUMMED reward
# WITHIN each prompt group (scale_rewards="group"): advantage_i =
# (r_i - group_mean) / group_std. A component that is CONSTANT across a group
# cancels EXACTLY in that subtraction and adds nothing to the std -- so once the
# cold-start model emits the envelope every time, think_format + extractable are
# inert (zero gradient), not diluting. Keep them as cheap REGRESSION guards
# (0.05 each), and put the reward budget where within-group VARIANCE lives: the
# hard sub-skill (numeric correctness). That is the actual ceiling-attack lesson
# (docs/production/rl/grpo/ceiling-attack.md) -- its winning component was DENSE
# on the hard skill, not on the already-saturated easy skills. So we add a dense
# relative-closeness partial-credit term that gives all-wrong groups (~75% of
# groups at cold start, frac_reward_zero_std~0.75) some variance to learn from.
_W_FORMAT = 0.05  # guard: only signals if the model REGRESSES on the envelope
_W_EXTRACT = 0.05  # guard: only signals if the answer stops being parseable
_W_CLOSE = 0.20  # dense partial credit -> within-group variance among wrong rollouts
_W_CORRECT = 0.70  # dominant exact-match RLVR signal


def think_format_reward(completions, **kwargs) -> list[float]:
    """Guard: reward if a well-formed <think>...</think><answer>...</answer>
    envelope was emitted. Pinned at max once the cold-start model is in-envelope,
    so it only signals on a REGRESSION (inert for the gradient otherwise)."""
    out = []
    for completion in completions:
        text = get_completion_text(completion)
        out.append(_W_FORMAT if _FORMAT_RE.search(text) else 0.0)
    return out


def answer_extractable_reward(completions, **kwargs) -> list[float]:
    """Guard: reward if a parseable numeric answer can be extracted from the
    <answer> span. Also pinned at max -> a regression guard, not a live signal."""
    out = []
    for completion in completions:
        _, ans = _answer_span(get_completion_text(completion))
        out.append(_W_EXTRACT if ans is not None else 0.0)
    return out


def _relative_closeness(pred: str | None, gold) -> float:
    """Dense [0, 1] partial credit for an extractable-but-WRONG numeric answer,
    from relative error to gold. The GSM8K analog of ceiling-attack's dense
    'order fraction': it gives within-group variance among all-wrong rollouts
    (which otherwise share a constant 0 correctness -> zero GRPO advantage). NOT
    awarded on an exact match (that is scored by answer_correct_reward), so
    partial credit never exceeds correct credit. Relative (not absolute) error
    keeps it scale-invariant across GSM8K's wide answer range."""
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


def answer_close_reward(completions, answer, **kwargs) -> list[float]:
    """Dense partial credit for near-miss answers -- the key gradient source for
    all-wrong groups at cold start. Weighted below exact-match so it can never
    dominate; zeroed on an exact match so it is strictly a consolation gradient."""
    out = []
    for completion, gold in zip(completions, answer):
        _, ans = _answer_span(get_completion_text(completion))
        out.append(_W_CLOSE * _relative_closeness(ans, gold))
    return out


def answer_correct_reward(completions, answer, **kwargs) -> list[float]:
    """Dominant exact-match RLVR signal. GSM8K answers are integers so exact
    match is the right correctness signal; the dense answer_close term supplies
    the partial-credit variance that binary correctness lacks at cold start."""
    out = []
    for completion, gold in zip(completions, answer):
        _, ans = _answer_span(get_completion_text(completion))
        out.append(_W_CORRECT if (ans is not None and gold and ans == str(gold)) else 0.0)
    return out


register_task(
    RLTask(
        name="gsm8k_reason",
        build_dataset=build_dataset,
        reward_funcs=[
            think_format_reward,
            answer_extractable_reward,
            answer_close_reward,
            answer_correct_reward,
        ],
        description=(
            "GSM8K chain-of-thought RLVR (Stage 2). Componentized reward: "
            "0.05 format guard + 0.05 extractable guard + 0.20 dense closeness "
            "+ 0.70 exact-match correct. GRPO group-norm cancels the constant "
            "guards; the dense closeness term supplies variance for all-wrong "
            "groups. Set GSM8K_MAX_STEPS to filter to an easy subset (curriculum)."
        ),
    )
)
