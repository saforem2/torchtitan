# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Componentized (shaped) reward for alphabet_sort -- the ceiling-attack lever.

The upstream RewardAlphabetSort scores a `difflib.SequenceMatcher` char-ratio
raised to a power. That is (a) noisy (sensitive to name lengths/spellings, not
just order) and (b) saturating, so GRPO's mean reward plateaus ~0.25 even as
per-batch peaks rise with capacity. This reward instead scores the actual
SUB-SKILLS additively, giving a smooth gradient toward FULLY correct output:

  format       (0.2): a valid `<xml_tag>` block was emitted with >=1 name line
  completeness (0.2): exactly the expected name set (no missing, no extra/dupes)
  order        (0.6): fraction of correctly-ordered adjacent pairs among the
                      names that ARE present (a Kendall-tau-like, order-specific
                      signal -- decoupled from format/spelling noise)

An exact, correctly-ordered answer scores 1.0. A well-formatted but mis-ordered
answer still earns format+completeness, and partial ordering earns proportional
order credit -- so the gradient points at ordering, the hard sub-skill.
"""

from __future__ import annotations

import re

from torchtitan.experiments.rl.rubrics import RewardFn


def _answer_lines(text: str, *, xml_tag: str) -> list[str]:
    blocks = re.findall(
        rf"<\s*{xml_tag}\s*>(.*?)</\s*{xml_tag}\s*>", text, re.DOTALL | re.IGNORECASE
    )
    if not blocks:
        return []
    return [line.strip() for line in blocks[-1].splitlines() if line.strip()]


def score_components(predicted: list[str], expected: list[str]) -> float:
    """Additive sub-skill score in [0, 1]: 0.2 format + 0.2 completeness + 0.6 order."""
    if not predicted:
        return 0.0
    fmt = 0.2  # a non-empty block was emitted

    exp_set = set(expected)
    pred_set = set(predicted)
    missing = len(exp_set - pred_set)
    extra = len(pred_set - exp_set) + (len(predicted) - len(pred_set))  # extras + dupes
    denom = max(1, len(exp_set))
    completeness = 0.2 * max(0.0, 1.0 - (missing + extra) / denom)

    # order: among predicted names that ARE expected, fraction of adjacent pairs
    # whose expected-ranks are non-decreasing (Kendall-tau-like on the subsequence).
    rank = {name: i for i, name in enumerate(expected)}
    seq = [rank[p] for p in predicted if p in rank]
    if len(seq) <= 1:
        order_frac = 1.0 if len(seq) == 1 else 0.0
    else:
        correct = sum(1 for a, b in zip(seq, seq[1:]) if a <= b)
        order_frac = correct / (len(seq) - 1)
    order = 0.6 * order_frac

    return fmt + completeness + order


class ShapedRewardAlphabetSort(RewardFn):
    """Drop-in replacement for RewardAlphabetSort using the componentized score.

    Same per-turn / expected-turn averaging as upstream (so early-ending rollouts
    are penalized), but each turn is scored by ``score_components`` instead of the
    char-ratio-to-a-power.
    """

    from dataclasses import dataclass as _dataclass

    @_dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        """Only needs `weight` (inherited); own nested Config so Configurable
        binds _owner to ShapedRewardAlphabetSort (not the abstract RewardFn)."""

        pass

    def __init__(self, config: "ShapedRewardAlphabetSort.Config") -> None:
        self.config = config
        self.weight = config.weight

    async def __call__(self, rollout, env_input) -> float:
        turn_scores: list[float] = []
        for turn_idx, (rollout_turn, expected_names) in enumerate(
            zip(rollout.turns, env_input.expected_names)
        ):
            message = rollout_turn.completion_message
            text = (message.get("content") or "") if message else ""
            xml_tag = (
                "alphabetical_sorted"
                if turn_idx == 0
                else "combined_alphabetical_sorted"
            )
            predicted = _answer_lines(text, xml_tag=xml_tag)
            turn_scores.append(score_components(predicted, list(expected_names)))
        num_expected = len(env_input.expected_names)
        return sum(turn_scores) / num_expected if num_expected else 0.0


__all__ = ["ShapedRewardAlphabetSort", "score_components"]
