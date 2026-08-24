# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""GSM8K chain-of-thought dataset for the Monarch RL engine (Stage 2 of the CoT
plan, docs/live/chains/rl/plans/cot.md).

Ports the dataset half of the TRL task ``rl/tasks/gsm8k_reason.py`` to an
upstream ``Configurable`` dataset that the ``Rollouter`` iterates. Each sample is
a GSM8K question plus its gold numeric answer (parsed from the ``#### N`` marker);
the ``<think>``/``<answer>`` prompt envelope + one-shot exemplar are added by the
env (``env.py``), matching how ``alphabet_sort_agpt/few_shot_env.py`` owns the
turn-0 prompt.

Difficulty curriculum (carried over verbatim): GSM8K has no difficulty label, but
the number of ``<<...>>`` calculator annotations in the solution is a clean
reasoning-step proxy. ``max_steps=N`` keeps only problems with <=N steps -- an
easier subset raises the per-prompt solve rate, so more GRPO groups start MIXED
(>=1 correct, not all correct) and produce gradient. In the TRL task this was the
``GSM8K_MAX_STEPS`` env var; here it is a proper Config field (``max_steps``, 0 =
no filter) that can be baked into a config or overridden on the CLI.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator
from dataclasses import dataclass

from datasets import load_dataset

from torchtitan.config import Configurable

# Same prompt suffix as gsm8k-r1cot SFT + the Stage 0 eval (train/eval/RL match).
# ASCII-only; kept identical to rl/tasks/gsm8k_reason.py so a trace scored by the
# TRL reward and by this rubric sees the same envelope.
PROMPT_SUFFIX = (
    "\nReason step by step inside <think></think>, then give the final "
    "answer inside <answer>\\boxed{}</answer>."
)

_GOLD_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_STEP_RE = re.compile(r"<<[^>]*>>")


def _norm_num(s: str | None) -> str | None:
    """Normalize a numeric string to a canonical form (drop commas/$/trailing '.',
    collapse ``12.0`` -> ``12``). Shared with the rubric's answer parsing so a gold
    answer and a predicted answer compare on identical footing."""
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


@dataclass(frozen=True, kw_only=True, slots=True)
class GSM8KReasonSample:
    """One GSM8K problem: the raw question and its gold numeric answer.

    The env turns ``question`` into the turn-0 prompt (suffix + one-shot exemplar);
    the rubric scores the model's ``<answer>`` span against ``answer``.
    """

    question: str
    """The natural-language GSM8K word problem (without the prompt suffix)."""

    answer: str
    """Gold numeric answer, normalized (from the solution's ``#### N`` marker)."""


class GSM8KReasonDataset(Configurable):
    """Endless, seeded stream of GSM8K chain-of-thought samples.

    Loads ``openai/gsm8k`` (config ``main``), optionally filters to an easy subset
    by calculator-annotation count (``max_steps``), shuffles with ``seed``, and
    (for a smoke) truncates to ``num_samples``. Iterates endlessly, reshuffling on
    each wrap so a run sees a fresh permutation every epoch.

    Example::

        ds = GSM8KReasonDataset(GSM8KReasonDataset.Config(seed=42, max_steps=3))
        sample: GSM8KReasonSample = next(iter(ds))
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 42
        """Seed for the row-order shuffle."""

        split: str = "train"
        """GSM8K split to load: ``train`` or ``test`` (validation)."""

        max_steps: int = 0
        """Difficulty curriculum: keep only problems whose solution has <= this
        many ``<<...>>`` calculator steps. 0 (default) = no filter. An easy subset
        raises the per-prompt solve rate so more GRPO groups start MIXED."""

        num_samples: int = 0
        """Truncate to this many samples after shuffle (smoke). 0 = full split."""

        shuffle: bool = True
        """Shuffle row order (with ``seed``), reshuffling on each wrap. Set False
        for validation so every pass draws the same held-out samples."""

        hf_dataset: str = "openai/gsm8k"
        hf_config: str = "main"

        def __post_init__(self) -> None:
            if self.max_steps < 0:
                raise ValueError(f"max_steps must be >= 0; got {self.max_steps}")
            if self.num_samples < 0:
                raise ValueError(f"num_samples must be >= 0; got {self.num_samples}")

    def __init__(self, config: Config) -> None:
        self._config = config
        raw = load_dataset(config.hf_dataset, config.hf_config, split=config.split)

        if config.max_steps > 0:
            n_before = len(raw)
            raw = raw.filter(
                lambda ex: len(_STEP_RE.findall(ex["answer"])) <= config.max_steps
            )
            if len(raw) == 0:
                raise ValueError(
                    f"max_steps={config.max_steps} filtered out every problem "
                    f"in {config.hf_dataset}:{config.split} ({n_before} rows)"
                )

        self._questions: list[str] = []
        self._answers: list[str] = []
        for ex in raw:
            m = _GOLD_RE.search(ex["answer"])
            gold = _norm_num(m.group(1)) if m else None
            if gold is None:
                continue  # a row with no parseable gold has no learning signal
            self._questions.append(str(ex["question"]))
            self._answers.append(gold)
        if not self._questions:
            raise ValueError(
                f"no rows with a parseable '#### N' gold answer in "
                f"{config.hf_dataset}:{config.split}"
            )

        self._rng = random.Random(config.seed)
        self._shuffle = config.shuffle
        self._order = list(range(len(self._questions)))
        if self._shuffle:
            self._rng.shuffle(self._order)
        if config.num_samples and config.num_samples < len(self._order):
            self._order = self._order[: config.num_samples]
        self._pos = 0

    def __iter__(self) -> Iterator[GSM8KReasonSample]:
        return self

    def __next__(self) -> GSM8KReasonSample:
        if self._pos >= len(self._order):
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._pos = 0
        idx = self._order[self._pos]
        self._pos += 1
        return GSM8KReasonSample(
            question=self._questions[idx],
            answer=self._answers[idx],
        )

    def state_dict(self) -> dict:
        """Snapshot the RNG + position so a run can resume mid-stream."""
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "pos": self._pos,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._pos = state_dict["pos"]
