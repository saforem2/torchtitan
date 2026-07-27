# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# GSM8K test-set decontamination for reasoning-distillation SFT traces.
#
# WHY THIS EXISTS. The reasoning-distillation SFT experiment cold-starts a 2B
# on pre-distilled frontier CoT traces (OpenThoughts-114k, OpenR1-Math-220k,
# ...), then evaluates on the GSM8K test split (1319 problems, scripts/eval/
# eval_cot_gsm8k.py). Those distillation corpora are built from the SAME public
# math-problem pools GSM8K was drawn from, so a nonzero fraction of their
# training questions ARE GSM8K test questions verbatim (or trivially reworded).
# Training on a test question inflates the eval and the "gain" is a mirage. The
# team was burned by exactly this class of leak before, so decontamination is a
# hard prerequisite -- NOT an optional cleanup step.
#
# WHAT IT DOES. Given a corpus of traces (HF dataset name, an on-disk
# save_to_disk tree, or a local json/jsonl) and the GSM8K test questions, drop
# any trace whose question collides with ANY GSM8K test question under either of
# two conservative signals:
#   1. normalized-exact match: identical after lowercasing + stripping all
#      non-alphanumeric characters + collapsing whitespace (catches
#      near-duplicates differing only in case / punctuation / spacing).
#   2. 13-gram exact-match: the question shares a contiguous run of 13
#      normalized word tokens with a test question (catches a test question
#      embedded inside a larger prompt, and heavier reword-with-verbatim-spans).
# 13 tokens is long enough that same-topic-but-distinct problems essentially
# never collide, so false-positive drops are negligible while a copied problem
# statement is caught reliably.
#
# Pure-Python / CPU. Importable (GSM8KDecontaminator, filter_dataset,
# load_gsm8k_test_questions) and CLI-runnable (`python -m
# torchtitan.experiments.ezpz.rl.decontam_traces --help`). The n-gram / exact
# logic has no `datasets` dependency, so `--self-test` runs anywhere.

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

log = logging.getLogger(__name__)

__all__ = [
    "GSM8KDecontaminator",
    "DecontamReport",
    "normalize_text",
    "word_ngrams",
    "load_gsm8k_test_questions",
    "filter_dataset",
]

# The CoT envelope prompt suffix appended to every training question (must stay
# byte-identical to datasets_sft._OPENR1_SUFFIX and the eval's _PROMPT_SUFFIX).
# We strip it before matching so the suffix text can't perturb the n-gram set.
# Duplicated here (rather than imported) to keep this module free of a
# datasets_sft import -- datasets_sft imports THIS module, not the reverse.
_ENVELOPE_SUFFIX = (
    "\nReason step by step inside <think></think>, then give the final "
    "answer inside <answer>\\boxed{}</answer>."
)

_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    """Lowercase, replace every non-alphanumeric run with a single space, and
    collapse whitespace. Used for BOTH the exact-match key and the token stream
    the n-grams are built from, so the two signals see the same normalization.
    """
    if not text:
        return ""
    t = text.lower()
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def word_ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    """Set of contiguous n-token tuples. Empty when the token list is shorter
    than n (a short question contributes no n-grams; it can still be caught by
    the normalized-exact signal)."""
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def strip_envelope_suffix(question: str) -> str:
    """Remove the trailing CoT-envelope prompt suffix if present, so a question
    already wrapped for training matches the same question in raw form."""
    if question.endswith(_ENVELOPE_SUFFIX):
        return question[: -len(_ENVELOPE_SUFFIX)]
    return question


@dataclass
class DecontamReport:
    """Result of a decontamination pass."""

    total: int
    kept: int
    dropped: int
    by_reason: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped": self.dropped,
            "drop_rate": round(self.dropped / self.total, 6) if self.total else 0.0,
            "by_reason": dict(self.by_reason),
            "by_source": dict(self.by_source),
        }


class GSM8KDecontaminator:
    """Holds the GSM8K test-question fingerprints and tests candidate questions.

    Build once from the 1319 test questions, then call ``is_contaminated`` per
    candidate. Thread-safe / picklable (only holds two Python sets), so it can
    be handed to ``datasets.map(..., num_proc=N)`` workers.
    """

    def __init__(self, test_questions: Iterable[str], n: int = 13) -> None:
        if n <= 0:
            raise ValueError(f"n-gram size must be positive; got {n}")
        self.n = n
        self.exact: set[str] = set()
        self.ngrams: set[tuple[str, ...]] = set()
        for q in test_questions:
            norm = normalize_text(q)
            if not norm:
                continue
            self.exact.add(norm)
            self.ngrams |= word_ngrams(norm.split(), n)
        self.num_questions = len(self.exact)

    def is_contaminated(self, question: str | None) -> tuple[bool, str]:
        """Return ``(contaminated, reason)``; reason is "exact", "ngram", or ""
        (empty when clean). Exact match is checked first (cheaper + strictest)."""
        norm = normalize_text(question)
        if not norm:
            return False, ""
        if norm in self.exact:
            return True, "exact"
        cand = word_ngrams(norm.split(), self.n)
        if cand and not cand.isdisjoint(self.ngrams):
            return True, "ngram"
        return False, ""

    def __len__(self) -> int:
        return self.num_questions


def load_gsm8k_test_questions(dataset: str = "openai/gsm8k") -> list[str]:
    """Load the GSM8K 'main' TEST split (1319 problems) and return the question
    strings. Requires the ``datasets`` package + (network or a warm HF cache)."""
    from datasets import load_dataset

    ds = load_dataset(dataset, "main", split="test")
    return [r["question"] for r in ds]


def _default_get_question(row: dict) -> str:
    """Best-effort question extraction for arbitrary trace rows.

    Handles the three shapes seen in this pipeline:
      - chat-format ``{"prompt": [{"role": "user", "content": ...}], ...}``
        (the envelope-wrapped mix rows) -- read the user content, strip suffix.
      - raw ``{"problem": ...}`` (OpenR1) or ``{"question": ...}`` (GSM8K-like).
      - OpenThoughts ``{"conversations": [{"from": "user", "value": ...}, ...]}``.
    """
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        content = prompt[0].get("content", "") if isinstance(prompt[0], dict) else ""
        if isinstance(content, str) and content:
            return strip_envelope_suffix(content)
    convs = row.get("conversations")
    if isinstance(convs, list):
        for turn in convs:
            if isinstance(turn, dict) and turn.get("from") in ("user", "human"):
                return str(turn.get("value", ""))
    for key in ("problem", "question", "query", "instruction"):
        val = row.get(key)
        if isinstance(val, str) and val:
            return strip_envelope_suffix(val)
    return ""


def filter_dataset(
    dataset,
    detector: GSM8KDecontaminator,
    get_question: Callable[[dict], str] | None = None,
    *,
    source_column: str | None = None,
    num_proc: int | None = None,
):
    """Drop GSM8K-test-contaminated rows from an HF ``Dataset``.

    Returns ``(filtered_dataset, DecontamReport)``. ``get_question`` maps a row
    to its question text (defaults to ``_default_get_question``). When
    ``source_column`` is given, the report additionally breaks drops down per
    source value (useful when the corpus tags rows with their origin, e.g.
    OpenThoughts' ``domain``/``source``).
    """
    getq = get_question or _default_get_question
    total = len(dataset)

    # An empty input has no columns to index; return it as-is (map/getitem on a
    # 0-row Dataset would raise on the synthetic reason column below).
    if total == 0:
        return dataset, DecontamReport(total=0, kept=0, dropped=0)

    reason_col = "__decontam_reason"

    def _annotate(row: dict) -> dict:
        _, reason = detector.is_contaminated(getq(row))
        return {reason_col: reason}

    annotated = dataset.map(_annotate, num_proc=num_proc)

    reasons = annotated[reason_col]
    sources = annotated[source_column] if source_column else None

    by_reason: dict[str, int] = {}
    by_source: dict[str, int] = {}
    dropped = 0
    for i, reason in enumerate(reasons):
        if not reason:
            continue
        dropped += 1
        by_reason[reason] = by_reason.get(reason, 0) + 1
        if sources is not None:
            src = str(sources[i])
            by_source[src] = by_source.get(src, 0) + 1

    kept = annotated.filter(lambda r: not r[reason_col], num_proc=num_proc)
    kept = kept.remove_columns([reason_col])

    report = DecontamReport(
        total=total,
        kept=len(kept),
        dropped=dropped,
        by_reason=by_reason,
        by_source=by_source,
    )
    return kept, report


# ---------------------------------------------------------------------------
# Self-test -- no `datasets` dependency, runs anywhere.
# ---------------------------------------------------------------------------


def _self_test() -> bool:
    """Confirm the detector drops a GSM8K-test paraphrase / embed and keeps an
    unrelated question. Returns True on success. Exercised by `--self-test`.
    """
    # Three synthetic "GSM8K test" questions, each well over 13 tokens.
    test_questions = [
        (
            "Natalia sold clips to 48 of her friends in April, and then she "
            "sold half as many clips in May. How many clips did Natalia sell "
            "altogether in April and May?"
        ),
        (
            "Weng earns 12 dollars an hour for babysitting. Yesterday, she just "
            "did 50 minutes of babysitting. How much did she earn for that time?"
        ),
        (
            "Betty is saving money for a new wallet which costs 100 dollars. "
            "Betty has only half of the money she needs. Her parents decided to "
            "give her 15 dollars and her grandparents twice as much as her "
            "parents. How much more money does Betty need to buy the wallet?"
        ),
    ]
    det = GSM8KDecontaminator(test_questions, n=13)

    results: list[tuple[str, bool, str, bool]] = []  # (label, want_drop, reason, ok)

    # Case A: normalized-exact paraphrase -- same question, only case /
    # punctuation / whitespace changed. Must DROP (reason "exact").
    para = (
        "  natalia SOLD clips to 48 of her friends in april,,, and then she   "
        "sold half as many clips in may.  how many clips did natalia sell "
        "altogether in april and may?!?  "
    )
    c, r = det.is_contaminated(para)
    results.append(("paraphrase(exact)", True, r, c is True and r == "exact"))

    # Case B: a test question embedded verbatim inside a larger trace prompt.
    # The whole strings differ (so NOT exact) but a 13-gram run collides.
    # Must DROP (reason "ngram").
    embed = (
        "Solve the following word problem and show all steps. "
        "Weng earns 12 dollars an hour for babysitting. Yesterday, she just "
        "did 50 minutes of babysitting. How much did she earn for that time? "
        "Give your final answer in dollars."
    )
    c, r = det.is_contaminated(embed)
    results.append(("embed(ngram)", True, r, c is True and r == "ngram"))

    # Case C: an unrelated question. Must KEEP.
    unrelated = (
        "A train leaves Chicago traveling west at 60 miles per hour while a "
        "second train departs from Denver heading east at 75 miles per hour. "
        "After how many hours do the two trains meet if the cities are 1350 "
        "miles apart?"
    )
    c, r = det.is_contaminated(unrelated)
    results.append(("unrelated(keep)", False, r, c is False))

    # Case D: empty question -- must be treated as clean (kept), never crash.
    c, r = det.is_contaminated("")
    results.append(("empty(keep)", False, r, c is False))

    all_ok = all(row[3] for row in results)
    print("=== decontam_traces self-test ===")
    print(f"detector: {len(det)} test questions, n={det.n}, "
          f"{len(det.ngrams)} unique {det.n}-grams")
    for label, want_drop, reason, ok in results:
        verdict = "PASS" if ok else "FAIL"
        action = "drop" if want_drop else "keep"
        print(f"  [{verdict}] {label:<18} expect={action:<4} reason={reason or '-'}")
    print(f"RESULT: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_input_dataset(args):
    """Resolve --input into an HF Dataset: a hub name, an on-disk save_to_disk
    tree (--from-disk), or a local json/jsonl file."""
    from datasets import load_dataset, load_from_disk

    src = args.input
    if args.from_disk:
        return load_from_disk(src)
    if src.endswith((".json", ".jsonl")):
        return load_dataset("json", data_files=src, split="train")
    return load_dataset(src, args.config, split=args.split)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description=(
            "Drop GSM8K-test-contaminated rows from a trace corpus "
            "(13-gram + normalized-exact question match)."
        )
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run the built-in self-test (no datasets/network needed) and exit.",
    )
    ap.add_argument("--input", help="HF dataset name, save_to_disk dir, or *.jsonl")
    ap.add_argument("--config", default=None, help="HF dataset config/subset name")
    ap.add_argument("--split", default="train", help="HF split (default: train)")
    ap.add_argument(
        "--from-disk",
        action="store_true",
        help="Treat --input as a Dataset.save_to_disk directory.",
    )
    ap.add_argument(
        "--question-column",
        default=None,
        help=(
            "Column holding the question. Default: auto-detect "
            "(prompt / problem / question / conversations)."
        ),
    )
    ap.add_argument(
        "--source-column",
        default=None,
        help="Optional column to break the drop report down by (e.g. 'source').",
    )
    ap.add_argument("--n", type=int, default=13, help="n-gram size (default: 13)")
    ap.add_argument(
        "--gsm8k",
        default="openai/gsm8k",
        help="GSM8K dataset id for the test split (default: openai/gsm8k).",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Write the filtered dataset here (save_to_disk dir, or *.jsonl).",
    )
    ap.add_argument("--num-proc", type=int, default=None, help="datasets.map procs")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if _self_test() else 1

    if not args.input:
        ap.error("--input is required (or pass --self-test)")

    det = GSM8KDecontaminator(load_gsm8k_test_questions(args.gsm8k), n=args.n)
    log.info(
        f"[decontam] built detector: {len(det)} GSM8K test questions, "
        f"n={args.n}, {len(det.ngrams):,} unique n-grams"
    )

    ds = _load_input_dataset(args)
    log.info(f"[decontam] loaded {len(ds):,} rows from {args.input!r}")

    get_question = None
    if args.question_column:
        col = args.question_column

        def get_question(row):  # noqa: F811
            # Explicit column override: read that column, strip the envelope
            # suffix; fall back to auto-detect when the column is absent/empty.
            val = row.get(col)
            if isinstance(val, str) and val:
                return strip_envelope_suffix(val)
            return _default_get_question(row)

    kept, report = filter_dataset(
        ds,
        det,
        get_question,
        source_column=args.source_column,
        num_proc=args.num_proc,
    )
    log.info("[decontam] report:\n" + json.dumps(report.as_dict(), indent=2))

    if args.output:
        out = args.output
        if out.endswith((".json", ".jsonl")):
            kept.to_json(out)
        else:
            kept.save_to_disk(out)
        log.info(f"[decontam] wrote {len(kept):,} clean rows -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
