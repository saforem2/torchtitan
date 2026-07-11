#!/usr/bin/env python3
"""Quality-control filter for synthetic summaries.

Between summarize (step 2) and retok (step 3), drop degenerate generations. Even
a decent instruct model occasionally, at scale, (a) echoes the system/user
prompt back, (b) emits meta-commentary about the task instead of a summary
("The summary is too long...", "Here is a summary:"), or (c) produces something
far too short to carry the source's information. Feeding those into the training
shard would inject noise, so filter them here.

The POC's Llama-3.1-8B-exvocab smoke produced exactly these failure modes on 2
of 4 docs (prompt-echo + meta-commentary), which is what motivated this filter.

Usage:
  python -m torchtitan.experiments.ezpz.synthetic.qc_summaries \\
    --in outputs/synthetic/wiki_summaries.jsonl \\
    --out outputs/synthetic/wiki_summaries.qc.jsonl

Writes the kept rows to --out and prints a reject breakdown. --field selects
which JSONL field holds the summary text (default "summary").
"""
from __future__ import annotations

import argparse
import json

# Fragments of the system/user prompt in summarize_text.py. A generation that
# contains one of these is echoing the instructions rather than summarizing.
PROMPT_ECHO_MARKERS = (
    "You are a precise summarizer",
    "faithfully preserves the key facts",
    "do not include opinions or meta-commentary",
    "--- DOCUMENT ---",
    "--- END ---",
    "Summarize the following document",
)

# Lead-ins / phrasings that signal meta-commentary about the task rather than an
# actual summary. Matched case-insensitively against the (stripped) start.
META_PREFIXES = (
    "the summary",
    "here is a summary",
    "here's a summary",
    "sure,",
    "sure!",
    "i cannot",
    "i can't",
    "as an ai",
    "note:",
    "this summary",
    "the document is too",
    "the text is too",
)


def is_degenerate(text: str, *, min_chars: int, min_words: int) -> str | None:
    """Return a rejection reason if degenerate, else None."""
    t = text.strip()
    if len(t) < min_chars:
        return "too_short_chars"
    if len(t.split()) < min_words:
        return "too_short_words"
    for m in PROMPT_ECHO_MARKERS:
        if m in t:
            return "prompt_echo"
    low = t.lower()
    for p in META_PREFIXES:
        if low.startswith(p):
            return "meta_commentary"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="summaries JSONL")
    ap.add_argument("--out", required=True, help="filtered summaries JSONL")
    ap.add_argument("--field", default="summary")
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--min-words", type=int, default=20)
    args = ap.parse_args()

    kept = 0
    rejects: dict[str, int] = {}
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get(args.field) or ""
            reason = is_degenerate(text, min_chars=args.min_chars, min_words=args.min_words)
            if reason is None:
                fout.write(json.dumps(row) + "\n")
                kept += 1
            else:
                rejects[reason] = rejects.get(reason, 0) + 1

    total = kept + sum(rejects.values())
    rate = (kept / total * 100) if total else 0.0
    print(f"=== QC: kept {kept}/{total} ({rate:.1f}%) -> {args.out} ===")
    for reason, n in sorted(rejects.items(), key=lambda kv: -kv[1]):
        print(f"    rejected {n}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
