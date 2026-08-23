"""Bump stale ``> Last updated: YYYY-MM-DD`` markers to each doc's last-commit date.

``check_stale_docs.sh`` phase 2 flags any doc whose ``Last updated`` marker
predates that file's last git-commit date: content changed but the stamp was
never bumped. That check is read-only, and the markers are hand-written prose
outside ``fill_trajectory_fields.py``'s scalar-field scope, so the punch-list
just accumulated (9 stale docs as of 2026-08-05). This closes the loop so the
catch-all can fix what it detects.

The marker is set to the file's **last-commit date**, not today: the stamp
answers "when was this content last changed", and the commit date is the
recorded answer. Stamping today would claim a freshness the content does not
have -- and on a doc that has not changed in weeks, it would be a lie that
permanently silences the staleness check.

Docs with uncommitted working-tree changes are SKIPPED: their real "last
changed" date is not yet knowable from git, and stamping them would bake in a
date that the imminent commit invalidates. They are reported so the next run
(post-commit) picks them up.

Three marker spellings exist in the corpus and each is preserved verbatim
apart from the date::

    > Last updated: 2026-07-24
    > **Last updated: 2026-07-24
    > **Last updated:** 2026-07-24

Usage::

    python3 -m torchtitan.experiments.ezpz.utils.refresh_last_updated [--dry-run]

Wired into ``refresh_all.sh`` so it runs with every catch-all refresh.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

DOCS_ROOT = Path("torchtitan/experiments/ezpz/docs")

# Capture everything up to the date so the exact spelling (bold, colon
# placement) round-trips untouched; only the date group is rewritten.
MARKER_RE = re.compile(
    r"^(?P<prefix>> *\*{0,2}Last updated:?\*{0,2} *)(?P<date>\d{4}-\d{2}-\d{2})",
    re.MULTILINE,
)


def _git_last_commit_date(path: Path) -> str | None:
    """Last commit date for `path` as YYYY-MM-DD, or None if never committed."""
    out = subprocess.run(
        ["git", "log", "-1", "--format=%cs", "--", str(path)],
        capture_output=True,
        text=True,
    )
    date = out.stdout.strip()
    return date or None


def _has_uncommitted_changes(path: Path) -> bool:
    """True if `path` has staged or unstaged modifications."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", str(path)],
        capture_output=True,
        text=True,
    )
    return bool(out.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change, write nothing",
    )
    args = ap.parse_args()

    if not DOCS_ROOT.is_dir():
        print(f"refresh_last_updated: no {DOCS_ROOT} (run from repo root)", file=sys.stderr)
        return 2

    bumped = fresh = skipped = 0
    for doc in sorted(DOCS_ROOT.rglob("*.md")):
        text = doc.read_text(encoding="utf-8", errors="replace")
        m = MARKER_RE.search(text)
        if not m:
            continue

        marker_date = m.group("date")
        git_date = _git_last_commit_date(doc)
        if git_date is None:
            continue  # untracked: nothing authoritative to stamp from

        # String compare is correct for zero-padded YYYY-MM-DD.
        if marker_date >= git_date:
            fresh += 1
            continue

        if _has_uncommitted_changes(doc):
            skipped += 1
            print(f"  [SKIP] {doc}")
            print(f"    marker {marker_date} < commit {git_date}, but file is dirty")
            continue

        print(f"  [BUMP] {doc}")
        print(f"    {marker_date} -> {git_date}")
        if not args.dry_run:
            doc.write_text(
                MARKER_RE.sub(
                    lambda mm: f"{mm.group('prefix')}{git_date}", text, count=1
                ),
                encoding="utf-8",
            )
        bumped += 1

    verb = "would bump" if args.dry_run else "bumped"
    tail = f", {skipped} skipped (dirty)" if skipped else ""
    print(f"=== {verb} {bumped}, {fresh} already fresh{tail} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
