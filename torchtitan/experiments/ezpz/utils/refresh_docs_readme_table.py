#!/usr/bin/env python3
"""Auto-update the ``Notes`` and ``Modified`` columns in
``docs/README.md`` tables.

The top-level ``docs/README.md`` has ~9 tables of the form

    | Page | Notes | Modified |
    |------|-------|---------:|
    | [Production Index](./production/README.md) | ... | 2026-05-23 |

Both columns rot. This script:

1. Parses every table-row that has a markdown link in the first cell.
2. Resolves the link target relative to ``docs/`` (the README's dir).
3. For each resolved file path:
   * Rewrites the ``Modified`` column with the last-commit date
     (``git log -1 --format=%cs``).
   * If the linked file has a recognisable canonical-chain status block
     (``**Latest checkpoint:** ...``, ``**Tokens consumed:** ...``,
     etc.), synthesises a fresh one-liner for ``Notes``.
4. Rows whose linked file is not a canonical chain (no recognisable
   status block) are left alone in the Notes column — that's narrative
   judgment.
5. Rows containing the sentinel ``<!-- noauto -->`` are not touched
   at all. Use this to keep hand-curated descriptions stable.

Usage (from repo root):
    python3 -m torchtitan.experiments.ezpz.utils.refresh_docs_readme_table

By default operates on ``torchtitan/experiments/ezpz/docs/README.md``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

DEFAULT_README = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "README.md"
)

# Matches a table row whose first cell contains a markdown link.
ROW_RE = re.compile(r"^\|\s*\[[^\]]+\]\(([^)]+)\)\s*\|")

# Recognisable status-block fields inside per-trajectory READMEs.
LATEST_CKPT_RE = re.compile(r"^\*\*Latest checkpoint:\*\*\s*(.+)$", re.M)
CUMULATIVE_STEPS_RE = re.compile(r"^\*\*Cumulative(?: \*persisted\*)? steps:\*\*\s*(.+)$", re.M)
TOKENS_RE = re.compile(r"^\*\*Tokens consumed(?: \(persisted\))?:\*\*\s*(.+)$", re.M)
LOSS_RE = re.compile(r"^\*\*Loss:\*\*\s*(.+)$", re.M)

# Pull the trailing-percent/loss-number patterns out of a longer cell
# so the synthesised one-liner stays short.
# Handles both formats:
#   "= **4.047T tokens** (**86.6%** of 4.67T target)"
#   "= **3.07T tokens** (65.7% of 4.67T target)"
TOKEN_TOTAL_RE = re.compile(
    r"\*\*([\d.]+[KMBT])\s*tokens\*\*\s*\(\*{0,2}([\d.]+%)"
)
STEP_NUM_RE = re.compile(r"step-?([\d,]+)", re.I)
LOSS_NUM_RE = re.compile(r"^([\d.]+)")

# Opt-out sentinel
NOAUTO_SENTINEL = "<!-- noauto -->"


def git_last_commit_date(repo_root: Path, path: Path) -> str | None:
    """Return YYYY-MM-DD of the last commit touching ``path``."""
    rel = path.resolve().relative_to(repo_root)
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "log", "-1", "--format=%cs", "--", str(rel)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError:
        return None
    date = out.stdout.strip()
    return date or None


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise RuntimeError(f"no .git found walking up from {start}")


def synthesize_notes(target_path: Path) -> str | None:
    """Read the linked file and synthesise a one-line Notes blurb if
    it has the canonical-chain status block. Return None otherwise.
    """
    try:
        text = target_path.read_text()
    except (OSError, UnicodeDecodeError):
        return None

    latest = LATEST_CKPT_RE.search(text)
    if not latest:
        return None  # not a canonical-chain page

    latest_str = latest.group(1).strip()
    tokens_str = TOKENS_RE.search(text)
    loss_str = LOSS_RE.search(text)

    # Pull step number out of the latest-checkpoint sentence.
    step_m = STEP_NUM_RE.search(latest_str)
    step_part = f"step-**{step_m.group(1)}**" if step_m else latest_str.split(" ")[0]

    # Pull "X.XXT tokens (YY%)" tidily.
    tokens_part = ""
    if tokens_str:
        ts = tokens_str.group(1).strip()
        m = TOKEN_TOTAL_RE.search(ts)
        if m:
            tokens_part = f" ({m.group(1)} tokens, {m.group(2)} of 4.67T)"
        else:
            tokens_part = f" ({ts.split('(')[0].strip()})"

    # Loss — first numeric token only.
    loss_part = ""
    if loss_str:
        ls = loss_str.group(1).strip()
        lm = LOSS_NUM_RE.match(ls)
        if lm:
            loss_part = f", loss {lm.group(1)}"

    return f"{step_part}{tokens_part}{loss_part}."


def split_row(line: str) -> list[str] | None:
    """Split a markdown table row into its raw cell contents (between
    the leading and trailing ``|``). Returns None if the line isn't a
    table row.
    """
    if not line.startswith("|") or not line.rstrip().endswith("|"):
        return None
    s = line.strip()
    s = s[1:-1]  # drop leading + trailing |
    # Markdown tables don't support escaped pipes in our usage; safe split.
    return s.split("|")


def join_row(cells: list[str]) -> str:
    return "| " + " | ".join(c.strip() for c in cells) + " |"


def refresh_readme(readme_path: Path, *, dry_run: bool = False) -> tuple[int, int]:
    """Rewrite rows in ``readme_path``. Returns (rows_with_changes,
    notes_rewritten).
    """
    repo_root = find_repo_root(readme_path)
    readme_dir = readme_path.parent

    lines = readme_path.read_text().splitlines(keepends=False)
    updated_lines: list[str] = []
    n_changed = 0
    n_notes_rewrites = 0

    for line in lines:
        m = ROW_RE.match(line)
        if not m:
            updated_lines.append(line)
            continue

        if NOAUTO_SENTINEL in line:
            updated_lines.append(line)
            continue

        target_str = m.group(1)
        if "://" in target_str or target_str.startswith("mailto:"):
            updated_lines.append(line)
            continue

        cells = split_row(line)
        if cells is None or len(cells) < 3:
            updated_lines.append(line)
            continue

        target_path = (readme_dir / target_str).resolve()
        if not target_path.exists():
            print(f"  [missing] {target_str} — leaving row unchanged",
                  file=sys.stderr)
            updated_lines.append(line)
            continue

        new_date = git_last_commit_date(repo_root, target_path)
        new_notes = synthesize_notes(target_path)

        old_date = cells[-1].strip()
        old_notes = cells[1].strip()

        any_change = False

        if new_date and new_date != old_date:
            cells[-1] = f" {new_date} "
            any_change = True
            print(f"  [date {old_date} -> {new_date}] {target_str}")

        if new_notes and new_notes != old_notes:
            cells[1] = f" {new_notes} "
            any_change = True
            n_notes_rewrites += 1
            print(f"  [notes rewrite] {target_str}")
            print(f"      OLD: {old_notes[:80]}{'...' if len(old_notes) > 80 else ''}")
            print(f"      NEW: {new_notes}")

        if any_change:
            updated_lines.append(join_row(cells))
            n_changed += 1
        else:
            updated_lines.append(line)

    if n_changed and not dry_run:
        readme_path.write_text("\n".join(updated_lines) + "\n")

    suffix = " (dry-run)" if dry_run else ""
    print(
        f"\n=== {n_changed} row(s) updated, {n_notes_rewrites} notes-column "
        f"rewrites{suffix} ===",
    )
    return n_changed, n_notes_rewrites


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
    )
    parser.add_argument(
        "readme",
        nargs="?",
        type=Path,
        default=DEFAULT_README,
        help=f"path to README.md (default: {DEFAULT_README})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="show planned changes without rewriting the file",
    )
    args = parser.parse_args()

    refresh_readme(args.readme, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
