#!/usr/bin/env python3
"""Auto-update the ``Modified`` column in ``docs/README.md`` tables.

The top-level ``docs/README.md`` has ~9 tables of the form

    | Page | Notes | Modified |
    |------|-------|---------:|
    | [Production Index](./production/README.md) | ... | 2026-05-23 |

The ``Modified`` column rots fast because it is hand-curated — every
per-trajectory README refresh leaves it stale. This script:

1. Parses every table-row that has a markdown link in the first cell.
2. Resolves the link target relative to ``docs/`` (the README's dir).
3. For each resolved file path, gets the last-commit date via
   ``git log -1 --format=%cs <path>`` (YYYY-MM-DD).
4. Rewrites the last cell of the row to that date.

The ``Notes`` column is left alone — that's narrative judgment, not
something the script should rewrite. The script does emit a warning
when a linked file has been modified more recently than this README
itself (i.e. "the description is probably stale, go look").

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
# Captures: full row, link target, last cell content (the date).
ROW_RE = re.compile(
    r"^\|\s*\[[^\]]+\]\(([^)]+)\)\s*\|.*\|\s*([^|]*?)\s*\|\s*$"
)


def git_last_commit_date(repo_root: Path, path: Path) -> str | None:
    """Return YYYY-MM-DD of the last commit touching ``path``, or None
    if git has no history for it (e.g. uncommitted, or path doesn't
    exist in HEAD).
    """
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
    """Walk up from ``start`` until we find a directory containing
    ``.git`` (treat that as the repo root).
    """
    p = start.resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise RuntimeError(f"no .git found walking up from {start}")


def refresh_readme(readme_path: Path, *, dry_run: bool = False) -> int:
    """Rewrite the ``Modified`` column of every linked row in
    ``readme_path``. Returns the number of rows updated.
    """
    repo_root = find_repo_root(readme_path)
    readme_dir = readme_path.parent
    readme_date = git_last_commit_date(repo_root, readme_path)

    lines = readme_path.read_text().splitlines(keepends=False)
    updated_lines: list[str] = []
    n_changed = 0
    n_stale_warnings = 0

    for line in lines:
        m = ROW_RE.match(line)
        if not m:
            updated_lines.append(line)
            continue

        target_str, old_date = m.group(1), m.group(2)

        # Skip external links (http://, https://, mailto:, etc.)
        if "://" in target_str or target_str.startswith("mailto:"):
            updated_lines.append(line)
            continue

        # Resolve relative to the README's directory (matches how
        # GitHub renders the link).
        target_path = (readme_dir / target_str).resolve()

        if not target_path.exists():
            print(
                f"  [missing] {target_str} — leaving row unchanged "
                f"(was: {old_date})",
                file=sys.stderr,
            )
            updated_lines.append(line)
            continue

        new_date = git_last_commit_date(repo_root, target_path)
        if new_date is None:
            print(
                f"  [no git] {target_str} — leaving row unchanged "
                f"(was: {old_date})",
                file=sys.stderr,
            )
            updated_lines.append(line)
            continue

        if new_date == old_date:
            updated_lines.append(line)
        else:
            # Replace only the last `| ... |` cell content with the new
            # date. Rebuild conservatively: find the last `|` before the
            # trailing `|` and slice.
            # Pattern is reliable because ROW_RE matched.
            i_last_sep = line.rfind("|", 0, line.rstrip().rfind("|"))
            new_line = f"{line[:i_last_sep]}| {new_date} |"
            updated_lines.append(new_line)
            n_changed += 1
            print(f"  [{old_date} -> {new_date}] {target_str}")

        # Warn if the linked file is newer than the index page itself —
        # the Notes column is probably stale and worth a manual look.
        if (
            readme_date is not None
            and new_date is not None
            and new_date > readme_date
        ):
            n_stale_warnings += 1

    if n_changed and not dry_run:
        readme_path.write_text("\n".join(updated_lines) + "\n")

    suffix = " (dry-run)" if dry_run else ""
    print(
        f"\n=== {n_changed} row(s) updated; "
        f"{n_stale_warnings} linked file(s) newer than the index "
        f"(notes-column may be stale){suffix} ===",
    )
    return n_changed


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
