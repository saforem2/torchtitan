#!/usr/bin/env python3
"""Move a docs subtree and repair every link by PATH RESOLUTION, not string edit.

Step 3 of docs/notes/docs-reorg-plan.md. String substitution is unsafe here:
"experiments/" also appears in the repo path torchtitan/experiments/ezpz, so a
sed sweep would rewrite real code paths. This resolves each relative link
against its containing file, and only rewrites links whose resolved target
actually moved -- which also fixes links INSIDE the moved tree, the bug class
that bit the earlier moves.

Usage:
  reorg_move.py <src-rel-to-docs> <dst-rel-to-docs> [--apply]
"""
import os
import re
import subprocess
import sys
from pathlib import Path

EZPZ = Path(__file__).resolve().parents[2]
DOCS = EZPZ / "docs"
REPO = EZPZ.parents[2]

# markdown inline links + reference definitions
LINK_RE = re.compile(r'(\]\(\s*)([^)\s]+?)(\s*(?:"[^"]*")?\s*\))')


def is_external(t: str) -> bool:
    return (
        t.startswith(("http://", "https://", "#", "mailto:", "<"))
        or t.startswith("/")
    )


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src_rel, dst_rel = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
    apply = "--apply" in sys.argv

    src, dst = DOCS / src_rel, DOCS / dst_rel
    if not src.is_dir():
        print(f"ERROR: {src} is not a directory")
        return 1
    if dst.exists():
        print(f"ERROR: {dst} already exists")
        return 1

    # map every moved file: old abs path -> new abs path
    moved = {}
    for p in src.rglob("*"):
        if p.is_file():
            moved[p.resolve()] = (dst / p.relative_to(src)).resolve()
    print(f"{len(moved)} files will move: {src_rel}/ -> {dst_rel}/")

    # every markdown file in the repo that might link into the tree
    md_files = [
        p for p in REPO.rglob("*.md")
        if ".git/" not in str(p) and "node_modules" not in str(p)
    ]

    def newloc(p: Path) -> Path:
        return moved.get(p.resolve(), p.resolve())

    edits = {}
    for md in md_files:
        try:
            text = md.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        md_new_dir = newloc(md).parent

        def fix(m):
            pre, target, post = m.group(1), m.group(2), m.group(3)
            if is_external(target):
                return m.group(0)
            frag = ""
            if "#" in target:
                target, frag = target.split("#", 1)
                frag = "#" + frag
            if not target:
                return m.group(0)
            try:
                resolved = (md.parent / target).resolve()
            except (OSError, RuntimeError):
                return m.group(0)
            tgt_new = moved.get(resolved)
            # rewrite if the TARGET moved, or if THIS FILE moved and the
            # relative path would otherwise break
            if tgt_new is None:
                if md.resolve() not in moved:
                    return m.group(0)
                tgt_new = resolved
                if not resolved.exists():
                    return m.group(0)
            rel = os.path.relpath(tgt_new, md_new_dir)
            if rel == target:
                return m.group(0)
            return f"{pre}{rel}{frag}{post}"

        new_text = LINK_RE.sub(fix, text)
        if new_text != text:
            edits[md] = new_text

    print(f"{len(edits)} markdown files need link rewrites")
    if not apply:
        for f in sorted(edits)[:15]:
            print(f"  would edit {f.relative_to(REPO)}")
        if len(edits) > 15:
            print(f"  ... and {len(edits)-15} more")
        print("\n(dry run -- pass --apply to execute)")
        return 0

    # git mv first so paths exist at their new home
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", str(src), str(dst)], cwd=REPO, check=True)

    for md, new_text in edits.items():
        path = newloc(md)
        path.write_text(new_text, encoding="utf-8")
    print(f"moved + rewrote {len(edits)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
