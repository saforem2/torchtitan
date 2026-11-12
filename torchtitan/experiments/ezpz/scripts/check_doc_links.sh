#!/bin/bash
# Fail if any internal markdown link in experiments/ezpz/docs/ is dangling.
#
# Step 4 of docs/notes/docs-reorg-plan.md. This exists so step 3 (moving
# directories) is SAFE TO REPEAT: ~1100 internal links plus inbound references
# from code, scripts and CLAUDE.md mean a rename that misses one sweep is
# otherwise invisible until a reader hits a dead link.
#
# Checks BOTH directions:
#   1. docs -> docs   every ](...md) target resolves
#   2. code -> docs   every experiments/ezpz/docs/... path named in a .py/.sh
#                     still exists
#   3. depth-counted  no script UNDER docs/ computes a path by counting
#      paths          parents[N] -- that silently resolves elsewhere when the
#                     script moves and renders an EMPTY chart instead of
#                     failing, which (1) and (2) cannot see
#
# The plan's version only did (1). (2) matters more in practice: the pinned
# production clones pull separately, so a docs move that breaks a path
# referenced from a script goes unnoticed until that script runs at scale.
#
# Usage:  bash torchtitan/experiments/ezpz/scripts/check_doc_links.sh
# Exit 0 = clean, 1 = dangling references or unreadable paths (count printed).
# Links are checked case-sensitively, with optional titles, and #fragments are
# checked against the destination's GitHub-style heading slugs.

set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
exec python3 - "$ROOT" <<'PYEOF'
import os, re, sys
root = sys.argv[1]
docs = os.path.join(root, "torchtitan/experiments/ezpz/docs")
if not os.path.isdir(docs):
    print("docs dir not found:", docs, file=sys.stderr); sys.exit(2)

bad = []
walk_errors = []


def walk(top):
    """os.walk that records unreadable directories instead of skipping them."""
    return os.walk(top, onerror=walk_errors.append)


def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError as e:
        walk_errors.append(e)
        return ""


_listing = {}


def exists_exact(path):
    """os.path.exists, but case-sensitive on case-insensitive filesystems.

    The published URLs are case-sensitive, so a link that only resolves
    because macOS ignores case is still broken.
    """
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return False
    rel = os.path.relpath(path, root)
    if rel.startswith(".."):
        return True
    cur = root
    for part in rel.split(os.sep):
        if part in (".", ""):
            continue
        if cur not in _listing:
            try:
                _listing[cur] = set(os.listdir(cur))
            except OSError:
                return True
        if part not in _listing[cur]:
            return False
        cur = os.path.join(cur, part)
    return True


# ](target) or ](target "title") / 'title' / (title), optional <...> around target.
MD_LINK = re.compile(
    r"\]\(\s*<?([^)\s>]+)>?(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
)
URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
FENCE = re.compile(r"^\s*(```|~~~)")
_anchors = {}


def anchors(path):
    """GitHub-style heading slugs plus explicit id/name anchors in a .md file."""
    if path in _anchors:
        return _anchors[path]
    out, counts, fenced = set(), {}, False
    for line in read(path).splitlines():
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        for m in re.finditer(r'(?:id|name)=["\']([^"\']+)["\']', line):
            out.add(m.group(1))
        h = re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not h:
            continue
        text = re.sub(r"<[^>]+>", "", h.group(1))
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
        slug = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
        n = counts.get(slug, 0)
        counts[slug] = n + 1
        out.add(slug if n == 0 else "%s-%d" % (slug, n))
    _anchors[path] = out
    return out


def check_md_link(src, base_dir, target, md_only):
    """Append to bad if target (relative to base_dir) or its #fragment is missing."""
    if target.startswith(("http://", "https://", "#", "mailto:", "/")):
        return
    path, _, frag = target.partition("#")
    if not path or (md_only and not path.endswith(".md")):
        return
    dest = os.path.join(base_dir, path)
    if not exists_exact(dest):
        bad.append("%s -> %s" % (os.path.relpath(src, root), path))
    elif frag and dest.endswith(".md") and frag.lower() not in anchors(dest):
        bad.append("%s -> %s (no heading #%s)" % (os.path.relpath(src, root), path, frag))

# 1. docs -> docs
for r, dirs, files in walk(docs):
    dirs[:] = [d for d in dirs if not d.startswith(".")]
    for fn in files:
        if not fn.endswith(".md"): continue
        p = os.path.join(r, fn)
        for m in MD_LINK.finditer(read(p)):
            check_md_link(p, r, m.group(1), md_only=True)

# 1b. markdown OUTSIDE docs/ that links INTO it
# Check 1 only walks docs/. But experiments/ezpz/README.md is the landing page
# and its folder table is nothing but docs/ links -- after the reorg six of its
# nine rows pointed at retired directories and no check saw it, because the
# file lives one level up. Any .md under experiments/ezpz but outside docs/
# counts.
ezpz = os.path.join(root, "torchtitan/experiments/ezpz")
for r, dirs, files in walk(ezpz):
    dirs[:] = [d for d in dirs if not d.startswith(".") and d != "docs"]
    for fn in files:
        if not fn.endswith(".md"): continue
        p = os.path.join(r, fn)
        for m in MD_LINK.finditer(read(p)):
            check_md_link(p, r, m.group(1), md_only=False)

# 2. code -> docs
# Match BOTH reference spellings -- the full 'experiments/ezpz/docs/...' and
# the bare 'docs/...' most comments use. Requiring the long prefix was why
# eight stale references survived the earlier moves.
#
# Match .md files AND bare directory paths: a stale DIRECTORY reference is
# worse than a stale file one, because a writer (a plot script's output dir)
# silently recreates it rather than erroring.
pat = re.compile(
    r'(?:experiments/ezpz/)?docs/'
    r'([A-Za-z0-9_./-]+?\.md|[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*/)'
)
seen = set()
for r, dirs, files in walk(os.path.join(root, "torchtitan/experiments/ezpz")):
    dirs[:] = [d for d in dirs if not d.startswith(".") and d != "docs"]
    for fn in files:
        if not fn.endswith((".py", ".sh")): continue
        p = os.path.join(r, fn)
        src = read(p)
        # Skip lines opted out with `docs-link-check: ignore`. Some code names
        # a path that is SUPPOSED to be absent: a branch-tolerant resolver
        # lists both the pre- and post-move location and picks whichever
        # exists, and prose in a comment explaining the move names the retired
        # path by definition. Flagging those trains people to ignore the
        # checker, which costs more than the few false negatives an explicit,
        # greppable opt-out allows.
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if "docs-link-check: ignore" in line:
                continue
            for m in pat.finditer(URL.sub("", line)):
                key = (os.path.relpath(p, root), m.group(1))
                if key in seen: continue
                seen.add(key)
                if not exists_exact(os.path.join(docs, m.group(1))):
                    bad.append("(code) %s:%d -> experiments/ezpz/docs/%s"
                               % (key[0], i + 1, key[1]))

# 2b. BARE docs/ path mentions in prose and comments
# Checks 1/1b/2 all require a markdown link or a resolvable string reference.
# But most stale paths after a move are neither: they are a bare docs path
# written inline in a sentence, or a comment naming a directory. The link
# target next to them gets rewritten by path resolution while the visible text
# keeps saying the old location -- 147 such mentions survived the reorg, in 61
# files, including prose that named a chain subtree without its `chains/`
# level.
#
# Only flags a path whose FIRST segment names a real (or once-real) docs
# directory, so an unrelated docs path in an external URL and other coincidental
# matches stay quiet.
bare = re.compile(r'docs/([A-Za-z0-9_./-]+)')
known_tops = set(os.listdir(docs)) | {
    "production", "evals", "guides", "experiments", "summaries", "scaling",
    "configs", "meeting-notes", "upstream-issues", "competitions", "baselines",
    "journal", "upstream-sync",
}
for r, dirs, files in walk(os.path.join(root, "torchtitan/experiments/ezpz")):
    dirs[:] = [d for d in dirs if not d.startswith(".")]
    for fn in files:
        if not fn.endswith((".md", ".py", ".sh")): continue
        if fn in ("reorg_move.py", "check_doc_links.sh"): continue
        p2 = os.path.join(r, fn)
        for i, line in enumerate(read(p2).splitlines(), 1):
            if "docs-link-check: ignore" in line: continue
            # External URLs can contain docs/<top>/... paths that are not ours.
            for m in bare.finditer(URL.sub("", line)):
                rel = m.group(1).rstrip("/.,)`'\"*;:")
                if not rel or any(c in rel for c in "*{}$"): continue
                if rel.split("/")[0] not in known_tops: continue
                if exists_exact(os.path.join(docs, rel)): continue
                bad.append("(path) %s:%d -> docs/%s does not exist"
                           % (os.path.relpath(p2, root), i, rel))

# 3. PIECEWISE docs paths: a pathlib chain naming a retired dir segment.  docs-link-check: ignore
# Check 2 only sees a docs path written as ONE string. A pathlib chain built a
# segment at a time is invisible to it -- and that is not hypothetical: after
# the production/ split, three plotters kept writing to the retired tree
# because their output dir was assembled as / "production" / "agpt". Nothing
# raised; mkdir recreated the directory and the figures went there, newer than
# the ones anybody reads.
#
# So: flag a quoted segment that names a directory which no longer exists
# directly under docs/. Cheap and specific -- it only fires on names that USED
# to be real, which is exactly the post-move window where it matters.
# Anchor on a DOCS base. Without it this fires on outputs/evals/... too --
# a data directory that merely shares a name with a retired docs one.
seg = re.compile(
    r'(?:DOCS_BASE|"docs"|\'docs\'|/\s*"docs")\s*/\s*"([A-Za-z0-9_-]+)"'
)
live_dirs = {d for d in os.listdir(docs)
             if os.path.isdir(os.path.join(docs, d))}
retired = {"production", "evals", "guides", "experiments", "summaries",
           "meeting-notes", "scaling", "upstream-issues", "competitions",
           "configs"} - live_dirs
for r, dirs, files in walk(os.path.join(root, "torchtitan/experiments/ezpz")):
    dirs[:] = [d for d in dirs if not d.startswith(".")]
    for fn in files:
        if not fn.endswith((".py", ".sh")): continue
        p2 = os.path.join(r, fn)
        for i, line in enumerate(read(p2).splitlines(), 1):
            if "docs-link-check: ignore" in line: continue
            for m in seg.finditer(line):
                if m.group(1) in retired:
                    bad.append(
                        "(piecewise) %s:%d -> / \"%s\" names a retired docs "
                        "directory; a pathlib chain is invisible to the "
                        "string check" % (os.path.relpath(p2, root), i, m.group(1))
                    )

# 4. depth-counted repo-root paths in scripts under docs/
# A plotter that does Path(__file__).resolve().parents[N] keeps working until
# the file moves between directory levels; then it silently points somewhere
# else, loads nothing, and emits an empty figure that refresh_all.sh commits.
# Checks (1) and (2) are blind to it -- they only look at markdown links.
depth = re.compile(r'Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]')
for r, dirs, files in walk(docs):
    dirs[:] = [d for d in dirs if not d.startswith(".")]
    for fn in files:
        if not fn.endswith(".py"): continue
        p = os.path.join(r, fn)
        for m in depth.finditer(read(p)):
            bad.append(
                "(depth) %s -> parents[%s] breaks silently if this file moves; "
                "use a marker-anchored _repo_root() walk"
                % (os.path.relpath(p, root), m.group(1))
            )

if walk_errors:
    errs = sorted({str(e) for e in walk_errors})
    print("UNREADABLE paths (%d) -- results incomplete:" % len(errs))
    for e in errs: print("  ", e)
if bad:
    print("DANGLING references (%d):" % len(bad))
    for b in sorted(bad): print("  ", b)
    sys.exit(1)
if walk_errors:
    sys.exit(1)
print("all internal doc links resolve")
PYEOF
