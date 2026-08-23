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
#
# The plan's version only did (1). (2) matters more in practice: the pinned
# production clones pull separately, so a docs move that breaks a path
# referenced from a script goes unnoticed until that script runs at scale.
#
# Usage:  bash torchtitan/experiments/ezpz/scripts/check_doc_links.sh
# Exit 0 = clean, 1 = dangling references found (count printed).

set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
exec python3 - "$ROOT" <<'PYEOF'
import os, re, sys
root = sys.argv[1]
docs = os.path.join(root, "torchtitan/experiments/ezpz/docs")
if not os.path.isdir(docs):
    print("docs dir not found:", docs, file=sys.stderr); sys.exit(2)

bad = []

# 1. docs -> docs
for r, dirs, files in os.walk(docs):
    dirs[:] = [d for d in dirs if not d.startswith(".")]
    for fn in files:
        if not fn.endswith(".md"): continue
        p = os.path.join(r, fn)
        for m in re.finditer(r'\]\(([^)\s]+\.md)(?:#[^)]*)?\)',
                             open(p, errors="replace").read()):
            t = m.group(1)
            if t.startswith(("http://", "https://", "#")): continue
            if not os.path.exists(os.path.join(r, t)):
                bad.append("%s -> %s" % (os.path.relpath(p, root), t))

# 2. code -> docs
pat = re.compile(r'experiments/ezpz/docs/([A-Za-z0-9_./-]+\.md)')
seen = set()
for r, dirs, files in os.walk(os.path.join(root, "torchtitan/experiments/ezpz")):
    dirs[:] = [d for d in dirs if not d.startswith(".") and d != "docs"]
    for fn in files:
        if not fn.endswith((".py", ".sh")): continue
        p = os.path.join(r, fn)
        for m in pat.finditer(open(p, errors="replace").read()):
            key = (os.path.relpath(p, root), m.group(1))
            if key in seen: continue
            seen.add(key)
            if not os.path.exists(os.path.join(docs, m.group(1))):
                bad.append("(code) %s -> experiments/ezpz/docs/%s" % key)

if bad:
    print("DANGLING references (%d):" % len(bad))
    for b in sorted(bad): print("  ", b)
    sys.exit(1)
print("all internal doc links resolve")
PYEOF
