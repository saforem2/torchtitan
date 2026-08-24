#!/bin/bash
# Install the vendored Polaris bad-node scraper patterns into the active venv.
#
# WHY THIS EXISTS: ezpz is installed from a PINNED git commit and the venv is
# rebuilt/re-tarred periodically. A fix applied only to site-packages vanishes
# on the next rebuild, silently reverting Polaris failover to BLIND rotation
# (see torchtitan/experiments/ezpz/failover_patterns/README.md, job 7550301).
# Re-run this after any venv rebuild, BEFORE `ezpz tar-env`.
#
# Run from repo root:
#   bash torchtitan/experiments/ezpz/scripts/install_polaris_failover_patterns.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
SRC="${REPO_ROOT}/torchtitan/experiments/ezpz/failover_patterns/polaris.py"

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: vendored source not found: $SRC" >&2
    exit 1
fi

# Resolve the ezpz install location from the interpreter that will actually
# run the job, rather than guessing a python3.X path.
DST_DIR="$(python3 -c 'import ezpz, os; print(os.path.join(os.path.dirname(ezpz.__file__), "failover", "patterns"))')"

if [[ ! -d "$DST_DIR" ]]; then
    echo "ERROR: ezpz failover patterns dir not found: $DST_DIR" >&2
    echo "       Is the venv activated?" >&2
    exit 1
fi

cp "$SRC" "${DST_DIR}/polaris.py"
echo "installed -> ${DST_DIR}/polaris.py"

# Verify registration actually took effect. A silent no-op here would leave
# failover blind while looking like success, so gate on the ARTIFACT.
n=$(python3 -c 'from ezpz.failover.patterns import get_patterns_for_machine as g; print(len(g("polaris")))')
if [[ "$n" -eq 0 ]]; then
    echo "ERROR: polaris patterns still not registered (got 0)" >&2
    exit 1
fi
echo "verified: ${n} polaris patterns registered"
