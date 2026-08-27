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

# ---------------------------------------------------------------------------
# Patch the machine-DETECTION path, not just the patterns.
#
# Installing polaris.py is necessary but NOT sufficient. ezpz's
# failover/scrape.py resolves which pattern module to load via
# _detect_machine() -> ezpz.get_machine().lower(), and on Polaris that
# returns the FQDN of the login node, e.g.
#
#     polaris-login-04.hsn.cm.polaris.alcf.anl.gov
#
# The registry keys are bare machine names (aurora / perlmutter / polaris /
# sunspot), so the lookup raises
#
#     ModuleNotFoundError: No module named
#         'ezpz.failover.patterns.polaris-login-04'
#
# and the scraper falls back to blaming whatever the generic noise points at
# -- in job 7560196 that was "possible application crash on rank 0", i.e. the
# innocent rank-0 node, while both genuinely-bad nodes stayed in the
# allocation and attempt 2 failed identically.
#
# This is a SEPARATE bug from the blind-rotation one this script was written
# for. There the patterns were missing; here they are present and correct but
# UNREACHABLE. Both present as "failover named the wrong node".
#
# Normalizing to the longest registered key contained in the detected string
# keeps every other machine working (aurora/sunspot already return bare names,
# so the substring match is the identity there).
python3 - <<'PY'
import inspect
import pathlib

from ezpz.failover import scrape

path = pathlib.Path(inspect.getfile(scrape))
src = path.read_text()

MARK = "# --- torchtitan/ezpz: normalize FQDN -> registered machine key ---"
if MARK in src:
    print("detection patch already present")
    raise SystemExit(0)

OLD = """    try:
        return ezpz.get_machine().lower()
    except Exception:
        return ""
"""
NEW = '''    try:
        detected = ezpz.get_machine().lower()
    except Exception:
        return ""
    # --- torchtitan/ezpz: normalize FQDN -> registered machine key ---
    # get_machine() can return a login-node FQDN
    # ("polaris-login-04.hsn.cm.polaris.alcf.anl.gov"); registry keys are
    # bare ("polaris"). Match the longest registered key so "polaris" wins
    # over any shorter accidental substring.
    try:
        keys = _registered_machines()
    except Exception:
        return detected
    hits = [k for k in keys if k and k in detected]
    if hits:
        return max(hits, key=len)
    return detected
'''

if OLD not in src:
    raise SystemExit(
        "ERROR: could not find _detect_machine body to patch; ezpz "
        "changed upstream -- re-check scrape.py before trusting failover"
    )

path.write_text(src.replace(OLD, NEW, 1))
print(f"patched detection -> {path}")
PY

# Gate on the ARTIFACT: re-detect in a FRESH interpreter and require that the
# auto-detect path (no explicit machine=) now resolves to "polaris". The
# get_patterns_for_machine("polaris") check above passes an explicit key and
# therefore CANNOT catch this failure -- that blind spot is why job 7560196
# looped for 3 hours with a green-looking install.
detected=$(python3 -c 'from ezpz.failover.scrape import _detect_machine; print(_detect_machine())')
if [[ "$detected" != "polaris" ]]; then
    echo "ERROR: auto-detect still resolves to '${detected}', not 'polaris'" >&2
    echo "       failover would blame the wrong node -- do NOT launch" >&2
    exit 1
fi
echo "verified: auto-detect resolves to '${detected}'"
