#!/usr/bin/env python3
"""Flag stale STEP counts in the hand-narrated production dashboard.

check_stale_docs.sh's "Last updated" check only compares a doc's marker date
to its git-commit date -- it CANNOT catch content that drifted without the
file being re-committed. The top-level docs/production/README.md dashboard is
hand-narrated (not auto-filled by fill_trajectory_fields.py, whose rollup
propagator is intentionally NOT pointed at it -- its column schema differs and
once mangled the Loss cells). So its per-chain step counts can silently fall
behind disk truth while the marker still looks fresh (exactly what happened:
20B rows sat at 4,400 / 2,100 after disk reached 5,400 / 3,100).

This cross-checks the dashboard against disk. For each LIVE trajectory it
inspects only the dashboard's *table rows* that link that trajectory's leaf
README (rows start with '|'), and reads the step from the row's dedicated
step CELL -- the first cell that is a bare step-count like `**5,400**` or
`5,400 (persisted)`. Prose mentions of a step (e.g. "frozen since step-4400")
are ignored because they are not in a table cell. A row whose step cell does
not equal the disk-valid step -> [DRIFT]. Read-only; always exits 0.
"""
from __future__ import annotations

import re

from torchtitan.experiments.ezpz.utils.trajectories import (
    REPO_ROOT,
    live_trajectories,
)
from torchtitan.experiments.ezpz.utils.fill_trajectory_fields import (
    largest_valid_step,
)

DASHBOARD = REPO_ROOT / "torchtitan/experiments/ezpz/docs/production/README.md"

# A table cell that is *just* a step count: >=4 significant digits
# (optionally comma-grouped, optionally bolded, optionally followed by a
# "(persisted)"/"(in-RAM)" qualifier). The >=4-digit floor excludes the
# Nodes column (<=2048) and the Loss column (has a decimal point); matching
# the WHOLE cell between pipes excludes prose ("frozen since step-4400 ...").
# Steps in these tables span 1,125..92,859, so require either a comma group
# or >=4 bare digits.
_STEP_CELL = re.compile(
    r"^\s*\*{0,2}(\d{1,3}(?:,\d{3})+|\d{4,})\*{0,2}\s*(?:\((?:persisted|in-RAM)\))?\s*$"
)


def _link_suffix(traj: dict) -> str:
    r = traj["readme"]
    i = r.find("docs/production/")
    rel = r[i + len("docs/production/"):] if i != -1 else r
    return rel  # e.g. agpt/20b/n512/README.md


def _row_step(row: str) -> int | None:
    """Return the step from a markdown table row's dedicated step cell, or
    None if no cell is a bare step count."""
    for cell in row.split("|"):
        m = _STEP_CELL.match(cell)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def main() -> int:
    if not DASHBOARD.is_file():
        print("  (dashboard not found; skipping drift check)")
        return 0
    rows = [ln for ln in DASHBOARD.read_text().splitlines() if ln.lstrip().startswith("|")]
    drift = 0
    ok = 0
    unreadable: list[str] = []
    unlinked: list[str] = []
    # NB: live_trajectories() filters on Path(ckpt_dir).is_dir(), so on a
    # machine where the cluster filesystem is not mounted it returns [] and
    # every loop below is skipped. That made this tool print "0 fresh, 0
    # stale" -- indistinguishable from a clean dashboard -- while checking
    # nothing at all. Count that case explicitly instead.
    from torchtitan.experiments.ezpz.utils.trajectories import TRAJECTORIES

    live_records = live_trajectories()
    declared_live = [t for t in TRAJECTORIES if t.get("cls") == "live"]
    off_host = len(declared_live) - len(live_records)

    for traj in live_records:
        step = largest_valid_step(traj["ckpt_dir"])
        if step is None:
            # Could not read the checkpoint dir -- almost always because it
            # lives on a cluster filesystem not mounted here. Record it: a
            # trajectory we could not check is NOT a trajectory that is fresh,
            # and silently continuing let this tool report "0 fresh, 0 stale"
            # while every single chain was unverified.
            unreadable.append(traj["key"])
            continue
        link = _link_suffix(traj)
        linked_rows = [r for r in rows if link in r]
        if not linked_rows:
            unlinked.append(traj["key"])
            continue
        stale_here = []
        fresh_here = 0
        for r in linked_rows:
            cited = _row_step(r)
            if cited is None:
                continue  # a linked row with no bare step cell (e.g. links-only)
            if cited == step:
                fresh_here += 1
            else:
                stale_here.append(cited)
        if stale_here:
            print(f"  [DRIFT] {link}")
            print(f"    disk-valid step: {step:,}   dashboard step-cell(s): "
                  f"{', '.join(f'{c:,}' for c in stale_here)}")
            drift += 1
        elif fresh_here:
            ok += 1
    print(
        f"=== dashboard drift: {ok} fresh, {drift} stale, "
        f"{len(unreadable)} unreadable, {len(unlinked)} unlinked, "
        f"{off_host} off-host ==="
    )
    if off_host:
        print(
            f"  [OFF-HOST] {off_host} of {len(declared_live)} live trajectories "
            "have a ckpt_dir that is not mounted here, so they were NOT "
            "checked. Run this on the cluster to verify them."
        )
    if unreadable:
        print(
            "  [UNREADABLE] checkpoint dir not accessible from here -- these "
            "were NOT checked, which is not the same as fresh:"
        )
        for k in unreadable:
            print(f"    {k}")
    if unlinked:
        print("  [UNLINKED] no dashboard row references these trajectories:")
        for k in unlinked:
            print(f"    {k}")
    # Non-zero when nothing could be verified at all, so a caller cannot read
    # a clean exit as a clean dashboard.
    if ok == 0 and drift == 0 and (unreadable or off_host):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
