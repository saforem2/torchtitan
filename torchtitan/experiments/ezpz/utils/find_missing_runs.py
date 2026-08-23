#!/usr/bin/env python3
"""Find W&B runs that wrote a chain's checkpoint dir but are not registered.

`trajectories.py` carries a hand-maintained `wandb_run_ids` list per chain, and
a missing id is invisible: the chain still plots, still shows a live state on
the board, it just silently omits every step that run logged. That has now
produced three separate classes of wrongness -- chart gaps, a chain showing
state `R` with a six-day-old heartbeat, and a 21k-step fork with no entry at
all -- each found by accident rather than by looking.

This asks W&B the question directly instead. `metadata.args` records the literal
argv a run executed, so `--checkpoint.folder` says which chain a run belongs to,
authoritatively -- not the submit script, not the clone's defaults, not what the
run was named. Any run pointing at a chain's ckpt dir and absent from its
`wandb_run_ids` is a hole.

Read-only: it prints what to add, and never edits `trajectories.py`. Adding an
id is a judgement call (a smoke test or an aborted relaunch legitimately writes
the same dir), so it stays a human decision.

Usage (run on Aurora, where W&B credentials live):
    ./.venv/bin/python3 -m torchtitan.experiments.ezpz.utils.find_missing_runs
    ./.venv/bin/python3 -m torchtitan.experiments.ezpz.utils.find_missing_runs \\
        --chain 20b_v2_512 --days 30
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

PROJECT = "aurora_gpt/torchtitan.ezpz.train"


def _ckpt_arg(args: list) -> str | None:
    """Pull --checkpoint.folder out of a run's recorded argv.

    Handles both `--checkpoint.folder=VALUE` and the space-separated form.
    """
    for i, a in enumerate(args):
        if not isinstance(a, str):
            continue
        if a.startswith("--checkpoint.folder="):
            return a.split("=", 1)[1]
        if a == "--checkpoint.folder" and i + 1 < len(args):
            nxt = args[i + 1]
            if isinstance(nxt, str):
                return nxt
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", help="only check this trajectory key")
    ap.add_argument(
        "--days",
        type=int,
        default=60,
        help="how far back to scan W&B (default 60; older runs predate the "
        "current chain layout and mostly add noise)",
    )
    args = ap.parse_args()

    import wandb  # noqa: PLC0415

    from torchtitan.experiments.ezpz.utils.trajectories import (  # noqa: PLC0415
        TRAJECTORIES,
    )

    chains = [
        t
        for t in TRAJECTORIES
        if t.get("ckpt_dir")
        and t.get("cls") in ("live", "wandb_only")
        and (not args.chain or t["key"] == args.chain)
    ]
    if not chains:
        raise SystemExit(f"no chain matched {args.chain!r}")

    # Match on the ckpt dir's basename: the same logical chain is reachable by
    # several absolute prefixes (main repo vs a fork's own clone under
    # /flare/AuroraGPT/foremans/runs/), and runs record whichever path their
    # clone used.
    by_base = {os.path.basename(str(t["ckpt_dir"]).rstrip("/")): t for t in chains}

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    api = wandb.Api()
    runs = api.runs(PROJECT, filters={"createdAt": {"$gte": since.isoformat()}})

    found: dict[str, list] = {k: [] for k in by_base}
    n_scanned = 0
    for run in runs:
        n_scanned += 1
        meta = run.metadata or {}
        folder = _ckpt_arg(meta.get("args") or [])
        if not folder:
            continue
        base = os.path.basename(folder.rstrip("/"))
        if base in found:
            found[base].append(run)

    print(f"scanned {n_scanned} run(s) created since {since.date()}\n")
    total_missing = 0
    for base, traj in by_base.items():
        registered = set(traj.get("wandb_run_ids") or [])
        hits = found[base]
        missing = [r for r in hits if r.id not in registered]
        mark = "OK " if not missing else "GAP"
        print(
            f"[{mark}] {traj['key']:<32} {len(hits):>3} run(s) wrote this dir, "
            f"{len(registered):>3} registered, {len(missing):>3} missing"
        )
        for r in sorted(missing, key=lambda r: r.created_at):
            steps = r.summary.get("_step")
            print(
                f"        + {r.id}  {r.state:<9} created {str(r.created_at)[:16]}"
                f"  last _step={steps}"
            )
        total_missing += len(missing)

    if total_missing:
        print(
            f"\n{total_missing} unregistered run(s). Add the real ones to the "
            "chain's wandb_run_ids in utils/trajectories.py -- but check each "
            "first: a smoke test or an aborted relaunch can legitimately write "
            "the same ckpt dir without belonging to the trajectory."
        )
    else:
        print("\nNo gaps: every run writing these dirs is registered.")


if __name__ == "__main__":
    main()
