#!/usr/bin/env python3
"""Check that every chain's `wandb_run_ids` list is in chronological order.

`concat_chain` does `by_step[step] = row` walking the list in order, so when
two runs logged the SAME step the last one listed wins. That is deliberate: a
crashed run's overlapping steps get re-trained by its relaunch, and the
relaunch owns the surviving trajectory. It also makes list ORDER SEMANTIC.

Appending a newly-discovered run to the end of a list is therefore only safe if
that run is the newest. Appending an OLDER run silently puts its stale values
on top of newer ones -- measured 2026-08-17, when adding `djmhgmmq` (Aug 3) to
the end of `20b_v2_256` moved loss at steps 7,510-7,590 from ~2.33 to ~2.47
with no error anywhere. It surfaced only by diffing a re-exported store against
the committed one, which is not a reliable way to catch it.

An inversion only *matters* when the two runs actually share steps; otherwise
order is cosmetic. `--impact` checks that, at the cost of pulling history.

Usage (on Aurora, where W&B credentials live):
    ./.venv/bin/python3 -m torchtitan.experiments.ezpz.utils.audit_run_order
    ./.venv/bin/python3 -m torchtitan.experiments.ezpz.utils.audit_run_order --impact

Exit status is 1 if any inversion is found, so it can gate a chart or store
regeneration.
"""

from __future__ import annotations

import argparse

PROJECT = "aurora_gpt/torchtitan.ezpz.train"
LOSS_KEY = "loss_metrics/global_avg_loss"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", help="only audit this trajectory key")
    ap.add_argument(
        "--impact",
        action="store_true",
        help="for each inversion, pull history and report whether the two runs "
        "actually share steps (slow, but distinguishes a real value change "
        "from a cosmetic ordering wart)",
    )
    args = ap.parse_args()

    import wandb  # noqa: PLC0415

    from torchtitan.experiments.ezpz.utils.trajectories import (  # noqa: PLC0415
        TRAJECTORIES,
    )

    api = wandb.Api()
    hist_cache: dict[str, dict] = {}

    def history(rid: str) -> dict:
        if rid not in hist_cache:
            run = api.run(f"{PROJECT}/{rid}")
            hist_cache[rid] = {
                x["_step"]: x.get(LOSS_KEY)
                for x in run.scan_history(keys=["_step", LOSS_KEY])
                if x.get("_step") is not None and x.get(LOSS_KEY) is not None
            }
        return hist_cache[rid]

    total = 0
    for traj in TRAJECTORIES:
        if args.chain and traj["key"] != args.chain:
            continue
        ids = traj.get("wandb_run_ids") or []
        if len(ids) < 2:
            continue

        dated = []
        for rid in ids:
            try:
                dated.append((rid, api.run(f"{PROJECT}/{rid}").created_at))
            except Exception:
                # A purged or unreachable run cannot be ordered; skip it rather
                # than guess, and say so instead of silently dropping it.
                print(f"    (skipped {rid}: not reachable from the API)")
        inversions = [
            (dated[i], dated[i + 1])
            for i in range(len(dated) - 1)
            if dated[i][1] > dated[i + 1][1]
        ]
        tag = "OK " if not inversions else "OUT-OF-ORDER"
        print(
            f"[{tag}] {traj['key']:<32} {len(ids)} run(s), "
            f"{len(inversions)} inversion(s)"
        )
        total += len(inversions)

        for (a, ta), (b, tb) in inversions:
            print(f"         {a} ({str(ta)[:16]}) listed before {b} ({str(tb)[:16]})")
            if not args.impact:
                continue
            ha, hb = history(a), history(b)
            shared = sorted(set(ha) & set(hb))
            if not shared:
                print("           no shared steps -> cosmetic, no value changes")
                continue
            diffs = [s for s in shared if abs(ha[s] - hb[s]) > 1e-4]
            print(
                f"           {len(shared)} shared step(s), "
                f"{len(diffs)} with different values"
            )
            for s in diffs[:5]:
                print(
                    f"             step {s}: {ha[s]:.4f} (1st) vs "
                    f"{hb[s]:.4f} (2nd, WINS)"
                )

    if total:
        print(
            f"\n{total} inversion(s). Reorder the offending lists in "
            "trajectories.py so they read oldest -> newest. Only inversions "
            "with shared steps change any value (use --impact to tell), but "
            "fix them all: the next run appended to a mis-ordered list "
            "inherits the trap."
        )
        return 1
    print("\nEvery chain's run list is chronological.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
