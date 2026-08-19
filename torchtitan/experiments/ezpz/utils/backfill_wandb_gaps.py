#!/usr/bin/env python3
"""Create one synthetic W&B run per gap so the UI curves are continuous.

WHY: 4,407 metric points exist only in PBS `.o` logs. Our charts show them --
`concat_chain` merges `.o` records with W&B history locally -- but that merge
never leaves the machine, so the W&B workspace still shows broken curves. A run
that crashed without flushing its tail leaves a hole nothing in the UI can fill.

WHAT: for each contiguous run of `.o`-only steps, create ONE new W&B run and
replay those records at their original `_step`. One run per gap keeps
provenance clean and mutates nothing historical.

THE CONFIG IS CLONED, NOT REBUILT. A real run carries 43 top-level config keys
and a 283-entry `env` subtree. Hand-writing the dozen keys a saved view happens
to filter on produces a run that passes the filters and is wrong everywhere
else -- and worse, silently: it would sit in the workspace looking legitimate
while missing `parallelism`, `optimizer.lr`, `torch_version`, and so on. So we
copy the config of a REAL run from the same chain and override only what is
genuinely gap-specific.

Every synthetic run is marked `backfill: true` in config and tagged
`backfill` + `synthetic`. No filter in the production view references either,
so the view is unaffected -- but nobody later has to guess which curves were
measured and which were replayed.

Usage:
    python3 -m torchtitan.experiments.ezpz.utils.backfill_wandb_gaps --dry-run
    python3 -m torchtitan.experiments.ezpz.utils.backfill_wandb_gaps --chain 2b_v2_256
    python3 -m torchtitan.experiments.ezpz.utils.backfill_wandb_gaps --execute
"""

from __future__ import annotations

import argparse
import copy

PROJECT = "aurora_gpt/torchtitan.ezpz.train"

# A real, healthy run per chain whose config we clone. Chosen because each is a
# normal production run of that chain -- same parallelism, optimizer, dataset.
TEMPLATE_RUN = {
    "20b_v2_512": "ctbs1be4",
    "20b_v2_256": "lc9oukel",
    "2b_v2_256": "fm3gzdxt",
}

# Config keys that MUST NOT be inherited from the template: they describe the
# template's own execution, not the gap's. Left in place they would attribute a
# backfill to the wrong PBS job and host.
PER_RUN_KEYS = ("jobid", "hostname", "tstamp", "dist")


def _gap_runs(records, max_stride):
    """Split .o-only records into contiguous gaps.

    A jump larger than `max_stride` means a separate hole, not one long one --
    logging cadence varies per chain, so the caller supplies the threshold.
    """
    if not records:
        return []
    groups, cur = [], [records[0]]
    for prev, rec in zip(records, records[1:]):
        if rec["_step"] - prev["_step"] > max_stride:
            groups.append(cur)
            cur = []
        cur.append(rec)
    groups.append(cur)
    return groups


def build_config(template_cfg, chain, first_step, last_step):
    """Clone the template config, strip per-run identity, mark as backfill."""
    cfg = copy.deepcopy(dict(template_cfg))
    for k in PER_RUN_KEYS:
        cfg.pop(k, None)
    cfg["backfill"] = True
    cfg["backfill_source"] = "pbs-olog"
    cfg["backfill_chain"] = chain
    cfg["backfill_step_range"] = f"{first_step}-{last_step}"
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", help="only this trajectory key")
    ap.add_argument("--max-stride", type=int, default=500,
                    help="step jump above which a gap is split in two")
    ap.add_argument("--execute", action="store_true",
                    help="actually create runs (default is a dry run)")
    args = ap.parse_args()

    import wandb  # noqa: PLC0415

    from torchtitan.experiments.ezpz.utils.trajectories import (  # noqa: PLC0415
        TRAJECTORIES,
    )
    from torchtitan.experiments.ezpz.utils.wandb_fetch import (  # noqa: PLC0415
        OLOG_KEYS,
        concat_chain,
        parse_olog,
    )

    api = wandb.Api()
    total_runs = total_pts = 0

    for traj in TRAJECTORIES:
        key = traj["key"]
        if args.chain and key != args.chain:
            continue
        fallbacks = traj.get("olog_fallbacks") or {}
        if not fallbacks or key not in TEMPLATE_RUN:
            continue

        # Steps W&B already has for this chain -- anything here needs no backfill.
        merged = concat_chain(
            traj["wandb_run_ids"], None, OLOG_KEYS, api=api, log=lambda *_: None
        )
        have = {int(r["_step"]) for r in merged if r.get("_step") is not None}

        # Steps a PREVIOUS backfill already published. Without this the tool is
        # not idempotent: re-running after registering one new .o fallback
        # would recreate every earlier synthetic run and double-publish their
        # points. The trajectory's wandb_run_ids do not list synthetic runs, so
        # concat_chain above cannot see them -- they have to be queried by tag.
        for prev in api.runs(
            PROJECT, filters={"config.backfill": True, "config.backfill_chain": key}
        ):
            have |= {
                int(x["_step"])
                for x in prev.scan_history(keys=["_step"])
                if x.get("_step") is not None
            }

        olog_records, _ = parse_olog(list(fallbacks.values()))
        missing = [r for r in olog_records
                   if r.get("_step") is not None and int(r["_step"]) not in have]
        if not missing:
            print(f"{key}: nothing missing")
            continue

        template = api.run(f"{PROJECT}/{TEMPLATE_RUN[key]}")
        gaps = _gap_runs(sorted(missing, key=lambda r: r["_step"]), args.max_stride)
        print(f"\n=== {key}: {len(missing)} missing point(s) in {len(gaps)} gap(s)"
              f"  [template {TEMPLATE_RUN[key]}, {len(dict(template.config))} cfg keys]")

        for g in gaps:
            lo, hi = int(g[0]["_step"]), int(g[-1]["_step"])
            name = f"backfill-{key}-{lo}-{hi}"
            print(f"  {name}: {len(g)} point(s)")
            total_runs += 1
            total_pts += len(g)
            if not args.execute:
                continue

            run = wandb.init(
                project=PROJECT.split("/", 1)[1],
                entity=PROJECT.split("/", 1)[0],
                name=name,
                config=build_config(template.config, key, lo, hi),
                tags=["backfill", "synthetic", key],
                notes=(
                    f"Synthetic backfill for {key} steps {lo}-{hi}. These points "
                    "exist only in the PBS .o log -- their original run crashed "
                    "without flushing. Config cloned from real run "
                    f"{TEMPLATE_RUN[key]}; metrics replayed at original _step."
                ),
                reinit=True,
            )
            for rec in g:
                payload = {k: v for k, v in rec.items()
                           if k != "_step" and v is not None}
                if payload:
                    run.log(payload, step=int(rec["_step"]))
            run.finish()

    verb = "created" if args.execute else "WOULD create"
    print(f"\n{verb} {total_runs} run(s) covering {total_pts} point(s)")
    if not args.execute:
        print("dry run -- pass --execute to actually create them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
