#!/usr/bin/env python3
"""Which RoPE flavor was a given checkpoint step trained with?

The agpt chains changed RoPE convention mid-flight: commit 5ffb850a1
(2026-06-25) flipped ``CONFIG_SUFFIX`` to ``_real`` and the running chains
picked it up on their next resume. The checkpoint records nothing about this
(both rope caches are ``persistent=False``), so converting with the wrong
flavor silently corrupts the export -- it loads fine and only fails as
gibberish at generation.

W&B run metadata is the authoritative record: every run stores the exact argv
it executed, including ``--config=agpt_2b`` vs ``--config=agpt_2b_real``. This
walks a chain's runs, maps step ranges to the flavor in force, and answers for
a specific step.

Usage::

    # what flavor for step 9100 of the 20B-512 chain?
    python3 rope_flavor_for_step.py --chain 20b_v2_512 --step 9100

    # print the whole step -> flavor map for a chain
    python3 rope_flavor_for_step.py --chain 20b_v2_512

    # every chain, transitions only
    python3 rope_flavor_for_step.py --all

Exit codes: 0 answered, 2 bad usage, 3 could not determine (W&B unreachable,
or the step falls in a gap between runs).
"""

from __future__ import annotations

import argparse
import os
import sys

# trajectories.py owns the chain -> run-id lists
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "utils"))

WANDB_PATH = "aurora_gpt/torchtitan.ezpz.train"


def _flavor_of(run) -> str | None:
    """Extract the flavor from a run's recorded argv.

    metadata.args is what the process actually ran, which is why this is
    trusted over config.model_spec (equivalent here, but a level further from
    the command line) and over any script default (which lies -- a clone can
    carry several submit scripts and the one you read may not be the one that
    ran).
    """
    md = getattr(run, "metadata", None) or {}
    for arg in md.get("args") or []:
        if arg.startswith("--config=agpt_"):
            cfg = arg.split("=", 1)[1]
            # agpt_2b -> 2b ; agpt_20b_real -> 20b_real
            return cfg[len("agpt_") :]
    return None


def _step_range(run) -> tuple[int, int] | None:
    """First and last logged step of a run, or None if it logged nothing."""
    steps = [
        r["_step"]
        for r in run.scan_history(keys=["_step"], page_size=5000)
        if r.get("_step") is not None
    ]
    return (min(steps), max(steps)) if steps else None


def build_map(api, run_ids: list[str], *, want_steps: bool) -> list[dict]:
    """[{run_id, flavor, created, lo, hi}] in chain order."""
    out = []
    for rid in run_ids:
        try:
            run = api.run(f"{WANDB_PATH}/{rid}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            out.append({"run_id": rid, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rec = {
            "run_id": rid,
            "flavor": _flavor_of(run),
            "created": (getattr(run, "created_at", "") or "")[:10],
            "lo": None,
            "hi": None,
        }
        if want_steps:
            rng = _step_range(run)
            if rng:
                rec["lo"], rec["hi"] = rng
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", help="trajectory key, e.g. 20b_v2_512")
    ap.add_argument("--step", type=int, help="checkpoint step to resolve")
    ap.add_argument("--all", action="store_true", help="summarise every live chain")
    args = ap.parse_args()

    if not args.chain and not args.all:
        ap.error("pass --chain <key> (optionally with --step), or --all")

    try:
        import wandb
    except ImportError:
        print("wandb not importable; cannot resolve flavor", file=sys.stderr)
        return 3
    try:
        import trajectories as T
    except ImportError as exc:
        print(f"cannot import trajectories.py: {exc}", file=sys.stderr)
        return 3

    api = wandb.Api(timeout=40)
    chains = {
        t["key"]: (t.get("wandb_run_ids") or [])
        for t in T.TRAJECTORIES
        if t.get("wandb_run_ids")
    }

    if args.all:
        for key, ids in chains.items():
            recs = build_map(api, ids, want_steps=False)
            flavors = [r.get("flavor") for r in recs if r.get("flavor")]
            if not flavors:
                print(f"{key:22s} <no flavor recorded>")
                continue
            if len(set(flavors)) == 1:
                print(f"{key:22s} {flavors[0]} (constant across {len(flavors)} runs)")
            else:
                print(f"{key:22s} SWITCHED:")
                prev = None
                for r in recs:
                    f = r.get("flavor")
                    if f and f != prev:
                        print(f"    {r['created']}  {r['run_id']}  -> {f}")
                        prev = f
        return 0

    if args.chain not in chains:
        print(f"unknown chain {args.chain!r}; known: {', '.join(sorted(chains))}", file=sys.stderr)
        return 2

    recs = build_map(api, chains[args.chain], want_steps=True)

    if args.step is None:
        print(f"{'run':10s} {'created':11s} {'steps':>18s}  flavor")
        for r in recs:
            if r.get("error"):
                print(f"{r['run_id']:10s} {'':11s} {'':>18s}  ERROR {r['error']}")
                continue
            rng = f"{r['lo']}..{r['hi']}" if r["lo"] is not None else "-"
            print(f"{r['run_id']:10s} {r['created']:11s} {rng:>18s}  {r['flavor']}")
        return 0

    # Resolve one step. Later runs win: a chain that resumes re-logs earlier
    # steps, and the most recent writer is the one whose weights are on disk.
    hit = None
    for r in recs:
        if r.get("lo") is None or not r.get("flavor"):
            continue
        if r["lo"] <= args.step <= r["hi"]:
            hit = r
    if hit is None:
        print(
            f"could not place step {args.step} in any run of {args.chain}. "
            "It may predate W&B logging, or fall in a gap between runs -- "
            "check the registry in docs/reference/known-bugs/rope-flavor-mismatch.md",
            file=sys.stderr,
        )
        return 3

    print(hit["flavor"])
    print(
        f"# step {args.step} of {args.chain} was trained by run {hit['run_id']} "
        f"({hit['created']}, steps {hit['lo']}..{hit['hi']})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
