#!/usr/bin/env python3
"""Flag checkpoint dirs that two writers produced, and other shape anomalies.

Recommendation #5 from `docs/reference/known-bugs/concurrent-job-ckpt-collision.md`.
Episode 2 of that incident was found 18 days late by hand-counting files; both
detectors below would have caught it the same day, and they are cheap.

WHAT A COLLIDED DIR LOOKS LIKE (measured on the two real cases):

    step-6200-20260729-142222   shards 0-191     07-28 20:37   252,561,448 B
                                shards 192-3071  07-28 17:34    97,997,402 B

Two mtime clusters, the split falling **exactly on a shard index**, with
different file sizes -- a 192-rank job wrote the low shards over the top of a
3072-rank job's. That index-alignment is the discriminator. Straggler ranks
also produce an mtime gap (three dirs in the same tree gapped 108-121s), but
their late files are a handful scattered anywhere, not a contiguous prefix.
So this reports the split index and cluster sizes rather than just "gap
detected", and only calls it MIXED when the split is index-aligned.

Also flagged, in decreasing severity:
  - shard count differing from the tree's mode (a scale change, legal but
    never silent -- recommendation #3)
  - out-of-order mtimes across step dirs (a later step written earlier)
  - empty dirs / dirs with no `.metadata` (interrupted saves; benign, and
    already skipped by the eval sweeps, but worth counting)

Read-only. Never deletes or renames -- reclaiming 906 GB of orphan shards is a
human decision, and the project rule is `backup`, not `rm`.

COST: this stats every shard file, and the production tree is ~1,500 step dirs
holding up to 3,072 files each -- order 1.5M stat calls on Lustre. A single 20B
chain (120 dirs) took ~40 minutes; `--all-chains` is a multi-hour job. Run it
detached, scoped to one chain, or on a schedule -- not interactively while
waiting. Progress is printed per chain and every 25 dirs, and stdout is
flushed, because the first version printed only on completion and a 1h41m run
was indistinguishable from a hang.

Usage:
    python3 -m torchtitan.experiments.ezpz.utils.audit_ckpt_dirs <ckpt_dir>
    python3 -m torchtitan.experiments.ezpz.utils.audit_ckpt_dirs --all-chains

Exit status is 1 if any MIXED dir is found, so it can gate a resume.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
from collections import Counter

STEP_RE = re.compile(r"^step-(\d+)")
SHARD_RE = re.compile(r"^__(\d+)_0\.distcp$")

# A gap smaller than this is normal save spread across thousands of ranks.
# The real collisions were ~3 HOURS apart; the benign straggler cases were
# 108-121s. 600s sits well clear of both.
CLUSTER_GAP_SECONDS = 600


def scan_dir(path: str) -> dict:
    """Shape summary for one step dir. No file contents are read."""
    try:
        names = os.listdir(path)
    except OSError as e:
        return {"error": repr(e)}

    shards = []
    for n in names:
        m = SHARD_RE.match(n)
        if not m:
            continue
        full = os.path.join(path, n)
        try:
            st = os.stat(full)
        except OSError:
            continue
        shards.append((int(m.group(1)), st.st_mtime, st.st_size))

    return {
        "n_shards": len(shards),
        "has_metadata": ".metadata" in names,
        "shards": sorted(shards),
    }


def find_split(shards: list[tuple[int, float, int]]) -> dict | None:
    """Detect two mtime clusters and report whether the split is index-aligned.

    ``shards`` is sorted by index. Returns None when every file lands in one
    time cluster.
    """
    if len(shards) < 2:
        return None
    times = sorted(s[1] for s in shards)
    gaps = [(times[i + 1] - times[i], i) for i in range(len(times) - 1)]
    gap, at = max(gaps)
    if gap < CLUSTER_GAP_SECONDS:
        return None

    cut = times[at]
    early = [s for s in shards if s[1] <= cut]
    late = [s for s in shards if s[1] > cut]

    # Index-aligned means one cluster is a contiguous prefix of shard indices.
    early_idx = {s[0] for s in early}
    prefix = set(range(len(early_idx)))
    aligned = early_idx == prefix or {s[0] for s in late} == set(
        range(len({s[0] for s in late}))
    )
    # Report the MEDIAN size, not the mode. Shard sizes are far from uniform
    # WITHIN a single writer's cluster: in the real collision the 192-rank
    # writer ranges from ~1.31 GB (shard 0) down to 252 MB (shard 191), and the
    # 3072-rank writer from ~98 MB to ~10 MB. A mode is then whichever exact
    # byte count happens to repeat, which is arbitrary. The median still
    # separates the two writers by an order of magnitude, which is all this
    # column needs to do -- but do NOT read either number as "the shard size".
    # The mtime split and index alignment are the evidence; size is a hint.
    return {
        "gap_seconds": gap,
        "n_early": len(early),
        "n_late": len(late),
        "index_aligned": aligned,
        "early_size": int(statistics.median([s[2] for s in early])) if early else 0,
        "late_size": int(statistics.median([s[2] for s in late])) if late else 0,
    }


def audit(ckpt_dir: str) -> int:
    """Audit every step dir under ``ckpt_dir``. Returns the MIXED count."""
    if not os.path.isdir(ckpt_dir):
        print(f"  (no such directory: {ckpt_dir})")
        return 0

    step_dirs = sorted(
        (d for d in os.listdir(ckpt_dir) if STEP_RE.match(d)),
        key=lambda d: int(STEP_RE.match(d).group(1)),
    )
    if not step_dirs:
        print(f"  (no step-* dirs under {ckpt_dir})")
        return 0

    counts, empties, no_meta, mixed, gapped = Counter(), [], [], [], []
    last_mtime, out_of_order = None, []

    for i, d in enumerate(step_dirs):
        # Heartbeat every 25 dirs: each holds up to 3072 files to stat, so a
        # single chain can run for tens of minutes with nothing to show.
        if i and i % 25 == 0:
            print(f"    ... {i}/{len(step_dirs)} dirs scanned", flush=True)
        full = os.path.join(ckpt_dir, d)
        info = scan_dir(full)
        if info.get("error"):
            continue
        n = info["n_shards"]
        if n == 0:
            empties.append(d)
            continue
        counts[n] += 1
        if not info["has_metadata"]:
            no_meta.append(d)

        split = find_split(info["shards"])
        if split:
            (mixed if split["index_aligned"] else gapped).append((d, split))

        mt = max(s[1] for s in info["shards"])
        if last_mtime is not None and mt < last_mtime:
            out_of_order.append(d)
        last_mtime = mt

    mode = counts.most_common(1)[0][0] if counts else 0
    print(
        f"  {len(step_dirs)} step dir(s); dominant shard count = {mode}",
        flush=True,
    )
    for n, c in sorted(counts.items()):
        tag = "" if n == mode else "   <-- DIFFERENT SHARD COUNT (scale change?)"
        print(f"    {c:>4} dir(s) with {n} shards{tag}")
    if empties:
        print(f"    {len(empties)} empty (interrupted saves): {empties[:4]}")
    if no_meta:
        print(f"    {len(no_meta)} without .metadata: {no_meta[:4]}")
    if out_of_order:
        print(f"    OUT-OF-ORDER mtimes: {out_of_order[:6]}")

    for d, s in gapped:
        print(
            f"    [gap ] {d}: {s['gap_seconds']:.0f}s between clusters "
            f"({s['n_early']}/{s['n_late']}), NOT index-aligned -> stragglers"
        )
    for d, s in mixed:
        print(
            f"    [MIXED] {d}: {s['n_early']} shard(s) @ {s['early_size']:,}B "
            f"and {s['n_late']} @ {s['late_size']:,}B, "
            f"{s['gap_seconds'] / 3600:.1f}h apart, split on shard index "
            f"-> TWO WRITERS"
        )
    return len(mixed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_dir", nargs="?", help="a checkpoint dir to audit")
    ap.add_argument(
        "--all-chains",
        action="store_true",
        help="audit every ckpt_dir registered in trajectories.py",
    )
    args = ap.parse_args()

    targets = []
    if args.all_chains:
        from torchtitan.experiments.ezpz.utils.trajectories import (  # noqa: PLC0415
            TRAJECTORIES,
        )

        targets = [(t["key"], t["ckpt_dir"]) for t in TRAJECTORIES if t.get("ckpt_dir")]
    elif args.ckpt_dir:
        targets = [(os.path.basename(args.ckpt_dir.rstrip("/")), args.ckpt_dir)]
    else:
        ap.error("pass a ckpt_dir or --all-chains")

    total = 0
    for i, (name, path) in enumerate(targets, 1):
        # Announce BEFORE the scan and flush: a chain with thousands of step
        # dirs takes many minutes to stat on Lustre, and printing only on
        # completion made a 1h41m run look identical to a hang (measured).
        print(f"\n=== [{i}/{len(targets)}] {name}", flush=True)
        total += audit(path)

    if total:
        print(
            f"\n{total} MIXED dir(s): two jobs wrote the same directory and their "
            "shards are physically interleaved. Do NOT delete blind -- each may "
            "still hold a coherent checkpoint for its step; extract the "
            "referenced shards first. See "
            "docs/reference/known-bugs/concurrent-job-ckpt-collision.md."
        )
        return 1
    print("\nNo mixed directories found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
