#!/usr/bin/env python3
"""Warn when two live jobs share one checkpoint directory.

Two concurrent jobs writing the same `checkpoint.folder` is silent, and it
physically mixes their shards. It happened at least twice on `20b_v2_256`,
leaving `step-6200-20260729-142222` and `step-6300-20260729-142222` holding
**3,072 files each where a healthy 20B-256 checkpoint has 192** -- 906 GB
combined, roughly 560 GB of it orphan bytes from the losing writer. Nothing
logged a warning, nothing named the other writer, and nothing flagged the
mixed directories; they were found weeks later by counting files.

This drops a small `.owner` file in the checkpoint dir recording jobid, rank
count, and start time, and looks for someone else's live claim first. A
mismatch is reported LOUDLY on rank 0 and nowhere else.

WARN, NOT REFUSE -- deliberately. A stale claim is common and harmless: any job
killed by walltime, a node fault, or a PBS `-14` leaves its `.owner` behind,
and there is no reliable way from inside the job to tell a stale claim from a
live one (PBS state is not visible here, and a jobid can be recycled). Refusing
would turn every crashed predecessor into a failed resume, which is a worse
failure than the one being prevented. The claim is a diagnostic that makes the
collision visible in the first 30 seconds of a log rather than in a file count
weeks later.

Read `docs/reference/known-bugs/concurrent-job-ckpt-collision.md` for the full
forensics.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

CLAIM_NAME = ".owner"

# A claim older than this is almost certainly from a finished job -- production
# umbrellas request 12-24h. Past it, say the claim looks stale rather than
# implying a live collision, so the warning stays worth reading.
STALE_AFTER_SECONDS = 36 * 3600


def _jobid() -> str:
    """PBS jobid, or a marker that this is not a scheduled job."""
    return os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID") or "interactive"


def claim_path(folder: str, dump_folder: str = "") -> str:
    """Resolve the `.owner` path the same way the checkpointer resolves saves.

    `dump_folder` is PREPENDED to `folder` by the checkpointer; omitting it
    here would point the claim at `<cwd>/checkpoints/...` while the real tree
    is `<cwd>/outputs/checkpoints/...`. That exact mistake made an earlier
    detector silently no-op, so the argument is not optional in practice.
    """
    rel = os.path.join(dump_folder, folder) if dump_folder else folder
    base = rel if os.path.isabs(rel) else os.path.join(os.getcwd(), rel)
    return os.path.join(base, CLAIM_NAME)


def read_claim(path: str) -> dict[str, Any] | None:
    """Return the existing claim, or None if absent/unreadable.

    A corrupt claim is treated as absent: this is a diagnostic, and failing a
    job over an unparseable advisory file would be worse than the collision.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def check_and_claim(
    folder: str,
    *,
    dump_folder: str = "",
    world_size: int | None = None,
    is_rank_zero: bool = True,
) -> bool:
    """Warn about a foreign claim on `folder`, then record ours.

    Returns True if a DIFFERENT jobid already held the directory. Only rank 0
    writes or warns -- thousands of ranks racing on one small file would be
    both noisy and a needless filesystem hot spot.
    """
    if not is_rank_zero:
        return False

    path = claim_path(folder, dump_folder)
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        # First save creates the directory; nothing to collide with yet.
        return False

    ours = _jobid()
    existing = read_claim(path)
    collided = False

    if existing and existing.get("jobid") and existing["jobid"] != ours:
        age = time.time() - float(existing.get("started", 0) or 0)
        stale = age > STALE_AFTER_SECONDS
        logger.warning(
            "CHECKPOINT DIR ALREADY CLAIMED by jobid=%s (%d rank(s), started "
            "%s, %.1fh ago)%s -- this job is %s. %s",
            existing["jobid"],
            existing.get("world_size", -1),
            existing.get("started_iso", "?"),
            age / 3600.0,
            " [likely STALE]" if stale else " [possibly LIVE]",
            ours,
            "A crashed predecessor leaves its claim behind, so this is probably "
            "harmless. But if the other job IS still running, both are writing "
            "the same directory and their shards will physically mix -- that "
            "produced two 453 GB dirs holding 3,072 files where 192 belong. "
            "Check `qstat` for the other jobid before letting this run.",
        )
        collided = True

    payload = {
        "jobid": ours,
        "world_size": world_size if world_size is not None else -1,
        "started": time.time(),
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "folder": folder,
    }
    try:
        # Write-then-rename so a reader never sees a half-written claim.
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001
        # Never fail a training job over an advisory file.
        logger.warning("could not write checkpoint claim %s: %r", path, e)

    return collided
