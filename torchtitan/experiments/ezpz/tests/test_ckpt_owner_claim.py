#!/usr/bin/env python3
"""Test the checkpoint-directory owner claim.

Runs standalone (no pytest, no torch, no cluster):
    python3 torchtitan/experiments/ezpz/tests/test_ckpt_owner_claim.py
"""
import json
import os
import shutil
import sys
import tempfile
import time

# Load the module by PATH, not by package import: torchtitan.experiments.ezpz
# imports torch at package-init, and this module is deliberately stdlib-only so
# the test runs anywhere (a laptop with no torch, CI, a login node).
import importlib.util  # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "ckpt_owner_claim", os.path.join(_here, "..", "ckpt_owner_claim.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CLAIM_NAME = _mod.CLAIM_NAME
STALE_AFTER_SECONDS = _mod.STALE_AFTER_SECONDS
check_and_claim = _mod.check_and_claim
claim_path = _mod.claim_path
read_claim = _mod.read_claim

fails = []


def check(name, cond, detail=""):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


tmp = tempfile.mkdtemp(prefix="ckpt-claim-test-")
try:
    dump = os.path.join(tmp, "outputs")
    folder = "checkpoints/agpt-test"
    ckpt_dir = os.path.join(dump, folder)
    os.makedirs(ckpt_dir)

    print("=== path resolution ===")
    p = claim_path(folder, dump)
    check("claim path includes dump_folder", p == os.path.join(ckpt_dir, CLAIM_NAME), p)
    # The bug this guards against: omitting dump_folder pointed an earlier
    # detector at <cwd>/checkpoints/... while the tree is <cwd>/outputs/...
    check(
        "omitting dump_folder resolves somewhere ELSE",
        claim_path(folder, "") != p,
    )

    print()
    print("=== first claim on an unclaimed dir ===")
    os.environ["PBS_JOBID"] = "1111.aurora"
    collided = check_and_claim(folder, dump_folder=dump, world_size=3072)
    check("no collision reported", not collided)
    claim = read_claim(p)
    check("claim written", claim is not None)
    check("records jobid", claim and claim.get("jobid") == "1111.aurora")
    check("records world_size", claim and claim.get("world_size") == 3072)
    check("no .tmp left behind", not os.path.exists(p + ".tmp"))

    print()
    print("=== same job re-claiming (a resume) is NOT a collision ===")
    collided = check_and_claim(folder, dump_folder=dump, world_size=3072)
    check("same jobid -> no collision", not collided)

    print()
    print("=== a DIFFERENT job collides ===")
    os.environ["PBS_JOBID"] = "2222.aurora"
    collided = check_and_claim(folder, dump_folder=dump, world_size=3072)
    check("different jobid -> collision reported", collided)
    check(
        "claim is taken over by the newer job",
        (read_claim(p) or {}).get("jobid") == "2222.aurora",
    )

    print()
    print("=== a stale claim still reports, but is labelled stale ===")
    old = read_claim(p) or {}
    old["jobid"] = "3333.aurora"
    old["started"] = time.time() - (STALE_AFTER_SECONDS + 3600)
    with open(p, "w") as f:
        json.dump(old, f)
    os.environ["PBS_JOBID"] = "4444.aurora"
    collided = check_and_claim(folder, dump_folder=dump, world_size=3072)
    check("stale foreign claim still reports a collision", collided)

    print()
    print("=== non-rank-0 does nothing ===")
    before = read_claim(p)
    os.environ["PBS_JOBID"] = "5555.aurora"
    collided = check_and_claim(
        folder, dump_folder=dump, world_size=3072, is_rank_zero=False
    )
    check("non-rank-0 reports no collision", not collided)
    check("non-rank-0 does not rewrite the claim", read_claim(p) == before)

    print()
    print("=== robustness ===")
    with open(p, "w") as f:
        f.write("{not json at all")
    os.environ["PBS_JOBID"] = "6666.aurora"
    collided = check_and_claim(folder, dump_folder=dump, world_size=8)
    check("corrupt claim treated as absent (no crash, no collision)", not collided)
    check("corrupt claim is replaced", (read_claim(p) or {}).get("jobid") == "6666.aurora")

    missing = "checkpoints/does-not-exist"
    collided = check_and_claim(missing, dump_folder=dump, world_size=8)
    check("missing dir is a no-op, not a crash", not collided)
    check(
        "no claim written into a nonexistent dir",
        not os.path.exists(claim_path(missing, dump)),
    )
finally:
    shutil.rmtree(tmp, ignore_errors=True)
    os.environ.pop("PBS_JOBID", None)

print()
if fails:
    print(f"FAILED: {fails}")
    sys.exit(1)
print("ALL PASS")
