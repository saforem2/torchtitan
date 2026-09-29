#!/bin/bash
# Is the RoPE re-eval sweep actually finished?
#
# The right test is PER TASK: every targeted task at every post-switch step in
# the original eval dir must exist in the -ropefix result. Counting step dirs
# can report COMPLETE when each corrected file contains only one task.
#
# A raw count threshold gave a false COMPLETE on 2026-08-17: 32+34+8 crossed
# 74 while both 20B arms were only half done. 74 was the count of targeted
# checkpoints, not a finish line.
#
# Switch points: 20b_v2_512 step 4401, 20b_v2_256 step 3101,
# 2b_v2_512 step 30401 (see guides/known-bugs/rope-flavor-mismatch.md).
R="${R:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
R="$R" python3 <<'PYEOF'
import json
import os
from pathlib import Path

root = Path(os.environ["R"]) / "outputs" / "evals"
arms = (
    ("20b-v2-512n", 4401, {"arc_challenge", "hellaswag", "arc_easy"}),
    ("20b-v2-256n", 3101, {"arc_challenge", "hellaswag", "arc_easy"}),
    ("2b-v2-512n", 30401, {
        "arc_easy", "hellaswag", "winogrande", "piqa", "openbookqa",
        "boolq", "mmlu", "arc_challenge",
    }),
)

print(f"{'arm':<14} {'switch':<8} {'need':<6} {'have':<6} status")
for arm, switch, tasks in arms:
    need = have = 0
    for source in sorted((root / f"agpt-{arm}").glob("step-*/results/results.json")):
        try:
            step = int(source.parents[1].name.removeprefix("step-"))
        except ValueError:
            continue
        if step < switch:
            continue
        target = root / f"agpt-{arm}-ropefix" / f"step-{step}" / "results" / "results.json"
        try:
            corrected = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError):
            corrected = {}
        present = set(corrected.get("results", corrected))
        for task in tasks:
            need += 1
            if task in present or any(key.startswith(task + "_") for key in present):
                have += 1
    missing = need - have
    status = "COMPLETE" if missing == 0 else f"pending ({missing} task result{'s' if missing != 1 else ''} left)"
    print(f"{arm:<14} {switch:<8} {need:<6} {have:<6} {status}")
PYEOF
