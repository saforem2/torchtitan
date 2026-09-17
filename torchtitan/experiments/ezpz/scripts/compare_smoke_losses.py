"""Compare per-step losses between a merged-tree smoke and its pre-merge baseline.

Usage:
    python compare_smoke_losses.py <merged-logdir> <premerge-logdir>

Both runs must have used the same seed, the same configs in the same order, and
--debug.deterministic. Under those conditions the arms should agree BITWISE:
this sync changes arithmetic GROUPING (fused QKV / gate-up), and grouping only
moves results when the fused path is actually exercised, so any difference here
is a real finding rather than expected float noise.

Exit code is 0 when every arm matches, 1 otherwise, so this can gate a script.
"""

import re
import sys
from pathlib import Path

# observability/metrics.py:539-540 emits
#     f"{color.red}step: {step:2}  {color.green}loss: {global_avg_loss:8.5f}  "
# so ANSI codes land BETWEEN the label and its number. Strip ANSI first, then
# match -- a regex written against the rendered line will silently find nothing.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:\s*(\d+)\s+loss:\s*([0-9.]+)")


def arm_losses(logdir: Path) -> dict[str, list[tuple[int, str]]]:
    """Map arm label -> [(step, loss-as-written), ...]."""
    out: dict[str, list[tuple[int, str]]] = {}
    for f in sorted(logdir.glob("[0-9]-*.log")):
        text = ANSI_RE.sub("", f.read_text(errors="replace"))
        steps = [(int(s), v) for s, v in STEP_RE.findall(text)]
        # keep the first occurrence of each step; ranks can duplicate lines
        seen: dict[int, str] = {}
        for s, v in steps:
            seen.setdefault(s, v)
        out[f.stem] = sorted(seen.items())
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    merged, pre = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = arm_losses(merged), arm_losses(pre)

    arms = sorted(set(a) | set(b))
    if not arms:
        print("no arm logs found in either directory -- nothing to compare")
        return 1

    ok = True
    for arm in arms:
        la, lb = a.get(arm, []), b.get(arm, [])
        if not la or not lb:
            print(f"  {arm:28} MISSING  (merged={len(la)} steps, pre={len(lb)} steps)")
            ok = False
            continue
        if la == lb:
            tail = ", ".join(f"{s}:{v}" for s, v in la[:4])
            print(f"  {arm:28} IDENTICAL  {len(la)} steps  [{tail}]")
            continue
        ok = False
        print(f"  {arm:28} DIFFERS")
        for (sa, va), (sb, vb) in zip(la, lb):
            mark = "" if va == vb else "   <-- differs"
            print(f"      step {sa:>3}  merged={va:<12} pre={vb:<12}{mark}")

    print()
    print("RESULT: all arms identical" if ok else "RESULT: differences found")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
