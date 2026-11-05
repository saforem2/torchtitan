#!/usr/bin/env python3
"""Compare loss and grad-norm histories of TT runs from TensorBoard events.

Reads full-precision scalars (the stdout log rounds to a few digits), using
the tags of core ``scripts/loss_compare.py``. Each ``--run`` is NAME=DIR,
where DIR is a run dump folder; all event files below it are read.

    python compare_agpt_tb_runs.py --reference a --run a=OUT_A --run b=OUT_B
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TAGS = {"loss": "loss_metrics/global_avg_loss", "grad_norm": "grad_norm"}


def read_scalars(directory: Path) -> dict[str, dict[int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_dirs = sorted({path.parent for path in directory.rglob("events.out.tfevents.*")})
    if not event_dirs:
        raise FileNotFoundError(f"no TensorBoard events below {directory}")
    result: dict[str, dict[int, float]] = {key: {} for key in TAGS}
    for event_dir in event_dirs:
        accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = set(accumulator.Tags().get("scalars", []))
        for key, tag in TAGS.items():
            if tag in tags:
                for event in accumulator.Scalars(tag):
                    if event.step in result[key] and result[key][event.step] != event.value:
                        raise ValueError(f"{directory}: conflicting {tag} at step {event.step}")
                    result[key][event.step] = event.value
    return result


def compare(reference: dict, other: dict) -> dict:
    report = {}
    for key in TAGS:
        steps = sorted(set(reference[key]) & set(other[key]))
        if not steps:
            report[key] = None
            continue
        deltas = [other[key][s] - reference[key][s] for s in steps]
        absolute = [abs(d) for d in deltas]
        worst = max(range(len(steps)), key=absolute.__getitem__)
        rel = [abs(d) / abs(reference[key][s]) for d, s in zip(deltas, steps) if reference[key][s]]
        entry = {
            "steps": [steps[0], steps[-1], len(steps)],
            "bitwise_equal_steps": sum(d == 0 for d in deltas),
            "first_differing_step": next((s for s, d in zip(steps, deltas) if d != 0), None),
            "max_abs_delta": absolute[worst],
            "max_abs_delta_step": steps[worst],
            "max_rel_delta": max(rel) if rel else None,
            "mean_abs_delta": sum(absolute) / len(absolute),
            "final_delta": deltas[-1],
            "mean_delta_last10": sum(deltas[-10:]) / len(deltas[-10:]),
            "rms_delta_last10": math.sqrt(sum(d * d for d in deltas[-10:]) / len(deltas[-10:])),
        }
        for threshold in (1e-4, 1e-3, 1e-2):
            entry[f"first_step_abs_delta_gt_{threshold:g}"] = next(
                (s for s, d in zip(steps, absolute) if d > threshold), None
            )
        report[key] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=DIR")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-steps", type=int, nargs="*", default=[1, 2, 5, 10, 25, 50, 75, 100])
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        name, _, directory = item.partition("=")
        runs[name] = read_scalars(Path(directory))
    reference = runs[args.reference]
    result = {"reference": args.reference, "runs": {}, "series": {}}
    for name, scalars in runs.items():
        result["series"][name] = {key: {str(s): v for s, v in sorted(values.items())} for key, values in scalars.items()}
        if name != args.reference:
            result["runs"][name] = compare(reference, scalars)
    print(json.dumps(result["runs"], indent=2))
    for key in TAGS:
        print(f"\n{key}: " + " ".join(f"{name:>14}" for name in runs))
        for step in args.print_steps:
            values = [runs[name][key].get(step) for name in runs]
            print(f"  step {step:>4}: " + " ".join(f"{v:14.9f}" if v is not None else f"{'-':>14}" for v in values))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
