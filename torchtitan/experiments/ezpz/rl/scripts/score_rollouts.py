#!/usr/bin/env python3
"""Recover per-step GRPO metrics from a run's `rollout_samples.jsonl`.

Why this exists: a GRPO run whose LoRA checkpoints did not survive is
usually written off as unmeasurable -- you cannot merge and re-eval a
checkpoint that is gone. But the rollout log is a *complete* record of
what the policy actually emitted, scored by the same reward components
the trainer used, with a policy version stamped on every turn. That is
the real metric, recorded at generation time, and it survives the
checkpoints.

The `cot-long` run (400 steps, 2026-07-21) was documented for weeks as
"never evaluated on the real metric; likely drifting" purely because its
checkpoints were gone. Its rollout log had been sitting on disk the whole
time, and scoring it flatly contradicted the inference. See
`docs/live/chains/rl/plans/cot.md`.

What it reports, binned by policy version (the GRPO step):
  - `acc`       -- mean `AnswerCorrectReward`, i.e. the task metric
  - `format_ok` -- fraction emitting both a reasoning span and `<answer>`
  - `reward`    -- the shaped scalar GRPO actually optimized

`format_ok` is computed from the generated text rather than read from
`ThinkFormatReward`, because that component reads only `content`, and
vLLM strips `<think>...</think>` out into a separate `reasoning_content`
field. A run can therefore log `ThinkFormatReward == 0.0` on every single
rollout while emitting perfectly well-formed output. Treating that zero
as real reads as total format collapse. See
`memory/project_grpo_vllm_reasoning_content_split.md`.

Usage:
    python3 -m torchtitan.experiments.ezpz.rl.scripts.score_rollouts \\
        outputs/rl_lora_agpt2b_cot_long/rollout_samples.jsonl --bins 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> list[dict]:
    """Parse the rollout log into flat per-rollout records.

    Truncated trailing lines are skipped: the log is append-only and a
    killed job leaves a partial last line.
    """
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            breakdown = d.get("reward_breakdown") or {}
            acc = breakdown.get("AnswerCorrectReward")
            if acc is None:
                # Unscored rollout (crashed generation, filtered sample).
                continue
            turns = d.get("turns") or []
            # A turn's max_policy_version is the weight version that
            # produced it; the rollout's version is the newest turn.
            version = max((t.get("max_policy_version", 0) for t in turns), default=0)
            msg = (turns[0].get("completion_message") if turns else None) or {}
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            # Reasoning may arrive inline (<think> in content) or split
            # out by vLLM into reasoning_content -- accept either.
            reasoned = "<think>" in content or bool(reasoning.strip())
            rows.append(
                {
                    "version": version,
                    "is_validation": bool(d.get("is_validation")),
                    "acc": float(acc),
                    "reward": float(d.get("reward") or 0.0),
                    "format_ok": float(reasoned and "<answer>" in content),
                }
            )
    return rows


def _report(label: str, rows: list[dict], bins: int, vmax: int) -> None:
    if not rows:
        print(f"--- {label}: no rollouts")
        return
    width = max(1, (vmax + 1) // bins)
    print(f"--- {label}  (n={len(rows)})")
    print("  pv-range        n     acc   format_ok   reward")
    for b in range(bins):
        lo, hi = b * width, (b + 1) * width
        if b == bins - 1:
            hi = vmax + 1
        group = [r for r in rows if lo <= r["version"] < hi]
        if not group:
            continue
        n = len(group)

        def mean(key: str) -> float:
            return sum(r[key] for r in group) / n

        print(
            f"  {lo:>4}-{hi - 1:<5} {n:>6}  {mean('acc'):.4f}  "
            f"{mean('format_ok'):.4f}     {mean('reward'):.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rollouts", type=Path, help="path to rollout_samples.jsonl")
    ap.add_argument("--bins", type=int, default=10, help="policy-version buckets")
    args = ap.parse_args()

    rows = _load(args.rollouts)
    if not rows:
        raise SystemExit(f"No scored rollouts found in {args.rollouts}")
    vmax = max(r["version"] for r in rows)
    n_val = sum(1 for r in rows if r["is_validation"])
    print(f"scored={len(rows)}  validation={n_val}  policy versions 0..{vmax}\n")

    _report("TRAIN", [r for r in rows if not r["is_validation"]], args.bins, vmax)
    print()
    _report("VALIDATION", [r for r in rows if r["is_validation"]], args.bins, vmax)


if __name__ == "__main__":
    main()
