# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare arms of the master-weight-dtype ablation.

Reads the per-arm JSON dumps written by ``fp32_norms_ablation.py`` and prints
the three tables the question needs:

1. Norm-weight liveness per arm -- ``frac_changed`` and variance for every
   RMSNorm.weight. The v1 bug signature is ``frac_changed == 0`` /
   ``var == 0``.
2. Non-norm parameter agreement between arms B and C. This is the actual open
   question: if B matches C on the bulk parameters, norms-only fp32 is
   sufficient; if they diverge, the full fp32 master is doing real work.
3. Loss curves and wall-clock per arm.

Usage:
    python3 compare_fp32_norms_ablation.py armA.json armB.json armC.json
"""

from __future__ import annotations

import json
import sys


def _load(paths: list[str]) -> dict[str, dict]:
    arms = {}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        arms[d["arm"]] = d
    return arms


def _is_norm(name: str) -> bool:
    return "norm" in name.split(".")[-2:][0] or name.endswith("norm.weight")


def _by_name(arm: dict) -> dict[str, dict]:
    return {p["name"]: p for p in arm["params"]}


def _exact_divergence(paths: list[str], arms: dict[str, dict]) -> None:
    """Element-wise arm-vs-arm comparison from the optional .pt dumps.

    Reports, per arm pair, the max/mean relative difference over non-norm
    parameters. Requires ``--dump-params`` on the ablation runs; silently
    skipped otherwise.
    """
    import os

    try:
        import torch
    except ImportError:
        return

    tensors: dict[str, dict] = {}
    for p in paths:
        pt = p + ".pt"
        if os.path.exists(pt):
            with open(pt, "rb"):
                pass
            d = json.load(open(p))
            tensors[d["arm"]] = torch.load(pt, map_location="cpu")
    if len(tensors) < 2:
        return

    print()
    print("=" * 78)
    print("2b. EXACT ELEMENT-WISE DIVERGENCE (from --dump-params .pt files)")
    print("=" * 78)
    have = [a for a in ("A", "B", "C") if a in tensors]
    for i, a1 in enumerate(have):
        for a2 in have[i + 1 :]:
            t1, t2 = tensors[a1], tensors[a2]
            for group, keep in (
                ("non-norm", lambda n: not _is_norm(n)),
                ("norm", _is_norm),
            ):
                names = [n for n in t1 if n in t2 and keep(n)]
                if not names:
                    continue
                max_rel = 0.0
                max_name = ""
                tot_abs = 0.0
                tot_n = 0
                n_bitident = 0
                for n in names:
                    x, y = t1[n].double(), t2[n].double()
                    diff = (x - y).abs()
                    denom = y.abs().clamp_min(1e-12)
                    rel = (diff / denom).max().item()
                    if rel > max_rel:
                        max_rel, max_name = rel, n
                    tot_abs += diff.sum().item()
                    tot_n += diff.numel()
                    if torch.equal(x, y):
                        n_bitident += 1
                print(
                    f"  {a1} vs {a2} [{group:<8}] "
                    f"bit-identical {n_bitident}/{len(names)} tensors; "
                    f"max rel diff {max_rel:.3e}"
                    + (f" ({max_name[-40:]})" if max_name else "")
                    + f"; mean abs diff {tot_abs/max(tot_n,1):.3e}"
                )


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    arms = _load(sys.argv[1:])
    order = [a for a in ("A", "B", "C") if a in arms]

    print("=" * 78)
    print("ARMS")
    print("=" * 78)
    for a in order:
        d = arms[a]
        print(
            f"  {a}: training.dtype={d['training_dtype']:<9} "
            f"fp32_norms={d['fp32_norms']}  steps={d['steps']}  "
            f"seed={d['seed']}  lr={d['lr']}  wall={d['wall_seconds']:.1f}s "
            f"({d['sec_per_step']*1000:.0f} ms/step)"
        )
        if d.get("peak_memory"):
            print(
                f"      peak memory: {d['peak_memory']['max_reserved_gib']:.3f} GiB "
                f"({d['peak_memory']['max_reserved_pct']:.2f}%)"
            )

    print()
    print("=" * 78)
    print("1. NORM WEIGHTS -- frac_changed (0.0 == FROZEN == the v1 bug) / variance")
    print("=" * 78)
    names = [p["name"] for p in arms[order[0]]["params"] if _is_norm(p["name"])]
    hdr = f"{'parameter':<58}"
    for a in order:
        hdr += f"{('arm ' + a):>19}"
    print(hdr)
    for n in names:
        row = f"{n[-57:]:<58}"
        for a in order:
            p = _by_name(arms[a]).get(n)
            if p is None:
                row += f"{'-':>19}"
            else:
                row += f"{p.get('frac_changed', float('nan')):>8.3f} {p['var']:>10.2e}"
        print(row)

    # Per-arm summary over norm weights.
    print()
    print(f"{'arm':<6}{'dtype':<12}{'n_frozen':>10}{'n_norms':>10}{'mean var':>14}{'std(all)':>12}")
    for a in order:
        ps = [p for p in arms[a]["params"] if _is_norm(p["name"])]
        frozen = sum(1 for p in ps if p.get("frac_changed", 0) == 0.0)
        mean_var = sum(p["var"] for p in ps) / max(len(ps), 1)
        mean_std = sum(p["std"] for p in ps) / max(len(ps), 1)
        print(
            f"{a:<6}{ps[0]['dtype'].replace('torch.',''):<12}"
            f"{frozen:>10}{len(ps):>10}{mean_var:>14.3e}{mean_std:>12.3e}"
        )

    print()
    print("=" * 78)
    print("2. NON-NORM PARAMETERS -- does B match C? (the open question)")
    print("=" * 78)
    if "B" in arms and "C" in arms:
        b, c = _by_name(arms["B"]), _by_name(arms["C"])
        nonnorm = [n for n in b if not _is_norm(n)]
        print(
            f"{'parameter':<58}{'B std':>12}{'C std':>12}"
            f"{'rel diff':>11}{'B fc':>8}{'C fc':>8}"
        )
        worst = 0.0
        worst_name = ""
        for n in sorted(nonnorm):
            pb, pc = b[n], c.get(n)
            if pc is None:
                continue
            denom = abs(pc["std"]) or 1.0
            rel = abs(pb["std"] - pc["std"]) / denom
            if rel > worst:
                worst, worst_name = rel, n
            print(
                f"{n[-57:]:<58}{pb['std']:>12.5f}{pc['std']:>12.5f}"
                f"{rel:>10.2%} {pb.get('frac_changed', -1):>7.3f}"
                f"{pc.get('frac_changed', -1):>8.3f}"
            )
        print()
        print(f"  worst relative std difference (non-norm): {worst:.3%} on {worst_name}")
        # Same comparison for arm A as a reference "how big is a real effect".
        if "A" in arms:
            a_ = _by_name(arms["A"])
            worst_a = max(
                (
                    abs(a_[n]["std"] - c[n]["std"]) / (abs(c[n]["std"]) or 1.0)
                    for n in nonnorm
                    if n in a_ and n in c
                ),
                default=0.0,
            )
            print(f"  for scale, worst A-vs-C non-norm std difference:   {worst_a:.3%}")

    # Exact element-wise divergence, when the .pt dumps are present.
    # Summary std is far too coarse: two arms can agree to 0.00% on std while
    # every element differs. This is the measurement that actually answers
    # "does B == C on non-norm params".
    _exact_divergence(sys.argv[1:], arms)

    print()
    print("=" * 78)
    print("3. LOSS")
    print("=" * 78)
    for a in order:
        ls = arms[a].get("losses") or []
        if not ls:
            print(f"  arm {a}: no loss curve recorded")
            continue
        tail = ls[-10:]
        print(
            f"  arm {a}: first={ls[0]:.5f}  last={ls[-1]:.5f}  "
            f"mean(last10)={sum(tail)/len(tail):.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
