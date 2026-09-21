# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collect the OLMo-2 ladder LR sweeps into one table.

Twelve sweeps (3 sizes x 4 optimizers) are too many to read off plots and
hand-transcribe into a report -- that transcription is where a digit gets
dropped and a training arm launches at 10x its measured LR. This reads the
CSVs the finder writes and reports each arm's blow-up and Smith-2015
suggestion (blow-up / 10) using the finder's OWN analysis function, so the
number here cannot disagree with the number the finder computed.

An arm whose sweep never reached its cliff reports NO BLOW-UP rather than a
suggestion. That is the case worth catching: the finder exits 0 in that
situation, so an empty result otherwise looks like a healthy run. Muon's
cliff at 30B/GBS=960 was 5.68e-03, 5.7x above the 1e-3 window the other
optimizers use, which is exactly how a window ends up too narrow.

Usage:
  python -m torchtitan.experiments.ezpz.scripts.summarize_lr_finder_ladder \
      [--root outputs] [--markdown]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

from torchtitan.experiments.ezpz.lr_finder import find_optimal_lr

SIZES = ("5b", "10b", "30b")
OPTIMIZERS = ("adamw", "mano", "muon", "sophiag")


def _read(csv_path: str) -> tuple[list[float], list[float]]:
    lrs: list[float] = []
    losses: list[float] = []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            try:
                lr = float(row["learning_rate"])
                loss = float(row["loss"])
            except (KeyError, TypeError, ValueError):
                continue
            # A non-finite loss is the blow-up itself; keep it, the
            # derivative analysis needs the rise to find the crossing.
            lrs.append(lr)
            losses.append(loss)
    return lrs, losses


def _find_csv(root: str, size: str, opt: str) -> str | None:
    # run_lr_finder.sh writes under <dump>/lr_finder/ezpz/ezpz.agpt/<flavor>/<opt>/.
    # Glob rather than reconstruct: the flavor segment has varied across
    # campaigns (30b vs 30b_olmo2tok), and a wrong guess reads as "no data".
    pat = os.path.join(
        root,
        f"lr_finder_{size}_olmo2tok_gbs6144_{opt}",
        "**",
        opt,
        "lr_finder_data.csv",
    )
    hits = sorted(glob.glob(pat, recursive=True), key=os.path.getmtime)
    if not hits:
        # adamw's first submissions predate the per-optimizer dump suffix.
        pat = os.path.join(
            root,
            f"lr_finder_{size}_olmo2tok_gbs6144",
            "**",
            opt,
            "lr_finder_data.csv",
        )
        hits = sorted(glob.glob(pat, recursive=True), key=os.path.getmtime)
    return hits[-1] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--smooth-frac", type=float, default=0.05)
    args = ap.parse_args()

    rows = []
    for size in SIZES:
        for opt in OPTIMIZERS:
            path = _find_csv(args.root, size, opt)
            if path is None:
                rows.append((size, opt, None, None, None, "no csv yet"))
                continue
            lrs, losses = _read(path)
            if len(lrs) < 5:
                rows.append(
                    (size, opt, None, None, None, f"{len(lrs)} rows -- too few")
                )
                continue
            cands = find_optimal_lr(lrs, losses, smooth_frac=args.smooth_frac)
            finite = [x for x in losses if x == x and abs(x) != float("inf")]
            min_loss = min(finite) if finite else None
            if not cands:
                rows.append(
                    (
                        size,
                        opt,
                        None,
                        None,
                        min_loss,
                        f"NO BLOW-UP in {min(lrs):.1e}-{max(lrs):.1e}"
                        " -- widen LRF_MAX_LR",
                    )
                )
                continue

            # find_optimal_lr returns EVERY negative-to-positive crossing, not
            # just the cliff -- any noise wiggle early in the sweep is a local
            # minimum too. On a real 20B adamw sweep it returned
            # [2.31e-06, 3.95e-03]: the first is noise near the sweep's 1e-6
            # start, the second is the true cliff at the loss minimum
            # (3.98e-03). Taking cands[0] understated the LR by ~1700x.
            #
            # The blow-up is the crossing at or after the loss minimum: the
            # curve must descend to its best value before the divergence that
            # ends the usable band. Candidates below the minimum are on the
            # descending branch and are not cliffs.
            lr_at_min = lrs[min(range(len(losses)), key=lambda i: losses[i])]
            after = [c for c in cands if c >= lr_at_min * 0.5]
            blow = after[0] if after else cands[-1]
            note = f"{len(lrs)} rows"
            if len(cands) > 1:
                note += f"; {len(cands)} candidates, took {blow:.2e}"
            rows.append((size, opt, blow / 10.0, blow, min_loss, note))

    if args.markdown:
        print("| size | optimizer | suggested LR | blow-up | min loss | note |")
        print("|---|---|---:|---:|---:|---|")
        for size, opt, sug, blow, ml, note in rows:
            f = lambda v, p=".3e": f"{v:{p}}" if v is not None else ""
            print(
                f"| {size} | {opt} | {f(sug)} | {f(blow)} | "
                f"{f(ml, '.4f')} | {note} |"
            )
    else:
        print(
            f"{'size':>5} {'opt':>8} {'suggested':>11} {'blow-up':>11} "
            f"{'min loss':>9}  note"
        )
        for size, opt, sug, blow, ml, note in rows:
            f = lambda v, p=".3e": f"{v:{p}}" if v is not None else "-"
            print(
                f"{size:>5} {opt:>8} {f(sug):>11} {f(blow):>11} "
                f"{f(ml, '.4f'):>9}  {note}"
            )

    missing = [r for r in rows if r[2] is None]
    if missing:
        print(f"\n{len(missing)} of {len(rows)} arms have no suggestion yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
