#!/usr/bin/env python3
"""Build a W&B report for the fixed-batch optimizer comparison.

Committed as a SCRIPT rather than run once by hand: the chain is still adding
links, so the report has to be regenerable. Re-running overwrites the same
report (matched by title) instead of accumulating duplicates.

Usage:
    python3 make_optcmp_report.py [--dry-run]
"""

import argparse
import sys

import wandb
import wandb_workspaces.reports.v2 as wr

ENTITY = "aurora_gpt"
PROJECT = "agpt-30b-optcmp"
TITLE = "30B optimizer comparison: AdamW vs Mano vs SophiaG (GBS=960, constant LR)"

# Chain links per arm, oldest -> newest. Order matters for the reader, not for
# W&B: each link is a separate run because the job is walltime-bounded and
# resumes from the previous link's checkpoint.
ARMS = {
    "adamw":   ("AdamW",   "3.05e-05",
                ["9r60gcoc", "b58dh9pe", "75w7d5nz", "t0f38yi7", "qcha7hxk"]),
    "mano":    ("Mano",    "5.61e-05",
                ["gipl1yu8", "01w94y2f", "k5tum9ye", "boo5194c", "81yrfd1n"]),
    "sophiag": ("SophiaG", "3.55e-05",
                ["hacandli", "9g9hj6ft", "jp7e6k8h", "atqm4546", "uve9aqcq"]),
}

INTRO = """
## What this measures

Three optimizers on **agpt 30B** (26.2B params, OLMo-2 tokenizer), differing
ONLY in the optimizer and its learning rate. Everything else is held fixed:

| | |
|---|---|
| batch | **GBS=960** (LBS 5 x 192 ranks x GAS 1), seq 4096 -> 3.93M tok/step |
| LR schedule | 20-step warmup, then **constant** |
| data | fineweb-edu-100BT, 16 local parquet shards (~22B tokens, no repeats) |
| nodes | 16N per arm, all three run concurrently |

**Each arm runs at its own measured LR**, from an LR-finder sweep at this exact
batch size. That matters more than it sounds: the suggested LR moves three
orders of magnitude with batch for a single optimizer (Mano: 4.79e-03 at 2B,
5.61e-05 here, ~3e-06 at 80B/GBS=6144). Inheriting one constant would have made
this measure which optimizer got the luckier number rather than which optimizer
is better.

The inherited placeholder was in fact on the cliff: 3.0e-4 sat 8.5x above
SophiaG's suggestion and only 1.18x below its measured blow-up.

## Why constant LR

It makes the optimizer the only variable, and it is resume-safe -- a chained
job cannot reshape the schedule. An earlier 64N attempt was discarded precisely
because a decaying schedule, paced off a walltime ceiling, jumped the LR 1.83x
on resume.

The cost: absolute losses here are NOT comparable to the decayed 30B AdamW
baseline (2.115 at 3.93B tokens). Only the three arms are comparable to each
other.

## Reading the curves

Each arm is several runs, one per chained job -- the walltime-bounded job stops
and the next resumes from its checkpoint. Group by `optimizer` to see one line
per arm rather than one per link.
"""

FINDING = """
## Result so far

**Mano leads, but the gap peaked and started closing.**

| tokens | AdamW | Mano | SophiaG | Mano - AdamW |
|---:|---:|---:|---:|---:|
| 0.39B | **6.139** | 6.315 | 7.122 | +0.176 |
| 1.18B | 4.768 | **4.662** | 6.272 | -0.106 |
| 2.36B | 4.027 | **3.664** | 5.260 | **-0.364 (peak)** |
| 3.15B | 3.717 | **3.376** | 4.657 | -0.341 |
| 3.54B | 3.596 | **3.275** | 4.353 | -0.321 |

Mano crossed AdamW between 0.79B and 1.18B and widened to -0.364 nats at 2.36B,
then narrowed. Whether that continues is the open question -- extrapolating the
per-100-step rates around 3.5B, AdamW would catch Mano near 6-7B tokens, which
is inside the planned 10B budget.

**SophiaG is third but descending fastest**, having closed from 1.51 to ~0.80
nats behind Mano. Its ordering is the least settled of the three.

## Caveats

- One seed per arm; no seed variance measured.
- Constant LR, so no decay phase. The documented prior from earlier
  competitions is "Mano/Muon win short runs, AdamW wins in cosine decay" --
  the crossover-then-reconvergence here is consistent with that prior playing
  out even without a decay phase.
- Still short of the 10B budget, and the runs stopped once at 3.54B on a
  filesystem quota rather than for any scientific reason.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="build the report object but do not save it")
    args = ap.parse_args()

    # Verify every run id actually resolves BEFORE building panels: a typo'd id
    # yields a silently empty line rather than an error, which is the worst
    # outcome for a report someone else reads.
    api = wandb.Api()
    missing = []
    for arm, (_, _, ids) in ARMS.items():
        for rid in ids:
            try:
                api.run(f"{ENTITY}/{PROJECT}/{rid}")
            except Exception:
                missing.append(f"{arm}/{rid}")
    if missing:
        print("MISSING run ids (report would render empty lines):")
        for m in missing:
            print("   ", m)
        return 1
    print(f"verified {sum(len(v[2]) for v in ARMS.values())} run ids resolve")

    all_ids = [r for _, _, ids in ARMS.values() for r in ids]
    # `filters` is parsed as a Python expression by wandb_workspaces (it walks
    # the AST), NOT as a mongo-style dict -- passing {"name": {"$in": ...}}
    # raises "Unsupported expression type: ast.Dict".
    runset = wr.Runset(
        entity=ENTITY, project=PROJECT, name="all arms",
        filters=f"ID in {all_ids!r}",
    )

    def lines(metric, title, log_y=False):
        return wr.LinePlot(
            title=title, x="_step", y=[metric],
            log_y=log_y, smoothing_factor=0.0,
            groupby="optimizer", legend_position="east",
        )

    report = wr.Report(
        entity=ENTITY, project=PROJECT, title=TITLE,
        description="Fixed-batch optimizer comparison at 30B. Regenerate with "
                    "scripts/make_optcmp_report.py",
        blocks=[
            wr.MarkdownBlock(text=INTRO),
            wr.PanelGrid(runsets=[runset], panels=[
                lines("loss_metrics/global_avg_loss", "loss (global avg)"),
                lines("loss_metrics/global_max_loss", "loss (global max)"),
            ]),
            wr.MarkdownBlock(text=FINDING),
            wr.PanelGrid(runsets=[runset], panels=[
                lines("optimizer/grad_norm", "grad norm (pre-clip)", log_y=True),
                lines("lr", "learning rate (constant by design)"),
            ]),
            wr.MarkdownBlock(text="## Throughput\n\nAll three arms hold ~28% "
                                  "MFU, so no arm is paying a speed penalty "
                                  "for its optimizer."),
            wr.PanelGrid(runsets=[runset], panels=[
                lines("throughput/tps", "tokens/sec/GPU"),
                lines("throughput/mfu(%)", "MFU (%)"),
            ]),
        ],
    )
    if args.dry_run:
        print("dry run: report built, not saved")
        print(f"  title  : {TITLE}")
        print(f"  runs   : {len(all_ids)}")
        print(f"  blocks : {len(report.blocks)}")
        return 0
    url = report.save().url
    print(f"REPORT URL {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
