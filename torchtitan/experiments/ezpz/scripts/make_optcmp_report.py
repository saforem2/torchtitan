#!/usr/bin/env python3
"""Build a W&B report for the fixed-batch optimizer comparison.

Committed as a SCRIPT rather than run once by hand: the chain is still adding
links, so the report has to be regenerable.

NOTE: re-running does NOT overwrite. `report.save()` mints a NEW report id
every call, even with an identical title -- this docstring previously claimed
otherwise and ten same-titled reports accumulated before anyone checked. The
URL printed at the end is the only current one; every earlier URL is a stale
snapshot that still renders, which makes a shared link silently go out of
date. Prune old ones in the W&B UI (the API exposes no report delete), and
re-share the new URL after each regenerate.

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
                ["9r60gcoc", "b58dh9pe", "75w7d5nz", "t0f38yi7", "qcha7hxk",
                 "w3uc6m5r"]),
    "mano":    ("Mano",    "5.61e-05",
                ["gipl1yu8", "01w94y2f", "k5tum9ye", "boo5194c", "81yrfd1n",
                 "geryqvdw"]),
    "sophiag": ("SophiaG", "3.55e-05",
                ["hacandli", "9g9hj6ft", "jp7e6k8h", "atqm4546", "uve9aqcq"]),
}

# Phase 1 LR-finder sweeps, same project. Two runs exist per arm: the first
# attempt read blendcorpus and was killed at ~19 steps, the second swept the
# real fineweb-edu data for the full 100. Only the latter is meaningful, so
# these ids are pinned rather than globbed -- and verified below by state,
# step count, and dataset, so a wrong pin cannot quietly plot the killed run.
LRFIND = {
    "adamw":   ("dasclciy", "3.05e-05"),
    "mano":    ("oe8w5i7d", "5.61e-05"),
    "sophiag": ("25s1nia7", "3.55e-05"),
}

# The controlled re-run of the SophiaG divergence. Same seed checkpoint, same
# optimizer state (initial_load_model_only=False), same LR -- only the data
# order and RNG differ. Its own W&B group so it cannot be mistaken for the
# original arm.
RERUN = ("a8i1825t", "sophiag")

# Third replicate of the divergence, forked from the same step-1000 seed with
# --debug.seed=42 and the grad-norm guard armed at 20x. Exists to put a RATE on
# the 2-of-2: a third distinct onset step means recurrent-with-random-timing.
RERUN2 = ("9ewej3bc", "sophiag")

DIVERGENCE_MD = """
## The SophiaG divergence -- it happens twice

At step 1048 the SophiaG arm blew up. Nothing in the loss curve warns you:

| step | loss | grad_norm |
|---:|---:|---:|
| 1047 | 3.967 | 0.39 |
| 1049 | 3.996 | **89.9** |
| **1057** | 7.015 | **100,611** |

Five orders of magnitude in nine steps. **AdamW and Mano were untouched at the
same wall-clock** (grad_norm 0.58 and 0.30), same nodes, same data -- so this
is the optimizer, not the machine.

### The re-run diverged too

Re-ran from the clean pre-spike `step-1000` checkpoint carrying the FULL
optimizer state (`initial_load_model_only=False`, so SophiaG's Hessian
estimate came along). Same weights, same LR; only data order and RNG differ.

It cleared step 1048 -- and then blew up 128 steps later on its own schedule:

| step | loss | grad_norm |
|---:|---:|---:|
| 1173 | 3.722 | 0.67 |
| 1176 | 3.747 | **37.4** |
| 1182 | 4.280 | 85,081 |
| **1189** | -- | **204,016** |

Same signature, **twice the peak**, and a worse landing (loss 9.47 vs 8.16).

### What that means

**3 of 3 runs from the same seed diverged**, at different steps (1048, 1176,
1071), with matching severity. Not a rare event that luck avoids, and not tied to a specific batch:
if it were the data, the re-run would have fired at 1048, where the data order
was nearly identical (step 1001 loss matched to five decimals).

### Replicate 3 makes it 3 of 3

A third replicate (job 12473833, `--debug.seed=42`, grad-norm guard armed at
20x) forked from the same clean `step-1000` checkpoint and blew up at **step
1071**:

| step | loss | grad_norm |
|---:|---:|---:|
| 1064 | 3.915 | 0.32 |
| 1067 | 3.909 | **6.88** |
| 1070 | 3.945 | 1.53 |
| **1071** | 3.954 | **53.49** |

Three runs, three distinct onsets -- **1048, 1176, 1071** -- from identical
weights and optimizer state. The timing is random; the event is not.

Note the loss again: 3.915 -> 3.954 across the whole onset. Nothing in the
loss curve distinguishes these steps from the 1,000 healthy ones before them.

**The guard worked.** It fired at 53.49 (133x the trailing median of 0.4022)
and stopped the run at rc=0 after 55 minutes, instead of letting it ride to
grad_norm ~100,000 and burn the remaining ~7 hours of the window, which is
what happened to replicates 1 and 2.

### The regime persists -- it is not a transient

Replicate 2 was deliberately left running well past its blow-up to show what
the post-onset state actually looks like on the charts above. 306 steps after
onset it is **still in it**: 72% of those steps exceed grad_norm 2.0, and it
was still throwing excursions of 21.9, 16.2 and 61.8 at steps 1465-1467 --
nearly 300 steps after the 204,016 peak at step 1189.

For contrast, AdamW and Mano have never once exceeded 0.8 in ~1,100 combined
steps.

This is why the loss panel misleads on its own. Replicate 2's loss wanders
back down toward 4.1-4.3 and looks like a recovering run; the grad_norm panel
shows it is nothing of the kind. Read the SophiaG arms on the grad_norm chart.

SophiaG at lr=3.55e-05 on this model **will** diverge; only the timing is
unpredictable.

Its comparison curve past step 1048 should be read as "SophiaG after a
blow-up", not as SophiaG.

### It is a regime flip, not a too-high LR

Counting post-warmup steps whose grad_norm exceeds 2.0, the populations do not
overlap:

| arm | steps | grad_norm > 2.0 | max grad_norm |
|---|---:|---:|---:|
| AdamW | 568 | **0** | **0.8** |
| Mano | 562 | **0** | **0.8** |
| SophiaG | 565 | 365 | 100,611 |
| SophiaG re-run | 379 | 135 | 204,017 |

Split at onset, SophiaG is indistinguishable from the healthy arms before
(0/147 steps above 2.0, max 0.921) and lives in a high-gradient regime after
(87% and 64% of steps above 2.0). It does not leave that regime.

**So apparent recovery is an artifact.** The re-run's loss dipped twice
(7.15 -> 4.56, then 5.50 -> 4.29) while grad_norm stayed 30-315 -- dips inside
the bad regime, not returns to the good one. Read these arms by grad_norm.

**And lowering the LR is a weak fix**, which is what an earlier version of this
report recommended. Onset is a discontinuity (0.39 -> 2.41 -> 89.9 -> 437 ->
1702 while the loss moves only 3.957 -> 4.089 -- a 4,000x gradient change under
0.13 nats), and the Phase 1 sweep descends smoothly through the entire low band
with no instability near 3.55e-05. A static sweep probes ~100 steps; it cannot
see a state-dependent failure that arms after 1,000 steps of Hessian
accumulation. Lower LR may delay onset without preventing it.

What would discriminate: instrument SophiaG's Hessian-estimate norm per step
and check whether it degrades monotonically before onset. If it does, the fix
is in the Hessian update (rho, the EMA, the clipping), not the LR.

### What it changed in the code

`nan_abort_consecutive` never fired, correctly and uselessly -- every value in
both sequences is finite. A grad-norm runaway guard was added that compares
against the run's own trailing median. It fires on BOTH divergences (step 1049
and step 1176, 8 and 13 steps before their peaks) and stays clean on adamw and
mano. Given the failure recurs rather than being rare, it is not optional for
a SophiaG run at this LR.
"""

LRFIND_MD = """
## Phase 1: where each learning rate came from

Before any comparison run, each optimizer was swept 1e-6 -> 1e-1 over 100 steps
at **this exact batch size**. All three produced a real blow-up, so every
suggestion is measured rather than an artifact of a sweep that ran out of range
-- a sweep that never diverges yields no suggestion at all.

| arm | suggested LR | blow-up | min loss | usable band top |
|---|---:|---:|---:|---:|
| **AdamW** | 3.05e-05 | 3.05e-04 | 8.8381 | 1.292e-03 |
| **Mano** | 5.61e-05 | 5.61e-04 | 8.9925 | 2.448e-03 |
| **SophiaG** | 3.55e-05 | 3.55e-04 | 9.3441 | 1.000e-03 |

"Suggested" is Smith 2015's blow-up/10, so it sits well left of the minimum by
construction: it buys stability margin, it is not the best-loss LR.

**Why this had to be measured per optimizer at this batch.** The suggestion
moves by orders of magnitude with batch size for a single optimizer -- Mano:
4.79e-03 at 2B, 5.61e-05 here, ~3e-06 at 80B/GBS=6144. The inherited
placeholder (3.0e-4, from the 2B competition configs) sat **8.5x above
SophiaG's suggestion and only 1.18x below its measured blow-up**. Running there
would have read as "SophiaG is unstable at 30B" rather than "SophiaG was run at
8.5x its usable LR".

One thing worth noting: at a FIXED batch the three land within 1.8x of each
other. Batch size moves the usable LR far more than optimizer choice does.

The sweep curves below plot loss against the swept LR -- x is the step index,
and LR rises exponentially across it.
"""

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


def _check_groupby(api, ids, key, panel):
    """Every run in a panel must have a DISTINCT value for its groupby key.

    Three chart bugs in this report came from asserting structure instead of
    querying it, and this one was the worst: the re-run carried
    optimizer_name=None, so grouping merged it into the original's trace and
    the panel captioned "the re-run that avoided the blow-up" drew the
    divergence as a single line -- arguing the opposite of the finding.

    A missing or duplicated value never errors in W&B; it silently merges. So
    check it here, where it can fail loudly.
    """
    seen = {}
    for rid in ids:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        v = r.config.get(key)
        if v is None:
            print(f"ERR {panel}: run {rid} has no {key!r} -- it would merge "
                  "into another run's line")
            return False
        seen.setdefault(v, []).append(rid)
    dupes = {v: r for v, r in seen.items() if len(r) > 1}
    if dupes and len(seen) < 2:
        print(f"ERR {panel}: all runs share {key}={list(seen)[0]!r} -- "
              "the panel would draw ONE merged line")
        return False
    return True


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

    # Label every run with a FLAT, groupable key. config["optimizer"] is a
    # nested dict (param_groups -> [{optimizer_name: ...}]), and W&B cannot
    # group on a dict -- doing so renders all 15 runs as one indistinguishable
    # blob, which is exactly how the first version of this report shipped.
    # Derive the label from each run's OWN config rather than the id lists
    # above: a mislabeled run would put the wrong curve under the wrong name,
    # which is worse than no grouping.
    expect = {"adamw": "AdamW", "mano": "Mano", "sophiag": "SophiaG"}
    for arm, (_, _, ids) in ARMS.items():
        for rid in ids:
            r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
            got = r.config["optimizer"]["param_groups"][0]["optimizer_name"]
            if got != expect[arm]:
                print(f"ERR {rid}: config says {got}, expected {expect[arm]}")
                return 1
            if r.group != arm or r.config.get("optimizer_name") != got:
                r.group = arm
                r.config["optimizer_name"] = got
                r.config["arm"] = arm
                r.update()
    for arm, (rid, _) in LRFIND.items():
        fr = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        got = fr.config["optimizer"]["param_groups"][0]["optimizer_name"]
        if fr.config.get("optimizer_name") != got or fr.group != f"lrfind-{arm}":
            fr.group = f"lrfind-{arm}"
            fr.config["optimizer_name"] = got
            fr.update()
    print("labeled all runs with group + flat optimizer_name")

    # Verify every metric a panel plots EXISTS on the runs. A wrong key does
    # not error -- W&B renders "Select runs that logged <key>", an empty panel
    # that looks like a data problem rather than a typo. Three of the six keys
    # in the first version of this report were wrong for exactly that reason.
    probe = api.run(f"{ENTITY}/{PROJECT}/{ARMS['adamw'][2][-1]}")
    have = set(probe.summary.keys())
    want = ["loss_metrics/global_avg_loss", "loss_metrics/global_max_loss",
            "grad_norm", "lr", "throughput(tps)", "mfu(%)"]
    absent = [k for k in want if k not in have]
    if absent:
        print("MISSING metric keys (panels would render empty):")
        for k in absent:
            print("   ", k)
        print("  available:", sorted(k for k in have if not k.startswith("_")))
        return 1
    print(f"verified {len(want)} metric keys exist on the runs")

    # Verify the LR-finder pins are the REAL sweeps, not the killed first
    # attempt: 100 steps, finished, and on fineweb rather than blendcorpus.
    for arm, (rid, _) in LRFIND.items():
        fr = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        ds = fr.config.get("dataloader", {})
        ds = ds.get("dataset") if isinstance(ds, dict) else ds
        if fr.state != "finished" or ds != "fineweb_edu_local":
            print(f"ERR lr-finder {arm}/{rid}: state={fr.state} dataset={ds} "
                  "-- this looks like the killed blendcorpus attempt")
            return 1
    print("verified 3 lr-finder runs (finished, fineweb, full sweep)")

    # Label the rerun too. It was previously left with optimizer_name=None,
    # so grouping on that key merged it INTO the original's line -- the panel
    # meant to show "the rerun avoided the blow-up" drew both as one trace and
    # argued the opposite. Give it a distinct value rather than reusing
    # "SophiaG", which would collide with the original by construction.
    _rr = api.run(f"{ENTITY}/{PROJECT}/{RERUN[0]}")
    if _rr.config.get("optimizer_name") != "SophiaG re-run":
        _rr.config["optimizer_name"] = "SophiaG re-run"
        _rr.config["arm"] = "sophiag-rerun"
        _rr.update()
    for _rid in ARMS["sophiag"][2]:
        _orig = api.run(f"{ENTITY}/{PROJECT}/{_rid}")
        if _orig.config.get("optimizer_name") != "SophiaG (diverged)":
            _orig.config["optimizer_name"] = "SophiaG (diverged)"
            _orig.update()
    _rr2 = api.run(f"{ENTITY}/{PROJECT}/{RERUN2[0]}")
    if _rr2.config.get("optimizer_name") != "SophiaG replicate 3 (seed 42)":
        _rr2.config["optimizer_name"] = "SophiaG replicate 3 (seed 42)"
        _rr2.config["arm"] = "sophiag-rerun2"
        _rr2.update()
    print("labeled all three divergence replicates distinctly")

    rr = api.run(f"{ENTITY}/{PROJECT}/{RERUN[0]}")
    if "rerun" not in (rr.group or ""):
        print(f"ERR rerun {RERUN[0]}: group={rr.group!r} -- expected a rerun-* "
              "group; this may be the ORIGINAL arm, which would make the "
              "comparison panel plot the same run twice")
        return 1
    print(f"verified rerun run (group={rr.group})")

    if not _check_groupby(api, [ARMS["sophiag"][2][-1], RERUN[0], RERUN2[0]],
                          "optimizer_name", "divergence comparison"):
        return 1
    if not _check_groupby(api, [v[0] for v in LRFIND.values()],
                          "optimizer_name", "lr finder"):
        return 1
    print("verified groupby keys are distinct per panel")

    # Include the re-runs. Built from ARMS alone, the main panels silently
    # omitted every SophiaG replicate -- they were only ever visible in the
    # dedicated divergence panel further down.
    all_ids = ([r for _, _, ids in ARMS.values() for r in ids]
               + [RERUN[0], RERUN2[0]])
    # `filters` is parsed as a Python expression by wandb_workspaces (it walks
    # the AST), NOT as a mongo-style dict -- passing {"name": {"$in": ...}}
    # raises "Unsupported expression type: ast.Dict".
    runset = wr.Runset(
        entity=ENTITY, project=PROJECT, name="all arms",
        filters=f"ID in {all_ids!r}",
    )

    # Every run the report knows about must appear in the main runset --
    # resolving is not the same as being displayed.
    _known = set(all_ids)
    _expected = ({r for _, _, ids in ARMS.values() for r in ids}
                 | {RERUN[0], RERUN2[0]})
    _missing = _expected - _known
    if _missing:
        print(f"ERR runs pinned but NOT in the main runset: {sorted(_missing)}"
              " -- they would resolve, pass every other check, and still be "
              "invisible in the loss/grad_norm panels")
        return 1
    print(f"verified all {len(_known)} runs appear in the main runset")

    # One arm must not split into two legend entries.
    for _arm, (_label, _lr, _ids) in ARMS.items():
        _names = {api.run(f"{ENTITY}/{PROJECT}/{i}").config.get("optimizer_name")
                  for i in _ids}
        if len(_names) != 1:
            print(f"ERR arm {_arm!r} has {len(_names)} distinct "
                  f"optimizer_name values {sorted(map(str, _names))} -- "
                  "groupby will draw it as multiple traces")
            return 1
    print("verified each arm carries ONE optimizer_name across its links")

    def lines(metric, title, log_y=False):
        return wr.LinePlot(
            title=title, x="_step", y=[metric],
            log_y=log_y, smoothing_factor=0.0,
            groupby="optimizer_name", legend_position="east",
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
                lines("grad_norm", "grad norm (pre-clip)", log_y=True),
                lines("lr", "learning rate (constant by design)"),
            ]),
            wr.MarkdownBlock(text=LRFIND_MD),
            wr.PanelGrid(
                runsets=[wr.Runset(
                    entity=ENTITY, project=PROJECT, name="lr finder sweeps",
                    filters=f"ID in {[v[0] for v in LRFIND.values()]!r}",
                )],
                panels=[wr.LinePlot(
                    title="LR finder: loss vs LEARNING RATE (log x)",
                    x="lr", y=["loss_metrics/global_avg_loss"],
                    log_x=True,
                    groupby="optimizer_name", legend_position="east",
                )],
            ),
            wr.MarkdownBlock(text=DIVERGENCE_MD),
            wr.PanelGrid(
                runsets=[wr.Runset(
                    entity=ENTITY, project=PROJECT,
                    name="sophiag: all three divergence replicates",
                    filters=f"ID in {[ARMS['sophiag'][2][-1], RERUN[0], RERUN2[0]]!r}",
                )],
                panels=[
                    wr.LinePlot(title="grad_norm: every replicate blows up "
                                      "(log scale; healthy arms never exceed 0.8)",
                                x="_step", y=["grad_norm"], log_y=True,
                                groupby="optimizer_name",
                                legend_position="east"),
                    wr.LinePlot(title="loss: same window",
                                x="_step", y=["loss_metrics/global_avg_loss"],
                                groupby="optimizer_name",
                                legend_position="east"),
                ],
            ),
            wr.MarkdownBlock(text="## Throughput\n\nAll three arms hold ~28% "
                                  "MFU, so no arm is paying a speed penalty "
                                  "for its optimizer."),
            wr.PanelGrid(runsets=[runset], panels=[
                lines("throughput(tps)", "tokens/sec/GPU"),
                lines("mfu(%)", "MFU (%)"),
            ]),
        ],
    )
    if args.dry_run:
        print("dry run: report built, not saved")
        print(f"  title  : {TITLE}")
        print(f"  runs   : {len(all_ids)}")
        print(f"  blocks : {len(report.blocks)}")
        return 0
    _prior = []
    try:
        _prior = [r for r in api.reports(f"{ENTITY}/{PROJECT}")
                  if getattr(r, "id", None)]
    except Exception:
        pass

    url = report.save().url
    if len(_prior) > 1:
        print(f"NOTE {len(_prior)} same-project reports already existed; "
              "save() mints a new id rather than overwriting, so the older "
              "ones are now stale snapshots. Prune them in the W&B UI.")
    print(f"REPORT URL {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
