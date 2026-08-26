#!/usr/bin/env python3
"""Build a W&B report for the fixed-batch optimizer comparison.

Every arm is a CHAIN of runs -- the jobs are walltime-bounded and each link
resumes from the previous link's checkpoint -- so the report has to stitch
links together and has to be regenerable as new links land.

It does that by GROUP, not by pinned run ids. Each run already carries a
wandb group ("adamw", "mano", "sophiag", "lrfind-<arm>", "rerun-<arm>"), set
at launch by the PBS scripts, so the runset filters on group and the panels
group by it. New chain links join their arm the moment they start; nothing
here needs editing.

The previous version pinned an explicit list of ~19 run ids. Every bug this
report has had came from that list drifting out of date:

  - the newest chain link was missing, so panels showed a truncated curve
  - both SophiaG re-runs resolved and were labeled but were never added to
    the main runset, so they appeared in NO main panel for the report's life
  - only the LAST link of the sophiag arm got relabeled "SophiaG (diverged)",
    so groupby split one arm into two legend entries

None of those are possible now: there is no list to drift. The guards that
existed to catch list drift are gone with it.

NOTE: re-running does NOT overwrite. report.save() mints a NEW report id every
call, even with an identical title, so each regenerate leaves a stale snapshot
that still renders -- which is how a shared link goes quietly out of date.
This script now deletes the older duplicates after a successful save (pass
--keep-old to skip). Fifteen accumulated before that was wired up.

Deletion goes through the `deleteView` GraphQL mutation: reports are `View`
objects internally, which is why searching the public API for "report" finds
no delete method and why an earlier version of this docstring wrongly claimed
none exists.

Usage:
    python3 make_optcmp_report.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys

import wandb
import wandb_workspaces.reports.v2 as wr

ENTITY = "aurora_gpt"
PROJECT = "agpt-30b-optcmp"
TITLE = "30B optimizer comparison: AdamW vs Mano vs SophiaG (GBS=960, constant LR)"

# Groups, not run ids. These are set at launch (WANDB_RUN_GROUP in the PBS
# scripts), so a new chain link joins its arm automatically.
ARM_GROUPS = ["adamw", "mano", "sophiag"]
LRFIND_GROUPS = ["lrfind-adamw", "lrfind-mano", "lrfind-sophiag"]
# Each SophiaG divergence replicate has its own rerun-* group, so they stay
# distinguishable from each other and from the original arm.
RERUN_GROUP_PREFIX = "rerun-"

# The panel legend key. groupby resolves against CONFIG keys, NOT run metadata:
# "group" looks right and is silently wrong (no run carries a config key by
# that name, so every run lands in one bucket and the panel renders a single
# aggregated line). `arm` is a real config key set at launch, and it matches
# the wandb group 1:1 for every comparison run -- verified before use below.
GROUPBY_KEY = "arm"

# Per-arm line colors are NOT settable from this library version. All three
# documented paths fail, and all three fail SILENTLY rather than raising:
#   - custom_run_colors with plain string keys: keys are read as RUN IDS and
#     merged into run_settings, so a group name matches nothing and the field
#     round-trips empty.
#   - the (grouping-key, value) tuple the field's own type annotation and
#     docstring advertise for grouped runs: passes Runset validation, then is
#     rejected by PanelGridMetadata at save ("Input should be a valid string").
#   - run_settings={run_id: RunSettings(color=...)}: saves without error and
#     also round-trips empty.
# Verified by saving throwaway reports and reading the spec back. Set colors
# in the W&B UI until this is fixed upstream.
# A replicate has to outlive the earliest observed divergence onset (step 1048)
# to say anything about divergence. Below this it is an aborted launch.
MIN_RERUN_STEPS = 1040

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

Both SophiaG arms were left running well past their blow-ups to see whether
the high-gradient state ever ends. **It does -- for one of them.** Excursion
rate per 100-step window (steps with grad_norm > 2.0):

| window | sophiag | sophiag re-run |
|---|---:|---:|
| 1200-1299 | 92% (max 9,613) | 68% (max 63,685) |
| 1400-1499 | 61% (max 419) | 88% (max 4,250) |
| 1500-1599 | 11% (max 89.7) | 98% (max 1,120) |
| 1600-1699 | **0% (max 0.7)** | 97% (max 1,023) |

sophiag recovered completely: by step 1600 it is at zero excursions with a
peak of 0.7, below Mano's lifetime average, ~600 steps after a 100,611 spike.
The re-run, from the same checkpoint at the same LR, is still at 93-98% and
still spiking past 1,000 at step 1750.

So the regime is **metastable, not absorbing** -- an earlier version of this
report said it "does not leave that regime", which was based on ~400 post-onset
steps that all happened to fall inside the bad window. Whether an arm escapes
looks like luck, the same stochastic signature as onset itself: two replicates
from one checkpoint diverged at different steps, and two replicates past
divergence had opposite recoveries.

For contrast, AdamW and Mano have never once exceeded 0.8 in ~1,100 combined
steps.

This is why the loss panel misleads on its own. Replicate 2 ended at loss
4.193 -- back in the range it held before the blow-up, and on the loss chart
alone indistinguishable from a run that recovered. Its grad_norm at that same
final step was 2.449, six times what AdamW or Mano have ever reached. Read the
SophiaG arms on the grad_norm chart.

SophiaG at lr=3.55e-05 on this model **will** diverge; only the timing is
unpredictable.

Its comparison curve past step 1048 should be read as "SophiaG after a
blow-up", not as SophiaG.

### It is a regime flip, not a too-high LR

Counting post-warmup steps whose grad_norm exceeds 2.0, the populations do not
overlap:

| arm | post-warmup steps | grad_norm > 2.0 | max grad_norm |
|---|---:|---:|---:|
| AdamW | 1,984 | 54 (**2.7%**) | 14.6 |
| Mano | 1,983 | 128 (**6.5%**) | 30.0 |
| SophiaG | 1,524 | 573 (**37.6%**) | 100,611 |
| SophiaG re-run | 644 | 380 (**59.0%**) | 204,017 |

An earlier version of this section reported **0 excursions and max 0.8** for
both healthy arms. That was wrong: it was computed from a single chain link's
log rather than the whole chain, so it missed every excursion outside that
window. The healthy arms DO spike -- Mano reached 30.0 -- and the corrected
numbers are above.

The separation survives the correction, but it is quantitative rather than
absolute: the healthy arms spend 3-7% of steps above 2.0 and peak in the tens,
while the SophiaG arms spend 38-56% there and peak in the hundred-thousands, a
factor of ~3,000 in magnitude. Post-onset SophiaG does not leave the regime;
replicate 2 ran 403 steps past onset with 77% of them above 2.0.

The distinguishing feature is the SHAPE of an excursion, not its existence.
Mano's largest late spike (3.84 at step 1944) ramped over six steps --
0.24, 0.39, 0.55, 1.03, 3.84 -- with the loss moving alongside it (2.89 ->
3.09), and was back under 1.0 within five steps. SophiaG's onsets are
discontinuities under a nearly flat loss: 0.39 -> 2.41 -> 89.9 -> 437 -> 1702
while the loss moves only 3.957 -> 4.089.

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
and the next resumes from its checkpoint. The panels group by wandb `group`,
so each arm is one line across all of its links.
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


def _discover(api) -> dict[str, list]:
    """Map group -> runs, for every group this report displays.

    Reads the project instead of trusting a hardcoded list. A group that is
    expected but absent is an error worth stopping on: it means a launch
    script changed its WANDB_RUN_GROUP and the report would silently render
    a missing arm.
    """
    by_group: dict[str, list] = {}
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        if not run.group:
            continue  # smoke/preflight runs, never part of the comparison
        by_group.setdefault(run.group, []).append(run)

    # Drop aborted launches. Discovery is only better than a pinned list if it
    # also declines to plot runs that never produced a comparable trajectory:
    # group `rerun-sophiag-seed42` is the first attempt at replicate 3, which
    # died at step 1001 with a NameError in the grad-norm guard. Plotted, it
    # reads as a replicate that survived where the others blew up. The
    # threshold is deliberately low -- it excludes launch failures, not short
    # runs -- and lr-finder sweeps are exactly 100 steps by design.
    kept = {}
    for group, runs in by_group.items():
        longest = max((r.summary.get("_step") or 0) for r in runs)
        if group.startswith(RERUN_GROUP_PREFIX) and longest < MIN_RERUN_STEPS:
            print(f"  SKIP {group}: longest run reached step {longest} "
                  f"(< {MIN_RERUN_STEPS}), an aborted launch rather than a "
                  "replicate")
            continue
        kept[group] = runs
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="build the report object but do not save it")
    ap.add_argument("--keep-old", action="store_true",
                    help="do not delete older same-project reports after saving")
    args = ap.parse_args()

    api = wandb.Api()
    by_group = _discover(api)

    rerun_groups = sorted(g for g in by_group if g.startswith(RERUN_GROUP_PREFIX))
    expected = ARM_GROUPS + LRFIND_GROUPS
    missing = [g for g in expected if g not in by_group]
    if missing:
        print(f"ERR expected groups absent from {PROJECT}: {missing}")
        print("    a launch script's WANDB_RUN_GROUP probably changed; the "
              "report would render an empty arm")
        return 1

    for g in expected + rerun_groups:
        runs = by_group[g]
        steps = max((r.summary.get("_step") or 0) for r in runs)
        print(f"  {g:24} {len(runs):2d} run(s), max step {steps}")
    print(f"discovered {len(expected) + len(rerun_groups)} groups, "
          f"{sum(len(by_group[g]) for g in expected + rerun_groups)} runs")

    main_groups_check = ARM_GROUPS + rerun_groups

    # groupby silently no-ops on a config key that does not exist, collapsing
    # every run into one line. Verify the key is present and 1:1 with the group.
    for g in main_groups_check + LRFIND_GROUPS:
        for r in by_group[g]:
            if r.config.get(GROUPBY_KEY) != g:
                r.config[GROUPBY_KEY] = g
                r.update()
                print(f"  backfilled config[{GROUPBY_KEY!r}]={g!r} on {r.id}")

    # Re-check rather than trust the writes: a failed update would otherwise
    # collapse the panel silently, which is the exact failure this guards.
    bad = []
    for g in main_groups_check + LRFIND_GROUPS:
        for r in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": g}):
            if r.config.get(GROUPBY_KEY) is None:
                bad.append(f"{r.id} (group={g})")
    if bad:
        print(f"ERR groupby key {GROUPBY_KEY!r} still missing after backfill "
              "-- panels would collapse into one aggregated trace:")
        for b in bad[:8]:
            print(f"    {b}")
        return 1
    print(f"verified config[{GROUPBY_KEY!r}] present on every displayed run")

    # The comparison panels show the three arms plus every divergence
    # replicate. Group is the legend key, so each arm is ONE trace across all
    # its chain links and each replicate stays separate.
    main_groups = main_groups_check
    runset = wr.Runset(
        entity=ENTITY, project=PROJECT, name="all arms + divergence replicates",
        filters=f"group in {main_groups!r}",
    )

    def lines(metric, title, log_y=False):
        return wr.LinePlot(
            title=title, x="_step", y=[metric],
            log_y=log_y, smoothing_factor=0.0,
            groupby=GROUPBY_KEY, legend_position="east",
        )

    report = wr.Report(
        entity=ENTITY, project=PROJECT, title=TITLE,
        description="Fixed-batch optimizer comparison at 30B. Regenerate with "
                    "scripts/make_optcmp_report.py (groups, not pinned ids).",
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
                    filters=f"group in {LRFIND_GROUPS!r}",
                            )],
                panels=[wr.LinePlot(
                    title="LR finder: loss vs LEARNING RATE (log x)",
                    x="lr", y=["loss_metrics/global_avg_loss"],
                    log_x=True,
                    groupby=GROUPBY_KEY, legend_position="east",
                )],
            ),
            wr.MarkdownBlock(text=DIVERGENCE_MD),
            wr.PanelGrid(
                runsets=[wr.Runset(
                    entity=ENTITY, project=PROJECT,
                    name="sophiag: original + every divergence replicate",
                    filters=f"group in {['sophiag'] + rerun_groups!r}",
                            )],
                panels=[
                    wr.LinePlot(title="grad_norm: every replicate blows up "
                                      "(log scale; healthy arms peak in the tens)",
                                x="_step", y=["grad_norm"], log_y=True,
                                groupby=GROUPBY_KEY, legend_position="east"),
                    wr.LinePlot(title="loss: same window",
                                x="_step", y=["loss_metrics/global_avg_loss"],
                                groupby=GROUPBY_KEY, legend_position="east"),
                ],
            ),
        ],
    )

    if args.dry_run:
        print("dry run: report built, not saved")
        print(f"  title  : {TITLE}")
        print(f"  groups : {main_groups}")
        print(f"  blocks : {len(report.blocks)}")
        return 0

    prior = list(api.reports(f"{ENTITY}/{PROJECT}"))
    saved = report.save()
    url = saved.url
    print(f"REPORT URL {url}")

    # Delete the snapshots this save superseded. They are renderings of THIS
    # generator, so nothing unique is lost -- but the ids are captured BEFORE
    # the save so a concurrent regenerate's report can never be a target, and
    # the just-saved id is excluded explicitly.
    if not args.keep_old and prior:
        stale = [r for r in prior if r.id != getattr(saved, "id", None)]
        if stale:
            mutation = ("mutation DeleteView($id: ID!) "
                        "{ deleteView(input: {id: $id}) { success } }")
            sa = api._service_api
            ok = 0
            for r in stale:
                try:
                    res = sa.execute_graphql(query=mutation,
                                             variables={"id": r.id})
                    if ((res.get("data") or res).get("deleteView") or {}).get(
                            "success"):
                        ok += 1
                except Exception as e:
                    print(f"  could not delete {r.id}: {type(e).__name__}")
            print(f"pruned {ok} superseded report snapshot(s); "
                  f"pass --keep-old to retain them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
