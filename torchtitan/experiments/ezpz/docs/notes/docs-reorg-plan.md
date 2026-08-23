# Plan: reorganize `experiments/ezpz/docs/`

> Written 2026-08-16. **PLAN ONLY -- nothing moved yet.**
>
> The corpus is 209 markdown files across 12 top-level directories. It is not
> neglected -- every file has been touched within 30 days and there are only 3
> orphans -- it has **outgrown its shape**. This proposes a new shape and, more
> importantly, an execution order that does not break the ~880 links pointing
> into it.

## Diagnosis

Measured, not asserted:

| metric | value |
|---|---|
| markdown files | 209 |
| top-level directories | 12 |
| `README.md` index files | 59 (28% of the corpus) |
| max path depth | 7 |
| files with no inbound link (orphans) | **3** |
| files not touched in 30 days | **0** |
| internal `*.md` -> `*.md` links | **630** |
| inbound links from OUTSIDE `docs/` (code, scripts, CLAUDE.md) | **~250** |

Three specific problems.

### P1. The top level mixes two incompatible axes

`production/`, `experiments/`, `evals/`, `scaling/`, `baselines/`,
`competitions/` classify by **kind of work**. `guides/`, `configs/`, `notes/`
classify by **kind of document**. So "where does an eval of a production
checkpoint go?" has no principled answer -- and empirically it goes in three
places.

### P2. Topics scatter with no declared canonical home

Umbrella `8756070` currently appears in six files across three directories:

```
README.md                                              (auto index)
production/README.md                                   (dashboard)
production/dispatch-log.md                             (cross-umbrella table)
experiments/agpt/aurora/20260816-umbrella-8756070.md   (the report)
reference/known-bugs/aurora-job-8756070-exit-14.md        (the -14 kill)
production/agpt/30b-exp/exp04-...md                    (passing reference)
```

Several of those are *correct* -- different layers of the same event. The
problem is that nothing declares which is authoritative, so a writer (me,
2026-08-16) added a dispatch-log row and a ticket before noticing the
per-umbrella report convention existed.

### P3. Three append-only files are ~40% of the corpus by size

| file | size | structure |
|---|---|---|
| `journal.md` | 252 KB | 77 `##` day sections, 2026-04 -> 2026-08 |
| `upstream-sync.md` | 184 KB | 81 sync sections |
| `meeting-notes/agpt-sync.md` | 76 KB | dated sections |

Nobody reads these top-to-bottom. They are also the **two most-linked files in
the entire repo** (84 and 28 inbound references), which constrains how they can
be split.

### What is NOT wrong

- **Depth.** `production/agpt/20b/n512/README.md` is depth 5 and correct -- it
  mirrors the actual chain topology, and every trajectory page is reachable
  from the dashboard. Do not flatten this.
- **Staleness.** Zero files older than 30 days.
- **The auto-generated index.** `refresh_docs_readme_table.py` works and should
  keep working; it just needs its paths updated once.

## Proposed shape: organize by LIFECYCLE, not by work-type

Every document answers one question: **does this expire?**

```
docs/
  live/         decays; must be refreshed to stay true
                  dashboard, dispatch-log, TODO, current-status pages
  reference/    durable; corrected in place, never dated
                  guides, known-bugs, configs, how-to
  records/      dated; immutable once written
                  experiments, evals, summaries, meeting-notes, journal
  outbound/     leaves the repo
                  upstream-issues, upstream-sync
```

The test is mechanical: *if this document is six months old and nobody has
touched it, is it wrong?* Live -> yes, that is a bug. Reference -> no, but it
may need correcting. Records -> no, it is a historical fact. Outbound -> no.

Today `production/` contains all four kinds at once: a live dashboard, dated
per-umbrella reports, durable failure taxonomy, and a draft ticket.

## File-by-file mapping

### -> `live/`

| from | to | note |
|---|---|---|
| `production/README.md` | `live/dashboard.md` | the chain status dashboard |
| `production/dispatch-log.md` | `live/dispatch-log.md` | appended every umbrella |
| `production/loss-dashboard.md` | `live/loss-dashboard.md` | |
| `TODO.md` | `live/TODO.md` | |
| `production/agpt/{2b,20b,80b}/**` | `live/chains/agpt/{2b,20b,80b}/**` | **keep depth**; these mirror chain topology |
| `production/{sft,rl,cpt,moe,polaris}/**` | `live/chains/{sft,rl,cpt,moe,polaris}/**` | |
| `production/queue-wait-analysis.md` | `live/queue-wait-analysis.md` | refreshed periodically |
| `production/scaling-performance.md` | `reference/scaling/performance.md` | see note below |
| `production/POST-TRAINING-2B.md` | `live/chains/agpt/2b/post-training.md` | |
| `production/figures/` | `live/figures/` | referenced by charts |

`scaling-performance.md` moves to reference rather than live because it is a
measured baseline, not a status page -- but it **is** currently stale for
20B/80B, which is exactly the kind of thing this split makes visible.

### -> `reference/`

| from | to |
|---|---|
| `guides/*.md` | `reference/guides/*.md` |
| `reference/known-bugs/**` | `reference/known-bugs/**` |
| `guides/training/**` | `reference/guides/training/**` |
| `configs/**` | `reference/configs/**` |
| `scaling/**` | `reference/scaling/**` |
| `baselines/**` | `reference/baselines/**` |
| `TREE.md` | `reference/TREE.md` (regenerated) |

### -> `records/`

| from | to |
|---|---|
| `experiments/**` | `records/experiments/**` |
| `evals/**` | `records/evals/**` |
| `summaries/**` | `records/summaries/**` |
| `meeting-notes/**` | `records/meeting-notes/**` |
| `competitions/**` | `records/competitions/**` |
| `journal.md` | `records/journal/` (split, see below) |
| `claude-sessions.md` | `records/claude-sessions.md` |
| `production/agpt/30b-exp/**` | `records/proposals/30b-exp/**` |

`30b-exp/` is a proposal plus its experiment log -- a record, not live status.
It sits under `production/` today only because that is where I put it.

### -> `outbound/`

| from | to |
|---|---|
| `upstream-issues/**` | `outbound/upstream-issues/**` |
| `upstream-sync.md` | `outbound/upstream-sync/` (split, see below) |

### Stays at root

`README.md` (the auto-generated index), `notes/` (scratch, including this file).

## Execution order -- this is the part that matters

**Do not do this as one commit.** 630 internal links plus ~250 from code,
scripts, and `CLAUDE.md` -- and the pinned production clones pull separately,
so a big-bang move leaves them with dangling links mid-run.

### Step 1 -- split the three append-only files (ZERO link risk)

Highest value, no risk, do it first. Split by the `##` sections that already
exist, and **leave a stub at the original path** so all 84 + 28 + 15 inbound
links keep resolving.

```
journal.md  (252 KB, 77 sections)  ->  records/journal/2026-04.md   (8 sections)
                                       records/journal/2026-05.md   (21)
                                       records/journal/2026-06.md   (32)
                                       records/journal/2026-07.md   (14)
                                       records/journal/2026-08.md   (2, current)
                                       journal.md  -> stub: index + link to current

upstream-sync.md (184 KB, 81 syncs) -> outbound/upstream-sync/NN-<range>.md
                                       upstream-sync.md -> stub index

meeting-notes/agpt-sync.md (76 KB)  -> records/meeting-notes/agpt-sync/YYYY-MM.md
                                       agpt-sync.md -> stub index
```

Removes ~512 KB of unreadable scroll and breaks nothing. **If only one step
ever happens, make it this one.**

### Step 2 -- declare canonical homes (ZERO risk, no moves)

One line at the top of each recurring-topic doc:

```markdown
> **Canonical record:** this page is authoritative for <X>.
> Other mentions cross-reference it; do not duplicate status here.
```

Apply to: `dispatch-log.md` (cross-umbrella taxonomy), each per-umbrella report
(that run's result), each chain README (that chain's status), each known-bug
page (that failure mode). Fixes P2 without moving a byte, and prevents the
mistake I made on 08-16.

### Step 3 -- move directories, ONE PER COMMIT

For each directory, in this order (least-linked first):

1. `git mv` the directory
2. rewrite internal links: `grep -rl 'old/path' docs | xargs sed -i 's|old/path|new/path|g'`
3. rewrite external refs: same sweep over `*.py`, `*.sh`, `.claude/CLAUDE.md`,
   `experiments/ezpz/.claude/CLAUDE.md`
4. verify: every `](...md)` target resolves (script below)
5. `refresh_all.sh`
6. commit

Suggested order, by inbound-link count ascending:

```
baselines/  configs/  competitions/  scaling/  evals/
meeting-notes/  summaries/  upstream-issues/
experiments/          <- 73 files, heavily linked
guides/               <- 25 inbound from code on one file alone
production/           <- LAST, most linked, and live during production
```

**`production/` moves last and only during a quiet window** -- it is read by
the dashboard tooling and referenced from the pinned clones.

### Step 4 -- link checker in `refresh_all.sh`

```bash
# fail if any internal md link is dangling
find docs -name '*.md' -print0 | while IFS= read -r -d '' f; do
  grep -oE '\]\([^)#]+\.md' "$f" | sed 's/^](//' | while read -r t; do
    [ -e "$(dirname "$f")/$t" ] || echo "DANGLING  $f -> $t"
  done
done
```

Cheap, and it makes step 3 safe to repeat.

## Recommended stopping point

Steps 1 and 2 deliver most of the value for almost none of the risk:

- **512 KB** of append-only scroll becomes navigable
- topic scatter gets a declared owner
- nothing moves, no link breaks, no clone gets a dangling reference

Step 3 is worth doing, but only with the link checker from step 4 in place
first, and `production/` should wait for a window with no umbrella running.

## Open questions

1. **Does `notes/` survive?** It has 2 files and is a scratch drawer. Could
   fold into `records/`, or keep deliberately as the "not yet classified" spot.
2. **`experiments/` vs `records/experiments/`** -- the extra nesting buys
   consistency but costs a path segment on 73 files. Worth it only if the
   four-way split is actually adopted.
3. **Should `live/` pages carry a machine-checkable freshness stamp?**
   `check_stale_docs.sh` already exists; a lifecycle split would let it run
   *only* against `live/`, where staleness is genuinely a bug, instead of
   flagging records that are correctly frozen.
