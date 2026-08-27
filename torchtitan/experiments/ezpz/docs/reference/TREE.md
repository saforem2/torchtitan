# `docs/` tree map

> One-page index of every directory and key file under
> `torchtitan/experiments/ezpz/docs/`. When you're not sure where
> something lives or where new content should go, scan this page
> instead of `cd`'ing around.
>
> See [`README.md`](../README.md) for the prioritized landing-page
> view (sections ordered by importance, with last-modified dates).
> This file is the structural map.

## Organized by lifecycle

Every document answers one question: **does this expire?** The test is
mechanical -- *if this page is six months old and nobody has touched it, is it
wrong?*

| Top level | Expires? | Contents | When to look here |
|---|---|---|---|
| `live/` (181 files) | **yes -- decays** | `chains/`, `figures/`, `metrics/`, plus the dashboard and dispatch log | What is running right now? Where is each chain? |
| `reference/` (56 files) | durable; corrected in place, never dated | `baselines/`, `configs/`, `guides/`, `known-bugs/`, `scaling/` | How do I do X? Why does Y break? |
| `records/` (234 files) | dated; immutable once written | `competitions/`, `evals/`, `experiments/`, `journal/`, `meeting-notes/`, `proposals/`, `summaries/`, `upstream-sync/` | What happened on date X? What did we measure? |
| `outbound/` (27 files) | leaves the repo | `upstream-issues/` | What are we filing upstream? |
| `notes/` (6 files) | scratch; not yet classified | (files only) | Working plans and drafts |

A page that decays goes in `live/`. A page that is a dated observation goes in
`records/` and is never edited afterward. A page that should stay true goes in
`reference/` and is corrected in place rather than superseded.

## Root files

Only three, all indexes into the trees above.

| File | Purpose |
|---|---|
| `README.md` | Prioritized landing page |
| `journal.md` | Day-by-day session log (index; monthly files under `records/journal/`) |
| `upstream-sync.md` | Upstream merge log (index; monthly files under `records/upstream-sync/`) |

Append-only logs are split by month rather than kept as one file: `journal.md`
had reached 252 KB and `upstream-sync.md` 184 KB, which is past the point where
anyone scrolls them. Both keep their path, so inbound links still resolve.
Individual syncs and journal days are `##` sections inside the monthly file,
not separate files.

## Full tree

```
docs/
├── live/                   (181 files)
│   ├── chains/                 (159 files)
│   ├── figures/                (6 files)
│   └── metrics/                (11 files)
├── notes/                  (6 files)
├── outbound/               (27 files)
│   └── upstream-issues/        (27 files)
├── records/                (234 files)
│   ├── competitions/           (20 files)
│   ├── evals/                  (15 files)
│   ├── experiments/            (159 files)
│   ├── journal/                (5 files)
│   ├── meeting-notes/          (6 files)
│   ├── proposals/              (10 files)
│   ├── summaries/              (13 files)
│   └── upstream-sync/          (5 files)
└── reference/              (56 files)
    ├── baselines/              (3 files)
    ├── configs/                (2 files)
    ├── guides/                 (12 files)
    ├── known-bugs/             (27 files)
    └── scaling/                (10 files)
```

> Regenerate the counts above from disk rather than editing them by hand --
> every directory row in the previous version of this page was stale.
