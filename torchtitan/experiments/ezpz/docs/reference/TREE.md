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
| `production/` (187 files) | **mixed -- not yet split** | `agpt/`, `cpt/`, `figures/`, `metrics/`, `moe/`, `polaris/`, `rl/`, `sft/` | Chain status today; moving to `live/` + `reference/` + `records/proposals/` -- see [`../notes/production-move-plan.md`](../notes/production-move-plan.md) |
| `reference/` (51 files) | durable; corrected in place, never dated | `baselines/`, `configs/`, `guides/`, `known-bugs/`, `scaling/` | How do I do X? Why does Y break? |
| `records/` (217 files) | dated; immutable once written | `competitions/`, `evals/`, `experiments/`, `journal/`, `meeting-notes/`, `summaries/`, `upstream-sync/` | What happened on date X? What did we measure? |
| `outbound/` (27 files) | leaves the repo | `upstream-issues/` | What are we filing upstream? |
| `notes/` (5 files) | scratch; not yet classified | (files only) | Working plans and drafts |

## Root files

| File | Purpose |
|---|---|
| `README.md` | Prioritized landing page |
| `SESSION-RESUME-2026-08-21.md` | Session handoff note |
| `TODO.md` | Open work items |
| `journal.md` | Day-by-day session log (stub; monthly files under `records/journal/`) |
| `upstream-sync-79.md` | One upstream sync, kept at root for its inbound links |
| `upstream-sync.md` | Upstream merge log (stub; per-sync files under `outbound/upstream-sync/`) |

## Full tree

```
docs/
├── notes/                   (5 files)
├── outbound/                (27 files)
│   └── upstream-issues/         (27 files)
├── production/              (187 files)
│   ├── agpt/                    (92 files)
│   │   ├── 20b/                     (19 files)
│   │   ├── 2b/                      (29 files)
│   │   ├── 2b-mds/                  (12 files)
│   │   ├── 30b-exp/                 (10 files)
│   │   ├── 80b/                     (3 files)
│   │   └── historical/              (17 files)
│   ├── cpt/                     (13 files)
│   │   └── figures/                 (8 files)
│   ├── figures/                 (6 files)
│   ├── metrics/                 (11 files)
│   ├── moe/                     (1 file) 
│   │   └── 10b_2b_sdpa_ep/          (1 file) 
│   ├── polaris/                 (6 files)
│   │   └── figures/                 (5 files)
│   ├── rl/                      (26 files)
│   │   ├── grpo/                    (12 files)
│   │   ├── history/                 (9 files)
│   │   └── plans/                   (1 file) 
│   └── sft/                     (26 files)
│       └── agpt/                    (25 files)
├── records/                 (217 files)
│   ├── competitions/            (20 files)
│   │   ├── agpt2b-n2-1000steps/     (4 files)
│   │   ├── agpt2b-n2-gas8-1000steps/ (7 files)
│   │   ├── agpt2b-n8-10BT/          (4 files)
│   │   └── agpt2b-n8-r5/            (3 files)
│   ├── evals/                   (17 files)
│   │   ├── agpt/                    (14 files)
│   │   └── figures/                 (1 file) 
│   ├── experiments/             (150 files)
│   │   ├── agpt/                    (65 files)
│   │   ├── lr-finder/               (69 files)
│   │   ├── moe/                     (14 files)
│   │   └── synthetic/               (1 file) 
│   ├── journal/                 (5 files)
│   ├── meeting-notes/           (6 files)
│   │   └── agpt-sync/               (4 files)
│   ├── summaries/               (13 files)
│   └── upstream-sync/           (5 files)
└── reference/               (51 files)
    ├── baselines/               (3 files)
    ├── configs/                 (2 files)
    ├── guides/                  (11 files)
    │   └── training/                (1 file) 
    ├── known-bugs/              (25 files)
    └── scaling/                 (9 files)
        └── yeet_env/                (4 files)
```
