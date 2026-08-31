# `docs/` tree map

> One-page index of every directory and key file under
> `torchtitan/experiments/ezpz/docs/`. When you're not sure where
> something lives or where new content should go, scan this page
> instead of `cd`'ing around.
>
> See [`README.md`](./README.md) for the prioritized landing-page
> view (sections ordered by importance, with last-modified dates).
> This file is the structural map.
>
> Last updated: 2026-08-31 (full tree regenerated from `find` against the
> real directory; the previous revision predated `notes/`, `ops/`,
> `production/{cpt,rl,sft,polaris,metrics}/`, `experiments/{mup,optimizer-comparison,synthetic}/`
> and several top-level pages, and carried per-chain step counts that had
> gone thousands of steps stale)

## At a glance

| Directory | Contents | When to look here |
|---|---|---|
| `production/` | Live training-job tracking -- per-model, per-node-count; also holds `cpt/`, `sft/`, `rl/` (GRPO), `polaris/` and the `metrics/` CSV store | "How is the canonical chain doing right now?" |
| `evals/` | lm-eval results — per-model, with v1-vs-v2 plots | "What benchmark scores has the v2 run hit?" |
| `guides/` | Big-finding writeups, operational notes, how-to docs | "How do I do X?" or "Why does Y break?" |
| `experiments/` | Per-machine smoke / benchmark / LR-finder reports | "What did we measure on Sunspot 8N?" |
| `scaling/` | Per-model scaling-study results (TPS/MFU vs N) | "How does throughput scale at 64N vs 256N?" |
| `competitions/` | Optimizer speedrun leaderboards | "Which optimizer won the GBS=384 sprint?" |
| `meeting-notes/` | AuroraGPT sync agendas + action items | "What did we agree to in last week's sync?" |
| `summaries/` | 2-week / monthly retrospectives | "What happened in the past 2 weeks?" |
| `upstream-issues/` | Repros + patches we're filing back to `pytorch/torchtitan` | "What PRs are we trying to land upstream?" |
| `baselines/` | Reference training curves + benchmarks | "What's the AdamW baseline at GBS=48?" |
| `configs/` | Model config docs (architecture, registered names) | "What's the dim/n_heads of `agpt_50b_wide`?" |
| `notes/` | Planning + strategy notes (data strategy, reorg plan, slides) | "What was the plan for X?" |
| `ops/` | ALCF tickets and operational escalations | "Did we file a ticket for this?" |
| `journal.md` | Day-by-day session log | "What did we do on date X?" |
| `TODO.md` | Open work items | "What's pending?" |
| `upstream-sync.md` | Log of upstream merges + replays | "What did the last merge from main pull in?" |
| `README.md` | Prioritized landing page | First-time landing |

## Full tree

```
docs/
├── README.md              <- Prioritized landing page + auto-generated "Recently Updated" table
├── TREE.md                <- (this file) structural map of the docs tree
├── journal.md             <- Day-by-day session log; most recent first
├── TODO.md                <- Open work items
├── upstream-sync.md       <- What we pulled from pytorch/torchtitan main; what was replayed onto agpt/moe
├── upstream-sync-79.md    <- 79th sync, written up separately (four stacked defects from one PR)
├── upstream-sync-80th-status.md  <- 80th sync: what works, what is deferred
├── claude-sessions.md     <- Per-session Claude work log
├── HANDOFF-aurora-2026-08-26.md      <- Point-in-time session handoff (Aurora)
├── HANDOFF-20b-polaris-2026-08-26.md <- Point-in-time session handoff (Polaris 20B chain)
├── SESSION-RESUME-2026-08-21.md      <- Point-in-time session-restart notes
│
├── production/            * LIVE training tracking. Updated every session.
│   ├── README.md                  <- Top-level snapshot of every active trajectory
│   ├── scaling-performance.md     <- Apr 18-21 production scaling experiments (historical)
│   ├── dispatch-log.md            <- Per-umbrella seat dispatch record
│   ├── loss-dashboard.md          <- Cross-chain loss view
│   ├── queue-wait-analysis.md     <- PBS queue-wait study
│   ├── POST-TRAINING-2B.md        <- 2B post-training plan
│   ├── figures/                   <- Cross-chain production figures
│   ├── metrics/                   <- Per-chain CSV export + manifest.json
│   ├── agpt/                      <- Dense (agpt) production
│   │   ├── README.md              <- Index: 2B/20B/80B chain tables, v1-vs-v2 overlays
│   │   ├── 2b/{README.md,figures/,n256/,n512/,n1024/}
│   │   ├── 20b/{README.md,figures/,n256/,n512/,n1024/}
│   │   ├── 80b/{README.md,n4/,n512/}
│   │   ├── 30b-exp/               <- 30B-exp proposal + exp01..exp08 writeups
│   │   ├── 2b-mds/                <- Pre-torchtitan Megatron-DeepSpeed 2B SophiaG baseline
│   │   └── historical/v1-bf16/    <- Collapsed v1 (frozen-norm) trajectories
│   ├── cpt/                       <- Continued pre-training (olmo x dolmino ratio sweep)
│   ├── sft/agpt/                  <- Supervised fine-tuning recipes + evals
│   ├── rl/                        <- GRPO production work (grpo/, history/, plans/, monarch.md, trl.md)
│   ├── polaris/                   <- Polaris (A100) production chains
│   └── moe/10b_2b_sdpa_ep/        <- 10B-2B MoE SDPA + EP experiment
│
├── evals/                 * lm-eval results.
│   ├── README.md                  <- Top-level eval index
│   ├── eval-landscape-2026-07.md  <- Modern-suite review
│   ├── mmlu-letter-prior-at-chance.md
│   ├── figures/
│   └── agpt/{2b,20b,2b-mds}/      <- Per-model results + figures/
│
├── guides/                * Big findings + operational notes. Check before suggesting work.
│   ├── aurora-quickstart-frameworks-rc.md  <- START HERE for new Aurora setups (frameworks/2026.1.0)
│   ├── aurora-quickstart-tarball.md        <- The pre-RC shared-tarball path
│   ├── frameworks-rc-validation.md
│   ├── bad-node-failover.md       <- Failover wrapper v2. See ../../tests/failover/
│   ├── checkpoint-on-signal.md
│   ├── hf-dataset-offline-cache.md
│   ├── known-issues.md            <- Catch-all live workarounds
│   ├── loss-reporting-tp-dist-reduce.md
│   ├── perlmutter-debug-host.md
│   ├── polaris-fresh-venv.md
│   ├── running-with-newer-pytorch.md
│   ├── spmd-backend-status.md
│   ├── training-dtype-bf16-norm-freeze.md  <- Root cause of the v1 -> v2 restart
│   ├── xpu-attention-issues.md
│   ├── training/agpt_80b.md
│   └── known-bugs/                <- ~33 per-bug deep dives (moe-tp2-wo-placement,
│                                     blendcorpus-*, fw-rc-*, sophiag-*, polaris-*, ...)
│
├── experiments/           <- Per-run reports. Every job that produced data lands one here.
│   ├── README.md
│   ├── agpt/{aurora,polaris,sunspot}/   <- Per-machine smoke / benchmark / incident reports
│   ├── moe/{aurora,polaris,sunspot}/
│   ├── lr-finder/agpt/{2b,20b,80b}/ + lr-finder/moe/{debugmodel,500m,2b,4b,7b}/
│   ├── mup/                       <- muP ladder: audit, design, staged plan
│   ├── optimizer-comparison/      <- Fixed-batch AdamW vs Mano vs SophiaG
│   └── synthetic/aurora/
│
├── scaling/               <- Per-model scaling-study results (TPS/MFU vs node count)
│   ├── README.md, agpt-2b.md, agpt-20b.md, agpt-80b.md, moe.md
│   └── yeet_env/                  <- yeet-env tarball broadcast scaling (8N -> 4096N)
│
├── competitions/          <- Optimizer speedrun leaderboards (W&B link in each)
│   └── agpt2b-{n2-1000steps,n2-gas8-1000steps,n8-10BT,n8-r5}/
│
├── meeting-notes/         <- agpt-sync.md (one stable file with ## YYYY-MM-DD sections)
├── summaries/             <- Weekly / 2-week retrospectives, named by END date
├── notes/                 <- Planning + strategy notes (data strategy, reorg plan, slides)
├── ops/                   <- ALCF tickets and operational escalations
├── upstream-issues/       <- Repros + patches we file back to pytorch/torchtitan (.md + repro *.py/*.sh)
├── baselines/             <- Reference training curves + benchmarks (README.md + *.json)
└── configs/               <- Model config docs
    ├── dense.md                   <- 2B / 20B / 50B_wide / 80B
    └── moe.md                     <- 500M-10B MoE variants
```

## Where to put new content

| Type of content | Goes under |
|---|---|
| New training run that produced data | `experiments/<module>/<machine>/<YYYYMMDD>-<purpose>.md` |
| New active production trajectory | `production/<module>/<model>/n<NODES>/README.md` |
| New lm-eval result for a checkpoint | Append to `evals/<module>/<model>/README.md` results table |
| New big finding (post-mortem, root cause writeup) | `guides/<finding>.md` (or `guides/known-bugs/<bug>.md` for narrower scope) |
| Per-day status update | Append top of `journal.md` |
| Upstream PR draft / repro | `upstream-issues/<PR-name>.md` + log in `upstream-sync.md` |
| Anything ephemeral (temp diagnostics, scratch notes) | NOT here — use `~/scratch/` or a TODO; don't litter the docs tree |

## Conventions

- **Per-day artifacts** use `YYYY-MM-DD` filenames (or `YYYYMMDD-HHMMSS-<purpose>` for experiments).
- **Per-recurring-meeting docs** use a stable filename with `## YYYY-MM-DD` sections inside (see `meeting-notes/agpt-sync.md`).
- **Figures** live under `<page>/figures/` next to the page that references them.
- **Cross-links**: production READMEs link to evals READMEs and vice versa; the bf16 freeze guide links to both training overlays and lm-eval figures.
- **Append-only** for production progress tables — don't rewrite history when a new chain link finishes.
- **Collapse stale content** into `<details>` blocks rather than deleting (e.g. v1 trajectories under each per-trajectory README).
