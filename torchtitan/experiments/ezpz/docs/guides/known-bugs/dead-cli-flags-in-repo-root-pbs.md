# Dead CLI flags: 99 PBS scripts at the repo root would abort today

**Status:** documented, deliberately NOT mass-fixed. Read before reusing any
`.pbs` at the repo root.

## What

Upstream `#4088` (Grain dataloader) and `#4121` (sequence -> token units)
removed four CLI flags. **99 of the ~100 `.pbs` scripts at the repo root still
pass at least one of them on a live (non-comment) line**, and every one of
those scripts would die in flag parsing before reaching a single training step:

| dead flag | removed by | replacement |
|---|---|---|
| `--training.seq-len` | #4121 | `--training.max-context-length` |
| `--training.local-batch-size` | #4121 | `--training.num-tokens-per-microbatch-per-dp-rank` |
| `--training.global-batch-size` | #4121 | `--training.num-tokens-per-train-step` |
| `--dataloader.num-workers` | #4088 | (gone; `GrainDataLoader.Config` has no such field) |

Post-#4121 the two token fields must be EQUAL:
`num_tokens_per_microbatch_per_dp_rank == max_context_length`. Production has
zero margin here.

## Why this keeps biting

The failure is loud when it happens -- `Unrecognized options: ...` -- but it is
**invisible until the job runs**, and by then the allocation is spent. It has
now cost real time four separate ways:

- **Phase 2 of the optimizer comparison**: five consecutive launch failures at
  full scale, the first of which was exactly this
  (`Unrecognized options: --training.seq-len, --training.local-batch-size,
  --training.global-batch-size`). See the "five failures" section of
  `docs/experiments/optimizer-comparison/README.md`.
- **`lrfind_opt.pbs`** still carried `--dataloader.num-workers=2` on 2026-08-30,
  seven days after the Phase 2 cascade fixed the same class of bug in
  `optcmp.pbs`. It was caught by a preflight before the Muon sweep launched,
  and it would have killed an adamw/mano/sophiag re-sweep just as dead.
- **`sync79_smoke.pbs`** would have produced three VOID arms against a clean
  merge and read as a merge failure. `sync82_smoke.pbs` was written from
  scratch instead.
- **`diagsmoke.pbs`** is stale on the same flags.

The common thread: a script that worked when it was written, was not re-run
for weeks, and was then reached for as a template.

## Why they are not all fixed

Most of these are one-off diagnostics from investigations that have concluded
(`det_*.pbs`, `parity*.pbs`, `spmd*.pbs`, `tok_*.pbs`, `mesh*.pbs`, ...). The
newest of the 99 is dated **2026-08-23**, so none of them have been touched
since the renames landed. Rewriting ~100 scripts blind -- most of which will
never run again, none of which can be tested without an allocation -- risks
introducing errors in files nobody reads, to fix a failure mode that announces
itself immediately and costs one queue slot.

A naive bulk `sed` is also wrong: `sync82_smoke.pbs` matches the grep only
because a comment *explains* the removal. Any automated fix must distinguish
comment lines from live flag lines.

## What to do instead

**Before reusing any repo-root `.pbs`, grep it:**

```bash
grep -nE '^[^#]*--(dataloader\.num-workers|training\.seq-len|training\.local-batch-size|training\.global-batch-size)' <script>.pbs
```

Non-empty output means the script is stale. Fix that script, not all of them.

**Use a preflight that parses the flags extracted from the script itself**, not
a retyped copy -- `optcmp.pbs` does this (see its `preflight.py` heredoc), and
that is what caught the `lrfind_opt.pbs` staleness. A retyped list passes while
the script still carries the fatal flag.

**Known-good templates as of 2026-08-30:** `optcmp.pbs`, `sync82_smoke.pbs`,
`lrfind_opt.pbs`.
