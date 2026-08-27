# Handoff -- Aurora, 2026-08-26

Session `b7be7bc5-96dd-437e-a979-6734b7d7b0d4` (`aurora-tt-ezpz`), handed to
`mbph`. Resume with:

```bash
cd ~/projects/saforem2/torchtitan && claude --resume b7be7bc5-96dd-437e-a979-6734b7d7b0d4
```

Cluster repo: `/flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz`,
branch `ezpz`. Everything below is committed and pushed.

## Production: nothing running

| job | what | state |
|---|---|---|
| `8784460` | umbrella, 2098 nodes, 12h | Q, `score_boost=0`, ~9h eligible |
| `8784462` | continuation | H on `afterany:8784460` |

Comment is "Not enough free nodes available"; free nodes went 329 -> 95 over
the afternoon. The boost was lost on resubmit (a fresh `qsub` starts at 0) --
restoring it is an ALCF request, not something we can set.

Five older chain jobs (`8687863`, `8752939`, `8752824`, `8756071`, `8756072`)
are `Hold_Types = u` -- plain USER holds. The 08-26 sync entry calls them
"held behind checkpoint conflicts", which does not match what PBS reports.
Left alone: releasing them could collide with the umbrella on the same ckpt
dirs.

## What landed today

**The umbrella lost 7 of 12h to a regex.** `ezpz` 0.21.x matched progress with
`r"\bstep=\d+"`; `torchtitan` prints `step: 21800`. Seats that trained
201/504/782 steps all scored zero progress, so failover filed
`stuck_pre_training` and abandoned them with 10 spares free. t3 was killed
immediately after checkpointing step 21,800. Fixed by upgrading both venv
tarballs to `ezpz` 0.27.3, regex verified INSIDE each archive.

> **Only two tarballs are real.** `agpt-20b-n256` and
> `agpt-2b-constlr-from9200` symlink to `agpt-20b-v2` / `agpt-2b-v2`, so a
> clone's `.venv` version is irrelevant at runtime. The tarball is what ships.

**Seat supervisor.** A seat that exited used to idle its slice for the rest of
the job -- the other 5h of `8773440`. `supervise_trainer()` relaunches, guarded
by deadline (45m) / two quick deaths / `rc=0` / `SEAT_RELAUNCH=0`. 5/5 against
a fake-trainer harness.

**Aurora frameworks RC (`2026.1.0`) trains** -- job `8784615`, 2N x 12, 10/10
steps, loss 10.865 -> 9.059. First training of any kind on that stack; the
2B/MoE/80B results in `frameworks-rc-validation.md` are Sunspot. Recipe and
seven traps are in the [experiment README](../README.md).

**prod_dash**: charts survive a width change; the backbone cache moved out of
`/tmp`; and the local cache writer, which never existed -- `_cached_payload`
read a file nothing in the codebase wrote.

## Open item 1: MoE at TP>1

`wo` receives `Shard(dim=0)` where `rowwise_config` expects `Partial(sum)`.
Measured (job `8785537`):

| arm | q entering the unflatten | q leaving | rc |
|---|---|---|---|
| agpt+det | -- (compile AOT assertion) | -- | 1 |
| agpt+det+nocompile | `(512, 8, 16)` **PLAIN** | `(1, 512, 8, 16)` | 0 |
| moe+det | `(512, 16, 192)` **`Shard(dim=1)`** | `(1, 1024, 8, 192)` | 143 |

**agpt's wrapper receives plain local tensors; moe's receives DTensors.** agpt
sees 8 heads (already local at tp=2) so the unflatten is exact. moe sees 16 --
the global count -- on a tensor holding half locally, so the reshape spills the
missing heads into the sequence axis, and the final `view` collapses
`Shard(dim=1)` -> `Shard(dim=0)`. `wo` rejecting it is the last link, not the
bug.

> [!WARNING]
> **Do not make the unflatten TP-aware.** It is correct given local tensors,
> which is what it is documented to receive and what agpt gets. Patching it
> papers over a `local_map` that is not converting, and breaks agpt.

`local_map` IS installed for moe (the traversal in
`protocols/module.py:263-286` is unconditional). The config is there; the
conversion is not happening. Fix that.

Full writeup, including everything eliminated:
[`known-bugs/moe-tp2-wo-placement.md`](guides/known-bugs/moe-tp2-wo-placement.md).

Side result: **agpt TP=2 needs `--compile.no-enable`** -- the torch-2.13
compile+AC+TP `DeviceMesh` AOT assertion, previously recorded only for the 80B
family, fires at `agpt_debugmodel` scale too.

## Open item 2: probes still in the tree

`EZPZ_MLA_PLACEMENT_PROBE=1`-gated instrumentation in **both**
`moe/model.py` and `agpt/__init__.py`. Silent when unset, but it is temporary
code sitting in the agpt path production runs on. Remove once item 1 is fixed.

## A documented "Intel limitation" that is not one

`known-bugs/xpu-graphs-block-oneccl-collectives.md` said XPU graphs cannot
capture oneCCL collectives, with an Intel ask drafted at the bottom. Job
`8785582`:

| `CCL_OP_SYNC` | `all_gather` inside capture |
|---|---|
| unset | **OK -- capturable** |
| `0` | **OK -- capturable** |
| `1` | FAILS with the documented error |

`ezpz_setup_env` exports `CCL_OP_SYNC=1` unconditionally (three sites in
`ezpz-utils`), so every run that reached the old conclusion had it set without
anyone choosing it.

- **Do not file that Intel ask.**
- **Do not blanket-remove the export** -- `bitwise_sync_check.sh` sets it
  deliberately for determinism. Capture and that flag are mutually exclusive;
  that is the finding.

## Backlog

`docs/reorg-lifecycle`: ~60 commits behind `ezpz`, 593 files diverged, and it
now overlaps `prod_dash.py`, which changed three times today. The user wants an
audit before merging -- they flagged low confidence in it after finding
timestamped directories. It gets worse every session.

## Environment gotchas that cost jobs today

- **`ZE_FLAT_DEVICE_HIERARCHY=FLAT`** -- a node is 6 cards x 2 tiles = 12 XPUs.
  Without FLAT torch sees 6 and `ezpz` rejects `--nproc 12` with
  `ngpus must be > 0 and <= 6`. Every other script in the tree exports it.
- **`dump_folder` defaults to `./outputs` for every model**, so two configs
  share `outputs/checkpoint/` and load each other's checkpoints
  (`Size mismatch ... [512, 2048] vs [256, 256]`). Use a per-run `mktemp -d`.
- **`--checkpoint.no-enable` disables SAVING, not loading.**
- **Rank tracebacks are in `logs/<module>/<ts>-rank0.jsonl`**, not stdout. A
  launch printing only `Execution finished with 143` has the real error there.
- **The umbrella `.o` banner stops early.** `8773440` banners 1h47m of a 12h
  run while seats keep training. Read
  `logs/multi-autoretry-<jobid>/trainer-N-*.console.log`.
- **`-P` does not protect torch.** The resolver pulled a PyPI `torch-2.13.0`
  over the XPU build with `-P torch -P pytorch-triton-xpu` set; all installs
  printed `ok` and only an explicit `torch.__version__` assert caught it. Take
  `triton` with it when evicting.

## Not transferred

- Background monitors. Re-arm if wanted.
- PBS state -- jobs keep running on ALCF, query with `qstat`.
- Anything said after the snapshot. Re-run the script with `FORCE=1` for a
  fresher one.
- ALCF SSH from `mbph` needs MFA; log in by hand once so the ControlMaster is
  warm, or the resumed session cannot reach `qstat`.
