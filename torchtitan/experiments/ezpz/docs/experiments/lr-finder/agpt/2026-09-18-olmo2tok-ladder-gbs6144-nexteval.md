# OLMo-3-vocab ladder LR finder: 4.64B / 9.48B / 26.2B at GBS=6144

**Date:** 2026-09-18 | **Jobs:** 8837334–8837399 (12 valid submissions; see map)
**Models:** agpt `{5,10,30}b_olmo2tok`, OLMo-3 tokenizer (vocab 100,352), seq 4096
**Tokenizer compatibility:** the retained `olmo2tok` config names and staged
`OLMo-2-1124-7B` asset path are historical; its core `tokenizer.json` is
byte-identical to `allenai/Olmo-3-1025-7B`.
**Data:** olmo-mix-1124 `wiki` subset, precached, via Grain + inline OLMo-3 tokenization
**Machine:** Aurora, `next-eval`, 64N x 3 jobs, 6h walltime | **Optimizer:** AdamW

Status: **FAILED BEFORE TRAINING.** All 12 jobs reached the runner but crashed
in 12–19 seconds with `ImportError: cannot import name 'SpmdType' from
'spmd_types'`. The borrowed image-independent venv carried `spmd-types 0.2.1`;
repository HEAD pins 0.2.5 and imports `SpmdType`. PBS still reported
`Exit_status=0` because the runner discarded the launcher status with
`|| true`. Commit `fa23dc6d1` adds an exact post-broadcast import/version
preflight and makes any non-OK arm fail the job. **No LR result was produced;
all 12 jobs must be resubmitted after the rebuilt tarball passes a compute-node
smoke.**

This file records the setup and the reasoning; numbers land in the Results
table when valid replacement jobs finish.

## Why

`agpt()` hands every size `lr=8e-4`, and the only guard (`_is_80b_config`)
gates on `flavor.startswith("80b")`, so none of these three is covered. For
scale: the 20B runs production at 2.28e-5, and the prior 30B olmo2tok sweep
(2026-08-23, GBS=960, Sunspot) put AdamW at **3.05e-05**. The inherited default
is therefore ~26x above the only measured number we have for this geometry.
The 80B shipped its inherited default against a documented ~7.4e-7 ceiling and
blew up at step 2.

## Geometry

| flavor | dim | layers | heads | kv | params | emb+head share |
|---|---:|---:|---:|---:|---:|---:|
| `5b_olmo2tok` | 3072 | 40 | 24 | 8 | 4.64B | ~14% |
| `10b_olmo2tok` | 4096 | 48 | 32 | 8 | 9.48B | |
| `30b_olmo2tok` | 6144 | 64 | 48 | 8 | 26.20B | |

Parameter counts are measured (meta-device build), not estimated.

## Setup notes that mattered

**Queue: next-eval, and it changes the node count.** `qstat -Qf next-eval`
shows NO nodect limits -- only walltime 00:05:00-06:00:00 and max_queued 20.
That dissolves the constraint `submit_lr_finder_30b_aurora.sh` was built
around, where 8-255 nodes above one hour was a queue dead zone forcing 256N
purely to reach a multi-hour queue. 64N is the measured throughput sweet spot
(exp06: 456 tps, 26.57% MFU) and is legal here. 64N x 12 = dp 768;
6144 = 768 * 2 * 4, so LBS=2/GAS=4.

**Runtime comes from the agpt-2b-v2 clone, not this repo.** next-eval runs a
TEST bkc image (26.181.0 / oneAPI 2026.1); this repo's `.venv` is based on a
`/opt/aurora/26.26.0/spack/...` python that does not exist there. Probe
(job 8837294) on a next-eval compute node:

- PROD spack python: **ABSENT**; the repo venv's interpreter is flatly
  "No such file or directory"
- clone venv: **OK** -- Python 3.14.2, torch 2.13.0.dev20260428+xpu, `xpu True`

So: code from this repo, runtime from that tarball, joined by `LRF_VENV_SRC`
plus an explicit `PYTHONPATH`. The tarball already carries torch 2.13 (not
2.10) and postdates the venv's `pyvenv.cfg`, so yeet will not skip it as stale.
Verified separately that HEAD's actual floor is satisfied on that runtime:
`from torch.distributed.fsdp import DataParallelMeshDims` -> IMPORT_OK, and
`torchtitan.distributed.fsdp` -> TT_FSDP_OK.

**Data: the config's own dataloader, NOT blendcorpus.** `run_lr_finder.sh`
hardcoded `--dataloader.dataset blendcorpus` on every invocation. Every
blendcorpus list on Aurora is gemma- or Llama-2-tokenized; feeding those ids to
a 100,352 embedding is the Polaris gibberish failure, and it fails SILENTLY --
ids below the embedding size index fine and the loss curve looks plausible.
`LRF_USE_CONFIG_DATALOADER=1` leaves the config's Grain path in place, which
tokenizes raw text inline with the OLMo-3 tokenizer.

**LR window 1e-8 -> 1e-3**, 150 probes (fraction 0.15 of 1000 steps) =
30 points/decade. Inherited from the 30B script: the script default 1e-6 -> 1.0
spends most probes above 1e-5 where everything has already diverged, and puts a
plausible optimum AT the first sample where a minimum is unresolvable.

**All four optimizers swept.** An earlier version of this file said AdamW
only, on the grounds that SophiaG's curve is "smooth". That was wrong: SophiaG
produced a real measured blow-up at 3.55e-04 in the 30B/GBS=960 sweep, the same
as the other two arms. "Smooth through the low band" describes the region below
the cliff and is true of every optimizer. A sweep finds a cliff if the window
reaches it -- which is exactly why the window is now per-optimizer.

SophiaG's 4/4 training divergences remain a reason not to LAUNCH it without
--grad-norm-abort=20.0, but they were never a reason not to MEASURE it.

**One size per job**, so a failure in one geometry does not cost the other two
their walltime.

## Preconditions verified before submitting

- LR-finder scheduler suspension still holds (`tests/test_lr_finder_scheduler.py`):
  suspended sees the full 4-decade spread, unsuspended collapses 5 decades to
  5x. Worth knowing: that file is a script, not a pytest module -- `pytest` on
  it collects nothing and exits 0, so this guard is NOT running in CI.
- ALCF preflight: PASS (worktree current, tokenizer, data-list, torch floor,
  ZE_FLAT_DEVICE_HIERARCHY=FLAT).
- OLMo-3 tokenizer assets staged at the historical
  `assets/hf/OLMo-2-1124-7B/` path: `tokenizer.json` plus three companions; the
  core tokenizer file is byte-identical to `allenai/Olmo-3-1025-7B`.
- olmo-mix wiki precached at 6.1 GB / 2 shards. (`du -sh` on an HF snapshot
  reports ~20K because snapshots are symlinks into `blobs/`; use
  `--dereference`.)

## Job map

3 sizes x 4 optimizers = 12 sweeps.

| size | adamw | mano | muon | sophiag |
|---|---|---|---|---|
| 5b  | 8837334 | 8837373 | 8837397 | 8837391 |
| 10b | 8837335 | 8837375 | 8837398 | 8837392 |
| 30b | 8837336 | 8837377 | 8837399 | 8837393 |

**Ignore any output from 8837374 / 8837376 / 8837378.** Those were the first
muon submissions and carry a 1e-3 window, 5.7x BELOW muon's measured 5.68e-03
cliff (30B/GBS=960). They will sweep, never diverge, and exit 0 with NO
suggestion -- an empty result that looks like a failed optimizer rather than a
mis-set window. PBS snapshots the script at qsub so they could not be fixed in
place, and `qalter -v` cannot help either: the old script sets LRF_MAX_LR with
an unconditional `export`, which overwrites anything injected. 8837397/8/9 are
the corrected replacements.

## Results

Pending.

| size | suggested LR | blow-up | min loss | LR at min |
|---|---:|---:|---:|---:|
| 5b | | | | |
| 10b | | | | |
| 30b | | | | |

Prior for comparison: 30B olmo2tok at GBS=960 on Sunspot gave AdamW 3.05e-05
(2026-08-23-30b-gbs960-three-optimizers.md). Optimal LR is batch-size
dependent, so the GBS=6144 answer is expected to differ -- the 80B's
small-batch finder said 1.1e-5 while its real production-batch ceiling was
~14x lower.

## Artifacts

`outputs/lr_finder_{5b,10b,30b}_olmo2tok_gbs6144/`
