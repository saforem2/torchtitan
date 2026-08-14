# `'Config' object has no attribute 'job'` (ezpz branch, 2026-08-13)

> [!IMPORTANT]
> **Fixed in `388a2bcbf`. If you are on the `ezpz` branch and every run dies
> before step 1, `git pull`.** Do not revert to `057eb8c0a` instead -- that
> restores a different, quieter bug (see "Do not revert" below).

## Symptom

Every run on the `ezpz` branch dies at the top of `train()`, on all ranks,
before step 1:

```
[rank0]:   File ".../torchtitan/experiments/ezpz/trainer.py", line 884, in train
[rank0]:     dump_folder=config.job.dump_folder,
[rank0]: AttributeError: 'Config' object has no attribute 'job'
```

Reported by @xuehu (`AuroraGPT_validation`, `torchtitan-ezpz-0813`).

## Cause

`dump_folder` is a **flat** field on `Trainer.Config`
(`torchtitan/trainer.py:77`). There is no `job` namespace on that config at
all, so `config.job` could never resolve.

`ae880c32d` (2026-08-13 09:53 CT) added the `dump_folder=` argument to the
flat-attention compat call and spelled it `config.job.dump_folder`. The
misspelling came from the helper's own docstring, which described the value as
`job.dump_folder`; that wording has been corrected too.

The seven other `dump_folder` call sites in the same file already used the flat
form -- including line 898, ten lines below. Line 884 was the lone outlier.

## Fix

```python
dump_folder=config.dump_folder,   # was config.job.dump_folder
```

## Verified on hardware

Aurora job **8754591** (debug, 1N/12 ranks, agpt-2b, seq 4096, 3 steps), both
refs on the same node, back to back:

| ref | line 884 (echoed by the job) | steps | AttributeError | rc |
|---|---|---:|---:|---:|
| `057eb8c0a` (broken) | `dump_folder=config.job.dump_folder,` | 0 | **11** | 143 |
| `388a2bcbf` (fixed) | `dump_folder=config.dump_folder,` | **3** | 0 | 0 |

Fixed half: `Training starts at step 1`, then loss 12.94 -> 13.54 -> 20.95 at
22.9% MFU. (Loss rising over 3 steps is expected -- warmup was compressed from
200 steps to 3, so the LR is far too high. Irrelevant here: the question was
whether `train()` is reachable at all.)

Broken half: 11 of 12 ranks raised at `trainer.py:884` before step 1; no rank
reached `Training starts`.

The A/B is guarded two ways: the job aborts if `torch` does not import (so a
setup failure cannot masquerade as the broken ref failing), and the broken half
only counts if the AttributeError string itself appears.

> [!NOTE]
> Job **8754490** was an earlier attempt that produced **no verdict** -- the
> scratch clone had no `.venv` (untracked, so `git clone` did not bring it) and
> `ezpz_setup` bailed before either half ran. The torch gate above was added in
> response. A first pass at 8754591 also mis-scored the fixed half as a failure:
> the scoring regex expected `step: N | loss:` but the real metrics format is
> `step:  1  loss: ...` with no pipe. The run was fine; the grep was wrong.

Static confirmation, independent of the run: parsing `Trainer.Config` lists
`dump_folder` as a field and contains no `job` field, and no `config.job.`
reference remains anywhere in the tree.

## Blast radius

Any `ezpz` checkout that pulled between 09:53 and 14:53 CT on 2026-08-13. It
fires before step 1, so it cannot corrupt a checkpoint or silently degrade a
run -- a job either started before the bad commit landed or never started.

**Long-running auto-retry jobs are the sharp edge.** Production job 8744247
(2098N, 5 trainers) had already imported the good code and kept stepping, but
its clone on disk was broken -- so any auto-retry relaunch would have re-imported
from disk and died on every rank. The fixed files were checked out into that
clone mid-flight (`git checkout origin/ezpz -- trainer.py ckpt_key_compat.py`,
both verified unmodified first) without disturbing the running trainers.

If you run long auto-retry jobs, remember the retry path re-reads the working
tree. A clone being "the one a healthy job is using" does not mean the code on
disk is the code that job is running.

## Do not revert

`ae880c32d` was itself a real fix: the flat-attention compat shim had been
silently no-op'ing because it resolved `<cwd>/<checkpoint.folder>` while the
checkpointer prepends `dump_folder`, making the real tree
`<cwd>/outputs/<checkpoint.folder>`. `read_metadata()` raised on the
non-existent path and a blanket `except` swallowed it.

Reverting to `057eb8c0a` trades a loud startup crash for that silent no-op,
which resurfaces later and much less legibly when resuming a pre-refactor
checkpoint:

```
Missing key in checkpoint state_dict: layers.0.attention.qkv_linear...
```

Take the fix, not the revert.
