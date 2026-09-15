# Sync 84 XPU: NOT blocked -- retracted

> [!CAUTION]
> **This page previously claimed the XPU smoke was blocked on an Aurora torch
> floor. THAT WAS WRONG, and wrong for three compounding reasons, all mine.
> The merged tree imports 12/12 on Aurora XPU with torch 2.13.** The original
> analysis is kept below the line because the three mistakes are worth not
> repeating.

## What is actually true (2026-09-15)

```
/flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/.venv
    torch 2.13.0.dev20260520+xpu          <- the dedicated XPU venv
    DataParallelMeshDims: PRESENT
    core torchtitan.distributed.fsdp: imports clean

after adding sync 84's two new deps (uv, --no-deps on torch_remat):
    12/12 ezpz modules import            agpt, moe, trainer, validator, train,
                                         zloss, mup, sharding, both registries
    torch unchanged: 2.13.0.dev20260520+xpu
```

Nothing about the torch version blocks the smoke. It is the ordinary sync-84
dependency install, the same one documented for every other host.

## The three mistakes

**1. I looked in the wrong directory for venvs.** I checked
`torchtitan/venvs/aurora/*` (two Feb/Mar 2026 dirs that fail on an MKL loader
error), hit the error, and moved on to the system module instead of resolving
it. `find -name pyvenv.cfg` across the project tree returns **19** venvs. The
live one is `projects/saforem2/torchtitan-ezpz/.venv`.

**2. I measured torch on a LOGIN node.** Login nodes run neither the prod nor
the test compute image. This is not a small discrepancy:

```
/opt/aurora/26.181.0   login node:  DOES NOT EXIST
                       next-eval compute node (x4000c5s6b0n0): EXISTS
```

That is the release holding `frameworks/2026.1.0`. I declared it "retired"
from a login-node `ls`. It is compute-node-only. Probe job 8828980 showed it.

**3. I truncated my own queue listing.** `qstat -Q | head -22` cuts off above
`next-eval`, and I read that as "no such queue" while holding the name. See
[[project_aurora_next_eval_queue]] -- it has no ACL, 6h walltime, 20 queued
per user, and runs a TEST bkc.

Any one of these alone would have been caught by the others. Together they
produced a confident, fully-documented, wrong conclusion.

## The 2N smoke: submitted 2026-09-15

Two matched jobs on `next-eval`, 2 nodes each, same venv, same script:

| job | tree | |
|---|---|---|
| `8829080` | `tt-sync84` (merged) | |
| `8829104` | `tt-sync84-pre` (merge commit's first parent) | baseline |

`scripts/sync_smoke_aurora.sh`, three configs x 3 deterministic steps at seed
42: `agpt_debugmodel` TP=1, `agpt_debugmodel` **TP=2**, `moe_debugmodel`.

The TP=2 arm is the point. #4533 rekeyed the local_map/SPMD contract, which
matches by positional-arg NAME and asserts ONLY under TP>1 -- a TP=1-only
smoke passes straight over the exact class of break this sync was most likely
to introduce. See [[project_tp_contract_rekeyed_third_time]].

Both trees were confirmed to import 12/12 on the same venv before submitting,
so a difference in the run is a difference in the code and not in the
environment.

## Run log

| job | queue | tree | result |
|---|---|---|---|
| `8829080` | next-eval | merged | `import_failed` @10s -- **environmental** |
| `8829104` | next-eval | pre-merge | `import_failed` @10s -- **identical**, rules out the merge |
| `8829136` | debug-scaling | merged | `IMPORT_OK`, training arms running |

### The next-eval failure was the venv, not the sync

Both trees died with

```
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'importlib.metadata'
```

A Python 3.12 stdlib module cannot be missing from a working interpreter --
that second line means the venv's python was not executing at all. Cause, from
`pyvenv.cfg`:

```
home = /opt/aurora/26.26.0/spack/.../python-3.12.12-nvje3vk/bin
```

Base interpreter from the **prod** image; `next-eval` runs the **test** BKC
(`/opt/aurora/26.181.0`). Verified the complement on a prod node: the same
interpreter exists and `importlib.metadata` imports fine.

So `next-eval`'s distinct image -- the thing that makes it valuable, since
`frameworks/2026.1.0` lives only there -- is the same thing that breaks a
prod-built venv. One fact, two consequences. See
[[project_aurora_venv_is_bkc_bound]].

**The matched baseline is what made this readable in one glance.** Only the
merged tree was under test; the pre-merge tree ran purely as a control and
failed byte-identically. Without it, a 10-second import failure on a fresh
merge looks like the merge.

## The rule

**Do not characterize a cluster's software stack from a login node**, and when
a venv path fails, find the right one (`find -name pyvenv.cfg`) rather than
falling back to the system module and concluding from that.

---

# Original (WRONG) analysis, retained

**Date:** 2026-09-15 · **Status:** cannot run today, and the reason is not the queue.

## The blocker, measured

```console
$ module load frameworks            # Aurora default, 2025.3.1
$ python -c "import torch; print(torch.__version__)"
2.10.0a0+git449b176

$ python -c "import torchtitan.distributed.fsdp"
ImportError: cannot import name 'DataParallelMeshDims' from 'torch.distributed.fsdp'
```

**The merged tree cannot be imported on Aurora at all.** HEAD needs torch 2.13
(`DataParallelMeshDims`, added there); every torch reachable on Aurora today is
2.10.

Available modules are `frameworks/2025.3.1` and `frameworks/2025.2.0`, both
torch 2.10. **`frameworks/2026.1.0` -- the RC with torch 2.13 that the ezpz
docs describe, and where compiled agpt TP=2 was validated -- is GONE.** It does
not appear in `module avail`, `/opt/aurora/*/frameworks/*2026*` does not exist,
and the `venvs/fw-2026.1-rc2` the journal references is not on disk. It was an
RC and has been retired.

Nothing else on the machine helps: no venv under
`AuroraGPT/foremans/.../torchtitan/venvs/` imports torch >= 2.13, and there is
no 2.13 XPU wheel staged to build one from.

## The `validation` queue does not help either -- two independent reasons

Checked because the RC quickstart
([aurora-quickstart-frameworks-rc.md](../guides/aurora-quickstart-frameworks-rc.md))
is written around it ("qsub -q validation ... -l select=2 -I").

**1. No access.** `foremans` is not in its `acl_users` (35 users, exact-matched
two ways). Queues actually open to us: `debug`, `debug-scaling`, `small`,
`medium`, `large`, `alcf_daos_cn`. Denied: `validation`, `run_next`.

**2. The module it depends on is gone anyway.** That guide pins
`FW=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0`. The entire
`/opt/aurora/26.181.0` release no longer exists -- only `25.190.0`, `26.26.0`,
and `default` remain. So even with queue access, `module load
frameworks/2026.1.0` has nothing to load.

**The RC quickstart guide is therefore stale end to end** -- both its queue and
its module are unavailable. It should not be followed as written until a
torch-2.13 stack returns.

> Parsing note: `qstat -Qf` wraps `acl_users` across lines with a leading tab.
> A naive `grep -c foremans` on the wrapped output returns 1 (substring hit on
> another username) and reads as "you have access". Unwrap with
> `tr -d '\n\t '` and match with `grep -x`. Same trap as the Sunspot
> `qstat` line-wrapping bug.

## What is NOT the problem

- **Queue access.** 977 nodes free; `debug-scaling` enabled, 1h max walltime,
  1 concurrent job per user. A 2N smoke would schedule fine.
- **`next-eval`.** CORRECTION: this queue EXISTS and is the best one available
  to us. An earlier revision of this page said "there is no such queue on
  Aurora" -- that was wrong, and wrong through carelessness: `qstat -Q | head -22`
  truncates the list above `next-eval`, and I read the truncated output as
  absence instead of grepping for the name. `~/test.sh` submitted to it fine
  (job 8828969). It was also already documented in
  [aurora-quickstart-tarball.md](../guides/aurora-quickstart-tarball.md) since
  2026-03-04.

  ```
  acl_user_enable = False                  no ACL -- open, unlike validation/run_next
  resources_max.walltime = 06:00:00        vs debug-scaling's 01:00:00
  max_queued = [u:PBS_GENERIC=20]          vs 1 concurrent job
  bkc_definition = compute_aurora_test_20260831T195218_1c6eebc_93ee049
  ```

  **The BKC line is the important one.** `next-eval` runs a TEST bkc where
  every other queue pins `compute_aurora_prod_20260828`. A different compute
  image can carry a different software stack, so the torch-2.10 finding below
  -- measured on a LOGIN node, which runs neither image -- may not hold there.
  Probe job 8828980 was submitted to settle it. Do not treat the torch floor as
  established until that reports.
- **The clone.** `sync84-trial` is checked out cleanly at
  `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/tt-sync84`. The
  production clone was left untouched (branch `ezpz`, 40 dirty files).

## What unblocking requires

One of, in rough order of cost:

1. **`frameworks/2026.1.0` (or later) comes back as a module.** This is the
   cheapest path and is not in our control. Worth asking ALCF whether a
   torch-2.13 stack is scheduled.
2. **Build a torch-2.13 XPU venv.** The last one took 4 packages plus 11
   transitive deps to get a rank up (journal 2026-08-30), and per
   [project_aurora_torch213_python314_weld] there is no shareable public wheel:
   ours was cp314-only and the public 2.14 nightly needs `libsycl.so.9` while
   Aurora has only `.8`.
3. **Use Sunspot instead** -- but it was unreachable all session
   (`Connection closed by UNKNOWN port 65535`, three attempts).

## Do not "work around" this by pinning the tree back

Reverting the sync to import on 2.10 would defeat the purpose: the whole point
of the smoke is to test THIS tree. A run on a downgraded tree proves nothing
about the merge.

## Related

Perlmutter cleared the numerics
([sync84-numerics-perlmutter.md](./sync84-numerics-perlmutter.md)) but cannot
answer the XPU question: its `pytorch/2.13.0` has `_fsdp_param.py` at 1095
lines with zero `_resolve_spmd_types_for_storage`, the SAME gap as ALCF, so
`spmd_types` is a torch version floor rather than an XPU bug. No NERSC run
settles it.
