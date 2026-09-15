# Sync 84 XPU smoke: BLOCKED on the Aurora torch floor

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
- **`next-eval`.** There is no such queue on Aurora. The closest name is
  `run_next`, which is ACL-restricted to `allcock`, `richp`, `ylan` -- not
  available to `foremans`. `debug-scaling` is the right queue for a 2N smoke
  anyway.
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
