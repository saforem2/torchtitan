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

## THE REAL FINDING: #4419 blocks the merge on Aurora

Job `8829185` (merged tree, debug-scaling, 2N) got past import, past the
tokenizer, built the model, and died in FSDP setup. All three arms, 23 ranks
each, identically:

```
ValueError: When dp_mesh_dims is provided, all parameters must be DTensors on
the full SPMD mesh (e.g. via distribute_module). Got plain tensor for
parameter 'weight'.
```

**This is the exact failure the page flagged as UNVERIFIED before the run.**
It was pre-registered in both config registries, and it reproduced.

### Mechanism, measured on both machines

Aurora's XPU torch, checked directly:

```
torch 2.13.0.dev20260520+xpu
_fsdp_param.py                     1093 lines
_resolve_spmd_types_for_storage       0
self.is_spmd_types                    0
get_local_type                        0
```

pytorch `da19cbd78` (#181519, 2026-06-23) added the `FSDPParam.__init__` block
that converts an annotated plain tensor into a DTensor before the
`is_spmd_mesh and not is_dtensor` check can fire. **Our torch predates it**, so
an annotated plain tensor falls straight through to the raise. Perlmutter's
`pytorch/2.13.0` shows the same gap at 1095 lines -- two independent builds,
same conclusion. See `known-bugs/spmd-types-plain-tensor.md`, which
root-caused this on 2026-08-20.

ezpz's workaround was to pin `partial_dtensor`. **#4419 deleted that backend**,
so the escape hatch is gone.

### The baseline settles it: pre-merge TRAINS, merged does not

`8829243`, pre-merge tree, same machine / venv / script / seed as `8829185`:

```
VERDICT: ok        all three arms rc=0, ZERO DTensor errors

1-agpt_debugmodel  TP=1   10.88382  10.75382  10.48337
2-agpt_debugmodel  TP=2   10.88071  10.70248  10.56243
3-moe_debugmodel          12.90379  12.57026  11.45320
```

Merged, identical conditions: dead in FSDP setup on all three arms, 23 ranks
each. **The trees differ on one line** -- `config_registry.py:252` pins
`partial_dtensor` pre-merge, and #4419 deleted both the pin and the backend.

The TP=2 arm passing matters on its own: that is the configuration the #4533
local_map contract asserts under, so there is now a known-good reference to
diff the merged tree against once it can run.

### No configuration workaround exists

Re-pinning is not an option. In merged core:

```
partial_dtensor references:  0
spmd_backend  references:  0   (config field deleted outright)
```

The backend is removed, not merely unselected -- there is no flag, no config
field, nothing to set. The merged tree cannot be made to train on Aurora by
configuration alone.

### CLOSED 2026-09-16: #181519 is ABSENT on all FOUR reachable torch builds

| build | where | `_fsdp_param.py` | #181519 |
|---|---|---|---|
| `2.13.0.dev20260520+xpu` | aurora `projects/saforem2/.venv` | 1093 | ABSENT |
| `2.13.0.dev20260428+xpu` | aurora `runs/agpt-2b-v2/.venv` (what prod yeets) | 1017 | ABSENT |
| `2.13.0+cu130` | perlmutter `pytorch/2.13.0` | 1095 | ABSENT |
| `2.13.0a0+gitcf30153` | aurora `frameworks/2026.1.0` (test BKC) | 1095 | ABSENT |

The fourth was the live candidate -- a genuinely different, newer runtime on the
test image, invisible from login. It does not have the fix either.

**A false PRESENT was caught and retracted.** A probe grepped for
`"full SPMD"` / `"plain tensor"` and reported PRESENT on 2 and 1 hits. Those
strings are **the raise text of the error the patch removes**
(`_fsdp_param.py:345-350`), so a build scores 2/1 BECAUSE it still raises.
Disproved two ways: a laptop CPU torch that certainly lacks the fix scores
identically, and the 2026.1.0 tree printed the surrounding code showing both
markers inside the `raise`. Grep for symbols the patch INTRODUCES, never prose.
See [[project_present_in_both_states_is_not_evidence]].

**Everything that looked like a stack obstacle was a probe bug.** Getting a
trustworthy answer took five probes and cost more than the answer:

```
module: command not found      no module function in a PBS shell
libglog.so.0                   module does not set its own libdir
libmkl_intel_lp64.so.3         oneMKL libdir also absent
libpti_view.so.0               and a third library
edited-after-qsub              PBS froze the script; the fix never ran
```

The form that works, and which sidesteps all of it including the
`default`-resolves-per-image trap:

```bash
module load frameworks/2026.1.0
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
```

`CONDA_PREFIX` is set by the module, so it names whichever tree actually
loaded rather than one we guessed, and its `lib` covers all three libraries.

### Three distinct failures, deliberately not blurred

1. **`aurora_moe`'s hardcoded `.so.5` check** (3 files) refuses a working
   oneMKL. Fixable upstream; `aurora-tt-ezpz` owns it.
2. **`gemm_bf16bf16bf16: unsupported device`** with image-matched oneMKL and
   `.so.6` present (job 8829790). Kernel/device support, not a loader problem.
3. **#181519 absent on all four builds.** The sync-84 blocker. Not patchable
   locally by either session.

A write-up that fixed (1) would imply (2) and (3) were handled. They are not.

### This is a TORCH FLOOR, not a defect in the sync ports

The two trees differ on exactly one relevant line:

```
tt-sync84-pre  config_registry.py:252  cfg.parallelism.spmd_backend = "partial_dtensor"
tt-sync84      (removed by #4419)      zero such pins
```

Baseline `8829243` runs the pre-merge tree on the same machine, same venv, same
script, to discriminate: if it trains, #4419 is the cause and the ports are
clean; if it fails identically, the cause is elsewhere.

### What this means for landing

Sync 84 **cannot land on Aurora** until one of:

1. A torch carrying #181519 lands in an Aurora stack (the real fix).
2. Upstream widens `resolve_fsdp_mesh`'s guard -- it still covers only
   `storage_mesh.size() == 1` (`distributed/fsdp.py:48`), which the bug doc
   records as insufficient at TP=1 with FSDP>1.
3. ezpz carries a local shim that distributes the affected parameters. A
   workaround in experiments/, not core -- and it needs its own justification,
   since it re-implements what upstream torch will provide.

Everything else in the sync is verified: 12/12 imports, 12/12 agpt + 14/14 moe
builds, 82 tests, byte-identical checkpoint keys, and numerics cleared on
Perlmutter. This one blocker is upstream of all of it.

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
