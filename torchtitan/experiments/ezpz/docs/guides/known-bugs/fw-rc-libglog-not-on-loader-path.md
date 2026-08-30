# `frameworks/2026.1.0` on Aurora: `import torch` dies on `libglog.so.0`

> Status: **open upstream, workaround is one line.** Filed here 2026-08-30.
> First hit 2026-08-25, the maintenance in which the module reached Aurora.
>
> Every claim below re-verified from inside a validation-node job on
> 2026-08-30 (`8792135`, `8792148`) rather than carried from notes.

## Observed

Load the module, import torch, and every import dies:

```
OSError: libglog.so.0: cannot open shared object file: No such file or directory
```

Measured traceback, module environment untouched:

```
torch/distributed/distributed_c10d.py:151  ->
torchcomms/__init__.py:45                  ->
torchcomms/__init__.py:42  _load_libtorchcomms()
OSError: libglog.so.0: cannot open shared object file
```

`torch/__init__.py` imports `distributed_c10d` unconditionally, so this gates
the entire stack rather than just collectives. `libgflags.so.2.2` ships in the
same directory.

The library is **not missing**. It ships inside the module's own tree:

```
$FW/lib/libglog.so.0          # 0.4.0
$FW/lib/libgflags.so.2.2
FW=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0
```

The modulefile does not export that directory onto `LD_LIBRARY_PATH`.

## Workaround

```bash
module load frameworks/2026.1.0
FW=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0
export LD_LIBRARY_PATH="$FW/lib:$LD_LIBRARY_PATH"
```

With that one line the stack works. Job `8781129` (1N): torch imports, 12 XPUs
visible, bf16 matmul, SDPA forward and backward, and the compiled SDPA backward.
Job `8789506` (2N x 4) trains all five agpt/moe corners under it.

## Expected

The modulefile should put its own `lib/` on the loader path. A module that
ships a shared object its bundled python cannot load is not self-contained --
the RC's own `python3` (3.12.12) bundles `torch 2.13.0a0+gitcf30153`, and that
torch cannot import under the module that provides it.

Nothing on the user side is misconfigured, which is what makes it expensive:
the error names a real library, at a path the user never set, from a module
that loaded without complaint.

## Version skew worth knowing

| module | soname | real file |
|---|---|---|
| `frameworks/2025.3.1` | `libglog.so.2` | `libglog.so.0.7.1` |
| `frameworks/2026.1.0` | `libglog.so.0` | `libglog.so.0.4.0` |

Both moved backwards: the soname `.so.2 -> .so.0` and the version
`0.7.1 -> 0.4.0`. Anything built against the 2025.3.1 glog looks for
`libglog.so.2`, which does not exist under the RC even with the path fixed.

## Two traps that cost an allocation each

Both are about the diagnosis, not the bug.

**A piped `module load` silently no-ops.** It runs in a subshell, so python
stays 3.6.15 and the failure presents as `No module named torch` -- which sends
you looking for a missing package instead of a module that never loaded.

**`set -o pipefail` without `set -e` reports success.** `val-xccl.sh` returned
`Exit_status=0` while `mpiexec` had failed. Same shape as every other exit-0
no-op on this project: verify the artifact, not the return code.

## Where this is applied

- [`scripts/rc-matrix.sh`](../../../scripts/rc-matrix.sh) -- the five-corner
  coverage matrix
- [`docs/guides/aurora-quickstart-frameworks-rc.md`](../aurora-quickstart-frameworks-rc.md)
  -- step 3 of the RC quickstart
- [`README.md`](../../../README.md) -- the frameworks RC setup block

## Not established

`/opt/aurora/26.181.0` exists only inside the validation-node image, so none of
this is checkable from a login node -- `module avail frameworks` there shows
only 2025.3.1, and an `ls` of the paths above finds nothing.

Whether the missing export is intentional (a staging artifact of an RC) or an
oversight has not been raised with ALCF.

## Verification

Job `8792135`, validation queue, 1 node, tests each claim rather than the happy
path -- the failure is reproduced BEFORE the fix is applied, in the same run.

| claim | result |
|---|---|
| `libglog.so.0 -> libglog.so.0.4.0` in `$FW/lib` | confirmed |
| modulefile does not export that dir | confirmed, 0 matches in `LD_LIBRARY_PATH` |
| failure is `torchcomms/__init__.py:42` | confirmed, exact frames |
| module python is 3.12.12 from the module tree | confirmed |
| with the fix: torch `2.13.0a0+gitcf30153`, 12 XPUs, xccl True | confirmed |
| 2025.3.1 ships `libglog.so.2` / `0.7.1` | confirmed |

One false lead, recorded because it is easy to repeat: probing with
`env -u LD_LIBRARY_PATH` reports the first failure as `libmkl_intel_lp64.so.3`
from `torch/__init__.py:380`, suggesting several of the module's libraries are
missing. That is an artifact of the probe -- unsetting the variable also
discards paths the module legitimately sets. With the module environment
untouched (`8792148` section A), the first and only failure is `libglog.so.0`.
The one-line workaround is complete, not a partial fix.
