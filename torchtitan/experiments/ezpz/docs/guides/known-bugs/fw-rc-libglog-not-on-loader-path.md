# `frameworks/2026.1.0` on Aurora: `import torch` dies on `libglog.so.0`

> Status: **open upstream, workaround is one line.** Filed here 2026-08-30.
> First hit 2026-08-25, the maintenance in which the module reached Aurora.

## Observed

Load the module, import torch, and every import dies:

```
OSError: libglog.so.0: cannot open shared object file: No such file or directory
```

The traceback comes from `torchcomms/__init__.py`, which `torch/__init__.py`
imports unconditionally -- so this gates the entire stack, not just
collectives. `libgflags.so.2.2` has the same problem behind it.

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

| module | libglog |
|---|---|
| `frameworks/2025.3.1` | `libglog.so.2` (0.7.1) |
| `frameworks/2026.1.0` | `libglog.so.0` (0.4.0) |

The soname went **backwards**. Anything built against the 2025.3.1 glog will
not find its library under the RC even with the path fixed, and vice versa.

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
only 2025.3.1. Everything above was measured from inside a job. Whether the
missing export is intentional (a staging artifact of an RC) or an oversight has
not been raised with ALCF.
