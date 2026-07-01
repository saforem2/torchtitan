# Fresh, Self-Contained `.venv` on Polaris (no conda inheritance)

> [!NOTE]
> This guide builds a **standalone** torchtitan/ezpz environment on Polaris
> that does NOT layer on top of a conda env. It mirrors the known-good
> `datascience/.venv` (uv-managed CPython 3.12, `include-system-site-packages
> = false`, torch 2.10.0+cu128, `mpi4py` built from source against Cray
> MPICH).

## Symptom this fixes

Running `ezpz launch python3 -m ezpz.examples.test` crashes at distributed
init with every non-zero rank dying on:

```
master_port = int(_get_env_or_raise("MASTER_PORT"))
ValueError: invalid literal for int() with base 10: 'None'
```

...even though ezpz's own log line shows rank 0 computed a real port
(`master_port=56465, world_size=4, rank=0`).

## Root cause

`mpi4py` was installed as a **portable PyPI binary wheel** (a
`manylinux_2_5_x86_64`-tagged wheel that bundles its own generic MPICH)
instead of being built against Polaris's Cray MPICH.

ezpz derives the rendezvous port on rank 0 and broadcasts it to the other
ranks with an mpi4py collective:

```python
# ezpz/distributed.py, _setup_ddp()
free_port   = str(_get_free_port()) if rank == 0 else None
master_port = os.environ.get("MASTER_PORT", free_port) if rank == 0 else None
master_port = broadcast(master_port, root=0)   # <- MPI.COMM_WORLD.bcast
os.environ["MASTER_PORT"] = str(master_port)
```

With a non-Cray mpi4py, `mpiexec`/PALS hands each process correct rank IDs
(so rank numbering looks fine and only rank 0 has a port), but the MPI
communicator is broken -- every process acts like its own 1-rank world.
So `broadcast(..., root=0)` never crosses ranks: ranks 1..N-1 keep their
local `None`, `str(None)` -> `'None'`, and torch's rendezvous chokes on
`int('None')`.

`--system-site-packages` does not save you here: the venv's own broken
`mpi4py` wheel shadows anything in the base env, and the conda env may not
even provide an mpi4py.

## How to tell if you have the bug

```bash
cat .venv/lib/python3.12/site-packages/mpi4py-*.dist-info/WHEEL | grep Tag
```

| Tag | Verdict |
| --- | --- |
| `cp312-cp312-linux_x86_64` | native, built on-node -- good |
| `cp312-cp312-manylinux_*` | portable PyPI wheel -- BROKEN on Polaris |

---

## Full recipe: fresh standalone venv

Run these from a **normal Polaris login shell** (interactive `ssh polaris`),
so the default Cray PE modules are loaded. Substitute your own project path
for `$PROJDIR`.

> [!IMPORTANT]
> Polaris needs the ALCF proxy for any download (login and compute nodes):
>
> ```bash
> export http_proxy="http://proxy.alcf.anl.gov:3128"
> export https_proxy="http://proxy.alcf.anl.gov:3128"
> export no_proxy="localhost,127.0.0.1,*.alcf.anl.gov,*.anl.gov"
> ```

### 1. Start from a clean shell (deactivate conda)

```bash
# Run until the prompt has NO (env) prefix left
conda deactivate 2>/dev/null; conda deactivate 2>/dev/null
```

Confirm the Cray programming environment is active and `cc` is the Cray
wrapper (it is by default on Polaris login nodes):

```bash
which cc            # -> /opt/cray/pe/craype/2.7.35/bin/cc
echo "$MPICH_DIR"   # -> /opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3
```

If `which cc` shows `/usr/bin/cc` instead, load the PE explicitly (harmless
if already loaded):

```bash
module load craype cray-mpich PrgEnv-nvidia
```

### 2. Create the standalone venv (uv-managed Python; no conda, no system)

```bash
cd "$PROJDIR"     # e.g. /eagle/datascience_collab/<user>/torchtitan

# --python-preference only-managed  -> use uv's own CPython, never conda's
# (no --system-site-packages)       -> nothing leaks in from a base env
uv venv --python 3.12 --python-preference only-managed --no-project .venv

source .venv/bin/activate
```

Verify it did NOT inherit conda:

```bash
grep -E "home|include-system-site-packages" .venv/pyvenv.cfg
```

You want:

```
home = /home/<user>/.local/share/uv/python/cpython-3.12.*-linux-x86_64-gnu/bin
include-system-site-packages = false
```

If `home` points at `~/.conda/envs/...`, you started from an active conda
env -- delete `.venv`, `conda deactivate`, and redo step 2.

### 3. Install PyTorch (CUDA 12.9 build)

Load the CUDA 12.9 toolkit (keep it loaded at run time too), then install the
cu129 wheels:

```bash
module load cuda/12.9

uv pip install --no-cache --link-mode=copy --force-reinstall --upgrade \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129
```

> [!NOTE]
> The cu129 wheels bundle their own CUDA runtime (`nvidia-*-cu12`), so the
> pip install succeeds even without the module. Load `cuda/12.9` anyway so
> the system CUDA matches at run time. (Verified 2026-07-01: torch resolves
> to `2.12.1+cu129`.)

### 4. Build `mpi4py` from source against Cray MPICH (THE fix)

> [!IMPORTANT]
> Switch to the **GNU** programming environment first. Polaris defaults to
> `PrgEnv-nvidia`, which makes `cc` wrap the NVIDIA HPC compiler (`nvc`).
> The uv-managed CPython was built with GCC, so its `sysconfig` CFLAGS
> include GCC-only flags (`-fno-strict-overflow`, `-Wunreachable-code`)
> that `nvc` rejects -- the mpi4py build then dies with
> `nvc-Error-Unknown switch: -fno-strict-overflow` /
> `error: Cannot compile MPI programs`. `PrgEnv-gnu` makes `cc` wrap
> `gcc`, which accepts those flags.

```bash
module swap PrgEnv-nvidia PrgEnv-gnu

MPICC=cc uv pip install --no-cache --no-binary mpi4py --force-reinstall mpi4py
```

- `module swap PrgEnv-nvidia PrgEnv-gnu` makes `cc` the GCC-backed Cray
  wrapper (also switches `MPICH_DIR` to `.../mpich/9.0.1/ofi/gnu/12.3`).
- `--no-binary mpi4py` forces a source build (no portable wheel).
- `MPICC=cc` links it against the Cray MPICH compiler wrapper.

Confirm `cc` is GCC, then build. Verify it built native:

```bash
cc --version | head -1     # -> gcc-14 (SUSE Linux) 14.3.0
cat .venv/lib/python3.12/site-packages/mpi4py-*.dist-info/WHEEL | grep Tag
# WANT: Tag: cp312-cp312-linux_x86_64
```

### 5. Install ezpz + torchtitan

```bash
# ezpz (editable from git, as in the known-good venv)
uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz.git"

# torchtitan itself, editable from the repo root
uv pip install --no-cache --link-mode=copy -e .
```

### 6. Download tokenizer assets (if training)

```bash
python3 scripts/download_hf_assets.py --repo_id google/gemma-7b --assets tokenizer
```

---

## Verify the fix

### 6a. Confirm the MPI collective actually crosses ranks

This is the exact operation that was broken. Run it on a **compute node**
(grab a debug allocation first: `qsub -I -A <acct> -q debug -l select=1
-l walltime=00:30:00 -l filesystems=home:eagle`) with the same environment
active (`source .venv/bin/activate && module load cuda/12.9 &&
module swap PrgEnv-nvidia PrgEnv-gnu`):

```bash
mpiexec -n 4 --ppn 4 python3 -c \
  "from mpi4py import MPI; c=MPI.COMM_WORLD; print(c.rank, c.bcast(56465 if c.rank==0 else None, root=0))"
```

- **Fixed:** all 4 ranks print `56465`.
- **Still broken:** only rank 0 prints `56465`; the rest print `None`
  (that is the `MASTER_PORT='None'` bug).

### 6b. End-to-end

```bash
ezpz launch python3 -m ezpz.examples.test
```

Should get past `init_process_group` and run to completion.

---

## Why each choice matters

| Step | Reason |
| --- | --- |
| `conda deactivate` first | Prevents `uv` from seeding the venv off a conda interpreter (the original mistake) |
| `--python-preference only-managed` | Guarantees a standalone CPython, never conda's |
| no `--system-site-packages` | Nothing from a base env can shadow or leak in |
| `module swap PrgEnv-nvidia PrgEnv-gnu` | Makes `cc` wrap `gcc` (not `nvc`), so mpi4py's GCC-built CPython CFLAGS compile |
| `MPICC=cc --no-binary mpi4py` | Compiles against `cray-mpich/9.0.1` so `bcast` actually works under PALS `mpiexec` |
| `cuda/12.9` + torch from the `cu129` index | Matches the system CUDA 12.9 toolkit with the cu129 wheels |

> [!WARNING]
> Do NOT later run a bare `uv pip install mpi4py` (or a `uv sync` /
> lockfile resolve) without the `MPICC=cc --no-binary mpi4py` flags -- it
> will silently swap the native build back to the portable wheel and
> reintroduce this exact bug.

## Reference: known-good venv this mirrors

`datascience/.venv` on Polaris (`foremans`):

- `pyvenv.cfg`: uv-managed CPython 3.12.10, `include-system-site-packages = false`
- `torch` from the pytorch cu129 index (`manylinux_2_28` wheel -- normal)
- `mpi4py`, native `linux_x86_64` tag (built from source against Cray MPICH)
- `ezpz` editable from `github.com/saforem2/ezpz`

## Verification

This recipe was run end-to-end from scratch in an isolated directory on a
Polaris compute node on 2026-07-01 (job 7232069, node x3001c0s19b1n0):

- venv created uv-managed, `include-system-site-packages = false` (no conda)
- `torch==2.12.1+cu129` installed and imported
- `mpi4py==4.1.2` built from source under `PrgEnv-gnu`, native
  `cp312-cp312-linux_x86_64` tag
- **Step 6a:** `mpiexec -n 4` bcast returned `56465` on all 4 ranks
- **Step 6b:** `ezpz launch python3 -m ezpz.examples.test` ran to
  `Execution finished with 0` (200 iters, loss 1.04 -> 0.16), no
  `MASTER_PORT='None'`
