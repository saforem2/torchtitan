# Installing PyTorch in a Fresh, Self-Contained `.venv` on Polaris

> [!NOTE]
> This guide installs PyTorch into a **standalone** torchtitan/ezpz
> environment on Polaris that does NOT layer on top of a conda env:
> uv-managed CPython 3.12, `include-system-site-packages = false`,
> `torch 2.13.0+cu129`, `mpi4py` built from source against Cray MPICH, plus
> ezpz/torchtitan/blendcorpus. Verified end-to-end from scratch twice:
> 2026-07-01 (`torch 2.12.1+cu129`) and 2026-07-26 (`torch 2.13.0+cu129`).
> See the [Verification](#verification) section. The exact `torch` version
> tracks whatever the cu129 index currently serves, since step 3 uses
> `--upgrade`; both patch versions passed every check.
>
> If you just want the commands, jump to
> [All-in-one script](#all-in-one-copy-paste).

## Overview

The install has five ordered steps:

1. Start from a clean shell (no conda active).
2. Create a standalone uv-managed venv.
3. Install PyTorch from the CUDA 12.9 (`cu129`) index.
4. Install ezpz, torchtitan, and blendcorpus.
5. Build `mpi4py` from source against Cray MPICH -- **last**.

The ordering matters: `mpi4py` must be built from source against Polaris's
Cray MPICH (so MPI collectives work under PALS `mpiexec`), and it must be
installed **last** because blendcorpus (and other deps) list `mpi4py` and
would otherwise pull a portable PyPI wheel that overwrites the native build.

---

## Step-by-step

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

Confirm it is standalone (not seeded off a conda interpreter):

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
> the system CUDA matches at run time. `--upgrade` pulls the latest cu129
> wheel, so the exact patch version drifts over time: 2026-07-01 resolved to
> `2.12.1+cu129`, 2026-07-26 to `2.13.0+cu129`. Pin a specific version
> (`torch==2.13.0`) if you need reproducibility.

### 4. Install ezpz, torchtitan, and blendcorpus

> [!IMPORTANT]
> Install everything that depends on `mpi4py` **before** the mpi4py source
> build in step 5. `blendcorpus` (and others) list `mpi4py` as a dependency,
> so `uv pip install` will pull in the portable PyPI wheel; the source build
> in step 5 then replaces it. Doing it in this order keeps the native build
> in place. (If a later install pulls the portable wheel back in, you get
> `RuntimeError: cannot load MPI library` / `libmpi.so.12: cannot open
> shared object file` at import -- rerun step 5.)

```bash
# ezpz (editable from git)
uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz.git"

# torchtitan itself, editable from the repo root
uv pip install --no-cache --link-mode=copy -e .

# blendcorpus -- the agpt/moe configs default to `--dataloader.dataset blendcorpus`.
# Use the remove-deepspeed branch: the upstream zhenghh04/blendcorpus pulls in
# `deepspeed` as a transitive import, which is not needed here.
uv pip install --no-cache --link-mode=copy \
    "git+https://github.com/saforem2/blendcorpus@feat/remove-deepspeed"
```

### 5. Build `mpi4py` from source against Cray MPICH -- do this LAST

> [!IMPORTANT]
> Switch to the **GNU** programming environment first. Polaris defaults to
> `PrgEnv-nvidia`, which makes `cc` wrap the NVIDIA HPC compiler (`nvc`).
> The uv-managed CPython was built with GCC, so its `sysconfig` CFLAGS
> include GCC-only flags (`-fno-strict-overflow`, `-Wunreachable-code`)
> that `nvc` rejects (`nvc-Error-Unknown switch: -fno-strict-overflow` /
> `error: Cannot compile MPI programs`). `PrgEnv-gnu` makes `cc` wrap
> `gcc`, which accepts those flags.

```bash
module swap PrgEnv-nvidia PrgEnv-gnu

MPICC=cc uv pip install --no-cache --no-binary mpi4py --force-reinstall mpi4py
```

- `module swap PrgEnv-nvidia PrgEnv-gnu` makes `cc` the GCC-backed Cray
  wrapper (also switches `MPICH_DIR` to `.../mpich/9.0.1/ofi/gnu/12.3`).
- `--no-binary mpi4py` forces a source build (no portable wheel).
- `MPICC=cc` links it against the Cray MPICH compiler wrapper.

Confirm `cc` is GCC and mpi4py built native:

```bash
cc --version | head -1     # -> gcc-14 (SUSE Linux) 14.3.0
cat .venv/lib/python3.12/site-packages/mpi4py-*.dist-info/WHEEL | grep Tag
# WANT: Tag: cp312-cp312-linux_x86_64  (NOT manylinux)
python3 -c "from mpi4py import MPI; print('mpi4py OK', MPI.COMM_WORLD.rank)"
```

A native mpi4py has a `cp312-cp312-linux_x86_64` wheel tag; a portable PyPI
wheel is `cp312-cp312-manylinux_*` (that one does not work with Cray MPICH
under PALS `mpiexec`).

### 6. Download tokenizer assets (if training)

```bash
python3 scripts/download_hf_assets.py --repo_id google/gemma-7b --assets tokenizer
```

---

## Verify

Run these on a **compute node** (grab a debug allocation first:
`qsub -I -A <acct> -q debug -l select=1 -l walltime=00:30:00
-l filesystems=home:eagle`) with the same environment active
(`source .venv/bin/activate && module load cuda/12.9 &&
module swap PrgEnv-nvidia PrgEnv-gnu`).

### a. MPI collective crosses ranks

```bash
mpiexec -n 4 --ppn 4 python3 -c \
  "from mpi4py import MPI; c=MPI.COMM_WORLD; print(c.rank, c.bcast(56465 if c.rank==0 else None, root=0))"
```

All 4 ranks should print `56465` (a portable-wheel mpi4py prints `56465` on
rank 0 and `None` on the rest, because its communicator does not cross
processes under PALS).

### b. ezpz self-test

```bash
ezpz launch python3 -m ezpz.examples.test
```

Should get past `init_process_group` and run to completion (exit 0).

### c. agpt training smoke (full torchtitan + blendcorpus path)

```bash
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt --config=agpt_2b \
    --checkpoint.no-enable \
    --training.num-tokens-per-microbatch-per-dp-rank=8192 \
    --training.num-tokens-per-train-step=65536 \
    --training.max-context-length=8192 \
    --training.steps=5
```

Should build the blendcorpus index and log training steps, ending in
`Execution finished with 0`.

> [!NOTE]
> The batch flags above are the post-#4121 (80th sync) token-unit names.
> The old `--training.local-batch-size` / `--training.seq-len` /
> `--training.global-batch-size` were replaced: sizes are now counted in
> **tokens**, not sequences. `tokens = batch_size * seq_len`, so the
> `LBS=1, SEQ_LEN=8192` above becomes `8192` tokens per microbatch per DP
> rank. `submit_agpt_20b_autoretry.sh` is already ported;
> `submit_agpt_2b_autoretry.sh` is **not** and still passes the old names.

---

## Why each choice matters

| Step | Reason |
| --- | --- |
| `conda deactivate` first | Prevents `uv` from seeding the venv off a conda interpreter |
| `--python-preference only-managed` | Guarantees a standalone CPython, never conda's |
| no `--system-site-packages` | Nothing from a base env can shadow or leak in |
| `module swap PrgEnv-nvidia PrgEnv-gnu` | Makes `cc` wrap `gcc` (not `nvc`), so mpi4py's GCC-built CPython CFLAGS compile |
| mpi4py built **last**, after blendcorpus | blendcorpus depends on mpi4py; installing it after would pull the portable wheel over the native build |
| `MPICC=cc --no-binary mpi4py` | Compiles against `cray-mpich/9.0.1` so MPI collectives work under PALS `mpiexec` |
| blendcorpus `@feat/remove-deepspeed` | agpt/moe default to `--dataloader.dataset blendcorpus`; the remove-deepspeed branch drops the unused `deepspeed` transitive import |
| `cuda/12.9` + torch from the `cu129` index | Matches the system CUDA 12.9 toolkit with the cu129 wheels |

> [!WARNING]
> A later bare `uv pip install mpi4py` (or a `uv sync` / lockfile resolve)
> without the `MPICC=cc --no-binary mpi4py` flags will swap the native build
> back to the portable wheel. Rerun step 5 if that happens.

---

## All-in-one (copy/paste)

Run from a fresh Polaris login shell, with `PROJDIR` pointing at your
torchtitan checkout. Steps 1-5 run on the login node; run the
[verification](#verify) commands afterward from a debug compute-node
allocation.

```bash
# --- config ---
export PROJDIR="/eagle/AuroraGPT/foremans/projects/saforem2/torchtitan"   # <- your checkout

# --- proxy (required for any download on Polaris) ---
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="localhost,127.0.0.1,*.alcf.anl.gov,*.anl.gov"

# --- 1. clean shell + Cray PE ---
conda deactivate 2>/dev/null; conda deactivate 2>/dev/null
module load craype cray-mpich PrgEnv-nvidia

# --- 2. standalone uv venv (no conda, no system site-packages) ---
cd "$PROJDIR"
uv venv --python 3.12 --python-preference only-managed --no-project .venv
source .venv/bin/activate
grep -E "home|include-system-site-packages" .venv/pyvenv.cfg   # sanity check

# --- 3. PyTorch (cu129) ---
module load cuda/12.9
uv pip install --no-cache --link-mode=copy --force-reinstall --upgrade \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

# --- 4. ezpz + torchtitan + blendcorpus (all mpi4py-dependent -> BEFORE step 5) ---
uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz.git"
uv pip install --no-cache --link-mode=copy -e .
uv pip install --no-cache --link-mode=copy \
    "git+https://github.com/saforem2/blendcorpus@feat/remove-deepspeed"

# --- 5. mpi4py from source against Cray MPICH -- LAST, under PrgEnv-gnu ---
module swap PrgEnv-nvidia PrgEnv-gnu
MPICC=cc uv pip install --no-cache --no-binary mpi4py --force-reinstall mpi4py

# --- 6. tokenizer assets (if training) ---
python3 scripts/download_hf_assets.py --repo_id google/gemma-7b --assets tokenizer

# --- sanity ---
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
cat .venv/lib/python3.12/site-packages/mpi4py-*.dist-info/WHEEL | grep Tag
python3 -c "from mpi4py import MPI; print('mpi4py OK', MPI.COMM_WORLD.rank)"
```

---

## Verification (captured output)

### 2026-07-01 -- `torch 2.12.1+cu129`

Run from scratch in an isolated directory
(`/eagle/AuroraGPT/foremans/tmp/venv-verify-20260701`) on a Polaris compute
node (job 7232069, node x3001c0s19b1n0).

```console
$ grep -E "home|include-system-site-packages" .venv/pyvenv.cfg
home = /home/foremans/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/bin
include-system-site-packages = false

$ python3 -c "import torch; print('torch', torch.__version__)"
torch 2.12.1+cu129

$ cat .venv/lib/python3.12/site-packages/mpi4py-*.dist-info/WHEEL | grep Tag
Tag: cp312-cp312-linux_x86_64        # native build, NOT manylinux

# a. mpiexec bcast
$ mpiexec -n 4 --ppn 4 python3 -c \
    "from mpi4py import MPI; c=MPI.COMM_WORLD; print(c.rank, c.bcast(56465 if c.rank==0 else None, root=0))"
0 56465
1 56465
2 56465
3 56465

# b. ezpz self-test
[I][ezpz/launch:855] Job ID: 7232069
[I][ezpz/distributed:1755:_setup_ddp] init_process_group: master_addr=x3001c0s19b1n0.hsn.cm.polaris.alcf.anl.gov, master_port=47949, world_size=4, rank=0, backend=nccl
[I][examples/test:403:train_step] iter=150  loss=0.179395 accuracy=0.945312
[I][ezpz/launch:913] Execution finished with 0.

# c. agpt_2b training smoke
[I][blendcorpus/blendcorpus_builder:311] Using BlendCorpus dataloader backend
[I][components/metrics:523] step: 1  loss: 12.94838  grad_norm:  1.9474  memory: 23.57GiB(59.68%)  tps: 172     mfu: 0.62%
[I][components/metrics:523] step: 5  loss: 13.54631  grad_norm: 30.9798  memory: 27.48GiB(69.59%)  tps: 15,926  mfu: 57.11%
[I][ezpz/launch:913] Execution finished with 0.
```

### 2026-07-26 -- `torch 2.13.0+cu129`

Same recipe re-run from scratch in a fresh isolated directory
(`/eagle/AuroraGPT/foremans/tmp/venv-verify-20260726`) on a Polaris debug
compute node (job 7295306, node x3002c0s7b0n0). Only change is `torch`
resolving to a newer patch (`--upgrade`).

```console
$ grep -E "home|include-system-site-packages" .venv/pyvenv.cfg
home = /home/foremans/.local/share/uv/python/cpython-3.12.10-linux-x86_64-gnu/bin
include-system-site-packages = false

$ python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
torch 2.13.0+cu129 cuda 12.9

$ cat .venv/lib/python3.12/site-packages/mpi4py-*.dist-info/WHEEL | grep Tag
Tag: cp312-cp312-linux_x86_64        # native build, NOT manylinux

$ cc --version | head -1
gcc-14 (SUSE Linux) 14.3.0

# a. mpiexec bcast
$ mpiexec -n 4 --ppn 4 python3 -c \
    "from mpi4py import MPI; c=MPI.COMM_WORLD; print(c.rank, c.bcast(56465 if c.rank==0 else None, root=0))"
0 56465
1 56465
2 56465
3 56465

# b. ezpz self-test
[I][ezpz/launch:917:launch] Execution finished with 0.

# c. agpt_2b training smoke
[I][ezpz/launch:917:launch] Execution finished with 0.
```
