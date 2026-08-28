# TorchTitan on Aurora -- quickstart (shared torch 2.13 tarball)

> Last tested: **2026-08-11**

Interactive single-job setup using the prebuilt venv tarball. For the
`frameworks/2026.1.0` RC instead, see
[aurora-quickstart-frameworks-rc.md](aurora-quickstart-frameworks-rc.md) --
that path needs no tarball and no relocation step, and is where Aurora is
heading. Keep using this page until the RC is the default module on the
nodes you land on.

## 1. Submit an interactive job

> **Queue history**
> - `[2026-03-04]` `-q next-eval` was required for PyTorch 2.10 on Aurora.
> - `[2026-08-11]` Rewritten to use the standard debug queue.

```bash
qsub -q debug-scaling -A <project> \
  -l walltime=00:60:00,filesystems=flare:home \
  -l select=2 -I
```

## 2. Clone TorchTitan

From `saforem2/torchtitan@ezpz`:

```bash
git clone https://github.com/saforem2/torchtitan --branch ezpz
cd torchtitan
```

## 3. Set up the base environment

```bash
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
```

## 4. Copy the prebuilt venv

torch 2.13 / Python 3.14. `relocate-venv.sh` rewrites the absolute paths baked
into the tarball at build time -- skipping it leaves a venv whose shebangs and
`pyvenv.cfg` point at someone else's directory.

```bash
cp /lus/flare/projects/Aurora_deployment/foremans/share-torch213/torchtitan-venv-torch2.13-py3.14.tar.gz .venv.tar.gz
tar xzf .venv.tar.gz
bash /lus/flare/projects/Aurora_deployment/foremans/share-torch213/relocate-venv.sh "$PWD/.venv"
source .venv/bin/activate
```

Confirm the XPU build survived:

```bash
python3 -c 'import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())'
```

> **Never `pip install torch`, or anything that resolves it.** A resolver run
> will pull a PyPI CUDA build over the XPU one while every line still prints
> `ok`. Re-run the check above after any install into this venv.

## 5. Get the tokenizer

```bash
python3 scripts/download_hf_assets.py --repo_id google/gemma-7b --assets tokenizer
```

## 6. Launch training

AuroraGPT-2B:

```bash
MODEL=2b
DFL=torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

ezpz launch python3 -m torchtitan.experiments.ezpz.train \
   --module ezpz.agpt \
   --config "ezpz_agpt_${MODEL}" \
   --training.dataset_path "${DFL}" \
   --debug.print_config
```
