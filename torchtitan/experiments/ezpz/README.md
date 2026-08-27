# TorchTitan + 🍋 `ezpz`

Pre-training AuroraGPT (dense + MoE) on ALCF systems
(Aurora / Sunspot / Polaris) with **PyTorch >= 2.13**. This folder is
an opinionated experiment harness on top of upstream `torchtitan`
that adds:

- Fault-tolerant training (bad-node failover wrapper)
- Per-machine launch scripts + venv broadcast (`ezpz yeet-env`)
- Custom optimizers (Mano, SPAM, Muon, SophiaG, ADOPT)
- A `BlendCorpusDataLoader` for olmo-mix-1124 + arbitrary HF datasets
- Eval pipeline (DCP → HF safetensors → lm-eval)
- Detailed production / eval / scaling tracking under [`docs/`](docs/)

This file is the **landing page** — quickstart + an index of where
everything lives. For day-to-day work, jump straight to the more
specific pages linked below.

> [!NOTE]
> This is the [`saforem2/torchtitan@ezpz`](https://github.com/saforem2/torchtitan/tree/ezpz)
> fork. Upstream is [`pytorch/torchtitan`](https://github.com/pytorch/torchtitan)
> and we [resync against it regularly](docs/upstream-sync.md).

> [!IMPORTANT]
> **torch 2.10 (the `frameworks/2025.3.1` module) no longer works.** Upstream
> `torchtitan/distributed/fsdp.py` imports `DataParallelMeshDims` from
> `torch.distributed.fsdp`, which does not exist before 2.13:
>
> ```
> ImportError: cannot import name 'DataParallelMeshDims' from 'torch.distributed.fsdp'
> ```
>
> Verified on Aurora 2026-08-11: torch 2.13.0.dev20260428+xpu has the symbol,
> torch 2.10.0a0+git449b176 (frameworks/2025.3.1) does not. This is CORE
> torchtitan, not ezpz -- it arrives via `agpt/parallelize.py` -> `distributed/fsdp.py`
> and came in with upstream #3159. Use the torch 2.13 venv (step 3 below), not a
> bare `module load frameworks/2025.3.1`.
>
> `ConfigManager._load_config` catches the `ImportError` and re-raises the
> generic `Cannot import config_registry for module 'ezpz.agpt'`, which hides
> this and every other missing-dependency cause. To see the real error, from the
> repo root run:
>
> ```bash
> python3 -c "import torchtitan.experiments.ezpz.agpt.config_registry"
> ```

## Quickstart (2B dense training, 2 nodes)

Full setup details — module loads, venv install, large-scale yeet-env
broadcast — are in
[`docs/reference/guides/running-with-newer-pytorch.md`](docs/reference/guides/running-with-newer-pytorch.md).
Minimum viable path on Aurora:

```bash
# 1. Allocate two nodes
qsub -q prod -A AuroraGPT -l walltime=06:00:00,filesystems=flare:home -l select=2 -I

# 2. Clone + enter
git clone https://github.com/saforem2/torchtitan --branch ezpz
cd torchtitan

# 3. Setup environment (loads modules + ezpz helper functions)
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env

# 4. Launch 2B training
MODEL=2b bash torchtitan/experiments/ezpz/run_train.sh
```

For other models / machines / data files, see
[`run_train.sh`](run_train.sh) and the detailed setup guide above.

## Aurora frameworks RC (`frameworks/2026.1.0`)

> [!IMPORTANT]
> **Validation-queue only, as of 2026-08-26.** `/opt/aurora/26.181.0` exists
> only in the validation-node image -- `module avail frameworks` on a login
> node shows just `2025.3.1`, and you cannot build or test against the RC from
> there. When the RC goes live cluster-wide this section becomes the normal
> path; until then every command below must run **on a validation node**.

The RC bundles its own python 3.12.12 with torch `2.13.0a0+gitcf30153`
(xpu available, 12 devices, `dist.is_xccl_available()` True). What it does
**not** bundle is our stack -- `ezpz`, `spmd_types`, `grain`, `blendcorpus`
and friends are all absent, so a bare `module load` gets you a working torch
and nothing that can train.

### Setup

```bash
# 1. Load the module at TOP LEVEL. Never pipe it (see traps below).
module load frameworks/2026.1.0

# 2. libglog.so.0 lives inside the module tree but is not on the path.
#    Without this every `import torch` dies in torchcomms.
FW=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0
export LD_LIBRARY_PATH="$FW/lib:$LD_LIBRARY_PATH"

# 3. Compute nodes need the proxy for any install.
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# 4. Build the overlay venv on the RC python.
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
#    -> venvs/aurora/torchtitan-ezpz-aurora_frameworks-2026.1.0

# 5. Dependencies. -P pins the torch family out of the resolver's reach;
#    the XPU build is on no index and would be replaced by a CUDA wheel.
uv pip install --no-cache --link-mode=copy \
    -P torch -P pytorch-triton-xpu -P torchvision -P torchaudio \
    tensorboard tyro grain 'etils[epath,epy]' xarray absl-py \
    array-record cloudpickle portpicker 'protobuf>=5.28.3' \
    'spmd_types==0.2.1' wandb \
    'git+https://github.com/saforem2/ezpz' \
    'git+https://github.com/saforem2/blendcorpus@feat/remove-deepspeed'

# 6. torchtitan is NEVER pip-installed -- it is imported from the repo tree,
#    and a PYTHONPATH export does not reach ranks. Drop a .pth instead.
SP=$(python3 -c 'import site; print(site.getsitepackages()[0])')
echo "$PWD" > "$SP/_torchtitan_repo.pth"
```

Then verify, **with the venv python** (not the login one):

```bash
python3 -c "import torch; print(torch.__version__, torch.xpu.is_available())"
python3 -c "import importlib; importlib.import_module(
    'torchtitan.experiments.ezpz.agpt.config_registry')"
```

A ready-made script that does all of the above plus a 2N collectives smoke and
a real training run is
[`scripts/val-rc-full.sh`](scripts/val-rc-full.sh).

### Why each dependency is here

| Package | Needed by |
|---|---|
| `grain` + `etils[epath,epy]`, `xarray`, `absl-py`, `array-record`, `cloudpickle`, `portpicker`, `protobuf` | core `torchtitan/components/data/loader.py` imports `grain.python`; the rest are grain's own deps |
| `tyro` (pulls `typeguard`) | config parsing -- `config_manager.parse_args()` |
| `tensorboard` | `torchtitan/components/metrics.py` |
| `spmd_types` | sharding specs (`spmd.S(n)`, `spmd.R`) |
| `blendcorpus` @ `feat/remove-deepspeed` | the production dataloader; that branch because the RC has no deepspeed |
| `wandb` | the RC base ships a `wandb` that is importable but has **no `.api`**, which `ezpz.verify_wandb()` calls. Installing a real one into the overlay shadows it |
| `ezpz` from **git** | PyPI `ezpz` is an unrelated package (v0.1.2) |

### Traps, all of which report success

Every one of these cost a job. They share a failure mode: the command exits 0
or prints nothing alarming, and you only find out at the next stage.

1. **`module load ... | tail -3`** runs the load in a subshell, which silently
   evaporates. Python stays 3.6.15 and every import fails with a misleading
   `No module named 'torch'`. Load at top level.
2. **`uvi` and `ezpz` are interactive-shell only.** `uvi` is an atuin alias
   (`uv pip install --no-cache --link-mode=copy`); neither is defined in a PBS
   batch shell, where you get `command not found` and rc 127. Use `uv pip`
   directly.
3. **`uv pip` from a login node** against the overlay venv fails with
   `No virtual environment found` -- the venv's interpreter symlinks into
   `/opt/aurora/26.181.0`, which is not there -- and **still exits 0**. Install
   on the node.
4. **Neither blanket install policy works.** `--no-deps` on everything skips
   legitimate pure-python deps -- chasing them one failed job at a time cost
   four runs (`etils`, `xarray`, `portpicker`, `typeguard`). But letting the
   resolver run pulled a generic PyPI **`torch-2.13.0`** into the overlay,
   shadowing the RC's XPU build, **even with `-P torch -P pytorch-triton-xpu`
   pins**: all 14 installs printed `ok` and only an explicit
   `torch.__version__` assertion caught it (job `8784535`). Use the resolver
   for leaf packages, `--no-deps` for anything that declares torch, and assert
   on the torch version afterwards.
5. **Evicting a stray torch is not enough -- take `triton` with it.** The same
   pull brings a generic `triton-3.7.1` that shadows the RC's Intel-enabled
   build; removing torch alone leaves `config_registry` dying on
   `No module named 'triton.backends.intel'` (job `8784572`).
6. **`import torchtitan.experiments.ezpz.agpt` is not a readiness gate** -- it
   succeeds while `config_registry` is still broken, because
   `config/manager.py` catches every exception and re-raises one generic
   `Cannot import config_registry`. Gate on `config_registry` itself, with
   `PYTHONPATH` unset from a neutral cwd, which is what ranks see.
7. **Rank errors are not on stdout.** A launch that prints only
   `Execution finished with 143` usually has the real traceback in
   `logs/<module>/<timestamp>-rank0.jsonl`.

### Status

| Check | Result |
|---|---|
| torch / XPU / XCCL on the RC | **PASS** -- `2.13.0a0+gitcf30153`, 12 devices, xccl True |
| compiled SDPA fwd+bwd | **PASS** (job `8781129`, 1N) |
| all deps import in the overlay | **PASS** (job `8784615`) |
| `config_registry` | **PASS** (job `8784615`) |
| 2N collectives (`ezpz.examples.test`) | **PASS** -- rc=0, 24 ranks, 268s (job `8784615`) |
| **2N x 12 distributed training** | **PASS** -- 10/10 steps, loss 10.865 -> 9.059 (job `8784615`) |
| W&B from a real run | **PASS** -- `torchtitan.ezpz.train/runs/70fumwgc` |

Full trajectory, `agpt_debugmodel`, 2 nodes x 12 ranks:

```
step:  1  loss: 10.86492  grad_norm: 0.5543
step:  5  loss: 10.50758  grad_norm: 0.7077
step: 10  loss:  9.05897  grad_norm: 1.0462
```

Monotonic descent, finite grad norms throughout. MFU is ~0.05% because
`agpt_debugmodel` at `max_context_length=512` is a plumbing check, not a
performance measurement -- do not read a throughput number off this run.

Note the 2B / MoE / 80B results in
[`docs/guides/frameworks-rc-validation.md`](docs/guides/frameworks-rc-validation.md)
are **Sunspot** (`1247xxxx`), a different machine. The rows above are the
first training on Aurora's RC.

> [!TIP]
> To suppress the `UserWarning: Torchinductor` error seen when using
> `--compile.enable` on Aurora:
> ```bash
> export SYCL_DISABLE_FSYCL_SYCLHPP_WARNING=1
> ```

## Documentation index

The full prioritized landing page (with last-modified dates and a
sentence per entry) is at [`docs/README.md`](docs/README.md). For a
structural map of every file under `docs/`, see
[`docs/reference/TREE.md`](docs/reference/TREE.md). The headline pages by topic:

### Live status

| Page | What's there |
|------|--------------|
| [Production index](docs/live/dashboard.md) | Snapshot of every active training trajectory — 2B / 20B / 80B at 256N / 512N / 1024N+ |
| [Eval index](docs/records/evals/README.md) | lm-eval scores per model with v1-vs-v2 plots (the bf16-master fix is decisively validated) |
| [Journal](docs/journal.md) | Day-by-day session log |

### Setup + running

| Page | What's there |
|------|--------------|
| [Running with newer PyTorch](docs/reference/guides/running-with-newer-pytorch.md) | Module loads, venv install, tokenizer download, large-scale (>512 nodes) workflow |
| [`scripts/submit_agpt_{2b,20b,80b}_aurora_venv*.sh`](scripts/) | Current (torch 2.13 venv) PBS production submitters |
| [`submit/README.md`](submit/README.md) | Legacy torch-2.10-conda submit scripts (kept for v1 reproduction only) |

### Big findings + workarounds

| Page | What's there |
|------|--------------|
| [Known issues](docs/reference/guides/known-issues.md) | Operational notes + workarounds for active bugs |
| [bf16 RMSNorm freeze](docs/reference/guides/training-dtype-bf16-norm-freeze.md) | The headline v1 bug — why we restarted as v2 with `dtype=float32` |
| [Bad-node failover wrapper](docs/reference/guides/bad-node-failover.md) | How the `failover_lib.sh` wrapper detects + swaps bad nodes mid-training |
| [TP loss-reporting bug](docs/reference/guides/loss-reporting-tp-dist-reduce.md) | Why TP > 1 loss is off by `dp_world_size` and how `EzpzValidator` fixes it |
| [XPU attention issues](docs/reference/guides/xpu-attention-issues.md) | No flash-attn, selective AC quirks, SDPA fallback |

### Per-feature subdirectories

| Folder | Contents |
|--------|----------|
| [`agpt/`](agpt/) | AuroraGPT dense model configs (2B / 20B / 80B) + parallelism |
| [`moe/`](moe/) | DeepSeek-style MoE model + custom `EzpzGroupedExperts` |
| [`moe_runs/`](moe_runs/) | JSON override configs + launcher for MoE experiments |
| [`optimizer/`](optimizer/) | Mano, SPAM, Muon, SophiaG, ADOPT |
| [`blendcorpus/`](blendcorpus/) | olmo-mix-1124 dataloader with train/validation splits |
| [`eval/`](eval/) | DCP → HF converter + lm-eval pipeline |
| [`rl/`](rl/) | GRPO experimental task registry |
| [`scripts/`](scripts/) | Production submission scripts + interactive launchers + benchmarks |
| [`competition/`](competition/) | Loss-speedrun harness for the optimizer competitions |
| [`tests/`](tests/) | CPU/XPU unit tests (run with `python3 -m unittest`) |

### Long-form documentation

| Folder | Contents |
|--------|----------|
| [`docs/live/`](docs/live/) | Chain status today: per-model trackers, the dashboard, the dispatch log |
| [`docs/reference/guides/`](docs/reference/guides/) | Big-finding writeups, operational notes, how-tos |
| [`docs/reference/known-bugs/`](docs/reference/known-bugs/) | Diagnosed failures and their workarounds |
| [`docs/reference/scaling/`](docs/reference/scaling/) | Per-model scaling-study results (TPS / MFU vs N) |
| [`docs/records/evals/`](docs/records/evals/) | Per-model eval results + plots |
| [`docs/records/experiments/`](docs/records/experiments/) | Per-machine smoke / benchmark / LR-finder reports |
| [`docs/records/journal/`](docs/records/journal/) | Day-by-day session log, by month |
| [`docs/records/meeting-notes/`](docs/records/meeting-notes/) | AuroraGPT sync agendas + action items |
| [`docs/records/summaries/`](docs/records/summaries/) | 2-week / monthly retrospectives |
| [`docs/records/upstream-sync/`](docs/records/upstream-sync/) | Upstream merge log, by month |
| [`docs/records/competitions/`](docs/records/competitions/) | Optimizer speedrun leaderboards |
| [`docs/outbound/upstream-issues/`](docs/outbound/upstream-issues/) | Repros + drafts for PRs we're filing back to `pytorch/torchtitan` |
| [`docs/reference/configs/`](docs/reference/configs/) | Model config docs (architecture, registered names) |
| [`docs/reference/baselines/`](docs/reference/baselines/) | Reference training curves + benchmarks |

## MoE training

The MoE harness is documented at [`moe_runs/README.md`](moe_runs/README.md)
(JSON override configs, launchers, per-machine smoke + perf + prod-sim
recipes for Aurora and Polaris). The MoE model + the
`EzpzGroupedExperts` compute-backend selector live in
[`moe/`](moe/).

## References

- 🍋 `ezpz`:
  - Documentation: [ezpz.cool](https://ezpz.cool)
  - GitHub: [saforem2/ezpz](https://github.com/saforem2/ezpz)
- Upstream torchtitan: [pytorch/torchtitan](https://github.com/pytorch/torchtitan)
- Datasets:
  - [olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) (production training)
  - [google/gemma-7b](https://huggingface.co/google/gemma-7b) (tokenizer, vocab_size=256128)
