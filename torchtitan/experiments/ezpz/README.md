# TorchTitan + 🍋 `ezpz`

Pre-training and post-training AuroraGPT (dense + MoE) on ALCF systems:
Aurora and Sunspot (XPU), Polaris and Sirius (CUDA). This folder is an
experiment harness on top of upstream `torchtitan` that adds:

- Fault-tolerant training (bad-node failover wrapper)
- Per-machine launch scripts + venv broadcast (`ezpz yeet-env`)
- Custom optimizers (Mano, SPAM, Muon, SophiaG, ADOPT)
- A `BlendCorpusDataLoader` for olmo-mix-1124 + arbitrary HF datasets
- Eval pipeline (DCP → HF safetensors → lm-eval)
- RL: Monarch GRPO with a vLLM generator and TorchStore weight sync, on XPU and CUDA
- Production / eval / scaling tracking under [`docs/`](docs/)

This file is the **landing page**: quickstart + an index of where
everything lives.

> [!NOTE]
> This is the [`saforem2/torchtitan@ezpz`](https://github.com/saforem2/torchtitan/tree/ezpz)
> fork. Upstream is [`pytorch/torchtitan`](https://github.com/pytorch/torchtitan).
> Last sync: upstream `8e63247c`, merged 2026-11-14 (#79). Every sync is logged in
> [`docs/upstream-sync.md`](docs/upstream-sync.md).

## Versions (as of 2026-11-14)

The venvs the last sync was validated on:

| Machine | Venv | torch |
|---|---|---|
| Sunspot | `/lus/tegu/projects/datascience/foremans/venvs/xpu-torch216-sync-5a7a622f16` | `2.16.0.dev20261015+xpu` |
| Aurora | `/flare/AuroraGPT/foremans/venvs/xpu-torch216-sync-5a7a622f16` | same |
| Polaris | `/eagle/AuroraGPT/foremans/venvs/cuda-torch215-cu132` | `2.15.0+cu132` |
| Sirius | `/home/foremans/torchtitan/.venv` | `2.15.0+cu132` |

- **XPU needs torch >= 2.14.** `frameworks/2026.1.0` ships `2.13.0a0+gitcf30153`, which
  lacks pytorch [#181519](https://github.com/pytorch/pytorch/pull/181519) and dies at FSDP
  wrapping ([`docs/experiments/sync84-xpu-smoke.md`](docs/experiments/sync84-xpu-smoke.md)).
- **torch 2.10 (`frameworks/2025.3.1`) does not work anywhere**: core `distributed/fsdp.py`
  imports `DataParallelMeshDims`, added in 2.13.
- **TP > 1 needs `spmd_types==0.3.1`.**
- `Cannot import config_registry for module 'ezpz.agpt'` hides the real import error.
  Run `python3 -c "import torchtitan.experiments.ezpz.agpt.config_registry"` from the
  repo root to see it.

## Quickstart (2B dense training, 2 nodes)

Full setup details — module loads, venv install, large-scale yeet-env
broadcast — are in
[`docs/guides/running-with-newer-pytorch.md`](docs/guides/running-with-newer-pytorch.md).
Minimum viable path on Aurora:

```bash
# 1. Allocate two nodes
qsub -q prod -A AuroraGPT -l walltime=06:00:00,filesystems=flare:home -l select=2 -I

# 2. Clone + enter
git clone https://github.com/saforem2/torchtitan --branch ezpz
cd torchtitan

# 3. Setup environment (loads modules + ezpz helper functions)
source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup_env

# 4. Launch 2B training
MODEL=2b bash torchtitan/experiments/ezpz/run_train.sh
```

For other models / machines / data files, see
[`run_train.sh`](run_train.sh) and the detailed setup guide above.

## Smoke test (all four machines)

[`scripts/sync_smoke_alcf.sh`](scripts/sync_smoke_alcf.sh) is the 2-node check run after
every upstream sync: agpt and MoE debug configs, local batch 2, and an exact resume
check. The wrappers `sync_smoke_{sunspot,aurora,polaris,sirius}.sh` carry only the
`#PBS` headers:

```bash
qsub -A <account> -q debug -l select=2 -l walltime=01:00:00 \
  -l filesystems=<fs> torchtitan/experiments/ezpz/scripts/sync_smoke_sunspot.sh
```

The last line is `VERDICT: ok`, `ok_with_known_failures` or `failed`.

## RL (Monarch GRPO)

Two-host GRPO (trainer on host 0, vLLM generator on host 1, TorchStore weight sync)
trains on all four machines. AGPT-2B LoRA on GSM8K, 3 steps, exit 0 on Sirius `33518`,
Sunspot `12483743` and Aurora `9089138`.

```bash
qsub -A <account> -q <queue> -l filesystems=<fs> \
  -v V=/path/to/rl-venv,CKPT=/path/to/hf-checkpoint \
  torchtitan/experiments/ezpz/rl/scripts/grpo/multihost_grpo_e2e.pbs
```

The RL venvs are separate from the training venvs above: `rl-monarch-torch216-sync-20261112`
on Aurora (`/flare/AuroraGPT/foremans/venvs/`) and Sunspot
(`/lus/tegu/projects/datascience/foremans/venvs/`), `rl-cuda132-sync-20261113` on Sirius
(`/home/foremans/venvs/`). XPU RL venvs need `vllm-xpu-kernels` built for PVC without its Xe2-only kernels
([`rl/scripts/build_vllm_xpu_kernels_pvc.pbs`](rl/scripts/build_vllm_xpu_kernels_pvc.pbs))
and `torchvision`. CUDA uses upstream's `scripts/install_rl_env.sh`. Details and
timings: the 2026-11-12 entry in [`docs/upstream-sync.md`](docs/upstream-sync.md) and
[`docs/production/rl/monarch.md`](docs/production/rl/monarch.md).

## Aurora `frameworks/2026.1.0`

The module is visible from Aurora login nodes as of 2026-11-14 (it was
validation-node only in August). It bundles torch `2.13.0a0+gitcf30153`, below the XPU
floor above, and none of our stack. The overlay-venv recipe, the seven traps that each
report success, and the August RC results are in
[`docs/guides/frameworks-rc-validation.md`](docs/guides/frameworks-rc-validation.md).

## Documentation index

The full prioritized landing page (with last-modified dates and a
sentence per entry) is at [`docs/README.md`](docs/README.md). For a
structural map of every file under `docs/`, see
[`docs/TREE.md`](docs/TREE.md). The headline pages by topic:

### Live status

| Page | What's there |
|------|--------------|
| [Production index](docs/production/README.md) | Snapshot of every active training trajectory — 2B / 20B / 80B at 256N / 512N / 1024N+ |
| [Eval index](docs/evals/README.md) | lm-eval scores per model with v1-vs-v2 plots (the bf16-master fix is decisively validated) |
| [Journal](docs/journal.md) | Day-by-day session log |
| [Upstream sync log](docs/upstream-sync.md) | What each upstream sync changed, how it was validated, what it left open |

### Setup + running

| Page | What's there |
|------|--------------|
| [Running with newer PyTorch](docs/guides/running-with-newer-pytorch.md) | Module loads, venv install, tokenizer download, large-scale (>512 nodes) workflow |
| [`scripts/submit_agpt_{2b,20b,80b}_aurora_venv*.sh`](scripts/) | Aurora PBS production submitters (venv + failover wrapper) |
| [`submit/README.md`](submit/README.md) | Legacy torch-2.10-conda submit scripts (kept for v1 reproduction only) |

### Big findings + workarounds

| Page | What's there |
|------|--------------|
| [Known issues](docs/guides/known-issues.md) | Operational notes + workarounds for active bugs |
| [bf16 RMSNorm freeze](docs/guides/training-dtype-bf16-norm-freeze.md) | The headline v1 bug — why we restarted as v2 with `dtype=float32` |
| [Bad-node failover wrapper](docs/guides/bad-node-failover.md) | How the `failover_lib.sh` wrapper detects + swaps bad nodes mid-training |
| [TP loss-reporting bug](docs/guides/loss-reporting-tp-dist-reduce.md) | Why TP > 1 loss is off by `dp_world_size` and how `EzpzValidator` fixes it |
| [XPU attention issues](docs/guides/xpu-attention-issues.md) | No flash-attn, selective AC quirks, SDPA fallback |

### Per-feature subdirectories

| Folder | Contents |
|--------|----------|
| [`agpt/`](agpt/) | AuroraGPT dense model configs (2B / 20B / 80B) + parallelism |
| [`moe/`](moe/) | DeepSeek-style MoE model + custom `EzpzGroupedExperts` |
| [`moe_runs/`](moe_runs/) | JSON override configs + launcher for MoE experiments |
| [`optimizer/`](optimizer/) | Mano, SPAM, Muon, SophiaG, ADOPT |
| [`blendcorpus/`](blendcorpus/) | olmo-mix-1124 dataloader with train/validation splits |
| [`eval/`](eval/) | DCP → HF converter + lm-eval pipeline |
| [`rl/`](rl/) | AuroraGPT RL registries, `train_upstream.py`, Monarch launchers, XPU compatibility shims |
| [`scripts/`](scripts/) | Production submission scripts + interactive launchers + benchmarks |
| [`competition/`](competition/) | Loss-speedrun harness for the optimizer competitions |
| [`tests/`](tests/) | CPU/XPU unit tests (run with `python3 -m unittest`) |

### Long-form documentation

| Folder | Contents |
|--------|----------|
| [`docs/production/`](docs/production/) | Live per-model / per-node-count training trackers |
| [`docs/evals/`](docs/evals/) | Per-model eval results + plots |
| [`docs/guides/`](docs/guides/) | Big-finding writeups, operational notes, how-tos |
| [`docs/experiments/`](docs/experiments/) | Per-machine smoke / benchmark / LR-finder reports |
| [`docs/scaling/`](docs/scaling/) | Per-model scaling-study results (TPS / MFU vs N) |
| [`docs/competitions/`](docs/competitions/) | Optimizer speedrun leaderboards |
| [`docs/meeting-notes/`](docs/meeting-notes/) | AuroraGPT sync agendas + action items |
| [`docs/summaries/`](docs/summaries/) | 2-week / monthly retrospectives |
| [`docs/upstream-issues/`](docs/upstream-issues/) | Repros + drafts for PRs we're filing back to `pytorch/torchtitan` |
| [`docs/configs/`](docs/configs/) | Model config docs (architecture, registered names) |
| [`docs/baselines/`](docs/baselines/) | Reference training curves + benchmarks |

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
