# RECIPE_PORT — Guoqiong XPU recipes → ezpz `xpu_trial`

> Port of Guoqiong `rl-xpu-aurora-torch214` @ `3358d8c873` (merge-base `d7e18743`)
> into `torchtitan/experiments/ezpz/rl/xpu_trial/` (AGENTS.md: no writes outside
> `experiments/ezpz/`). See `README.md` for full launcher/env docs and
> `/tmp/guoqiong-recipe-analysis.md` for the 132-line recipe delta analysis.

## 1. What was ported

| Source (`rl-xpu-aurora-torch214`) | Trial path | Notes |
|---|---|---|
| `torchtitan_recipes/rl/alphabet_sort.py` | `alphabet_sort.py` (verbatim copy) | 22 explicit `rl_grpo*` + ~696 dynamic host-layout names via `globals()[f"rl_grpo_qwen3_0_6b_..."]` |
| `scripts/rl/xpu/run_grpo.sh` | `run_grpo.sh` (+ `RECIPE_MODULE` param) | Default `RECIPE_MODULE=torchtitan.experiments.ezpz.rl.xpu_trial.alphabet_sort`; override with `qsub -v RECIPE_MODULE=...` |
| `scripts/rl/xpu/env_torch214.sh` | `env_torch214.sh` | oneAPI 2026.1, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, thread-pool caps |
| *(new)* | `layouts.py` | Single source of truth for DP layout: `ezpz_host_layout` / `resolve_dp_degrees` / `generator_count_for_tp`, parameterized over `ffn_dim1` |
| *(new)* | `recipe_adapter.py` | **This port's ConfigManager entry point** — imports `alphabet_sort.py` then re-exports only ezpz-compatible names (see §4) |
| *(new)* | `RECIPE_PORT.md` | This file |

## 2. DP_REPLICATE mapping (Guoqiong ↔ ezpz HSDP)

Both stacks implement the same invariant:

```
world == dp_replicate × dp_shard × cp × tp × pp   (this trial: cp=1, tp=1, pp=1)
mesh  == ["pp","dp_replicate","dp_shard","cp","tp"]
FSDP axis = dp_shard (dim-0 shard), replicate = dp_replicate (gradient all-reduce)
```

| Guoqiong field | ezpz field | Notes |
|---|---|---|
| `trainer.parallelism.data_parallel_replicate_degree` | `ParallelismConfig.data_parallel_replicate_degree` (`EzpzParallelismConfig` subclass under `experiments/ezpz/config.py`) | 1:1 — subclass passes `isinstance(..., ParallelismConfig)` so core annotations accept it unchanged; tyro exposes as `--parallelism.data-parallel-replicate-degree` |
| `trainer.parallelism.data_parallel_shard_degree` | `ParallelismConfig.data_parallel_shard_degree` | 1:1; complement of `dp_replicate` (`rep × shard == trainer_gpus`) |
| `generator.parallelism.tensor_parallel_degree` | `InferenceParallelismConfig.tensor_parallel_degree` | `_gtp` variant (see §3) |

Default layout is **node-local HSDP**: `dp_replicate = trainer_nodes` (one copy per trainer node), `dp_shard = gpus_per_host` so FSDP's per-layer all-gather stays intra-node:

```
trainer_nodes = num_hosts // 2   (1-host: trainer = gpus//2)
trainer_gpus  = trainer_nodes × gpus_per_host
generators    = (num_hosts - trainer_nodes) × gpus_per_host  (1-host: gpus - gpus//2)
walk replicas = (trainer_nodes, 1, 2, …, trainer_gpus)
  pick first rep where trainer_gpus % rep == 0, shard = trainer_gpus // rep,
  shard ≤ gpus_per_host and (shard ≤ 2 or ffn_dim1 % shard == 0)
```

Example 8n×ppn4: `trainer=16, rep=4, shard=4` → mesh `4×4`.

## 3. `_rep` / `_gtp` variants

### `_rep<R>` — DP_REPLICATE override (FSDP across nodes)

Selected when caller passes `DP_REPLICATE=<R>` to the launcher or constructs
`_rl_grpo_qwen3_0_6b_hosts(..., dp_replicate=R)` directly. Arithmetic:

```python
rep, shard = dp_replicate, rep * shard // dp_replicate   # preserves rep×shard == trainer_gpus
```

| Config suffix | Nodes | `dp_replicate × dp_shard` | Nodes per copy |
|---|---|---|---|
| (default) | 8n ppn4 | `4 × 4` | 1 |
| `_rep2` | 8n ppn4 | `2 × 8` | 2 (⚠️ full-arm hangs — see Known issues in README.md) |
| `_rep1` | 8n ppn4 | `1 × 16` | 4 |
| `_rep2` | 16n ppn4 | `2 × 16` | 4 (⚠️ both arms hang) |
| `_rep1` | 16n ppn4 | `1 × 32` | 8 |

Only emitted for `hosts > 2` and `hosts//2 * gpus % R == 0`.

Pure-CLI equivalent (no Python config names needed):

```bash
--parallelism.data-parallel-replicate-degree 1 --parallelism.data-parallel-shard-degree 16
```

### `_gtp<T>` — generator tensor-parallel degree

`GEN_TP=<T>` (or `generator_tp=T`) sets `generator.parallelism.tensor_parallel_degree = T`
and `num_generators = generators // T`. Emitted only when `generator_gpus % T == 0`
(`generator_gpus = gpus - gpus//2` on 1 host, else `gpus`). E.g. 2n×ppn4 `_gtp2` → 2 generators × TP2.

Requires launcher env `CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD=0` (oneCCL #222: TP
group's first communicator is intra-node; without it cross-node weight broadcasts
to generators deliver zeros — symptom `!!!!` completions, reward 0, 10 untrainable
batches). The trial launcher exports this (see `run_grpo.sh:55`).

Pure-CLI equivalent:

```bash
--generator.parallelism.tensor-parallel-degree 2   # and halve --num-generators if exposed
```

## 4. What `recipe_adapter.py` filters

Guoqiong's `alphabet_sort.py` also defines MoE/DeepEP (`rl_grpo_qwen3_moe*`),
GPT-OSS-20B (`rl_grpo_gpt_oss*`), Qwen3-30B-A3B and Qwen3.6-27B recipes that
require expert parallelism, separate renderers, or 64k context — never exercised
in the XPU regression and not wired to the trial launcher. `recipe_adapter.py`
imports the source module (which registers the dynamic host-layout names) then
re-exports only names matching:

* `rl_grpo*` and callable and not containing `moe`/`deepep`/`30b_a3b`/`gpt_oss`/`qwen3_6_27b`
* and starting with an allowed prefix (`rl_grpo_qwen3_0_6b*`, `rl_grpo_qwen3_1_7b`, `rl_grpo_qwen3_14b`, `rl_grpo_qwen3_5_*`)

That is ~700 names → ~696 kept (≈196 base + ~252 `_rep` + ~240 `_gtp` for Qwen3-0.6B, plus ~12 explicit Qwen3 base/Qwen3.5 variants). Validate with:

```bash
python3 - << 'PY'
import sys, types
sys.modules.setdefault('ezpz', types.ModuleType('ezpz'))
import sys; sys.modules['torch'] = types.ModuleType('torch')  # stub if not installed
# or just count via static scan:
import re, pathlib
text = pathlib.Path('torchtitan/experiments/ezpz/rl/xpu_trial/alphabet_sort.py').read_text()
# dynamic names are generated by the loop — count them separately
print('Explicit rl_grpo* defs:', len(re.findall(r'^def (rl_grpo\w+)', text, re.MULTILINE)))
PY
```

Use the adapter as the `--module`:

```bash
python -m torchtitan.experiments.ezpz.rl.train_upstream \
  --module torchtitan.experiments.ezpz.rl.xpu_trial.recipe_adapter \
  --config rl_grpo_qwen3_0_6b_full_8n_ppn4_10steps --output-dir outputs/...
# shorthand (ConfigManager prepends torchtitan.experiments.):
python -m torchtitan.experiments.ezpz.rl.train_upstream \
  --module ezpz.rl.xpu_trial.recipe_adapter \
  --config rl_grpo_qwen3_0_6b_full_8n_ppn4_10steps --output-dir outputs/...
```

## 5. Invariants to preserve

* `dp_replicate × dp_shard == trainer_gpus`, `dp_shard ≤ gpus_per_host`, `ffn_dim1 % dp_shard == 0 or dp_shard ≤ 2`.
* Launcher keeps `unset HOSTNAME` + `TMPDIR=/tmp` + `CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD=0` (don't reintroduce `TORCHINDUCTOR_CACHE_DIR` on `/home` — 16n race).
* No writes outside `torchtitan/experiments/ezpz/`.
