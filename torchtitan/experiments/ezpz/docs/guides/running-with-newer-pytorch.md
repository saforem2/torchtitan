# Running with newer PyTorch on Aurora `prod` (oneAPI 2026.1)

**Last validated:** 2026-09-30

> [!IMPORTANT]
> Submit new work to Aurora's `prod` routing queue. The production execution
> queues now use the current BKC and provide the oneAPI 2026.1 stack by default;
> the former `next-eval` workflow is no longer the operator entry point. Do not
> load `frameworks/2026.1.0`: its bundled PyTorch is a different runtime from
> the isolated Torch 2.15 environment validated here.
>
> This image-independent runtime remains separate from the older
> [site-Python recipe](running-with-newer-pytorch.md) and from
> [aurora-quickstart-frameworks-rc.md](aurora-quickstart-frameworks-rc.md), which
> layers a venv on the compute image's bundled framework module.

## Validated stack

The reusable environment used here lives at:

```text
/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/.venv
/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/.venv.tar.gz
```

| component | validated value |
|---|---|
| submission queue | `prod` (routing queue) |
| execution queues | `small`, `medium`, `large`, or matching `backfill-*` route |
| compute image | current production BKC (`compute_aurora_prod_20260928T161726_a233` when audited) |
| software tree | `/opt/aurora/26.181.0` |
| oneAPI | 2026.1, inherited by default |
| Python | 3.14.2 from `$HOME/.local/share/uv` |
| PyTorch | `2.15.0.dev20260919+xpu` |
| required PyTorch fix | `pytorch/pytorch#181519` (`_resolve_spmd_types_for_storage`) |
| `spmd-types` | **0.2.5** |
| Grain | **0.2.18** |
| XPU layout | 12 tiles/node with `ZE_FLAT_DEVICE_HIERARCHY=FLAT` |

The Python is intentionally independent of `/opt/aurora`: that makes the venv
start on both the login node and current production compute nodes. The tarball is
broadcast to node-local `/tmp/.venv` before launch.

## 1. Request `prod`

For an interactive check:

```bash
qsub -q prod -A AuroraGPT \
  -l walltime=00:30:00,filesystems=home:flare \
  -l select=2 -I
```

`prod` is a routing queue. PBS selects an execution queue according to node
count, walltime, and backfill eligibility. Queue access, limits, and the active
BKC can change; inspect both the route and its destination queues rather than
relying on a historical queue name:

```bash
qstat -Qf prod
qstat -Qf small
qstat -Qf medium
qstat -Qf large
```

## 2. Load the compute-node environment

Run this **inside the allocation**:

```bash
if ! command -v module >/dev/null 2>&1 || [[ -z "${MODULEPATH:-}" ]]; then
  source /etc/bash.bashrc.local
fi

# oneAPI 2026.1 is part of the default production allocation environment.
# Load only workload-specific additions that are not already present.
module load hdf5 pti-gpu

export PATH="/opt/pbs/bin:${PATH}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov"
```

The production image begins with the 2026.1 software tree. Keep its default
oneAPI 2026.1 runtime with the 2.15 XPU nightly, then prepend `$VIRTUAL_ENV/lib` to
`LD_LIBRARY_PATH`: the wheel's bundled Unified Runtime loader exports
`urGraphGetIdExp`/`urDeviceWaitExp`, while the system loader may advertise the
same symbol version without exporting those entry points. Record module reload
messages rather than assuming the routing queue alone identifies every library in
the process.

## 3. Use an image-independent venv

Do not activate this repository's production-image `.venv`. Broadcast the
known venv tarball instead:

```bash
cd /flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
source <(curl -fsSL https://bit.ly/ezpz-utils)
ezpz_setup_job

VENV_ROOT=/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz
source "${VENV_ROOT}/.venv/bin/activate"
ezpz yeet --src "${VENV_ROOT}/.venv.tar.gz"
deactivate
source /tmp/.venv/bin/activate
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Code comes from the current checkout; the tarball supplies the runtime.
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
```

`/tmp` is node-local. Keep the repository, datasets, checkpoints, and outputs
on `flare` or another shared filesystem.

## 4. Fail-closed preflight

Do not substitute a nearby import such as `DataParallelMeshDims` for the real
requirements. The September 18 LR sweep did that, passed its preflight, and
then all 12 jobs crashed because its borrowed venv still had `spmd-types 0.2.1`.
Repository HEAD imports `SpmdType`, which is present in the pinned 0.2.5 API.

Run the exact imports and assert the exact versions:

```bash
python3 - <<'PY'
import importlib.metadata as metadata
import sys

import grain
import torch
from spmd_types import SpmdType
from torchtitan.distributed.fsdp import DataParallelMeshDims

assert metadata.version("spmd-types") == "0.2.5"
assert metadata.version("grain") == "0.2.18"
if not torch.__version__.startswith("2.15.") or "+xpu" not in torch.__version__:
    raise RuntimeError(f"wrong torch build: {torch.__version__}")

import inspect
import torch.distributed.fsdp._fully_shard._fsdp_param as fsdp_param
assert "_resolve_spmd_types_for_storage" in inspect.getsource(fsdp_param)
assert torch.xpu.is_available()
assert torch.xpu.device_count() == 12

print("python:", sys.version)
print("torch:", torch.__version__)
print("spmd-types:", metadata.version("spmd-types"), SpmdType)
print("grain:", metadata.version("grain"), grain.__file__)
print("XPU devices:", torch.xpu.device_count())
print("DataParallelMeshDims:", DataParallelMeshDims)
PY
```

Expected essentials:

```text
spmd-types: 0.2.5
grain: 0.2.18
XPU devices: 12
```

A successful scheduler state or process exit is not sufficient evidence. The
preflight must run from `/tmp/.venv`, after broadcast, using the same
`PYTHONPATH` and modules as training.

## 5. Updating the reusable venv

Use `uv` from the login environment because this venv intentionally has no
`pip` module:

```bash
VENV_ROOT=/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz
uv pip install \
  --python "${VENV_ROOT}/.venv/bin/python" \
  --pre \
  --upgrade-package torch \
  --index-url https://download.pytorch.org/whl/nightly/xpu \
  --link-mode=copy \
  torch torchvision torchaudio

# The XPU nightly resolver installs Intel's pip MPI runtime. Aurora launches
# with the site MPICH/PMIx stack; leaving impi-rt in the venv can make even
# `ezpz yeet` abort in MPI_Init_thread before the archive is distributed.
uv pip uninstall \
  --python "${VENV_ROOT}/.venv/bin/python" \
  impi-rt

uv pip install \
  --python "${VENV_ROOT}/.venv/bin/python" \
  --link-mode=copy \
  "spmd_types==0.2.5" \
  "grain==0.2.18"

"${VENV_ROOT}/.venv/bin/python" -c \
  'import importlib.metadata as m; import grain; from spmd_types import SpmdType; print(m.version("spmd-types"), SpmdType, m.version("grain"), grain.__file__)'
```

Back up and rebuild the tarball after changing the source venv. Never replace
the working archive with a partial one:

```bash
cd "${VENV_ROOT}"
stamp=$(date +%Y%m%d-%H%M%S)
cp -p .venv.tar.gz ".venv.tar.gz.before-${stamp}"
tar -czf .venv.tar.gz.new .venv

tar -tzf .venv.tar.gz.new >/dev/null
mv .venv.tar.gz.new .venv.tar.gz
```

Then extract into a fresh directory and test the archive itself, not only the
source venv:

```bash
verify_dir="${TMPDIR:-/tmp}/prod-venv-verify-${USER}"
mkdir -p "${verify_dir}"
tar -xzf .venv.tar.gz -C "${verify_dir}"
"${verify_dir}/.venv/bin/python" -c \
  'import importlib.metadata as m; from spmd_types import SpmdType; assert m.version("spmd-types") == "0.2.5"; print("archive OK")'
```

Keep the dated backup until a compute-node smoke has passed.

## 6. Smoke before scaling

First prove the archived environment on one or two nodes. For the OLMo-3
ladder (the existing config names remain `*_olmo2tok`), use the same
config-owned Grain dataloader and tokenizer that the real run will use; do not
override it with a pretokenized `blendcorpus` list.

```bash
ezpz launch --nproc 24 --nproc_per_node 12 -- \
  python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt \
    --config agpt_5b_olmo2tok_smoke \
    --training.steps 2 \
    --checkpoint.no-enable \
    --metrics.log-freq 1 \
    activation-checkpoint:full
```

Required evidence:

- the exact `SpmdType` preflight passed from `/tmp/.venv`;
- at least one real training step was logged;
- the expected fresh output artifact exists;
- the log contains no traceback or non-finite values.

Only then scale to 64 nodes.

## 7. LR-finder launch

The repository runner now performs the exact `spmd-types` preflight and returns
nonzero if any arm is `CRASH`, `OOM`, `TIMEOUT`, or `UNKNOWN`:

```bash
qsub \
  -v MODEL_SIZE=5b,LRF_OPTIMIZERS=adamw \
  torchtitan/experiments/ezpz/scripts/submit_lr_finder_olmo2tok_aurora.sh
```

Submit one size and optimizer per job when independent failure and accounting
matter. For the full ladder, see the
[OLMo-3 LR-finder report and job map](../experiments/lr-finder/agpt/2026-09-18-olmo2tok-ladder-gbs6144-nexteval.md).

## Failure signatures

| symptom | meaning | action |
|---|---|---|
| venv Python reports `No such file or directory` | venv was built against the production `/opt/aurora` Python | use the image-independent Python 3.14 venv |
| `cannot import name 'SpmdType' from 'spmd_types'` | borrowed venv has 0.2.1; HEAD requires 0.2.5 | update source venv, rebuild tarball, test extracted archive |
| `No module named 'grain'` | the image-independent venv lacks the config-owned dataloader dependency | install `grain==0.2.18`, rebuild, and re-run the exact preflight |
| FSDP rejects a plain `weight` with `dp_mesh_dims` | torch predates `pytorch#181519`, even if `spmd-types` itself is current | install a 2.15 XPU nightly and assert `_resolve_spmd_types_for_storage` exists |
| `undefined symbol: urGraphGetIdExp` or `urDeviceWaitExp` while importing torch | the system Unified Runtime loader won over the newer loader bundled with the XPU wheel | prepend `$VIRTUAL_ENV/lib` to `LD_LIBRARY_PATH` after activating the broadcast venv |
| `PMIX_Init returned -25` during `ezpz yeet` | the XPU wheel resolver installed `impi-rt`, shadowing Aurora's site MPICH/PMIx | uninstall `impi-rt`, rebuild the archive, and verify it is absent |
| six XPU devices instead of twelve | composite device hierarchy | export `ZE_FLAT_DEVICE_HIERARCHY=FLAT` before importing torch |
| PBS says `Exit_status=0`, report says `CRASH` | wrapper swallowed the launcher status | use the fail-closed runner at or after `fa23dc6d1` |
| plausible loss with the wrong tokenizer | a pretokenized dataset silently overrode the config dataloader | keep the OLMo-3 config-owned Grain path |
| import succeeds on login but fails in job | login and production compute images differ | test inside the allocation after venv broadcast |

## Related documentation

- [Frameworks 2026.1 validation matrix](frameworks-rc-validation.md)
- [Frameworks RC quickstart](aurora-quickstart-frameworks-rc.md)
- [Large-scale venv broadcast](#running-at-large-scale--512-nodes)
- [OLMo-3 ladder plan](../experiments/optimizer-comparison/2026-09-18-olmo2tok-ladder-plan.md)

## Running at Large Scale (> 512 nodes)

At a few nodes, importing Python from a shared filesystem is a small tax —
~milliseconds per import, paid once.

At 6k workers all hammering Lustre with the same import waterfall, that small
tax becomes minutes of dead time before the first training step lands, plus
tail-latency stragglers that dominate collective wait time for the rest of the
run.

[`ezpz yeet`](https://ezpz.cool/cli/yeet/) sidesteps this by copying the active
venv (or a pre-built `.venv.tar.gz`) to node-local `/tmp/` storage on every
worker, so subsequent imports, checkpoint loads, and config reads hit local SSD
instead of Lustre.

The local copy is patched once on the source node (activate scripts, shebangs,
symlinks) and then distributed via a greedy `rsync` fan-out — each completed
node immediately becomes a source for others, so the broadcast tree grows in
roughly `O(log N)` time instead of saturating one NIC.

### Measured scaling on Aurora (8 → 4096 nodes)

| Nodes | yeet (s) | Per-node (ms) |
| ----: | -------: | ------------: |
|     8 |     69.7 |         8,712 |
|    16 |     89.7 |         5,606 |
|    32 |     89.2 |         2,788 |
|    64 |     91.2 |         1,425 |
|   128 |    110.4 |           862 |
|   256 |    132.9 |           519 |
|   512 |    174.5 |           341 |
|  1024 |    255.4 |           249 |
|  2048 |    421.4 |           206 |
|  4096 |    750.6 |           183 |

Two regimes:

- **< 128 nodes** the cost is dominated by the one-time local extract (~70-91 s
  flat)
- **≥ 128 nodes** the broadcast tree depth and per-leaf contention dominate,
  with each 2× in nodes adding ~1.5-1.8× wall-clock.

Even at full-Aurora 4096-node scale the pre-launch overhead is under 13 minutes,
versus the 1--2 hours the per-file `rsync` mode was projected to take.

For more detail (full sweep, plots, methodology) see the
[yeet CLI docs](https://ezpz.cool/cli/yeet/) and the
[benchmark harness](https://github.com/saforem2/torchtitan/tree/ezpz/torchtitan/experiments/ezpz/docs/scaling/yeet_env).

### Workflow

> [!TIP]
> If you already built a tarball with `ezpz tar-env`, pass it explicitly —
> tarball broadcast is ~10× faster than per-file `rsync` at scale because the
> Lustre side becomes one sequential read instead of millions of `stat()`s.
>
> Plain `ezpz yeet` will print a hint when it sees a same-named `.tar.gz`
> sitting nearby.

1. Be sure to load the appropriate modules and activate the `.venv` we just
   created:

   ```bash
   source <(curl -fsSL https://bit.ly/ezpz-utils)
   ezpz_setup_job
   ezpz_setup_xpu
   source .venv/bin/activate
   ```

1. Create a compressed tarball (`.tar.gz`) from the `.venv`:

   ```bash
   ezpz tar-env
   # or, alternatively:
   # tar czvf .venv.tar.gz --directory .venv .
   ```

   Note:
   This will take a ~few minutes but only needs to be done _once_ (and can be
   done on CPU).
   After that, we can simply reuse the `.venv.tar.gz` for subsequent
   runs[^tarball].

1. Distribute the tarball to every node's `/tmp/`:

   ```bash
   ezpz yeet .venv.tar.gz   # faster at scale

   # alternatively, yeet the uncompressed `.venv/`:
   # ezpz yeet              # slower at scale
   ```

   <details closed><summary>Output</summary>

   ```bash
   #[05/04/26,09:08:42][x4302c2s3b0n0][~/a/f/p/s/torchtitan-ezpz][ezpz][?]
   ; ezpz yeet .venv.tar.gz
   [2026-05-04 09:08:53][I][utils/yeet_env:331:_maybe_apply_hsn_suffix] HSN interface available on 2/2 nodes (-hsn0 suffix)
   [2026-05-04 09:08:53][I][utils/yeet_env:1013:run] Source: /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/.venv.tar.gz (2.7G)
   [2026-05-04 09:08:53][I][utils/yeet_env:1014:run] Target: /tmp/.venv/ on 2 node(s)
   [2026-05-04 09:08:53][I][utils/yeet_env:1016:run]   local:  x4302c2s3b0n0 (rsync to /tmp/.venv/)
   [2026-05-04 09:08:53][I][utils/yeet_env:1019:run]   remote: x4302c2s4b0n0-hsn0
   [2026-05-04 09:08:53][I][utils/yeet_env:1054:run] Syncing (2 nodes)...

       ✓ x4302c2s3b0n0 (local, tar.gz (pre-built)) — 48.5s
       ✓ x4302c2s4b0n0-hsn0 — 20.1s
   [2026-05-04 09:10:02][I][utils/yeet_env:1334:run] Done in 68.9s

   To use this environment:
     deactivate 2>/dev/null
     source /tmp/.venv/bin/activate

   Then launch your training (from a shared filesystem path):
     cd /path/to/your/project
     ezpz launch python3 -m your_app.train

   Note: /tmp is node-local. Make sure your working directory
   is on a shared filesystem (e.g. Lustre) before launching,
   so all ranks can access data and outputs.
   [2026-05-04-091002] Command: ezpz yeet .venv.tar.gz
   took: 1 min. 14 s.
   ```

   </details>

1. Deactivate the _current_ `.venv` and activate the one we just created at
   `/tmp/.venv/`:

   ```bash
   deactivate && source /tmp/.venv/bin/activate
   ```

   ```bash
   $ which python3
   /tmp/.venv/bin/python3
   ```

1. Launch training as usual; `ezpz launch` respects `$VIRTUAL_ENV`, so it picks
   up the `/tmp/` venv automatically:

   ```bash
   ezpz launch python3 -m torchtitan.experiments.ezpz.train \
       --module=ezpz.agpt \
       --config=agpt_2b \
       --training.steps=10 \
       --checkpoint.no-enable \
       --training.num-tokens-per-microbatch-per-dp-rank=8192 \
       --training.max-context-length=8192
   ```

> [!IMPORTANT]
> `/tmp/` is node-local.
> Keep your project directory (data, checkpoints) on a shared filesystem so
> all ranks can read inputs and write outputs.

[^tarball]: If we install additional packages or make changes to the `.venv/`, we will need
to repeat this step and create a new `.venv.tar.gz` to capture these.

<details closed>
<summary>Deprecated Instructions from <code>oneapi/2025.3.1</code></summary>

> Historical instructions retained for reproducibility only. Do not use this
> setup for new Aurora jobs; use the current `prod` / oneAPI 2026.1 procedure
> above.

> [!NOTE]
> We will use the alias `uvi`:
>
> ```bash
> alias uvi='uv pip install --no-cache --link-mode=copy'
> ```

> [!IMPORTANT]
> To access the internet, you need to set the following environment variables:
>
> ```bash
> export http_proxy="http://proxy.alcf.anl.gov:3128"
> export https_proxy="http://proxy.alcf.anl.gov:3128"
> export no_proxy="localhost,127.0.0.1,*.alcf.anl.gov,*.anl.gov"
> ```

1. Clone torchtitan:

   ```bash
   gh repo clone saforem2/torchtitan -- --branch ezpz
   cd torchtitan
   ```

1. Load modules and export environment variables[^ezpz-setup]:

   ```bash
   source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job && ezpz_load_modules
   ```

1. Create venv:

   ```bash
   # to use the python from `/opt/aurora/.../python-3.12.12-xxx/bin/python3`
   module load python

   # create venv
   uv venv \
       --system-site-packages \
       --relocatable \
       --no-cache \
       --link-mode=copy \
       --python=$(which python3)

   # activate venv
   source .venv/bin/activate
   ```

1. Install PyTorch:

   ```bash
   uvi torch torchvision torchaudio torchdata \
       --pre \
       --index-url https://download.pytorch.org/whl/nightly/xpu \
       --upgrade
   ```

   - **NOTE** (2026-06-09): The _nightly_ PyTorch 2.13 has missing symbols and
     is **currently** broken.
     The latest (confirmed) functional PyTorch 2.13 wheel is `torch==2.13.0.dev20260519+xpu`

1. Install dependencies:

   ```bash
   uvi spmd_types torchcomms tyro tensorboard deepspeed mpi4py
   uvi "git+https://github.com/zhenghh04/blendcorpus"
   uvi "git+https://github.com/saforem2/ezpz"
   ```

1. Remove Intel's MPI runtime (`impi-rt`):

   ```bash
   uv pip uninstall impi-rt
   ```

1. Download tokenizers:

   ```bash
   python3 scripts/download_hf_assets.py --repo_id google/gemma-7b --assets tokenizer
   ```

1. Run training:

   ```bash
   MODULE=ezpz.agpt
   CONFIG=agpt_2b
   ezpz launch python3 -m torchtitan.experiments.ezpz.train \
       --module="${MODULE}" \
       --config="${CONFIG}" \
       --training.steps=10 \
       --checkpoint.no-enable \
       --training.num-tokens-per-microbatch-per-dp-rank=8192 \
       --training.max-context-length=8192
   ```

   > [!NOTE]
   > Batch sizes are counted in TOKENS since upstream #4121 (80th sync).
   > `--training.local-batch-size`, `--training.global-batch-size` and
   > `--training.seq-len` no longer exist, and a command carrying any of them
   > dies in flag parsing with `Unrecognized options:` before step 1. The
   > mapping is `tokens = batch_size * seq_len`, and
   > `num_tokens_per_microbatch_per_dp_rank` must EQUAL `max_context_length`
   > on the blendcorpus path -- the loader folds `[B, L] -> [T]`, so a second
   > row overruns the RoPE cache. That is why the old `LBS=2` above becomes
   > `8192` (one row of `seq_len=8192`) and not `16384`. Raise the global
   > batch with `--training.num-tokens-per-train-step` instead, which sets
   > gradient accumulation. See
   > [known-bugs/dead-cli-flags-in-repo-root-pbs.md](known-bugs/dead-cli-flags-in-repo-root-pbs.md)
   > and
   > [known-bugs/blendcorpus-fold-batch-dim.md](known-bugs/blendcorpus-fold-batch-dim.md).

   - <details closed><summary>AuroraGPT-20B:</summary>

     ```bash
     MODULE=ezpz.agpt
     CONFIG=agpt_20b
     ezpz launch python3 -m torchtitan.experiments.ezpz.train \
         --module="${MODULE}" \
         --config="${CONFIG}" \
         --training.steps=10 \
         --checkpoint.no-enable \
         --training.num-tokens-per-microbatch-per-dp-rank=8192 \
         --training.max-context-length=8192
     ```

     </details>

   - <details closed><summary>AuroraGPT-80B:</summary>

     ```bash
     MODULE=ezpz.agpt
     CONFIG=agpt_80b
     ezpz launch python3 -m torchtitan.experiments.ezpz.train \
         --module="${MODULE}" \
         --config="${CONFIG}" \
         --training.steps=10 \
         --checkpoint.no-enable \
         --training.num-tokens-per-microbatch-per-dp-rank=8192 \
         --training.max-context-length=8192 \
         --optimizer=adamw \
         --optimizer.lr=1e-6 \
         --parallelism.tensor-parallel-degree=2 \
         --compile.no-enable
     ```

     See [`guides/training/agpt_80b.md`](training/agpt_80b.md) for the
     full 80B walkthrough (prerequisites, expected step-by-step
     numbers, scale-out, known issues).

     </details>

[^ezpz-setup]: Explicitly, the `ezpz_load_modules` sets:

     ```bash
     module load oneapi/release/2025.3.1 hdf5 pti-gpu
     export ZE_FLAT_DEVICE_HIERARCHY=FLAT
     export CCL_PROCESS_LAUNCHER=pmix
     export CCL_OP_SYNC=1
     export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
     export TORCH_CPP_LOG_LEVEL=ERROR
     ```

</details>
