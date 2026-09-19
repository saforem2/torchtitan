# Running with newer PyTorch on Aurora `next-eval`

**Last validated:** 2026-09-19

> [!IMPORTANT]
> `next-eval` does **not** use Aurora's normal production image. Its compute
> nodes currently expose the TEST BKC (`compute_aurora_test_20260831`, software
> tree `26.181.0`, oneAPI 2026.1), while login nodes and ordinary production
> queues expose a different stack. A venv built against the production image's
> `/opt/aurora/26.26.0/spack/...` Python will not start on `next-eval`.
>
> This is a separate recipe from
> [running-with-newer-pytorch.md](running-with-newer-pytorch.md), which describes
> the production-image path, and from
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
| queue | `next-eval` |
| compute image | `compute_aurora_test_20260831` |
| software tree | `/opt/aurora/26.181.0` |
| oneAPI | 2026.1 |
| Python | 3.14.2 from `$HOME/.local/share/uv` |
| PyTorch | `2.13.0.dev20260428+xpu` |
| `spmd-types` | **0.2.5** |
| XPU layout | 12 tiles/node with `ZE_FLAT_DEVICE_HIERARCHY=FLAT` |

The Python is intentionally independent of `/opt/aurora`: that makes the venv
start on both the login node and the TEST BKC compute nodes. The tarball is
broadcast to node-local `/tmp/.venv` before launch.

## 1. Request `next-eval`

For an interactive check:

```bash
qsub -q next-eval -A AuroraGPT \
  -l walltime=00:30:00,filesystems=home:flare \
  -l select=2 -I
```

`next-eval` permits multi-hour jobs and does not currently impose the node-count
dead zone that shaped the old production-queue LR-finder scripts. Queue access
and limits can change; inspect the live configuration rather than relying on
this page:

```bash
qstat -Qf next-eval
```

## 2. Load the compute-node environment

Run this **inside the allocation**:

```bash
if ! command -v module >/dev/null 2>&1 || [[ -z "${MODULEPATH:-}" ]]; then
  source /etc/bash.bashrc.local
fi

module load oneapi/release/2025.3.1 hdf5 pti-gpu

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

The TEST image begins with the 2026.1 software tree. The explicit module load
above is the environment used by the current ezpz runner; it may print that
oneAPI was reloaded from 2026.1.0 to 2025.3.1. Record that message rather than
assuming the queue name alone identifies every library in the process.

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

import torch
from spmd_types import SpmdType
from torchtitan.distributed.fsdp import DataParallelMeshDims

assert metadata.version("spmd-types") == "0.2.5"
assert torch.xpu.is_available()
assert torch.xpu.device_count() == 12

print("python:", sys.version)
print("torch:", torch.__version__)
print("spmd-types:", metadata.version("spmd-types"), SpmdType)
print("XPU devices:", torch.xpu.device_count())
print("DataParallelMeshDims:", DataParallelMeshDims)
PY
```

Expected essentials:

```text
spmd-types: 0.2.5
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
  --link-mode=copy \
  "spmd_types==0.2.5"

"${VENV_ROOT}/.venv/bin/python" -c \
  'import importlib.metadata as m; from spmd_types import SpmdType; print(m.version("spmd-types"), SpmdType)'
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
verify_dir="${TMPDIR:-/tmp}/next-eval-venv-verify-${USER}"
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
| six XPU devices instead of twelve | composite device hierarchy | export `ZE_FLAT_DEVICE_HIERARCHY=FLAT` before importing torch |
| PBS says `Exit_status=0`, report says `CRASH` | wrapper swallowed the launcher status | use the fail-closed runner at or after `fa23dc6d1` |
| plausible loss with the wrong tokenizer | a pretokenized dataset silently overrode the config dataloader | keep the OLMo-3 config-owned Grain path |
| import succeeds on login but fails in job | login and `next-eval` images differ | test inside the allocation after venv broadcast |

## Related documentation

- [Frameworks 2026.1 validation matrix](frameworks-rc-validation.md)
- [Frameworks RC quickstart](aurora-quickstart-frameworks-rc.md)
- [Production-image newer-PyTorch guide](running-with-newer-pytorch.md)
- [Large-scale venv broadcast](running-with-newer-pytorch.md#running-at-large-scale--512-nodes)
- [OLMo-3 ladder plan](../experiments/optimizer-comparison/2026-09-18-olmo2tok-ladder-plan.md)
