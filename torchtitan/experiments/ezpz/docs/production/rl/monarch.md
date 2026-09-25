# Production RL with Monarch, TorchStore, and vLLM on XPU

**Current production runbook. Last updated: 2026-09-25.**

> [!IMPORTANT]
> This page is the canonical operator guide for the current upstream-style
> TorchTitan RL path. Older TRL server-mode, oneAPI 2025.3, Torch 2.12/2.13,
> external-fork, and pre-`next-eval` instructions are retained only as historical
> reproductions. Do not copy their environment setup into a new production run.

## Current status

| Machine / queue | Status | Validated contract |
|---|---|---|
| Sunspot `workq` | **Validated end to end** | Two physical hosts; one trainer actor on host 0; one vLLM generator actor on host 1; Monarch scheduler-SPMD mesh; TorchStore Gloo transport; three finite GRPO updates; pre/post generation; checkpoints; clean shutdown. Job `12478711`, PBS exit 0. |
| Aurora `next-eval` | **Runtime contract established; multi-host RL gate still required** | oneAPI 2026.1 + an image-independent, versioned `.venv.next-eval` archive; no frameworks Conda environment; 12 flat XPU tiles per node. Do not call Aurora production-ready until the same actor, weight-sync, optimizer, artifact, and shutdown gates pass there. |

The merged implementation landed through PR #26. The authoritative Sunspot
hardware report is
[Sunspot multi-host Monarch, TorchStore, and vLLM validation](../../experiments/2026-09-25-sunspot-multihost-rl-validation.md).

## Architecture

The controller builds one Monarch actor graph over scheduler-launched host
workers:

```text
PBS allocation
  └─ ezpz launch: one rank per host
       └─ host_mesh_from_store(): one Monarch host mesh
            ├─ host 0 slice → trainer ProcMesh → TrainerActor
            └─ host 1 slice → generator ProcMesh → vLLM GeneratorActor

TrainerActor ── TorchStore/Gloo policy publication ──> GeneratorActor/vLLM
      │                                                    │
      └──────────── gradients + optimizer updates <─ rollouts/rewards
```

The production entry point is
`torchtitan/experiments/ezpz/rl/scripts/grpo/multihost_train_upstream.py`.
It supplies separate trainer/generator `HostMeshes` to the existing
`train_upstream.spawn_proc_mesh()` path; model, controller, reward, and
checkpoint logic remain the normal TorchTitan RL implementation.

## Why the validated cross-host launcher currently forces Gloo

TorchStore's intended automatic (`TransportType.Unset`) preference order is:

```text
same-host SharedMemory
→ TorchComms / MonarchRDMA
→ XCCL when available
→ Gloo
→ MonarchRPC
```

The successful one-host automatic runs (`12478538`, `12478621`) logged
`TransportType.Unset`. Because trainer and generator were colocated, the
expected first choice was SharedMemory; those jobs do **not** establish that
automatic selection chose XCCL.

The explicit-XCCL control `12478537` held the model and one-host actor topology
fixed but hung during the first publication after flatten/cast. Thus forced
XCCL is not a passing result on this stack.

For the two-host topology, automatic-selection job `12478709` incorrectly
entered the SharedMemory path even though the generator was remote. This points
to a locality/hostname-classification defect in the current integration, not a
design rule that cross-host TorchStore must use Gloo. The remote generator
correctly rejected the inaccessible shared-memory volume:

```text
Shared memory storage not found. This may indicate the storage volume is on a different host.
```

Until automatic locality resolution is fixed and revalidated, the committed
two-host launcher pins the known-good network fallback:

```bash
export TORCHTITAN_TORCHSTORE_TRANSPORT=gloo
```

Gloo is therefore the **currently validated workaround** for this cross-host
topology. It is not the preferred design endpoint, and it does not prove that
automatic XCCL cannot work after the locality bug is corrected. Do not silently
replace Gloo with XCCL or relabel the passing result as RDMA/XCCL without a new
hardware gate.

## Required evidence before promotion

A scheduler state or model load is not a pass. Require all of the following:

1. exact checkout SHA asserted by the batch script;
2. clean tracked worktree;
3. trainer and generator placement on distinct host identities;
4. explicit `TransportType.Gloo` in the controller log;
5. vLLM pre-training generation completes;
6. initial trainer policy publication and generator pull complete;
7. at least three finite optimizer updates with finite loss and gradient norm;
8. policy versions advance in retained rollout rows;
9. post-training generation completes;
10. non-empty rollout JSONL and DCP checkpoint metadata/data shards exist;
11. actor/process-mesh shutdown is clean;
12. scheduler exit is zero.

The passing Sunspot reference produced 40/40 completed nonzero-reward rollouts,
policy versions 0 through 3, and DCP checkpoints at steps 1, 2, and 3.

## Inspectable rollout examples

The authoritative raw artifact for job `12478711` is:

```text
/lus/tegu/projects/datascience/foremans/reproductions/
  agpt2b-mds154391-broad-grain-sft900/grpo/
  agpt2b-multihost-sync-12478711/rollout_samples.jsonl
```

It contains 40 rows: 16 at policy version 0 and eight each at versions 1, 2,
and 3. Every row has `status="completed"`; all 40 have nonzero total reward.
The renderer stores the reasoning body in `reasoning_content` and the answer
tail in `content`, so the examples below reconstruct the raw sampled response
as `<think>{reasoning_content}</think>{content}`.

### Exact-correct updated-policy rollout

Artifact row 16, policy version 1, training sample, reward `0.8`:

```text
User: A cake of 400 grams is divided into eight equal parts. Nathalie eats
one-eighth of the cake, and Pierre eats double what Nathalie ate. How much did
Pierre eat, in grams?

Assistant:
<think>
Nathalie eats 1/8 x 400g = 50g of cake.
Pierre eats 2 x 50g = 100g of cake.
</think>
<answer>\boxed{100}</answer><end_of_turn>
```

Reward components:

```text
ThinkFormatReward       1.0
AnswerExtractableReward 1.0
AnswerCloseReward       0.0
AnswerCorrectReward     1.0
```

This proves the remote vLLM actor consumed an updated policy version and
returned a bounded, correctly formatted, exact-answer completion.

### Well-formed partial-credit rollout

Artifact row 24, policy version 2, training sample, reward
`0.31666666666666665`:

```text
User: John can play 200 beats per minute. If he plays 2 hours a day for 3 days
how many beats does he play?

Assistant:
<think>
He played for 60*2=120 minutes
That means he played 120*200=24000 beats
</think>
<answer>\boxed{24000}</answer><end_of_turn>
```

The response is bounded and format-valid, but it omits the three-day factor.
The scorer therefore records:

```text
ThinkFormatReward       1.0
AnswerExtractableReward 1.0
AnswerCloseReward       0.33333333333333337
AnswerCorrectReward     0.0
```

This is useful negative evidence: the pipeline distinguishes successful
generation/formatting and partial numeric closeness from exact task correctness.

### Post-training validation can still be wrong

Artifact row 32, policy version 3, validation sample, reward `0.25`, emits a
well-formed but incorrect answer (`148`) after flawed traffic arithmetic. All
eight policy-version-3 validation rows were completed and bounded, but none had
`AnswerCorrectReward=1.0`. Therefore the job proves multi-host runtime and
policy-version advancement, **not** semantic improvement. Never promote a model
from these rollout snippets or aggregate reward alone; run a fixed paired
held-out evaluation.

## Sunspot: validated two-host reference

Use a clean detached worktree at an immutable pushed SHA. The merged validation
launcher is:

```bash
qsub \
  -v EXPECTED_COMMIT="$(git rev-parse HEAD)" \
  torchtitan/experiments/ezpz/rl/scripts/grpo/agpt2b_multihost_torchstore_validate.pbs
```

The launcher owns these machine-specific settings:

```bash
V=/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TORCHTITAN_TORCHSTORE_TRANSPORT=gloo
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=tcp
export CCL_KVS_IP_PORT="${head_node}_${port}"
```

It launches two scheduler ranks (`-n 2 -ppn 1`), then the entry point spawns one
trainer XPU actor on the first host and one generator XPU actor on the second.
The committed PBS script is a bounded production validation, not a long quality
campaign; change training budget only after the unmodified gate passes.

## Aurora `next-eval`: current runtime contract

Aurora `next-eval` uses the TEST compute image, not the ordinary production
image. New RL work must use all of the following:

```text
queue: next-eval
account: AuroraGPT
filesystems: home:flare
runtime module: oneapi/release/2026.1.0 (+ hdf5, pti-gpu as needed)
Python/PyTorch: image-independent, versioned .venv.next-eval archive
base frameworks Conda environment: forbidden
XPU hierarchy: ZE_FLAT_DEVICE_HIERARCHY=FLAT
```

A representative allocation header is:

```bash
#PBS -A AuroraGPT
#PBS -q next-eval
#PBS -l select=2
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:flare
```

Inside the allocation:

```bash
if ! command -v module >/dev/null 2>&1 || [[ -z "${MODULEPATH:-}" ]]; then
  source /etc/bash.bashrc.local
fi
module load oneapi/release/2026.1.0 hdf5 pti-gpu

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PATH="/opt/pbs/bin:${PATH}"

source <(curl -fsSL https://bit.ly/ezpz-utils)
ezpz_setup_job

VENV_ROOT=<shared directory containing the validated archive>
ARCHIVE="${VENV_ROOT}/.venv.next-eval.tar.gz"
ezpz yeet --src "${ARCHIVE}"
source /tmp/.venv.next-eval/bin/activate
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PBS_O_WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
```

The archive name and extraction directory must be versioned and asserted by the
wrapper; do not overwrite a production archive in place. If the current archive
extracts under a different node-local basename, derive and assert that path
instead of assuming `/tmp/.venv.next-eval`.

### Aurora fail-closed preflight

Before allocating a full RL run, execute this on every selected host using the
node-local interpreter that will launch the actors:

```bash
python3 - <<'PY'
import importlib.metadata as metadata
import inspect
import torch

import monarch
import torchstore
import vllm
from spmd_types import SpmdType
from torchtitan.experiments.ezpz.rl import train_upstream
from torchtitan.rl.controller import Controller

assert torch.__version__.startswith("2.15.") and "+xpu" in torch.__version__
assert metadata.version("spmd-types") == "0.2.5"
assert torch.xpu.is_available()
assert torch.xpu.device_count() == 12
assert callable(train_upstream.spawn_proc_mesh)
print(torch.__version__, monarch.__file__, torchstore.__file__, vllm.__file__, SpmdType, Controller)
PY
```

Also reject pip MPI runtimes such as `impi-rt`; site MPICH/PMIx must remain
authoritative. Import success on the login node is not a compute-node preflight.

### Aurora promotion sequence

Aurora has not yet passed the final multi-host RL gate. Promote in this order:

1. two-host actor-placement preflight with distinct host identities;
2. same two-host AGPT validation shape as Sunspot;
3. require explicit Gloo, pre/post vLLM generation, three finite updates,
   policy-version advancement, checkpoints, rollout artifacts, clean shutdown,
   and PBS exit 0;
4. only then increase actor counts, model size, or training budget.

Do not claim Aurora success from the generic communicator probe, scheduler
state, a parser check, or model construction.

## Operational invariants

- Use one scheduler allocation and one Monarch actor graph. Two independent MPI
  controllers are not an equivalent multi-host test.
- Keep checkpoints, datasets, repository, and outputs on the shared filesystem;
  only the runtime venv is node-local.
- Namespace output, TorchStore/FileStore rendezvous, and ports by PBS job ID.
- Retain the 120-second Monarch attach timeout and 30-minute coordination-store
  timeout: full RL/vLLM imports exceeded the defaults during healthy startup.
- Use dimension-preserving host slices (`slice(0, 1)`, not integer `0`) so role
  meshes retain the named `hosts` dimension.
- Scrub standalone vLLM subprocesses of machine-specific CCL/FI variables when
  the launcher does so; do not reintroduce inherited transport pollution.
- Never infer model-quality improvement from a mechanically successful GRPO
  run. Evaluate a fixed held-out set with retained raw generations.

## Failure signatures

| Symptom | Meaning | Action |
|---|---|---|
| attach config-push timeout | full RL imports exceeded Monarch's default attach window, or reverse channel is unroutable | retain the tested 120 s timeout; first confirm the lightweight two-host actor preflight |
| `KeyError: 'hosts'` | integer host slice dropped the named dimension | use `hosts.slice(hosts=slice(i, i + 1))` |
| `Shared memory storage not found` | automatic selection misclassified the remote volume as local and chose SharedMemory | use the validated Gloo workaround; separately fix/retest locality metadata before relying on `Unset` |
| non-controller rank times out after 300 s | wrapper's FileStore coordination timeout is shorter than healthy RL startup/training | retain the tested 30-minute timeout |
| forced XCCL hangs after flatten/cast | TorchStore XCCL transport remains unvalidated on this actor bootstrap | return to Gloo; treat XCCL as a separate experiment |
| vLLM engine fails after inheriting CCL/FI settings | standalone EngineCore inherited launcher transport state | use the committed XPU environment scrub; compare against the validated launcher |

## Historical and deprecated paths

- [`trl.md`](trl.md): legacy TRL `GRPOTrainer` + external vLLM-server path. Kept
  for reproduction; not the current production recommendation.
- [`2026-07-06_multinode-grpo-root-cause.md`](2026-07-06_multinode-grpo-root-cause.md):
  historical diagnosis of the legacy TRL multi-trainer-node path.
- [`monarch.md`](monarch.md): July-era one-host/two-tile Monarch result and reward
  studies. Useful history, superseded as an operator guide by this page.
- [`history/`](history/README.md): pre-current runtime bring-up records; never use
  their environment snippets as new-run instructions.
- [Aurora `next-eval` newer-PyTorch guide](../../guides/running-with-newer-pytorch-next-eval.md):
  detailed TEST-image environment and archive handling.
