# Sunspot multi-host Monarch, TorchStore, and vLLM validation

Date: 2026-09-25

## Result

Merged `ezpz` plus the multi-host validation additions at commit
`738109e8d4481ebb723622db0d1a34b9c8907203` completed a real two-host RL
actor graph on Sunspot. PBS job `12478711` placed the trainer on host index 0
and the vLLM generator on host index 1, transferred policy weights with
TorchStore's Gloo transport, completed three finite GRPO optimizer updates,
ran pre- and post-training generation, wrote three DCP checkpoints and a
40-row rollout artifact, shut down cleanly, and exited zero.

This closes the gap left by one-node job `12478621`: policy publication and
vLLM consumption now work when trainer and generator reside on different
physical hosts in one Monarch actor graph.

## Exact contract

- Commit: `738109e8d4481ebb723622db0d1a34b9c8907203`
- Job: `12478711`
- Runtime: `/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214`
- Worktree: `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan-multihost-rl-738109e8d`
- Topology: two PBS hosts, one trainer XPU actor on host 0, one vLLM generator
  XPU actor on host 1
- TorchStore transport: `TransportType.Gloo`
- Starting model:
  `/lus/tegu/projects/datascience/foremans/reproductions/agpt2b-mds154391-broad-grain-sft900/stage2/agpt2b-mds154391-step600-gsm8k-r1cot-2n-r2/final`
- Training: three GRPO updates, four prompts per update, four samples per prompt,
  zero target off-policy steps, LoRA rank 8, fp32 generation
- Output:
  `/lus/tegu/projects/datascience/foremans/reproductions/agpt2b-mds154391-broad-grain-sft900/grpo/agpt2b-multihost-sync-12478711`

## Acceptance evidence

```text
MULTIHOST_RL_PLACEMENT trainer_host_index=0 generator_host_index=1 trainer_world_size=1 generator_world_size=1
Forcing TorchStore weight transport to TransportType.Gloo
Validation | Step: 0  validation/response_length/mean: 117.9
Train | Step: 1  loss/mean: 0.14  rollout_reward/_mean: 0.50  trainer/grad_norm/mean: 0.32
Train | Step: 2  loss/mean: 0.15  rollout_reward/_mean: 0.48  trainer/grad_norm/mean: 0.34
Train | Step: 3  loss/mean: 0.13  rollout_reward/_mean: 0.46  trainer/grad_norm/mean: 0.33
Validation | Step: 3  validation/response_length/mean: 120.4
MULTIHOST_TORCHSTORE_VLLM_OK
MULTIHOST_RL_DONE rc=0
PBS Exit_status = 0
```

`rollout_samples.jsonl` contains 40/40 completed rows, all with nonzero reward.
Observed policy versions were 0, 1, 2, and 3. DCP checkpoints were written at
steps 1, 2, and 3; each data shard is approximately 7.96 GB and has matching
metadata.

## Controlled failures

The failed jobs were necessary to isolate the production requirements:

- `12478706`: stale import of upstream's removed
  `breakable_cuda_graph_env`; failed before actor attachment.
- `12478707`: full RL startup exceeded Monarch's default attach config-push
  timeout.
- `12478708`: attachment passed, but integer host slices removed the `hosts`
  dimension and caused `KeyError: 'hosts'` in role provisioning.
- `12478709`: role placement and vLLM initialization passed, but automatic
  TorchStore selection misclassified the remote storage volume as local and
  entered the SharedMemory path; the generator rejected it with
  `Shared memory storage not found`. This is a current locality-resolution bug,
  not evidence that TorchStore's intended automatic cross-host order prefers
  SharedMemory.
- `12478710`: forced Gloo completed cross-host vLLM pre-validation, but the
  non-controller rank's default five-minute `FileStore.get()` timeout killed the
  otherwise healthy MPI world before training completed.

The final implementation uses Monarch's scheduler-SPMD `host_mesh_from_store`,
dimension-preserving host slices, a 120-second attach timeout, a 30-minute
coordination-store timeout, and explicit Gloo as the validated workaround for
the current automatic locality-resolution bug.

## Scope of the claim

This proves cross-host actor routing, vLLM generation, TorchStore policy
synchronization, optimizer execution, checkpoint writing, and clean shutdown for
the two-host AGPT-2B validation topology. It does not claim that forced XCCL
TorchStore transfer works; prior XCCL controls remain failed/hung. It also does
not establish a model-quality improvement, which requires paired semantic
evaluation.
