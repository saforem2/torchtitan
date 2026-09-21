# PR #17 MoE integration: merge-readiness review

> **Last updated:** 2026-09-21
> **PR:** [saforem2/torchtitan#17](https://github.com/saforem2/torchtitan/pull/17)
> **Target:** `ezpz`
> **Reviewed head:** `44d6d5ab1` plus documentation follow-up
> **Current verdict:** **not ready to merge**. The restored full-Sonic training path
> and deterministic numerical/gradient comparison pass, but native DCP resume and
> final CI/review closure remain open.

## Executive summary

PR #17 now contains a working current-stack AGPT 2B/50K full-Sonic path. It is
no longer accurate to describe `aurora_full_sonic` as unported. The restored
factory builds, uses GQA-aware sharding, performs EP=12 training on Sunspot, and
matches a deterministic loop reference at DP=2 x EP=2 for outputs and first-order
gradients.

The review also found and repaired several independent integration problems:

- stale MLA-only attention assumptions in MoE sharding and FLOP accounting;
- SelectiveAC incompatibility with Sonic's nested custom autograd;
- missing EP-mesh and routing-contract validation;
- padding masks discarded before MoE routing;
- native-DDP code that advertised unsupported PP and mixed-precision modes;
- DDP wrapping after FSDP and validation after loss wrapping;
- launcher timeout status being converted to scheduler success;
- hard-coded launcher artifact paths that contradicted the documentation;
- lint, license-header, formatting, and broken-link failures;
- fork GPU jobs waiting for PyTorch-hosted runners unavailable to this repo.

## Current tested scope

### Host and CPU tests

The integrated focused suite completed:

```text
135 passed
trainer initialization parity: PASS
launcher contract tests: PASS
bash -n on modified launchers: PASS
pre-commit modified-file suite: PASS
repository-wide Lychee check: PASS locally
```

This includes:

- current AGPT/MoE registry and JSON factory loading;
- all retained active AGPT JSONs using the current schema;
- checkpoint rotation fail-closed at `keep_latest_k=0`;
- six registered expert backends;
- non-square Sonic weight layout;
- MoE routing-count behavior and padding-mask propagation;
- Sonic EP/routing validation before any collective;
- native-DDP validation and scope reduction;
- two-rank CPU/Gloo native-DDP update equivalence;
- trainer initialization parity;
- launcher factory, override, and return-code contracts.

## Sunspot XPU evidence

### Production-shaped AGPT 2B/50K full-Sonic smoke: PASS

Job `12478353`, commit `b8e070ba4`, Torch `2.14.0+xpu`, oneAPI `2026.1.0`:

- one Sunspot node;
- 12 XPU ranks;
- EP=12, DP-shard=12;
- 12.29B total / approximately 1.91B active parameters;
- full-Sonic forward and backward;
- optimizer updates;
- finite losses and gradient norms;
- exit status 0.

```text
step 1  loss 11.36224  grad_norm 3.0981  memory 19.34 GiB
step 2  loss 10.64051  grad_norm 5.0499  memory 20.81 GiB
Training completed
```

This was a functional smoke at context length 128 with compilation and W&B
disabled. It is not a throughput result and has no W&B URL.

Evidence on Sunspot:

```text
/lus/tegu/projects/datascience/foremans/agpt50k-sonic-12478353-result.txt
/lus/tegu/projects/datascience/foremans/agpt50k-sonic-12478353.log
```

### Deterministic Sonic-vs-loop numerical and gradient comparison: PASS

Job `12478371`, exit status 0:

- four XPU ranks arranged as DP=2 x EP=2;
- non-square expert dimensions `D=48`, `F=80`;
- fixed top-2 routing;
- imbalanced routes with global counts `[14, 7, 7, 0]`;
- one global expert receives no routes;
- full routed and shared expert paths;
- DP replicas bitwise identical.

Sonic matched the loop reference within the declared BF16 tolerances for:

- output;
- input gradient;
- router-score gradient;
- routed `w1`, `w2`, and `w3` gradients;
- all three shared-expert gradients.

Maximum absolute differences across ranks:

```text
output                 <= 1.53e-05
input gradient         <= 3.05e-05
router-score gradient  <= 5.72e-06
routed weight gradient <= 1.53e-05
shared weight gradient  = 0
```

Evidence on Sunspot:

```text
/lus/tegu/projects/datascience/foremans/sonic-validation-12478371/result.json
/lus/tegu/projects/datascience/foremans/sonic-validation-12478371/run.log
```

## Diagnostic failures and what they established

These jobs are not counted as successful validation, but each exposed a distinct
integration or harness defect:

| job | result | finding |
|---|---|---|
| `12478347` | failed during `config.build()` | MoE sharding assumed MLA `Attention.Config`; AGPT correctly uses `GQAttention.Config`. Fixed by dispatching to the shared GQA sharding helper. |
| `12478349` | failed during model construction | Reduced smoke wrapper omitted vendored `aurora_moe` from `PYTHONPATH`. |
| `12478350` | failed before Sonic execution | Reduced smoke wrapper omitted required `AURORA_MOE_ALLTOALLV=1`. The checked-in production launcher already had it. |
| `12478351` | reached step 1, then failed | Venv `bin` was not on `PATH`, so PyTorch could not discover the installed `ninja` executable. |
| `12478352` | reached real backward, failed | SelectiveAC is incompatible with Sonic's nested `torch.autograd.grad` backward. The restored full-Sonic factory now selects activation checkpointing `none`. |
| `12478364`/`12478365` | harness failure | Standalone tests assumed torchrun `WORLD_SIZE`; `ezpz launch` supplies PALS variables. |
| `12478366`/`12478367` | harness failure | The first PALS adapter used nonexistent `PALS_SIZE`, producing an empty `WORLD_SIZE`. |
| `12478372` | DCP test failure | Raw `dcp.load` into fresh AdamW did not materialize initially empty optimizer state; this was a harness/API misuse. |
| `12478376` | DCP test failure | The synthetic test represented different EP-local tensors under identical ordinary-tensor keys, which DCP correctly interpreted as replicas. |
| `12478377` | DCP test still failed | DTensor representation improved the test, but interrupted and uninterrupted state still diverged. Native-DCP resume equivalence is **not yet established**. |

## Important implementation constraints

### Full Sonic and activation checkpointing

Full Sonic currently requires activation checkpointing to be disabled. Its custom
autograd backward invokes `torch.autograd.grad`, which conflicts with
SelectiveAC's single-backward region constraint. The restored factory therefore
uses AC `none`.

### Expert-parallel process group

Full Sonic owns dispatch and combine. It must receive the actual EP mesh and must
not substitute the world group. The integration now fails before collectives
unless:

- an actual `DeviceMesh` is supplied;
- EP size is greater than one;
- global experts divide evenly across EP ranks;
- local weight shards contain exactly the expected number of experts;
- routing tensors have valid shape, dtype, device, counts, and expert IDs.

### Padding

The MoE transformer block now forwards `padding_mask` into the core MoE router.
Padded positions are excluded from routing-count accumulation and auxiliary-loss
statistics, matching the current core router contract.

### Native DDP

Native DDP has been reduced to a deliberately narrow supported scope:

- AGPT only;
- non-pipeline only;
- no TP, CP, or EP;
- replicated DP with DP-shard degree 1;
- BF16 autocast over FP32 master parameters and FP32 reductions;
- one gradient-accumulation step;
- no fault tolerance, CPU offload, seed-checkpoint creation, or custom optimizer
  groups.

The previous PP/native-DDP and experimental DDP mixed-precision branches were
removed rather than leaving unreachable or misleading code. The AGPT
parallelizer skips FSDP when native DDP is selected, and a CPU/Gloo two-rank test
checks update equivalence against a single-process global-batch reference.

A real XPU native-DDP smoke and checkpoint round trip are still desirable before
calling native DDP production-validated.

## Launcher and CI corrections

- Launchers continue to use `ezpz launch`.
- Inner launcher and timeout return codes now propagate to PBS.
- A resumable timeout remains nonzero rather than being reported as scheduler
  success.
- The two-node full-Sonic launcher honors documented `TT_*` path overrides and
  validates required files and hashes before training.
- Documentation now agrees that `checkpoint.keep_latest_k=0` retains all
  checkpoints.
- Ordinary lint, formatting, license-header, and codespell failures were fixed.
- Broken current documentation links were repaired; only explicitly enumerated
  historical records and literal internal proxy values receive narrow Lychee
  exclusions.
- PyTorch-hosted CUDA workflows are skipped for pull requests in forks where
  those runners are unavailable. CPU/lint coverage remains enabled.

## Remaining merge gates

1. **Resolve native-DCP resume equivalence.** Job `12478377` remains a failing
   validation. Determine whether the remaining divergence is another synthetic
   harness mismatch or a real state-dict/optimizer restore problem, then obtain
   a passing interrupted-versus-uninterrupted test.
2. **Obtain green GitHub lint on the latest head.** Local full Lychee and
   modified-file pre-commit pass, but the latest pushed CI result must be green.
3. **Run a real XPU native-DDP smoke.** The CPU/Gloo equivalence test validates
   reducer/loss scaling, but not XPU autocast and XCCL behavior.
4. **Update the PR description.** It still says full Sonic is unported and lists
   obsolete test results.
5. **Resolve all review threads** with fixing commits or explicit scope
   decisions, and request a fresh review.

## Follow-ups that need not block the narrowly validated Sonic merge

- Add an HF adapter for the GQA+MoE flavor, or explicitly reject HF conversion;
  current native DCP training does not imply HF export support.
- Validate `aurora_sycl` with FP32 master weights and BF16 compute, including
  input and weight gradients.
- Fingerprint prebuilt extension caches by source, Python ABI, Torch, compiler,
  oneAPI, target architecture, and build options.
- Measure the transient memory cost of materializing three transposed contiguous
  expert-weight copies per Sonic forward.
- Add stronger overflow and gradient assertions for the capacity-dropping `bmm`
  backend.
