# Upstream `f359667` merge readiness

Status as of 2026-09-29 for branch `sync/upstream-f359667`.

## Scope and provenance

The integration merges upstream TorchTitan through
[`f359667137438bef39274cb869626c3798512ecd`](https://github.com/pytorch/torchtitan/commit/f359667137438bef39274cb869626c3798512ecd)
at merge commit `780f0a73e2`, then replays the local configuration/topology, MoE,
RL, and regression-test changes. Sunspot hardware validation now includes the
MoE repair commit `6913990333`.

The merge preserves both dense AGPT and MoE experiment surfaces. It also adds
fail-closed handling for fake-SPMD real pipeline groups and repairs executable
validation paths that independent review found were calling stale APIs.

## Verified gates

| Gate | Result | Evidence |
|---|---|---|
| Pre-sync collectable ezpz suite | pass | 256 passed, 2 skipped |
| Post-integration collectable ezpz suite | pass | 260 passed, 2 skipped, 14 subtests passed |
| Focused post-review suite | pass | 50 passed, 2 skipped |
| Full `tests/unit_tests/ezpz` suite | pass | 75 passed, 2 skipped |
| Broad CPU collection classification | no replay regression found | 33 pre-sync collection failures were shared; five post-only files are newly added tests |
| Dense numerical parity | exact pass | `llama3/debugmodel`, two AdamW steps, `rtol=0`, `atol=0` |
| MoE numerical parity | exact pass | `deepseek_v3/debugmodel`, standard MoE communication, two AdamW steps, `rtol=0`, `atol=0` |
| Cross-version checkpoint resume | exact pass | pre -> post and post -> pre reproduced uninterrupted step 1 for dense and MoE |
| Independent review, first pass | findings fixed locally | validator stale API calls and fake-SPMD topology/serialization risks repaired |
| Sunspot dense TP=1 | pass | job `12479017`, three optimizer steps, arm exit 0 |
| Sunspot dense TP=2 | pass | job `12479017`, three optimizer steps, arm exit 0; exercises the TP sharding-contract path |
| Sunspot MoE TP=1 | pass after repair | job `12479018`, commit `6913990333`: three finite optimizer steps, arm exit 0, `VERDICT: ok`; losses 12.95227 -> 12.59636 -> 11.47138 |
| Sunspot distributed checkpoint resume | exact pass | job `12479022`, commit `c228830bd3`: uninterrupted control; full step-2 save; fresh-process load; resumed steps 3-4 loss/gradient exact; full step-4 save; PBS exit 0 |
| Sunspot RL/weight sync | pass | two-host job `12479032`, commit `a893208c8f`: explicit Gloo, four policy pushes and pulls, three nonzero-gradient updates, versions 0–3 consumed, 40/40 completed rollouts, checkpoints at steps 1–3, PBS exit 0 |

The numerical comparison used Python 3.14.7, PyTorch 2.13.0 CPU, one thread,
model seed `20260929`, input seed `20260930`, and AdamW with LR `8e-4` and
weight decay `0.1`. Dense covered 6,163,712 parameters and 39 state keys; MoE
covered 33,014,016 parameters and 77 state keys. Every scalar delta was zero and
every output, gradient, optimizer, and post-step tensor digest matched. The gate
script SHA-256 is
`5b5cfe04d075ad4ea4718d0c08e03dc4a7217bb5236adb92efc8659b71a9b9ae`.

Native `FlexInnerAttention` backward is unavailable on this CPU runtime. The
comparison replaced only each built model's inner-attention runtime module with
a deterministic causal eager implementation; model parameters, routing,
surrounding forward structure, loss, gradients, AdamW state, and checkpoint
state remained real. Distributed DCP resume is therefore still a separate
accelerator gate.

## Review fixes awaiting commit

- `EzpzValidator` now requests the valid `dp` mesh rather than nonexistent
  `batch` and enters `dist_utils.get_spmd_context(...)` instead of calling the
  nonexistent `self.validation_context()`.
- The validator regression executes preprocessing, forward, loss, and logging
  on CPU rather than inspecting source text.
- `ParallelDims` rejects multi-axis mesh requests containing `pp` when fake SPMD
  uses a separate real pipeline process group.
- Serialization fails explicitly while that live process group is attached;
  the field no longer participates in equality or repr.
- Two tautological strict-`xfail` controls were removed.

## Known environment limits

The broad local CPU suite cannot collect fully because the shared macOS runtime
lacks or mismatches optional/core dependencies including `expecttest`,
`torchvision`, `transformers`, `torch_checkpointing.default_resharder`, and
some Triton/Torch private APIs. An identical pre-sync control established the
baseline. In particular, `test_parallel_dims.py` currently stops at collection
because `expecttest` is absent; this is not recorded as a passing gate.

The broad experiment directory also contains 14 script-style `test_*.py` files
that execute and may call `sys.exit` at import time. The collectable manifest is
used for pytest; those scripts are classified and exercised separately rather
than hidden with broad skips.

## Required before PR

- Complete final independent review/security scan and fix/reverify any blocker.
- Sunspot dense, MoE, distributed checkpoint save/load/resume, and two-host
  RL/weight-sync gates are green on exact integration commits.
- Run representative Aurora dense and MoE gates through the established
  `aurora-tt-ezpz` ownership path.
- Re-run the complete collectable ezpz suite after the review fixes are
  committed.
- Push the integration branch, open the PR against `ezpz`, and require exact-head
  CI plus current review approval.

The branch is not merge-ready until those remaining accelerator and review gates
produce terminal artifacts. Live operational state is maintained in the private
TorchTitan Agent Vault (not reachable from public CI runners).
