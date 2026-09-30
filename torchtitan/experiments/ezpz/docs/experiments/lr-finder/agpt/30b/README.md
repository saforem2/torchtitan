# LR Finder — agpt 30B OLMo-tokenizer

Learning-rate and runtime evidence for the canonical 26.20B-parameter
`agpt_30b_olmo2tok` model. For the campaign-wide record, see the
[GBS=6144 report](../2026-09-18-olmo2tok-ladder-gbs6144-nexteval.md).
For methodology and acceptance criteria, see the
[LR-finder methodology](../../README.md).

## Status

**AdamW LR: open. SophiaG LR: open.**

No complete fine curve has produced real terminal training evidence.

| optimizer | evidence | interpretation |
|---|---|---|
| AdamW | `12478510`, 94/100 points | incomplete; excluded |
| SophiaG | `12478513`, 30-point coarse; `12478570`, 97/100 fine | coarse candidate `6.48e-7` is diagnostic only; incomplete fine curve excluded |
| Muon | excluded | first-update Newton–Schulz corruption remains unresolved |

## Runtime blocker

The Aurora and Sunspot failures are different layers.

- Aurora `8862818` failed before model construction in XCCL communicator
  initialization. The communicator ladder later passed at 48, 192, 384, and
  twice at 768 ranks, closing that gate.
- Sunspot `12478695`, TP=1 and `dp_shard=32`, reached first backward and failed
  in FSDP2 `torch._chunk_cat` with `UR_RESULT_ERROR_OUT_OF_RESOURCES`.
- Sunspot `12478805`/`12478806`, TP=1 and `dp_shard=64`, failed in pre-forward
  `all_gather_single` with `SIGSEGV`.
- Sunspot `12478813`/`12478814`, TP=2 and `dp_shard=32`, reduced model memory to
  18.48 GiB/rank but failed in the same pre-forward path.
- Sunspot `12478834`/`12478835`, TP=4 and `dp_shard=32`, failed in the same path.

Gradient accumulation, compilation, checkpoint corruption, and persistent HBM
alone do not explain the failure. More blind topology permutations are not LR
evidence.

The maintained reduced reproducer closed that lower-level gate on Sunspot job
`12479055` (commit `199563d0658c00b3b738fff1b597371a07b2a845`, PBS
exit 0). It used the exact `100352 x 6144` embedding storage shape and the
production XCCL/SUM-reduction policy. Pure FSDP (`dp_replicate=1`,
`dp_shard=16`, 16 ranks with Sunspot's 12+4 placement) and HSDP
(`dp_replicate=3`, `dp_shard=16`, 48 ranks on four nodes) each completed three
finite backward passes and three nonzero parameter updates. The HSDP terminal
gradient norm was `1330.21497` and update norm was `1.3302151`. Earlier job
`12479051` independently passed exact-size raw BF16 all-gather, FP32
reduce-scatter, and stride-16-replica FP32 all-reduce controls. This rules out a
deterministic failure in those raw collectives, the shard-16 FSDP2 storage path,
or the reduced HSDP replicate axis; it does not prove the 768-rank full model.

The next controlled escalation is a four-node full-model canary at
`dp_replicate=3`, `dp_shard=16`, TP=1. It must complete three finite AdamW
updates with positive update-ratio evidence before any 64-node LR sweep is
released.

That escalation failed before step 1 in both forms. HSDP job `12479056`
(`3 x 16`, 48 ranks) built the 26.20B model at 32.64 GiB/rank, entered step 1,
then rank 16 was killed after a Level Zero `MPL_gpu_imemcpy`/MPI pipeline
failure during a oneCCL scale-out all-reduce. Pure-FSDP job `12479057`
(`1 x 16`, 16 ranks) also built the model and entered step 1, then ranks 12 and
14 raised `UR_RESULT_ERROR_OUT_OF_RESOURCES` from `loss.backward()` before
gradient clipping or an optimizer update. Removing the HSDP replicate axis
therefore does not remove the full-model failure. Since the exact-size raw
collectives and one-embedding FSDP lifecycle pass, the remaining boundary is
the full model's repeated collective/resource pressure. The next control keeps
64 layers and the same shard-16 runtime while reducing model width to 20B.

That initial size ladder accidentally enabled the trainer's heavyweight
per-parameter diagnostics on every step and disabled compile. The diagnostics
launch an extra DTensor reduction per parameter, and the 10B run `12479064`
completed backward/optimizer work before aborting inside
`diagnostics.collect_param_stats`; it was not a training-path failure.
Corrected non-intrusive 10B job `12479065` completed three finite AdamW updates
(loss `12.01583 -> 11.97582`, gradient norm `2.3522 -> 2.4822`) with PBS exit
0. Corrected 20B job `12479066` still failed in first backward with
`UR_RESULT_ERROR_OUT_OF_RESOURCES`; `rabenseifner`, host-staged `ring`,
node-local shard-12, and IPC-cache controls (`12479059`-`12479062`) did not
remove that Torch 2.14 eager-path failure. Torch 2.13 job `12479063` was not a
runtime A/B: it failed earlier at FSDP construction because a plain parameter
violated the current `dp_mesh_dims` DTensor contract.

The eager controls also diverged from the production 30B launcher, which keeps
compile enabled. A 20B-width/48-layer eager rung (`12479067`) reached a genuine
63.98-GiB HBM OOM. The maintained canary now preserves production compile
before the next 30B topology decision; eager failures cannot by themselves
reject the compiled production path.

The production-compiled follow-up closes the remaining local hypotheses without
finding a working 30B update path. Compiled 30B `3 x 16` jobs `12479068` and
`12479072` failed in FSDP backward unshard/copy-in before step 1. Corrected 10B
job `12479065` completed three updates, while corrected 20B job `12479066`
failed in backward, bracketing the current Torch 2.14 full-model ceiling above
10B. Disabling per-collective SYCL output events let 20B job `12479071` finish
backward and clipping, but AdamW's lazy moment allocation then exhausted HBM;
the same flag did not clear 30B. Backward-prefetch-off jobs `12479074`/`12479075`
shifted the first failure but did not remove it. Dimension-safe `dp_shard=32`
job `12479076` failed in FSDP post-backward reduce-scatter copy-in at
`torch._chunk_cat`. A reviewed equivalent copy fallback (`12479078`), explicit
device synchronization (`12479079`), and the combined output-event control
(`12479080`) all failed at the fallback's first `zero_()`, proving resources
were already exhausted before packing. These unsuccessful monkeypatches were
removed from the maintained branch.

Subsequent matched-runtime work explains why the historical 30B campaign could
train while the current path cannot. The successful August campaign used Torch
2.13, `partial_dtensor`, and pure FSDP over 192 ranks; this is verified directly
from the preserved `12473743`/`12473744` logs. Current upstream removed both the
backend selector and `partial_dtensor` in #4419.

Current-head job `12479081` recreated the historical 16-node, 192-rank, LBS=5,
GBS=960 geometry under Torch 2.14 and still failed before step 1. Although the
mesh resolved as `dp_shard=192`, FSDP attempted a 36.35-GiB all-gather while
29.84 GiB was resident per rank. This is nearly the same resident memory seen
at `dp_shard=16`, so increasing shard degree did not restore the historical
memory behavior.

The Torch 2.15 environment was then repaired rather than discarded. A previous
cleanup had deleted 38 legitimate package files whose basenames matched
`core.py`, `core.pyi`, or `core.h`. Reinstalling the exact
`torch==2.15.0.dev20260915+xpu` wheel and affected packages, plus restoring the
exact cached Triton file, produced an integrity scan of 33,409 RECORD files with
zero missing and zero mismatched. Matched 192-rank job `12479083` nevertheless
failed with the same 36.35-GiB first-forward FSDP all-gather as Torch 2.14.

Frameworks `2026.1.0` Torch `2.13.0a0+gitcf30153` was also tested. Current head
already calls `_parallelize(parallel_dims)` unconditionally, but job `12479084`
still hit the pytorch/pytorch#181519 contract during FSDP construction: a plain
`weight` remained where `dp_mesh_dims` requires a full-mesh DTensor.

An opt-in ezpz compatibility experiment then recreated the old TP=1 contract:
leave parameters plain before FSDP, use the one-dimensional `dp_shard` mesh, and
omit `DataParallelMeshDims`. Torch 2.15 5B job `12479085` passed three updates
(`loss3=11.99557`, `grad3=1.8735`), proving the path is functional at smaller
scale. It did not recover 30B: Torch 2.15 job `12479086` still attempted the
36.35-GiB first-forward all-gather, while frameworks Torch 2.13 job `12479087`
attempted a 72-GiB all-gather during fused-FFN parameter initialization at
`gate_init(t[0])`. No 30B job in this sequence completed one update.

Exact historical-source job `12479088` used commit
`5a26d8e7c5c05cd38bec7ab4eb48036e5ba54c6d` and package versions recovered
from the successful jobs' W&B manifests. It did not reproduce the old success
under the subsequently modified shared runtime, so it remains provenance
evidence rather than a new known-good baseline.

The decisive source delta was upstream #4808: fused FFN `w13` changed from
`[2F,D]` to stacked `[2,F,D]`, and core added an FSDP placement override that
shards matrix rows (`Shard(1)`). AGPT maintains a copied FSDP wrapper and had
not replayed that override. Its default `Shard(0)` padded the two-element
projection axis to the shard degree. At shard degree 192 this predicts the
observed allocations exactly: about 36 GiB in BF16 and 72 GiB in FP32.

Commit `cde3c93227` mirrors core's `linear_param_shard_placements()` contract in
AGPT. The first shard-192 attempt, `12479089`, correctly replaced the giant OOM
with a fail-fast divisibility error because `16384` is not divisible by `192`.
The dimension-compatible HSDP topology is `dp_replicate=3`, `dp_shard=64`.
Job `12479090` used that topology on repaired Torch 2.15 and completed three
finite updates with PBS exit 0 and `FULL_MODEL_CANARY_PASS`; terminal metrics
were `loss3=11.8405`, `grad3=3.5988`.

Restoring `full_dtensor` is unnecessary and is not the historical solution:
the successful jobs used `partial_dtensor`, while compiled AGPT on
`full_dtensor` previously failed the `vc_check` `DeviceMesh` assertion. The 30B
runtime gate is now green on current head with Torch 2.15 and HSDP `3 x 64`.
Production LR work may resume only through this validated topology and wrapper;
shard-192 is invalid for stacked `w13` because its matrix-row dimension is
16,384.

## Artifacts

- [30B SophiaG coarse CSV](../data/2026-09-24-olmo2tok-gbs6144-verified/sunspot-12478513-30b-sophiag-coarse.csv)
- [Evidence manifest](../data/2026-09-24-olmo2tok-gbs6144-verified/README.md)

Partial fine curves are deliberately not committed as successful evidence.
No independently initialized points will be appended to an existing trajectory.

## 2026-09-29 fresh resumable AdamW sweep

Job `12478510` reached 94/100 finite AdamW updates and then exhausted walltime,
but it cannot be resumed: its command disabled checkpointing and its source
predated durable `LRFinderState`. No model, optimizer, dataloader, or finder
cursor exists on disk. W&B history alone cannot continue the trajectory, and
new independently initialized points will not be stitched onto it.

A replacement uses one fresh 75-point trajectory over the same `3e-7` to
`3e-4` window at the measured-good 64-node geometry: TP1,
`dp_replicate=48`, `dp_shard=16`, 768 ranks, sequence length 4096, GBS 6144,
and full activation checkpointing. Source is
`45f2e72d0522b1bd8457d0ad645533658f3ff293`, which includes resumable
LR-finder state. Checkpoints are written every 25 points without automatic
purging.

Compute-runtime preflight `12479013` finished with PBS exit 0 and
`LRF75_PREFLIGHT_PASS` under Torch `2.14.0+xpu`. Sweep `12479014` then failed
before training because the immutable checkout lacked the OLMo-2 tokenizer.
After staging the verified tokenizer and adding an explicit wrapper check,
preflight `12479019` passed. Replacement `12479020` reached distributed/model
startup but rank 35 segfaulted before step 1; PBS recorded exit 143 and no LR
point or checkpoint exists. This reproduces the native pre-step boundary on the
measured geometry, so another blind retry is not justified. No basin or LR
recommendation will be reported until a controlled native-runtime diagnostic
identifies a working path and a complete trajectory is produced.

Reduced-probe attempts `12479049`, `12479052`, `12479053`, and `12479054` were
harness-development failures (respectively unsupported uneven `ezpz` occupancy,
missing rendezvous variables, sparse-gradient validation, and omitted oneCCL
SUM-reduction policy). None is counted as model or LR evidence.
