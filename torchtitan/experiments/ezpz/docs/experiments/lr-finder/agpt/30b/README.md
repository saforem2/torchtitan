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
