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
evidence. The next model step is a minimal FSDP2/XCCL collective reproducer.

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
`LRF75_PREFLIGHT_PASS` under Torch `2.14.0+xpu`. Dependent sweep `12479014` is
queued. No basin or LR recommendation will be reported until the job produces
finite real updates, resumable checkpoints, and a complete 75-point curve.
