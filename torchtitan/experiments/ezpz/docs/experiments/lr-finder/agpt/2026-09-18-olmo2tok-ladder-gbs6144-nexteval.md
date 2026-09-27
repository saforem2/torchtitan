# OLMo-3-vocab LR finder: 5B / 10B / 30B at GBS=6144

**Campaign:** 2026-09-18 through 2026-09-26

**Models:** `agpt_{5,10,30}b_olmo2tok`

**Sequence length:** 4096

**Global batch size:** 6144 sequences / 25,165,824 tokens

**Runtime:** queue-provided oneAPI 2026.1.0 with an isolated PyTorch environment

## Result

A recommendation requires a complete sweep, real optimizer updates, finite
losses and gradients, and fresh terminal artifacts. PBS exit 0 or a finite
loss CSV alone is not enough.

| model | AdamW | SophiaG |
|---|---:|---:|
| 5B | **`7.17e-5`** | open |
| 10B | **`4.16e-5`** | open |
| 30B | open | open |

### Final recommendations

- **5B AdamW:** job `12478508`, 100/100 points, suggested LR `7.17e-5`,
  blow-up `7.17e-4`.
- **10B AdamW:** job `12478509`, 100/100 points, suggested LR `4.16e-5`,
  blow-up `4.16e-4`.
Each final arm started from a clean initialization. The fine sweeps did not
continue from coarse-run weights.

## Open results

### 5B SophiaG

Job `12478624` completed 100/100 points and bracketed an interior loss basin.
Its raw minimum is `5.63e-4` at sample 77. The detector returned crossings at
`7.04e-6` and `4.96e-4`. The previously reported `7.04e-7` recommendation used
the first crossing, which is a low-LR noise wiggle before the basin. The
corrected post-minimum rule selects `4.96e-4`, yielding a safety-scaled
candidate of `4.96e-5`. That corrected candidate still requires a clean
validation run before publication as a final recommendation.

### 10B SophiaG

Job `12478569` completed 100/100 points over `[1.66e-6, 1.66e-4]`. The
application detector returned `3.24e-7`, below the sampled interval. The
coarse candidate was `1.66e-6`. Neither is a final recommendation. A new fine
sweep must measure below `1.66e-6` from a clean initialization.

### 30B AdamW

Job `12478510` stopped at 94/100 points. Later 30B controls did not produce a
valid terminal LR curve. The canonical model currently fails before its first
optimizer update in FSDP2 collectives. No AdamW recommendation is published.

### 30B SophiaG

Job `12478513` completed a 30-point coarse sweep and emitted candidate
`6.48e-7`, with blow-up `6.48e-6`. This is retained as coarse diagnostic
evidence only.

Job `12478570` stopped at 97/100 fine points without complete resumable state.
Its partial curve is excluded. No SophiaG recommendation is published.

## 30B runtime diagnosis

The 30B failures are not one generic OOM.

- Aurora job `8862818` failed before training in XCCL communicator creation.
  A later communicator ladder passed at 48, 192, 384, and twice at 768 ranks.
- Sunspot job `12478695`, TP=1 and `dp_shard=32`, reached the first backward
  pass and failed in FSDP2 `torch._chunk_cat` with
  `UR_RESULT_ERROR_OUT_OF_RESOURCES`.
- Jobs `12478805` and `12478806`, TP=1 and `dp_shard=64`, failed during
  pre-forward `all_gather_single` with `SIGSEGV`.
- Jobs `12478813` and `12478814`, TP=2 and `dp_shard=32`, reduced resident model
  memory to 18.48 GiB/rank but failed in the same pre-forward all-gather.
- Jobs `12478834` and `12478835`, TP=4 and `dp_shard=32`, failed in the same
  path.

Gradient accumulation, compilation, checkpoint corruption, and persistent HBM
alone do not explain the failure. More blind topology permutations are not LR
evidence. The next experiment should isolate the FSDP/XCCL collective runtime.

## Exclusions

- Muon remains excluded. Its first-update Newton–Schulz path has not passed a
  canary with finite gradients, finite weights, and real optimizer updates.
- Partial 30B curves are not appended to newly initialized runs.
- Independently initialized points are not combined into one continuous LR
  trajectory.
- The coarse 30B SophiaG candidate is not presented as a final result.

## Evidence

Immutable source CSVs live in
[`data/2026-09-24-olmo2tok-gbs6144-verified/`](data/2026-09-24-olmo2tok-gbs6144-verified/README.md).
The CSV schema records LR, loss, job ID, host, world size, token batch size, and
sequence length. It does not record gradient or update health. Final validity
was therefore checked against the terminal application logs and scheduler
records before each artifact was admitted here.
