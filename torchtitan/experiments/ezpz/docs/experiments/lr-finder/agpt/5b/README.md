# LR Finder — agpt 5B OLMo-tokenizer

Learning-rate evidence for the 4.64B-parameter `agpt_5b_olmo2tok` / `agpt_5b_olmo3_polaris` model.
For the campaign-wide records, see the
[GBS=6144 report](../2026-09-18-olmo2tok-ladder-gbs6144-nexteval.md) and the
[GBS=128 Sirius report](../2026-11-12-olmo3-gbs128-sirius.md).
For methodology and acceptance criteria, see the
[LR-finder methodology](../../README.md).

## Status

**Production LR: open.**

The complete AdamW and SophiaG sweeps are real evidence, but their original
detector outputs are not final production recommendations. Fixed-LR training
is now validating sustained behavior.

| optimizer | evidence | interpretation |
|---|---|---|
| AdamW | `12478508`, 100/100 points over `1e-6`–`1e-3` | observed basin minimum `6.58e-4`; detector `7.17e-5` is crossing ÷ 10, not a measured optimum |
| SophiaG | `12478624`, 100/100 points over `2.8e-6`–`2.8e-3` | old `7.04e-7` selected a pre-basin noise crossing; corrected candidate `4.96e-5` remains unvalidated |
## OLMo-3-1K GBS=128 on Sirius (2026-11-11 to 2026-11-15) -- direct input to Polaris chain

Same model family at the **Polaris production batch** (GBS=128, 524k tok = 128 x 4096, 8N 32 GPUs Sirius H100).
Exponential sweep `1e-08 -> 1e-01`, 150 steps, `mult=1.114243`, `fraction=0.15` of `steps=1000`.
Seven of eight arms exited 0 with finite losses (10B SophiaG incomplete); FSDP2 on 32 GPUs clean.

| optimizer | job | suggested (blow-up) | min loss at LR | interpretation |
|---|---|---:|---:|---|
| **AdamW** | `33429` | **2.48e-04** (`2.48e-03`) | `9.38` at `2.53e-03` | deeper descent than Mano |
| **Mano** | `33429` | **1.81e-04** (`1.81e-03`) | `9.75` at `2.27e-03` | 27% lower than AdamW at 5B |
| **Muon** | `33542` | **6.54e-04** (`6.54e-03`) | `9.57` at `6.69e-03` | 2.6x higher than AdamW, widest basin |
| **SophiaG** | `33542` | degenerate | `10.00` at `1.06e-03` | detector noise at `7e-06`; real candidate `1.01e-04` not trusted |

![5B all optimizers GBS128](figures/sirius-5b-all-opts-gbs128.png)
![5B AdamW vs Mano GBS128](figures/sirius-33429-5b-adamw-mano-gbs128.png)

Cross-size: [10B page](../10b/README.md) shows 10B inversions -- Mano 76% higher than AdamW, Muon 8.3x higher than 5B AdamW.
Polaris 5B recommendation at GBS=128: **AdamW 2.48e-04 / Mano 1.81e-04 / Muon 6.54e-04** (blow-up/10, see [GBS128 report](../2026-11-12-olmo3-gbs128-sirius.md)). SophiaG not recommended at this batch.
Artifacts: [`sirius-33429-5b-adamw-150.csv`](../data/2026-11-12-olmo3-gbs128-sirius/sirius-33429-5b-adamw-150.csv) |
[`sirius-33429-5b-mano-150.csv`](../data/2026-11-12-olmo3-gbs128-sirius/sirius-33429-5b-mano-150.csv) |
[`sirius-33542-5b-muon-150.csv`](../data/2026-11-12-olmo3-gbs128-sirius/sirius-33542-5b-muon-150.csv) |
[`sirius-33542-5b-sophiag-150.csv`](../data/2026-11-12-olmo3-gbs128-sirius/sirius-33542-5b-sophiag-150.csv) | [manifest](../data/2026-11-12-olmo3-gbs128-sirius/README.md)

### Fixed-LR candidate table (GBS=128, validated LRs)

Four validated LRs feed the gated fixed-LR matrix (all blow-up/10):

| model | optimizer | fixed-LR candidate | blow-up LR | evidence job | status |
|---|---|---:|---:|---|---|
| 5B | AdamW | **2.48e-04** | `2.48e-03` | `33429` | validated |
| 5B | Muon | **6.54e-04** | `6.54e-03` | `33542` | validated |
| 10B | AdamW | **1.28e-04** | `1.28e-03` | `33430` | validated |
| 10B | Muon | **1.06e-03** | `1.06e-02` | `33543` | validated |

See [GBS128 report](../2026-11-12-olmo3-gbs128-sirius.md) and
[agpt index](../README.md#recommended-learning-rates) for full context.
Muon's basin is 2.6x-8.3x higher than AdamW at this batch.

### SophiaG degenerate -- excluded from fixed-LR matrix

SophiaG is **excluded** from the fixed-LR matrix until re-measurement:

- **5B** (`33542`, 150/150): detector reports two candidates -- `7.00e-07`
  (`7.00e-06` blow-up) and `1.01e-04` (`1.01e-03` blow-up). The first is a noise
  crossing in the nearly-flat early basin (loss barely moved at that LR); the
  second sits near the shallow observed minimum (`10.00` at `1.06e-03`) but the
  curve barely descends there (12.00 -> 10.00 vs AdamW 12.01 -> 9.38) and is too
  shallow to trust. The `~1e-5` noise and `~1e-3` drift span two decades, so
  detector output is untrusted -- neither LR can be safely fixed.
- **10B** (`33543`, 25/150): incomplete -- `trajectory_fingerprint mismatch`
  (`a002...` vs `8d2e...` in `checkpoints/lr_finder_10b_sophiag`) from a stale
  checkpoint on auto-retry. No complete CSV; fingerprint bug, not a measurement.

Do not use SophiaG for production at GBS=128 without a clean re-run.

### GBS scaling law (5B AdamW, Sirius H100, Nov 2026) -- measured 32/128/512

The GBS ladder tests how optimal LR varies with batch size at fixed model.
Per-GBS live output is isolated (`lr_finder.py:gbs{gbs}`,
`global_batch_size=GBS*4096`), so curves no longer collide. Sweep
`1e-08 -> 1e-01` 150 steps, `mult=1.114243`.

| GBS | global_batch_size | GAS | job | suggested (blow-up) | min loss | note |
|-----|-------------------|-----|-----|---------------------|----------|------|
| 32 | 131072 | 1 | 33628 / **33654** re-run | 3.49e-05 (3.49e-04) | 9.54 | clean, 12:00 |
| 128 | 524288 | 4 | 33429 (2026-11-12) | 2.48e-04 (2.48e-03) | 9.38 | baseline |
| 512 | 2097152 | 16 | 33629 / **33655** re-run 5.36e-04 | 5.36e-04 (5.36e-03) | 9.13 | clean, 12:00 |
| 2048 | 8388608 | 32 (LBS=2) | 33656/33630 | NCCL 300s, rc137 | -- | **data-path blocked, not model-limited** (see below) |

GBS=128 came from [2026-11-12 report](../2026-11-12-olmo3-gbs128-sirius.md) /
`data/2026-11-12-olmo3-gbs128-sirius/sirius-33429-5b-adamw-150.csv`.
Live curves: `outputs/lr_finder/ezpz/AgptModel/adamw/gbs*/lr_finder_data.csv`;
archive re-runs `33654/33655` confirmed the 32/512 numbers above.

**GBS2048 is not a model failure.** Both attempts (GAS=64 at 33630 and
corrected GAS=32/LBS=2 at 33656) NCCL-timed-out after 300s at step 3
(3 steps in ~11-16 min vs GBS512 16 steps in 74 min — see `2026-11-12-olmo3-gbs128-sirius.md`).
The abort stack is in Grain `prefetch.py` (`_put_iterator_elements_in_buffer`):
`json.gz` streaming + python tokenizer + Grain queue cannot feed 32
microbatches within the NCCL window on the wiki subset. Production jobs
avoid this by using pre-tokenized `BlendCorpus`/parquet. Treat GBS2048 as
not measured for the scaling law; the 32/128/512 trend is the valid basis
and already covers 16x batch. Walltime was 12:00 for 512 and 2048 (6h killed
early jobs).

10B arm: GBS32 was degenerate (1.98e-09, noise, job 33657) and is excluded
from any fit; 10B GBS512/2048 both NCCL-timed-out (33658/33659, same data-path
cause — 10b is ~4x slower per step than 5b at same GAS, ~180 tps vs ~790).
Treat the 10B scaling law as not yet measured.

![5B AdamW GBS scaling law — 32 / 128 / 512 clean, 2048 blocked](figures/sirius-5b-gbs-scaling-law.svg)

*Figure: 5B AdamW LR-finder curves at GBS 32 / 128 / 512 (Sirius H100, same 1e-8 -> 1e-1 sweep). Dashed lines mark blow-up/10 (suggested LR); dots mark minima. GBS 2048 both attempts NCCL-timed-out in the data loader (json.gz + tokenizer straggler), not a model limit — excluded from the trend. Style: `ambivalent` + Iosevka25 (Iosevka fallback). Source: `torchtitan/experiments/ezpz/utils/plot_style.apply_style()` via `.venv` `ambivalent.mplstyle`.*

## GBS=6144 evidence (2026-09-18 campaign, separate batch)

## AdamW fixed-LR validation

The coarse sweep `12478444` tested through `1e-1`. LRs `1e-3` and `1e-2`
remained finite for their individual sweep updates, but that does not prove
sustained fixed-LR stability.

The matched 100-update matrices use identical fresh initialization, seed 42,
GBS=6144, sequence length 4096, LBS=2/GAS=4, `dp_shard=4`,
`dp_replicate=192`, five warmup steps, then constant LR.

The first matrix completed 100 finite updates at every LR. Every arm wrote a
false `INVALID` marker because the wrapper searched stdout for an update-ratio
metric emitted only to W&B/TensorBoard. These are valid training trajectories
after post-hoc checks of scheduler exit, finite loss/gradients, and real
updates; the marker itself is not evidence of failure.

| LR | first job | step-100 loss | interpretation |
|---:|---:|---:|---|
| `1e-4` | `12478890` | `5.70206` | best endpoint in the first matrix |
| `3e-4` | `12478891` | `6.23678` | finite, worse than `1e-4` |
| `1e-3` | `12478892` | `6.89134` | finite, worse than `1e-4` |
| `3e-3` | `12478893` | `7.15027` | finite, worse than `1e-4` |
| `1e-2` | `12478894` | `14.96573` | finite but clearly unstable/poor |

The canonical rerun uses jobs `12478895`–`12478899`. Job `12478895`
reproduced the `1e-4` endpoint exactly: loss `5.70206` after 100 updates. It
also produced a fresh 768-shard checkpoint and real update-ratio evidence
(median `7.05e-5`, max `1.76e-3`). This is a deterministic confirmation, not
an independent seed replicate. The other rerun arms are monitored separately;
their live scheduler state belongs in the dashboard rather than this durable
page.

A winner requires 100 finite updates, finite gradients, real parameter-change
evidence, a fresh checkpoint, and comparison of the full matched trajectory.

### 2026-09-29 lower-control replacement chain

The copied isolated venv has a stale `bin/ezpz` shebang, so the replacement
wrapper invokes `ezpz.cli:main` through the verified interpreter instead of the
console script. Runtime/parser preflight `12479007` finished with exit 0.
Five-step `1e-4` canary `12479008` completed five finite updates and exited 0;
its inline `INVALID` marker repeats the known stdout-only predicate defect, while
W&B run [`qxkw7004`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/qxkw7004)
contains the optimizer diagnostics. The `3e-5` arm `12479009` completed 100 steps with PBS exit 0 under
W&B run [`wzpyq053`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/wzpyq053).
The `1e-5` arm `12479010` also completed 100 finite updates with PBS exit 0,
final loss `6.84922`, final gradient norm `0.5989`, and a fresh 768-shard DCP
checkpoint. Post-hoc validation found real parameter updates (median update
ratio `7.77e-6`, maximum `1.82e-4`) and wrote `VALIDATED_POSTHOC`.

These jobs close the missing lower-LR controls. Final selection still requires
comparison of the complete matched trajectories rather than endpoint loss alone.

## Artifacts

- [5B AdamW CSV](../data/2026-09-24-olmo2tok-gbs6144-verified/sunspot-12478508-5b-adamw-fine.csv)
- [5B SophiaG CSV](../data/2026-09-24-olmo2tok-gbs6144-verified/sunspot-12478624-5b-sophiag-fine.csv)
- [Evidence manifest](../data/2026-09-24-olmo2tok-gbs6144-verified/README.md)

The CSV schema does not encode gradient or optimizer-update health. Terminal
logs and fresh artifacts remain part of the acceptance contract.
