# Aurora production checkpoint migration matrix

**Status:** audit complete; candidate-SHA hardware validation pending.

The queued production umbrella `8879474` is intentionally frozen. It runs
checkout `816f46e1cc14e5c70eb83d0c4485a6de82578c37` and two checksum-pinned
Python archives containing Torch `2.13.0.dev20260428+xpu` and ezpz `0.29.2`.
It does not consume the current `ezpz` branch or the new `ConfigLoader` work.
Passing that job would advance the old production stack; it would not prove
that current software can resume these checkpoints.

## Active seats

| Seat | Production lineage | Frozen source | Latest complete checkpoint | Files | Metadata bytes |
|---|---|---|---|---:|---:|
| 1 | 20B-512 constant-LR fork | `f1c20ee76c` | `step-11100` | 6,145 | 1,252,226,679 |
| 2 | 20B-256 base | `086ff99618` | `step-17500` | 3,073 | 698,545,255 |
| 4 | 2B-256 stage-2 Dolmino | `f319e3fad2` | `step-41300` | 3,073 | 108,980,210 |

Incomplete next-step directories (`step-11200` and `step-17600`) contain no
files and are not restore candidates.

## Executed production contract

All three seats invoke `python -m torchtitan.experiments.ezpz.train` with
legacy dotted CLI overrides. Upstream `ConfigLoader` accepts only `--module`,
`--config`, repeatable `--override`, `--comm-backend`, `--output-dir`,
`--resume-step`, and `--print-config`; without the ezpz-local compatibility
loader, the active production invocation is rejected before initialization.

Shared settings:

- `checkpoint.enable=true`
- `checkpoint.interval=100`
- `checkpoint.keep_latest_k=0`
- `checkpoint.last_save_model_only=false`
- `checkpoint.async_mode=disabled`
- `dataloader.dataset=blendcorpus`
- `optimizer=sophiag`
- `lr_scheduler.decay_ratio=0.0`
- `lr_scheduler.min_lr_factor=1.0`
- `lr_scheduler.warmup_steps=20`
- `training.local_batch_size=2`
- `training.seq_len=8192`
- `validator.enable=true`, `freq=100`, `steps=10`

| Seat | Config | GBS | Steps | LR | Data | Restore mode |
|---|---|---:|---:|---:|---|---|
| 1 | `agpt_20b_real` | 12,288 | 46,429 | `2.28e-5` | `olmo-mix-1124` | resume latest from own folder |
| 2 | `agpt_20b_real` | 6,144 | 92,859 | `2.28e-5` | `olmo-mix-1124` | resume latest from own folder |
| 4 | `agpt_2b` | 6,144 | 47,492 | `2.17e-5` | `dolmino-mix-1124` | own folder, originally seeded weights-only from base `step-92859` |

Production passes explicit checkpoint folder, dataset path and cache, validator
dataset/cache, optimizer/LR, local/global batch sizes, sequence length, and
training steps. A compatibility loader must preserve each value exactly. It
must also preserve empty-initial-load semantics: a nonempty resumable
`checkpoint.folder` wins over `checkpoint.initial_load_path`.

## Required migration gate

Do not replace or release the production umbrella from a current checkout until
all gates pass at one immutable candidate SHA and one immutable Torch 2.15
runtime archive.

1. **Parser/config parity.** Feed each seat's actual legacy argv into the
   ezpz-local compatibility loader. Serialize the resolved config and compare
   every field listed above with this manifest. Unknown dotted paths, changed
   booleans, changed config flavor, or changed restore precedence are failures.
2. **Negative control.** Run the same argv with the compatibility layer disabled
   and require rejection on a representative dotted flag. This proves the gate
   is exercising the migration rather than an unrelated parser.
3. **Representative canary.** Run seat 4 first because it is smaller and tests
   the complex-RoPE stage-2 lineage. Restore `step-41300`, complete three finite
   optimizer updates, and write a fresh full-state checkpoint into a job-unique
   scratch directory.
4. **20B matrix.** Independently run seats 1 and 2 at the smallest
   dimension-compatible topology that fits. Restore their exact current heads,
   complete three finite updates with finite gradients, and write fresh
   full-state checkpoints. Do not serialize independent arms unnecessarily.
5. **Fresh-save resume.** Resume one additional step from each newly written
   checkpoint. This proves current-format save/load, not only historical input
   migration.
6. **Umbrella canary.** Only after all lineage arms pass, run the current
   umbrella launcher with all three seats, job-unique sinks, three finite
   updates, and no canonical writes. Release a new production umbrella only
   after this final concurrency gate.

Scheduler exit zero, a `VALIDATED` text marker, or a successful model-only load
is insufficient. Each arm requires a verified full-state restore, real
optimizer updates, nonempty current-format checkpoint metadata/shards, and a
fresh-save resume.
