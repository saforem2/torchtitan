# Known bugs

> **Canonical record:** this page is authoritative for the catalogue of known
> failure modes. Other pages cross-reference it; do not duplicate status here.

One page per failure mode. Each records the symptom, the root cause where it
is known, and the workaround in force. Check here before diagnosing a failure
that looks familiar -- several of these took multiple sessions to isolate and
read as something else entirely at first.

| page | what it covers |
|---|---|
| [agpt on full_dtensor: vc_check/DeviceMesh, and a pin that was justified uncompiled](agpt-full-dtensor-vc-check.md) | 1589 steps and a compiled resume, see the answer to open question 1 below. |
| [Aurora: 2098-node job killed at 9h13m of 24h with `Exit_status = -14`](aurora-job-8756070-exit-14.md) | The three live trainers were **mid-stride**, logging clean steps 2m39s before |
| [Blendcorpus EOFError Race in `_build_index_mappings`](blendcorpus-eoferror-race.md) | author: Sam Foreman |
| [`BlendCorpusDataLoader` aliases torchtitan parallelism axes onto Megatron knobs](blendcorpus-megatron-aliasing.md) | constructs a `bc_cfg` for the Megatron-derived `blendcorpus` library by |
| [Concurrent-job checkpoint collision on `20b_v2_256`](concurrent-job-ckpt-collision.md) | Investigated 2026-08-16. Read-only audit of the AuroraGPT production |
| [`'Config' object has no attribute 'job'` (ezpz branch, 2026-08-13)](config-job-dump-folder-attributeerror.md) | Every run on the `ezpz` branch dies at the top of `train()`, on all ranks, |
| [frameworks-RC torch: `torch.compile` at TP=4 crashes in the SDPA flash-backward](fw-rc-compile-sdpa-backward-tp4.md) | (`2.13.0a0+gitcf30153`, oneAPI 2026.1.0, XPU) any `torch.compile` agpt run at |
| [hybridep on XPU: not a version floor, not portable](hybridep-is-nvidia-only.md) | ImportError: cannot import name 'CustomClassBase' from |
| [`--debug.deterministic` costs ~27% device memory on the MoE path (2026-08-19)](moe-deterministic-memory.md) | Two different failure modes under `det=1`: |
| [MoE under EP aborts in `all_to_all_single`, worse with model size (2026-08-19)](moe-ep-a2a-degrades-with-size.md) | The two a2a arms abort with a specific runtime message, visible just above |
| [Flex attention on MoE: two stacked bugs, both fixed](moe-flex-attention-blockmask.md) | Every flex-attention MoE config died immediately: |
| [Polaris 20B "eval gibberish" root cause: training-data / model tokenizer mismatch](polaris-20b-tokenizer-mismatch.md) | Every Polaris 20B checkpoint scored at chance on every lm-eval task, and a |
| [Pre-#3623 checkpoints can't resume on current code: optimizer state-dict format migration](pre3623-optim-statedict-resume.md) | optimizer state-dict API was swapped: |
| [RoPE flavor mismatch: a mid-flight convention switch, and the exports it broke](rope-flavor-mismatch.md) | sound.** MEASURED: at each switch the training loss spiked (2.57 -> 6.03 on |
| [Trying a newer XPU torch against `spmd_types` (2026-08-20)](spmd-types-newer-torch-attempt.md) | config, same command -- only torch differs: |
| [Why `spmd_types` leaves parameters unconverted](spmd-types-plain-tensor.md) | predates the FSDP code that consumes spmd_types annotations. Nothing to fix in |
| [RETRACTED -- this was NOT a cluster fault](sunspot-ccl-allgatherv-outage-20260810.md) | Hello -- following up on the `x1921c4s3b0n0` mount issue (thank you for the |
| [Sunspot: bare `reduce_scatter_tensor` SIGSEGVs at 48 ranks (2026-08-14)](sunspot-reduce-scatter-segv-20260814.md) | 2026-08-03** (job `12472452`) -- now SIGSEGVs on every rank inside step 1: |
| [sunspot-x1921c4s3b0n0-bad-mount](sunspot-x1921c4s3b0n0-bad-mount.md) | Hello, |
| [Umbrella `std::bad_alloc` at init -- intermittent, not yet root-caused](umbrella-bad-alloc-init.md) | plausible mechanism (concurrent-init contention) is identified but NOT proven -- |
| [Unregistered W&B runs: the failure that never announces itself](unregistered-wandb-runs.md) | underlying drift is structural and will recur. |
| [The "validator CCL deadlock at 80B TP=4" was a phantom -- two unrelated bugs](validator-tp4-at-80b.md) | validator collective deadlock at 80B TP=4. The label conflated two *separate*, |
| [`--debug.deterministic` is not bit-reproducible on XPU (2026-08-16)](xpu-determinism-rank-seqlen-interaction.md) | agpt_20b, seq 2048, 10 steps, `--debug.seed=42 --debug.deterministic`, |
| [XPU graphs cannot capture oneCCL collectives (2026-08-16)](xpu-graphs-block-oneccl-collectives.md) | <details> |
