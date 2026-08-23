# agpt (Dense AuroraGPT) Benchmarks

Dense transformer training benchmarks across ALCF machines.

## Reports

### Aurora

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-03-30 | [80B results](../../../reference/scaling/agpt-80b.md) | 80B, 80B_alt, 80B_wide, 80B_deep | 2 | TP=2 best at 15.5% MFU |
| 2026-04-12 | [Smoke test (n2)](aurora/20260412-035148-smoke-n2.md) | debugmodel, 2b | 2 | 2b at 20% MFU, 59.6 TFLOPS |
| 2026-04-12 | [LR Finder (n2)](../lr-finder/agpt/2b/README.md#2026-04-12----2b-aurora-first-run) | 2b, 20b x {AdamW,Muon,SophiaG} | 2 | AdamW most tolerant; SophiaG needs 10x lower LR |
| 2026-04-12 | [80B Throughput (n2)](aurora/20260412-193100-throughput-80b-n2.md) | 80B_alt TP={3,6} x compile | 2 | TP=6+compile best: 45 TPS, 8.23% MFU |
| 2026-04-13 | [80B Leaderboard](aurora/80b-throughput-leaderboard.md) | 80B_wide/alt/deep x TP={2-12} | 2 | 80B_wide TP=4 compile: 68 TPS, 11.79% MFU |
| 2026-04-13 | [20B Throughput (n2)](aurora/20260413-143800-throughput-20b-n2.md) | 20B TP={1,2,4} x compile | 2 | TP=1 compile: 357 TPS, 17.82% MFU |
| 2026-04-14 | [20B Production (n512)](aurora/20260414-production-20b-n512.md) | 20B SophiaG LR=2.28e-5 | 512 | Verify: loss 12.92->10.50 in 146 steps |
| 2026-04-18 | [80B TP=2 Restored](aurora/20260418-80b-tp2-restored.md) | 80B TP=2 compile | 2 | 88 TPS, 16% MFU — regression fixed |
| 2026-04-18 | [Scaling & Production](../../../production/scaling-performance.md) | 2B, 20B, 80B | 4-512 | 80B scales perfectly to 128N; compile wall at 512N |
| 2026-04-25 | [Scaling Study (torch 2.13)](../../../reference/scaling/agpt-20b.md) | 20B | 2-4096 | 440 TPS @ 2N (+23% vs torch 2.10); in progress |

### Polaris

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-04-12 | [Smoke test (n2)](polaris/20260412-160749-smoke-n2.md) | debugmodel, 2b, 7b, 8b, 20b, 50b, 80b | 2 | 7b 8.8% MFU; 50b/80b OOM; 8b vocab mismatch |

### Sunspot

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-03-30 | [80B results](../../../reference/scaling/agpt-80b.md) | 80B variants x TP={2,3,6,12} | 2 | TP=2 best: 85 TPS, 15.5% MFU |
| 2026-04-12 | [LR Finder (n2)](../lr-finder/agpt/2b/README.md#2026-04-14----2b-sunspot-dim-aware-init) | 2B, 20B x {AdamW,Muon,SophiaG} | 2 | All 6 sweeps; 20B SophiaG suggested LR=1.5e-5 |
| 2026-04-12 | [Scaling Study](../../scaling/) | 2b, 20b, 80b at 1-64N | 1-64 | 20b 87% efficiency at 64N; 80b OK at 4-32N |
| 2026-04-13 | [Benchmark (n2)](sunspot/20260413-benchmark-n2.md) | All 11 agpt configs | 2 | 80b_deep best 80B variant: 83 TPS, 15.2% MFU |
| 2026-04-15 | [Full Benchmark (n2)](sunspot/20260415-benchmark-n2.md) | All 18 configs (agpt+MoE) | 2 | 80B compile regression found; fix in e8cbb8ef |
| 2026-04-18 | [Torch 2.12 Benchmark (n2)](sunspot/20260418-torch212-benchmark-n2.md) | 8 configs (agpt+MoE) | 2 | 2b +11% TPS, -49% mem; 20b +29% TPS; 80b AC regression |
| 2026-04-21 | [LR Finder 80B + GAS (n2)](../lr-finder/agpt/80b/README.md#2026-04-21----80b-gas-sweep-sunspot-small-batch-gbs192) | 80B x 3 opts, 2B/20B GAS sweep | 2 | 80B AdamW LR=1.1e-5; Muon/SophiaG broken at 80B |
| 2026-04-27 | [10B Optimizer Sweep (n8)](../../competitions/agpt2b-n8-10BT/) | 2B x {AdamW, AdamW+QKNorm, Mano, Mano+QKNorm} | 8 | AdamW wins at GBS=384; loss 2.711 |
| 2026-05-20 | [Post-resync smoke (n2)](sunspot/20260520-smoke-n2-postresync.md) | debugmodel, 2b (LBS=1, LBS=2) | 2 | PR #3159 replay verified: 2b LBS=2 matches Apr 25 baseline (7,224 TPS / 27.1% MFU) |
| 2026-05-20 | [PR #3386 merge follow-up smoke (n2)](sunspot/20260520-smoke-n2-pr3386-merge-followup.md) | 2b, 50b_wide | 2 | agpt_2b byte-identical to baseline (24.34 GiB / 38.04%); agpt_50b_wide re-confirms torch-2.13 `DeviceMesh`-in-saved-tensors crash (~121s to repro) |
| 2026-06-02 | [80B TP=2 + xccl workaround smoke (n4)](sunspot/20260602-smoke-n4-80b-tp2-xccl-workaround.md) | 80B (TP=2, AdamW LR=1e-6, compile=OFF) | 4 | Validates xccl_split_group workaround (`8031d1d3a`) does NOT regress 80B TP=2. 20 steps clean: loss 12.94→10.39 (Δ-2.55), ~17.8% MFU, 88.94% mem — numerically equivalent to May 5 Aurora baseline (Δ-2.52, 17.8% MFU, 88.94% mem). 5 nested PGs built cleanly under the workaround. |
| 2026-06-08 | [SFT 2B + metamathqa (n32)](sunspot/20260608-sft-2b-sophiag-metamathqa-n32.md) | AuroraGPT-2B-sophiag-138650 + MetaMathQA, FSDP1 + TRL SFTTrainer | 32 | First 32N production SFT pass — wired up TRL `SFTTrainer` + chat-template fallback + `assistant_only_loss`. Verified the pipeline end-to-end; consumed by the 06-10 tulu-mix scaleup. |
| 2026-06-10 | [SFT 2B + tulu_math_uc_mix, autoretry failover (n32)](../../../production/sft/agpt/2b-mds/tulu_math_uc_mix/README.md) | AuroraGPT-2B-sophiag-138650 + `tulu-3-sft-mixture:0.65 + metamathqa:0.15 + ultrachat-200k:0.20`, GBS=6144, FSDP1 | 32 (+ 4 spare) | **`Training complete.` at step 729 / epoch 3.0** after 4-job chain surviving 3 oneCCL SIGABRTs. Loss 1.16 → 0.77, `mean_token_accuracy` 0.7957, ~4.5B tokens. Final HF-format ckpt at `outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf/`. Promoted to a production-tracked recipe at [`docs/production/sft/agpt/2b-mds/tulu_math_uc_mix/`](../../../production/sft/agpt/2b-mds/tulu_math_uc_mix/README.md). Cleared two upstream blockers inline: [pytorch/pytorch#186938](https://github.com/pytorch/pytorch/issues/186938) (XPU FSDP resume) + [`6b4a00b`](https://github.com/saforem2/ezpz/commit/6b4a00b) (autoretry STUCK_PRE_TRAINING false-positive on TRL log format). |
| 2026-06-24 | [Native auto-retry 2B smoke (n2)](sunspot/2026-06-24-native-autoretry-2b-smoke.md) | debugmodel, 2b (LBS=2), native `ezpz launch --auto-retry` | 2 (+2 spare) | Validates the ezpz-native auto-retry replacement for `failover_lib.sh` (job 12469523): yeet, active/spare arithmetic, blind spare rotation, and the `STUCK_PRE_TRAINING` circuit-breaker all fire correctly. |
| 2026-06-25 | [80B GBS=1488 512N-batch sim (n62)](sunspot/2026-06-25-80b-gbs1488-512N-sim.md) | 80B TP=4/LBS=1, GAS=8 -> GBS=1488, dp=186, AdamW | 62 | 512N-batch simulation clean: 46/46 steps, zero NaN, loss 12.92 -> 8.84. The stable corner (dp_degree <= 186) holds at 4x the validated global batch. |
| 2026-06-25 | [80B TP=4 100-step validation (n62)](sunspot/2026-06-25-80b-tp4-100step-validation.md) | 80B TP=4/LBS=1, GAS=2 -> GBS=372, dp=186, AdamW | 62 | First non-smoke run of the TP=4/bf16 stable corner to 100 steps: 100/100 clean, zero NaN, loss 12.93 -> 7.72, step-100 ckpt saved. |
| 2026-06-26 | [80B GBS=2976 1024N-batch sim (n62)](sunspot/2026-06-26-80b-gbs2976-1024N-sim.md) | 80B TP=4/LBS=1, GAS=16 -> GBS=2976, dp=186 | 62 | 1024N-batch simulation clean: 34/34 steps, zero NaN, loss 12.92 -> 9.84. The corner holds at 8x the validated global batch. |
| 2026-06-26 | [80B GBS=5952 2048N-batch sim (n62)](sunspot/2026-06-26-80b-gbs5952-2048N-sim.md) | 80B TP=4/LBS=1, GAS=32 -> GBS=5952, dp=186 | 62 | First batch-scaling rung to break: 28 clean steps then grad_norm NaN at step 29 (loss still finite -- the grad-path-first signature). 16x GBS exceeds the corner. |
| 2026-06-26 | [80B LR / batch / dp-degree findings (n62-88)](sunspot/2026-06-26-80b-lr-batch-dpdegree-findings.md) | 80B TP=4/LBS=1, LR-finder + LR-scaling + dp-degree bisect | 62-88 | Five connected jobs: usable 80B LR is low (~1e-6); do NOT linearly scale LR with batch (16x-scaled LR NaN'd at step 7 vs step 29 flat); the "dp<=186 cliff" is not real (dp=192 and dp=264 both ran 30 steps clean). |
| 2026-06-27 | [80B LR-finder at production batch (n64)](sunspot/2026-06-27-80b-lr-finder-production-batch.md) | 80B TP=4/LBS=1, GAS=32 -> GBS=6144, LR-finder x {AdamW, mano, muon, sophiag} | 64 | At the real GBS=6144: AdamW usable-LR ceiling is ~7e-7 (default 1e-6 sits on the NaN cliff), explaining the nondeterministic NaNs; mano is far better-behaved (no cliff, loss min at lr=1.6e-5). |
| 2026-06-30 | [80B convergence at GBS=6144 (n64)](sunspot/2026-06-30-80b-convergence-gbs6144.md) | 80B TP=4/LBS=1, GAS=32 -> GBS=6144, constant LR x {mano, sophiag, AdamW} | 64 | Negative result that overturns the finder: at constant finder-recommended LR all three optimizers diverge to NaN within 5-12 steps. The 15-step finder's ramping LR masked the instability. |
| 2026-07-06 | [SFT prep on completed v2 256N base (step-92,859)](sunspot/2026-07-06-sft-2b-v2-256n-base-prep.md) | SFT AuroraGPT-2B v2 256N base + tulu_math_uc_mix; DCP->HF convert + transfer + smoke | 1 (smoke) | First SFT on the COMPLETED v2 256N base (4.674T tokens): Aurora DCP -> HF conversion, Aurora->Mac->Sunspot transfer, and a 10-step smoke clean (loss 2.08 -> ~1.7, no silent Qwen fallback, no oneCCL barrier). |
| 2026-07-10 | [SFT gs138650 + FULL tulu_math_uc_mix (n32)](sunspot/2026-07-10-sft-2b-gs138650-big-mix-32n.md) | SFT gs138650 base + full tulu_math_uc_mix (~93M rows, ~54B tok), GBS=6144 | 32 | Full-mix SFT push (job 12470281) blocked by a base-independent 384-rank oneCCL scale GPU-fault; identical data+config trains clean at 24 ranks. Vectorized interleave (4.8s vs ~90min) + on-disk cache validated en route. |
| 2026-07-28 | [2B MDS anneal-schedule A/B + data-mix experiment](sunspot/20260728-2b-mds-anneal-and-datamix.md) | MDS/olmo fork, 10B tok/arm: flat-vs-WSD schedule x2 bases; owm/edu/blend data mixes; held-out FineMath+wikitext val-loss | 32 | LR SCHEDULE is NOT the lever (flat beats WSD-decay-to-0 on both bases); DATA MIX is: edu-100 forgets math +0.308, 75%/25% owm/edu is the sweet spot (math intact, ~all of edu's general gain). Recipe: continue math-saturated base on ~75/25 math/edu at constant LR. |
