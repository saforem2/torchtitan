# Sunspot FLAT runs: timing complete, checkpoint weights differ

Follow-up: [2026-10-29 native-checkpoint comparison](2026-10-29-native-checkpoint-closure.md) verifies every starting parameter and the full 32-update token stream. The measurements below remain the original October 24 experiments; checkpoint names alone did not establish equal starting weights.

## Status

**Checkpoint weights differ across frameworks.** TorchTitan loaded a DCP converted from the 2026-02-09 HF export; MDS loaded its native 2025-12-22 archive. The `global_step138650` label did not make these identical weights. Timing and loss histories below are complete for the recorded configurations, but loss differences are descriptive and are not a numerical-parity result. The four TT arms control runtime and sharding changes against each other; the MDS comparison is not a same-weight comparison.

`12481360` completed the BF16 SDPA dispatch probe. Native GQA and explicit KV repetition selected fused Flash attention in bare mode and inside 16-rank FSDP at sequence length 512. Outputs matched exactly; the largest reported gradient absolute difference was `4.55e-13`.

`12481361` completed the 3-update Megatron-DeepSpeed FLAT canary with PBS and application exit status `0`. It loaded `global_step138650`, saved a checkpoint, ran validation, and synced its run to online `W&B`. The three iteration timings are canary samples, not a steady-state result. Job `12481364` completed all 32 updates with `WARMUP_ITERS=0`, one validation iteration, PBS exit status `0`, and a finished online `W&B` run. Its steps 21-32 averaged `24.284014 s` (sample stdev `0.017532 s`); loss moved from `17.542778` at step 1 to `10.302231` at step 32. The complete console and W&B histories agree on all 32 timings to within `0.000050 s`. The one `max allocated` memory reading was logged after iteration 1; it is an observed early high-water mark, not a whole-run peak.

`12481362` completed the strict-determinism sequence-512 probe. It selected the math path for native GQA and explicit KV repetition, both bare and inside 16-rank FSDP. `12481363` completed the non-deterministic, sequence-8192 FSDP probe. Native GQA and explicit KV repetition both selected Flash forward and backward. This result satisfies the launcher's sequence-8192 fused-attention proof gate for no-rematerialization arms.

`12481365` completed its 3-step profiled `matched-hsdp-no-remat-defer-retain` canary. The submitted shell revision matched the combined arm in the no-rematerialization branch before the deferred-reduction branch, so its actual `fsdp_cadence_args` were empty. It ran no-rematerialization with the default FSDP cadence; it did not measure deferred reduction or retained parameters. The three logged losses were `17.02153`, `17.02611`, and `16.68156`; all were finite. The first-step gradient norm was `1117.9272`. The online [`W&B` run](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/agpt-matched-hsdp-no-remat-defer-retain-12481365) synced with reported max active memory `38.7587 GiB` and max reserved memory `43.88867 GiB`. Its one-step profiler trace contains 288 Flash forward and 288 Flash backward ops, with no math SDPA op, across 2,208,517 events. The profiler override intentionally captured step 1; the trace flush took `27.68 s`, so this is not a timing sample. The process group was destroyed and `ezpz.launch` reported exit `0` after `190.19 s`; the local scheduler snapshot did not include a PBS `exit_status`. This short run is a canary, not a steady-state result.

Four 32-update matched-configuration arms were submitted after the canary: `12481366` is full-rematerialization HSDP, `12481367` is no-rematerialization, `12481368` is no-rematerialization with deferred gradient reduction and `reshard_after_forward=never`, and `12481370` is the same treatment with loss-only compilation. All four have PBS exit status `0` and finished online W&B runs. Complete timing and loss comparisons are recorded below. The profiler schedule is `profile_freq=7`, `warmup=3`, and `active=1`, with `skip_first=5`; this yields `wait=3`, skips steps 1-5, waits on steps 6-8, warms up on steps 9-11, then records step 12. The later measurement window is steps 21-32, after trace export overhead.

Job `12481369`, the compiled version of the combined arm, failed during CLI parsing before training. Its submitted value `--compile.components=[loss]` was parsed as an unknown component named `[loss]`. The corrected launcher passes `--compile.components=loss`; job `12481370` passed remote parser validation and started training. Compilation is limited to the loss component, not the full model. The completed compiled arm is included in the comparison below.

Job `12481371` was a 3-step legacy-RoPE compatibility canary using `agpt_2b_mds_anneal_flat_legacy_rope`. It failed on the second backward pass with invalid/OOM-sized allocations and GPU faults. Its finite first-step loss was `17.02509` with grad norm `1553.3768`; this is `0.51769` below MDS `12481364`'s `17.542778` and does not close the loss gap. The failure is a failed ablation, not evidence that RoPE caused the run-level loss difference. Do not extend it to 32 updates. The experiment-only compatibility path is quarantined pending a corrected numerical and memory gate.

## Dispatch probe `12481360`

The probe ran on `torch 2.14.0+xpu` with Intel Data Center GPU Max 1550, BF16 inputs, query shape `[1,16,512,128]`, and key/value shape `[1,4,512,128]`. The backend priority was `OVERRIDEABLE`, `CUDNN_ATTENTION`, `FLASH_ATTENTION`, then `MATH`.

| Mode | Native GQA | Explicit KV repeat |
|---|---|---|
| Bare, world size 1 | Flash forward/backward | Flash forward/backward |
| FSDP, world size 16 | Flash forward/backward | Flash forward/backward |

Native GQA versus repeated-KV outputs had maximum absolute error `0.0`; the q-gradient maximum absolute difference was `4.55e-13`, and k/v gradient differences were `0.0`. This directly disproves the blanket source comment that the requested fused backend is always ignored inside XPU FSDP for this shape and runtime. It does not establish dispatch at sequence length 8192 or inside the full training model. Artifacts: `/private/tmp/agpt-matched-20261024/bare-12481360.json` and `/private/tmp/agpt-matched-20261024/fsdp-12481360.json`.

The same probe in strict deterministic mode, job `12481362`, selected math for both call forms in bare and FSDP modes. Native GQA versus repeated-KV outputs had maximum absolute error `0.0`; q-gradient difference was `0.0`, with k/v gradient differences `1.19e-7` and `2.38e-7`. Strict deterministic mode changes dispatch for this sequence-512 probe. It does not prove numerical parity for full-model training. Artifacts: `/private/tmp/agpt-matched-20261024/bare-12481362.json` and `/private/tmp/agpt-matched-20261024/fsdp-12481362.json`.

Job `12481363` used non-deterministic mode, BF16, FSDP world size 16, and query/key/value sequence length 8192. Both native GQA and explicit KV repetition selected `aten::_scaled_dot_product_flash_attention` and its backward op. No math SDPA op appeared in either case. Native/repeated outputs differed by `0.0`; the q-gradient maximum absolute difference was `7.28e-12`, and k/v differences were `0.0`.

| Probe form | Mean forward + backward | Peak allocated |
|---|---:|---:|
| Native GQA | `0.02952 s` | `537,399,296 bytes` |
| Explicit KV repeat | `0.03265 s` | `688,395,264 bytes` |

These are isolated attention-probe measurements, not training step times. The proof artifact is `/private/tmp/agpt-matched-20261024/fsdp-12481363.json`; its remote path is `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan-agpt-matched-20261024/outputs/agpt-matched-attention-sunspot/probe-12481363/fsdp/results-rank0.json`.

## Megatron-DeepSpeed canary `12481361`

The submitted launcher used `ZE_FLAT_DEVICE_HIERARCHY=FLAT` after module setup. Its device preflight reported 12 visible XPU devices per node, each an Intel Data Center GPU Max 1550 with `65,520 MB` and `448` compute units. `ezpz` launched 16 ranks across two nodes, eight per node. This confirms the corrected rank layout for this canary; it does not establish a performance comparison against the previous composite-device run.

| Setting | Value |
|---|---|
| Initial checkpoint | `/lus/tegu/home/foremans/.cache/agpt-mds-baseline-20261023/load`, tag `global_step138650` |
| MDS source | `/lus/tegu/home/foremans/.cache/agpt-mds-baseline-20261023/source`, HEAD `b5617bac7b4e2303e738ef7db0307915da905941` |
| Data list and split | `torchtitan/experiments/ezpz/data-lists/sunspot/books.txt`, `95,5,0` |
| Model and sequence | 12 layers, hidden size 2048, 16 query heads, 4 KV heads, FFN 11008, sequence length 8192 |
| Batch | Global batch 384, microbatch 1, gradient accumulation 24 |
| Optimizer | AdamW, LR `2e-6`, constant schedule, warmup 1, seed 42 |
| Runtime | `torch 2.10.0a0+git449b176`, DeepSpeed `0.18.5`, oneAPI `2025.3.1` |
| W&B | Online; [`mds-canary-12481361`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/agpt-mds-canary-12481361) |

The complete local log reports all three training iterations, checkpoint save, and validation:

| Iteration | Loss | Logged iteration time |
|---:|---:|---:|
| 1 | 17.494349 | 26.8451 s |
| 2 | 17.598484 | 24.8924 s |
| 3 | 17.403709 | 24.9539 s |

Validation loss at iteration 3 was `17.237331`. The log confirms checkpoint loading and saving; the saved checkpoint directory is `/lus/tegu/home/foremans/.cache/agpt-zero1-hsdp-20261024/outputs/mds-canary-12481361/checkpoints`. Iteration 1 includes startup effects, and only two later samples exist. These numbers are a canary, not a steady-state baseline. No cross-framework loss or gradient agreement is established.

The corpus list, checkpoint tag, LR, seed, and global batch are recorded. An independent first-batch audit found all 384 indexed inputs and labels match. The direct checkpoint audit found that MDS used a native 2025-12-22 archive while TorchTitan used a DCP converted from the 2026-02-09 HF export. The TorchTitan DCP matches the HF export for the sampled tensors, but the native MDS checkpoint differs in Q/K/V/O projections, norms, and MLP tensors. Therefore the sampled bit-identical HF-to-DCP comparison did not establish MDS-to-TorchTitan weight parity. Earlier MDS used one warmup iteration; the follow-up baseline and matched TorchTitan arms explicitly use zero warmup to hold LR at `2e-6` from the first update. This does not retroactively make the canary's LR schedule constant.

## 32-update MDS baseline `12481364`

The complete log `/private/tmp/agpt-matched-20261024/mds-baseline-12481364.log` contains exactly iterations 1-32. The online [`W&B` history](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/agpt-mds-baseline-12481364) is finished and contains the same 32 training iterations. Direct log timings are printed to 0.1 ms; their steps 21-32 mean is `24.284008 s` with sample stdev `0.017543 s`. W&B's higher precision `training/iteration_time` values give mean `24.284014 s` and sample stdev `0.017532 s`. The maximum difference between W&B and the rounded console values is `0.000050 s`.

| Measure | Result |
|---|---:|
| Training-step coverage | 32/32, steps 1-32 |
| Mean iteration time, steps 21-32 | `24.284014 s` |
| Sample standard deviation, steps 21-32 | `0.017532 s` |
| Loss, step 1 | `17.542778` |
| Loss, step 32 | `10.302231` |
| Logged maximum allocated memory | `40,406.832 MB` (`39.460 GiB`) after iteration 1 only |

The memory figure is the available `report_memory` snapshot after iteration 1. It is not evidence of the run's maximum allocated memory over all 32 iterations. The W&B export is `/private/tmp/agpt-matched-20261024/mds-baseline-12481364.wandb.json`.

The completed MDS baseline's six-file source artifact was committed online as [`agpt-matched-flat-mds-12481364:v0`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/agpt-matched-flat-analysis-20261024). The analysis run attached that artifact to the original MDS run without resuming or rewriting its metrics.

An independent data audit compared the first global batch from the 32-step MDS and TorchTitan configurations. All 384 `(microbatch, dp_rank, dataset_index, input, label, raw)` rows matched by SHA-256. Both used the same Books list, `95,5,0` split, seed `42`, `GBS=384`, `microbatch=1`, `world_size=16`, `GAS=24`, sequence length `8192`, and zero consumed samples. Hash manifests are `/private/tmp/agpt-mds32-hashes.json` and `/private/tmp/agpt-tt32-hashes.json`. This establishes first-batch identity only, not later-batch identity or forward/loss/gradient agreement. The separate native-checkpoint audit below establishes that the starting weights differed.


## Completed 32-step timing, memory, and loss comparison

The complete online W&B histories cover steps 1-32 for all five runs. TT step time is the adjacent difference of cumulative `time_metrics/end_to_end(s)` for steps 21-32, using step 20 as the anchor. MDS uses the direct per-iteration seconds scalar. The MDS `elapsed_ms_per_iteration` legacy key contains seconds in this export; no unit conversion was applied.

| Arm / job | Mean step time, steps 21-32 | Sample stdev | Peak active / reserved memory |
|---|---:|---:|---:|
| TT full remat `12481366` | `37.053481 s` | `0.625439 s` | `28.622 / 34.195 GiB` |
| TT no remat `12481367` | `31.930570 s` | `0.178024 s` | `38.759 / 43.889 GiB` |
| TT defer/retain `12481368` | `28.552377 s` | `0.196053 s` | `47.959 / 53.820 GiB` |
| TT compile loss `12481370` | `25.255870 s` | `0.357036 s` | `32.334 / 38.195 GiB` |
| MDS ZeRO-1 `12481364` | `24.284014 s` | `0.017532 s` | `39.460 GiB allocated after iteration 1` |

The TT compiled arm's step-1 cumulative end-to-end time exceeded the eager defer/retain arm by `5.423494 s`. This is a startup-delta proxy, not an isolated measurement of compile cost. The MDS memory number is an iteration-1 snapshot, not a full-run peak. The plot therefore shows TT active/reserved memory only; it does not compare MDS and TT memory as equivalent peaks.

All runs have loss values for all 32 steps. Pairwise differences below compare each TT arm with MDS over those steps. They are descriptive; no tolerance was selected, so they do not establish numerical agreement.

| TT arm vs MDS | Shared steps | Maximum absolute loss difference | RMS loss difference |
|---|---:|---:|---:|
| TT full remat | `32/32` | `0.476486` | `0.191426` |
| TT no remat | `32/32` | `0.476486` | `0.212050` |
| TT defer/retain | `32/32` | `0.476486` | `0.271102` |
| TT compile loss | `32/32` | `0.476484` | `0.244636` |

The TT runs share a start checkpoint, but that checkpoint differs from native MDS weights. Do not treat the 32-step loss trajectories as parity evidence. Job `12481371` failed on its second backward pass and its first-step loss remained `0.51769` below the MDS baseline; it did not resolve the mismatch.

The plot and machine-readable summary are [`2026-10-24-matched-flat.png`](2026-10-24-matched-flat.png) and [`2026-10-24-matched-flat.json`](2026-10-24-matched-flat.json). The JSON records full per-step loss and timing coverage, all pairwise max-absolute/RMS loss differences, and the startup-delta definition.

## Profiled communication and attention

The table below summarizes the selected step-12 trace from each 32-step arm and the MDS baseline. `XPU kernel union` is the union of rank-0 device-kernel intervals. `Exposed collective` is the analyzer's rank-local estimate: the union of XCCL call annotations and oneCCL kernels minus intervals overlapping other XPU kernels. It is not a causal stall measurement. These are profiled steps and must not be substituted for the unprofiled 21-32 step-time window. CPU nested timing breakdowns are intentionally omitted.

| Arm / job | Profiled-step duration | XPU kernel union | Exposed collective | XCCL calls: all-gather / reduce-scatter / all-reduce | Attention forward / backward |
|---|---:|---:|---:|---:|---|
| MDS `12481364` | `24.498 s` | `23.822 s` | `0.233 s` | `3 / 0 / 105` (plus 2 barriers) | efficient SDPA `288 / 288` |
| Full rematerialization `12481366` | `55.079 s` | `33.350 s` | `5.429 s` | `648 / 336 / 20` | Flash `576 / 288` |
| No rematerialization `12481367` | `33.236 s` | `30.585 s` | `5.949 s` | `648 / 336 / 20` | Flash `288 / 288` |
| Deferred reduction + retain `12481368` | `28.409 s` | `26.888 s` | `1.696 s` | `15 / 14 / 20` | Flash `288 / 288` |
| Loss-compiled deferred + retain `12481370` | `25.181 s` | `24.590 s` | `0.643 s` | `15 / 14 / 20` | Flash `288 / 288` |

The `12481367` to `12481368` change reduced rank-0 all-gather annotations from 648 to 15 and reduce-scatter annotations from 336 to 14. The analyzer's exposed collective estimate fell from `5.949 s` to `1.696 s`; the number of all-reduce annotations remained 20. This is consistent with the deferred-reduction and retain-parameters settings changing communication cadence. The measurement does not establish the full-job causal speedup by itself.

The compiled `12481370` trace retained the same 15 all-gather and 14 reduce-scatter annotations and showed `288` Flash forward and `288` Flash backward ops. Its W&B `memory/max_reserved(GiB)` maximum was `38.19531`, compared with `53.82031` for `12481368`. Full-run timing and W&B memory summaries are below. Do not treat the profiled-step `25.181 s` as its steady-state timing result.

The MDS trace uses the efficient-attention path with 288 forward and 288 backward ops. The no-rematerialization TorchTitan traces each show 288 Flash forward and 288 backward ops; full rematerialization shows 576 Flash forward ops for 288 backward ops. This confirms the selected backend within the traces and the additional forward calls in the rematerialized trace. It does not establish whole-run loss agreement.

## Legacy MDS RoPE compatibility

The direct native-checkpoint audit found differences in Q/K/V/O projections, norms, and MLP tensors between the MDS 2025-12-22 archive and the HF export used to create the TT DCP. The sampled TT DCP tensors matched the HF export, not native MDS. The audited norm difference reached `0.0391`; the gate and down projections differed by about `0.0234` and `0.0117`. This invalidates the earlier inference that the loss gap isolated RoPE. Separately, a CPU reference audit found the experiment-only BF16 split-half RoPE implementation bit-identical to sampled MDS Q/K rotations. The MDS path casts inverse frequencies and positions to BF16, then computes BF16 cosine/sine and BF16 rotation multiply/adds. Comparing sampled MDS Q/K rotations with the current TorchTitan FP32 complex-RoPE path produced max absolute difference `8.25` and p95 `1.50`; the FP32 phase calculation itself differed from the ideal phase reference by about `7e-6`. These are implementation-level differences, not an explanation for the run-level loss gap because the model weights differed. Job `12481371` failed on its second backward pass with invalid/OOM-sized allocations and GPU faults; its first-step loss `17.02509` did not close the `0.51769` gap from MDS `12481364`. The legacy path remains a failed, quarantined ablation.

## Layout and attention interpretation

The earlier MDS baseline `12481353` used six ranks per node after its module setup reset `ZE_FLAT_DEVICE_HIERARCHY` to `COMPOSITE`. The FLAT canary `12481361` reasserted `FLAT` after module loading and used eight ranks per node. The prior `63 s` MDS versus roughly `98 s` TT profiled-step gap combined attention dispatch, rematerialization, communication cadence, and device hierarchy. The new probe and canary do not isolate their causes. In particular, the sequence-512 fused result does not show that full-model SDPA at sequence length 8192 uses Flash attention. Do not attribute the earlier `63 -> 25 s` difference to a single fix.

The strict deterministic sequence-512 probe identifies a dispatch switch: strict mode used math; non-deterministic mode used Flash. The sequence-8192 probe confirms the non-deterministic dispatch for both SDPA forms inside 16-rank FSDP. The matched TorchTitan launch arms disable strict deterministic mode while retaining seed `42`; the original eager/compiled benchmark arms remain strict. This aligns the candidate dispatch setting with the non-deterministic MDS runtime for the matched performance experiment. MDS still runs `torch 2.10`, while TorchTitan runs `torch 2.14`; this is not a common-runtime comparison. It does not establish training-loss agreement. Jobs `12481368` and `12481370` ran the no-rematerialization, deferred-reduction, retain-parameters combination; their traces confirm fused attention and the lower all-gather/reduce-scatter call counts. Job `12481365` used an earlier shell revision and had default cadence. The traces do not by themselves establish unprofiled timing gains or causal attribution.

PyTorch's XPU SDPA dispatch utility checks `ctx.deterministicAlgorithms` before selecting the fused implementation and returns false when strict deterministic algorithms are enabled. See [`sdp_utils.cpp` at the probed source revision](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/transformers/xpu/sdp_utils.cpp). The traces independently show math with strict mode and Flash without it.

The launcher's `matched-hsdp` arm is the full-rematerialization, non-compiled HSDP control. `matched-hsdp-no-remat-defer-retain-compile` adds loss-region compilation to the no-rematerialization, deferred-reduction, retain-parameters treatment. The launcher sets `compile_components=loss` for compiled arms and logs it; this does not compile the full model. Job `12481369` did not train because of the parse error above; corrected job `12481370` passed parsing and produced the selected step-12 trace. Its steady-state timing and complete W&B memory history are included in the completed comparison above.

## Provenance and limits

The TorchTitan checkout path used for the probes was `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan-agpt-matched-20261024`, HEAD `5be47f7cfa12cb47d4a774480fe9251fe1e53ff8`. The local hashes below identify the current candidate files. The corrected launcher hash is post-submission and does not identify the bytes submitted for job `12481369`; submitted launcher bytes were not captured in these local records:

| Launcher / probe | Local SHA-256 |
|---|---|
| `benchmark_agpt_compile_sharding_sunspot.sh` | `79d58647f89d582a8062aef2eb07ebaaf2e98a03a047ceef5c5820cd877373d3` current local candidate; includes config/data-iteration overrides and job-unique source/command provenance added after the submitted 32-step jobs |
| `benchmark_mds_agpt_sunspot.sh` | `4e33411b94dd1a0a42d7996ce3a5664a2b231be8af5b625d2baea5e9b69bf456` |
| `probe_agpt_gqa_sdpa_sunspot.sh` | `896734f5e6a83ed4331c8a65a5f8675a47fa209d7343acda5b305ac69cc1fd9e` |
| `probe_agpt_gqa_sdpa.py` | `062968c930e88be49a7eba049422b65140eee92a2f1e9a44df8081fa52c37de7` |

The MDS job log is `/private/tmp/agpt-matched-20261024/mds-canary-12481361.log`, SHA-256 `407ce90014b3121e012968e2f7260bd6924fe547382ec0df8e731eab8878875b`. The dispatch artifacts are `/private/tmp/agpt-matched-20261024/bare-12481360.json` and `/private/tmp/agpt-matched-20261024/fsdp-12481360.json`. The submitted MDS job record names `benchmark_mds_agpt_sunspot.sh`; this host could not read the remote SSH ControlPath, so the recorded local launcher hashes are not independently verified against remote submitted bytes.

The current evidence supports a successful 16-rank FLAT MDS baseline, fused SDPA dispatch at sequence lengths 512 and 8192 when strict deterministic mode is disabled, and identical first global batches. It also establishes that the TT DCP came from a different HF export than the native MDS archive. The 8192 dispatch result is from an isolated FSDP probe. Same-weight loss/gradient agreement, optimizer parity, and causal performance attribution remain unmeasured. The 32-step timing and loss windows are complete, but their loss differences are not parity evidence because the starting weights differ. The step-12 trace audits are `/private/tmp/agpt-matched-20261024/mds-12481364-profile-audit.json` and `/private/tmp/agpt-matched-20261024/TT-12481366-profile-audit.json` through `TT-12481370-profile-audit.json`.
