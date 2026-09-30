# INCITE Quarterly Report — Q3 2026 (Jul 1 – Sep 30)

> **Project:** AuroraGPT — A Large-Scale Foundation Model for Advancing Science
> (INCITE-2026, Allocation Year 2)
> **Author:** Sam Foreman
> **Period covered:** 2026-07-01 through 2026-09-30
> **Scope:** TorchTitan + `ezpz` training work on Aurora (primary), with Sunspot,
> Polaris, and Perlmutter used for development and portability validation.

Sources: Q3 retrospectives ([index](README.md)), the live
[production](../production/README.md) and [evaluation](../evals/README.md)
trackers, and the September 30 [development journal](../journal.md).

---

## 1. Executive Summary

The TorchTitan + `ezpz` stack now covers dense pre-training, CPT, SFT, GRPO,
and MoE execution on Intel XPU.

1. **Both 2B base-pretraining chains completed.** The 256-node chain
   finished at step 92,859 and 4.674T tokens; the 512-node chain finished at
   step 46,429 and 4.674T tokens. The latter completed on August 13.
   ([256N](2026-07-06.md#1-agpt-2b-256n-base-pre-training-complete-headline),
   [512N](2026-08-14.md#1-2b-512-canonical-chain-complete))
2. **The 20B chains reached steps 11,100 and 17,500.** Accepted endpoint
   results now cover step 11,100 for the 512-node chain and
   step 17,500 for the 256-node chain, each with six zero-shot tasks plus
   ARC-Challenge 25-shot. ([journal](../journal.md#2026-09-30-aurora----production-reporting-audit-and-tail-evaluation))
3. **The 2B post-training path runs end to end on XPU.** Two-stage SFT reached
   0.205 GSM8K-CoT accuracy with 0.985 format compliance. A gated 100-step GRPO
   continuation preserved format but moved accuracy only from 0.205 to 0.215;
   a separate 1,000-step arithmetic GRPO run solved its narrower task.
   ([post-training status](../production/POST-TRAINING-2B.md))
4. **MoE correctness is established on Intel XPU.** `bmm_nodrop` was bit-exact
   against the reference, `aurora_sycl` agreed within `6.104e-05`, and the full
   Sonic expert path trained within approximately `5e-05` of the EP=2
   reference. A production-shaped 12.29B-total/1.91B-active configuration
   completed finite EP=12 updates. Production-scale throughput remains open.
   ([four-machine report](../experiments/moe-expert-backends-4machine.md))
5. **The new dense-model ladder is 4.64B / 9.48B / 26.20B.** Models
   with 4.64B, 9.48B, and 26.20B parameters were built around the OLMo-3
   tokenizer. In the completed fixed-batch 26.2B comparison, Mano ended 0.075
   nats below AdamW at 23.59B tokens; the result is limited to one seed and a
   constant-LR schedule. SophiaG diverged unpredictably in every tested run.
   ([ladder](2026-09-18.md#6-the-olmo-3-tokenizer-ladder),
   [optimizer comparison](../experiments/optimizer-comparison/README.md))
6. **The 80B failure mechanism is still unknown.** Q3 controls invalidated
   the earlier bf16 range-overflow account and did not reproduce the historical
   failure deterministically. DP and seed remain live variables; there is no
   production-length stable 80B trajectory to report.
   ([September status](2026-09-18.md#5-the-80b-localization-is-not-a-mechanism))

---

## 2. Resource Usage & Allocation

| Item | Q3 evidence |
|---|---|
| Year-2 Aurora allocation | 6,620K node-hours (renewal milestone table, carried forward from the [Q2 report](2026-Q2-incite.md#2-resource-usage--allocation)) |
| Year-2 Polaris allocation | 150K node-hours (same source) |
| Aurora balance snapshot, Aug. 30 | 1,269,842.6 node-hours available ([meeting notes](../meeting-notes/agpt-sync.md)) |
| Implied Aurora usage at that snapshot | ~5.35M of 6.62M node-hours, or ~80.8% (derived from the documented balance and allocation) |
| Polaris scheduling status | Allocation balance was already negative (`burn_ratio=1.68`) in late August ([Polaris handoff](../HANDOFF-20b-polaris-2026-08-26.md)) |
| Main production packing | Five training seats in a 2,098-node umbrella allocation ([September summary](2026-09-18.md#1-production-two-55-umbrellas-then-three-distinct-launch-failures)) |
| Full-utilization demonstrations | Jobs `8808931` and `8812215` trained all 5/5 seats for full 12-hour allocations ([September summary](2026-09-18.md#1-production-two-55-umbrellas-then-three-distinct-launch-failures)) |
| Storage recovery | 8.4 TB reclaimed; project usage fell from 20.35 TB to 11.91 TB ([August 29 summary](2026-08-29.md#9-infrastructure)) |

Aurora usage moved from 28.5% at the Q2 reporting point to an implied 80.8% at
the August 30 snapshot. Concurrent 2B and 20B chains ran in five-seat,
2,098-node umbrella allocations. Two consecutive umbrellas trained 5/5 seats
for 12 hours. Other allocations still lost time to initialization failures and
queue pressure.

There is no reconciled September 30 accounting export in the repository. The
80.8% value is an August 30 snapshot, not a quarter-end total. The Polaris
record establishes a negative balance, not an exact quarter-end total.

---

## 3. Milestone Status

> **Mapping note.** The INCITE renewal milestones name a Megatron-DeepSpeed
> pipeline and a 70B dense / 300B MoE / 7B and 36B CPT model lineup. Q3 work
> used the PyTorch-native TorchTitan + `ezpz` stack and a 2B/20B/30B-class/80B
> validation ladder. The mapping is therefore approximate.

| Milestone | Proposal target | Q3 status | Evidence-based interpretation |
|---|---|---|---|
| **Y2:M1** | Dense ~70B: pretrain, CPT, post-train, eval | **Partially achieved; large-model risk remains** | Both 2B base chains completed; 20B advanced and received accepted endpoint evaluations; 2B CPT/SFT/GRPO paths ran. The closest large-dense analog, 80B, remains blocked by unresolved long-run instability. |
| **Y2:M2** | 300B MoE: pretrain, post-train, eval | **Correctness infrastructure advanced; production not started** | Portable and SYCL expert backends were integrated and checked across Aurora, Sunspot, Polaris, and Perlmutter. EP=2/8/12 geometry and a production-shaped 12.29B-total/1.91B-active model were validated, but no 300B production run occurred. |
| **Y2:M3** | 7B dense CPT | **Method demonstrated below target scale** | Two approximately 300B-token 2B CPT pilots completed and reduced in-distribution loss, but degraded downstream commonsense scores. A target-scale 7B CPT deliverable is not documented. |
| **Y2:M4** | 36B MoE CPT | **Not delivered** | MoE execution and data/optimizer infrastructure advanced, but no 36B MoE CPT run is documented. |
| **Y2:M5** | Development time | **Substantial progress** | Q3 delivered upstream compatibility, failover improvements, evaluation corrections, MoE backend integration, optimizer studies, and the OLMo-3-tokenizer model ladder. |

---

## 4. Production Training and Evaluation

### 4.1 Base and continued pre-training

| Trajectory | Quarter-end outcome |
|---|---|
| 2B, 256 nodes | Complete at step 92,859, 4.674T tokens, final loss 2.652 ([summary](2026-07-06.md)) |
| 2B, 512 nodes | Complete at step 46,429, 4.6737T tokens against a 4.6738T target ([summary](2026-08-14.md)) |
| 2B-512 corrected endpoint evaluation | MMLU 0.2579, ARC-Challenge 0.2978, HellaSwag 0.5384 after correcting the RoPE export path ([summary](2026-08-21.md)) |
| 20B, 512 nodes | Accepted endpoint evaluation at checkpoint 11,100 ([journal](../journal.md#2026-09-30-aurora----production-reporting-audit-and-tail-evaluation)) |
| 20B, 256 nodes | Accepted endpoint evaluation at checkpoint 17,500 (same source) |
| 2B-512 stage 2 | Complete at step 23,746; endpoint evaluated on September 30 (same source) |
| 2B-256 stage 2 | Complete at step 41,300; endpoint evaluated on September 30 (same source) |

The 2B base runs plateaued before their final tokens. On the 256-node chain,
HellaSwag-normalized, ARC-Easy, and PIQA were flat over approximately the last
635B tokens. The corrected 512-node endpoint showed the same broad conclusion:
the final training window produced no clear capability gain. This motivated
continued-pretraining experiments that changed the data mix rather than merely
extending the original schedule.

Two 2B CPT pilots each ran approximately 300B tokens. Dolmino-100 ended at
validation loss 2.492 and the 50/50 OLMo/Dolmino mix at 2.601, versus a base
plateau near 2.80. However, Dolmino-100 reduced HellaSwag from 0.560 to 0.486
and ARC-Easy from 0.651 to 0.547; the 50/50 mix reached 0.491 and 0.610.
Lower in-distribution loss did not preserve broad downstream capability.
([CPT tracker](../production/cpt/README.md))

### 4.2 Post-training

The best 2B reasoning checkpoint came from a two-stage SFT recipe: general math
followed by GSM8K chain-of-thought. It reached 0.205 strict-answer accuracy and
0.985 format compliance on 200 GSM8K-CoT problems. Single-stage alternatives
landed between 0.02 and 0.065. A gated GRPO continuation held format at 1.000
but moved accuracy only from 0.205 to 0.215 after 100 steps, within noise. A
separate 1,000-step arithmetic GRPO campaign did solve its narrower task, and
reward decomposition improved the alphabet-sorting score to 0.667, a 168%
increase over the prior plateau. ([post-training status](../production/POST-TRAINING-2B.md))

The final checkpoint of a long full-mix SFT run catastrophically forgot general
capability (HellaSwag 0.593 to 0.273; ARC-Easy 0.694 to 0.298). The retained
deliverable is the earlier checkpoint 900, not checkpoint 8,672.

### 4.3 Evaluation integrity

The quarter expanded evaluation to MMLU 5-shot, GSM8K 5-shot, and
ARC-Challenge 25-shot, and corrected two silent provenance errors: a shot-count
key collision and checkpoint export with the wrong RoPE convention. The
September 30 reconciliation rendered a 608-result corpus plus four completed
tail artifacts. Current accepted endpoints cover both 2B base chains, both 20B
base chains, and both 2B stage-2 chains.
([evaluation landscape](../evals/eval-landscape-2026-07.md),
[journal](../journal.md#2026-09-30-aurora----production-reporting-audit-and-tail-evaluation))

MMLU remains effectively at chance for the reported AuroraGPT checkpoints.
The completed 2B-512 endpoint scored 0.2579. The MDS stage-3 mixed and
math/code endpoints scored 0.2463 and 0.2591 at 7.771T tokens; the mixed
endpoint retained ARC-Easy 0.7138 and ARC-Challenge@25 0.4164, while the
math/code endpoint reduced HellaSwag from 0.587 to 0.422.
([September 26 summary](2026-09-26.md#evaluations))

---

## 5. Technical Highlights

### 5.1 Dense-model efficiency and optimizer research

A 26.2B-parameter model completed 2,000 steps on Sunspot with loss 12.028 to
2.115, zero NaN/inf, and 28.3% MFU. Its early evaluation reached HellaSwag
27.14% and MMLU 26.24%, with a combined Stouffer significance of +4.81 sigma
over random. This was a proposal-validation campaign, not Aurora production.
([September summary](2026-09-18.md#6-the-olmo-3-tokenizer-ladder))

At fixed global batch 960, Mano completed the 23.59B-token optimizer comparison
at loss 2.43889 versus AdamW at 2.51357, a 0.075-nat advantage. The comparison
used one seed and no decay phase, so it does not establish that Mano wins a
production schedule. SophiaG produced strong matched-step loss but diverged in
all tested runs and is not suitable for unattended production.
([optimizer comparison](../experiments/optimizer-comparison/README.md))

The 30B-class model held 25.54% MFU at 64 nodes. At 512 nodes, the measured
configuration was approximately 17% MFU because a fixed, useful global batch
forces local batch size down to one. This makes learning geometry, rather than
peak MFU alone, the scale-selection constraint.

### 5.2 MoE and cross-platform validation

The largest Q3 upstream integration absorbed 148 commits. Before landing,
14/14 target modules imported, 12/12 AGPT and 14/14 MoE flavors
built, 82 tests passed, and Perlmutter comparison showed maximum absolute delta
`1.907e-06` with 100% argmax and top-5 agreement. The integrated stack was then
exercised on Aurora, Sunspot, Polaris, and Perlmutter.
([September summary](2026-09-18.md#3-sync-84-148-commits-verified-before-landing))

For MoE expert execution, `bmm_nodrop` was bit-exact against the reference on
Intel XPU and NVIDIA CUDA, while `aurora_sycl` agreed within `6.104e-05`. The
full Sonic path required one-time shared-cache compilation of its SYCL
extensions; after that, it trained within approximately `5e-05` of the EP=2
reference. The production-shaped EP=12 smoke contained 12.29B total and 1.91B
active parameters and completed finite updates. The first matched small-shape
comparison was directionally neutral in early steady steps and 21.4% slower in
whole-run wall time because of compile stalls, so no production throughput
advantage is claimed. ([four-machine report](../experiments/moe-expert-backends-4machine.md))

### 5.3 Runtime, resilience, and production operations

The oneAPI/framework release candidate improved the 2B baseline by 9.2% at two
nodes and 4.5% at 64 nodes, reaching 7,800 and 7,003 TPS/GPU respectively.
Aurora XCCL mesh probes passed at 48, 192, 384, and 768 ranks, including an
independent second 768-rank pass. ([August summary](2026-08-14.md#92-rc-re-baseline-45-92-faster-no-regression),
[September summary](2026-09-26.md#aurora-xccl))

Production logs produced two failover fixes: correct selection of the
unreachable PALS child node, verified by 39 tests, and a narrow
`std::bad_alloc` classifier for a failure observed five times. Two consecutive
2,098-node umbrellas subsequently trained all five seats for their full
12-hour allocations. Initialization failures still waste capacity. The packed
production model works when initialization succeeds.

---

## 6. Risks and Open Issues

1. **80B long-run stability remains unexplained.** Clean controls do not
   reproduce the historical failure deterministically; DP and seed remain live
   variables. No production-length 80B trajectory is ready to claim.
2. **One 2B constant-LR trajectory became NaN-poisoned.** The first NaN occurred
   at step 39,943. Restoration from a later checkpoint propagated the poisoned
   state, and the first-update replay remained blocked by a collective stall at
   the September 26 cutoff. ([summary](2026-09-26.md#production))
3. **20B remains incomplete.** Quarter-end endpoint evaluations exist, but the
   model has not completed the 4.67T-token base target.
4. **MoE performance is not yet established.** Correctness is proven, but a
   compiled, production-representative EP=12 throughput comparison is still
   required.
5. **Large-allocation startup failures persist.** `std::bad_alloc`, oneCCL KVS
   timeout, and PALS RPC-forward failures all appeared during Q3 umbrella
   initialization.
6. **Evaluation provenance is uneven for older artifacts.** Older 2B results do
   not all carry self-describing shot metadata; endpoint claims in this report
   use corrected and accepted artifacts only.
7. **Resource accounting is incomplete at quarter end.** The latest documented
   Aurora balance is from August 30; an authoritative September 30 allocation
   export was not preserved in the repository.

---

## 7. Priorities for Q4 (Oct – Dec 2026)

1. Resolve the poisoned 2B trajectory's first-update replay and identify the
   first non-finite tensor at step 39,943.
2. Continue the 20B-512 and 20B-256 chains from their accepted quarter-end
   checkpoints and maintain sparse, token-aligned evaluation coverage.
3. Complete fixed-LR validation for the 5B/10B/30B OLMo-3-tokenizer ladder,
   then run the decay-phase comparison needed to determine whether Mano's
   constant-LR advantage survives a production schedule.
4. Convert MoE correctness into a production performance result with a compiled,
   production-representative 10B EP=12 comparison reporting MFU, step time, and
   loss agreement.
5. Reframe 80B work as controlled hypothesis testing over DP and seed rather
   than production readiness, retaining tensor-level pre-clip diagnostics.
6. Preserve complete, self-describing evaluation metadata and reconcile Aurora
   and Polaris allocation exports for the next program report.

---

## Appendix: Source Material and Coverage

The Q3 rollup draws from the retrospectives ending July 6, July 10, July 26,
August 10, August 14, August 21, August 29, August 31, September 6, September
18, and September 26, plus September 30 journal entries and the production,
evaluation, CPT, post-training, MoE, and optimizer trackers linked above.

The summary windows overlap and use different commit-count conventions, so no
aggregate Q3 commit total is claimed. The latest broad retrospective ends
September 26; September 27–30 outcomes are incorporated only where the
September 30 journal provides accepted artifacts. No 300B MoE pretraining, 36B
MoE CPT, target-scale 7B CPT completion, production-scale MoE throughput result,
or final 80B root cause is documented.

Branch: [`ezpz`](https://github.com/saforem2/torchtitan/tree/ezpz) ·
Docs root: [README.md](../README.md) ·
Live production: [production/README.md](../production/README.md)
