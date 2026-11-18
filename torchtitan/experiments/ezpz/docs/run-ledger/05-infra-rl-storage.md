# 05 — Infra / RL / Storage Ledger

> **Branch:** `ezpz-cpt111k-eval-sft` HEAD `a6135a79` (2026-11-16).
> **Scope:** VENV-per-workload map, Monarch GRPO infra, TRL legacy, RL-as-format-perfector evidence, storage routing, and what went wrong.
> **Every claim cites its source file or job ID.** Inferences are labeled `[INFERENCE]`.

---

## 1. What is running

### 1.1 RL — Monarch GRPO (live 3-step smoke + proven 2-host gate)

The current RL launcher is `torchtitan/experiments/ezpz/rl/scripts/grpo/multihost_grpo_e2e.pbs` — a 2-host Monarch job (trainer on host 0, vLLM generator on host 1, TorchStore weight sync every step) running `torchtitan/experiments/ezpz/rl/scripts/grpo/multihost_train_upstream.py` via `ezpz launch -n 2 -ppn 1` with LoRA on GSM8K. Config: `rl_grpo_lora_agpt_2b_gsm8k_b2smoke`, `STEPS=3` default, `num_prompts_per_train_step=4`, `num_samples_per_prompt=4`, `max_tokens=700`, shard/tensor degrees 1/1, checkpoint every step. Source: `multihost_grpo_e2e.pbs:1-52,139-164`.

The 3-step cross-machine validation landed on 2026-11-14 on the `8e63247c` sync branch (PR #79) — three machines, same commit, all `Exit 0`:

| Run | Loss (steps 1-3) | Reward (steps 1-3) | Wall | Exit |
|---|---|---|---|---|
| **Sirius `33518`** (A100) | `0.0022 / -0.037 / 0.062` | `0.43 / 0.54 / 0.44` | ~18 min | `0` |
| **Sunspot `12483743`** (XPU/PVC) | `0.024 / 0.054 / 0.093` | `0.47 / 0.52 / 0.42` | `851 s` | `0` |
| **Aurora `9089138`** (XPU/PVC) | `0.017 / 0.082 / 0.11` | `0.47 / 0.46 / 0.48` | `1150 s` | `0` |

Source: `torchtitan/experiments/ezpz/docs/journal.md:62-71` (Monarch GRPO 2026-11-14 entry). Validation response length `117.9` at steps 0 and 3 on each machine — held steady, not degraded.

The higher-value Monarch gate is older and still canonical: **Sunspot `12479177`**, `40/40` rollouts across policy versions `0-3`, `auto->Gloo` transport, 4 pushes/4 pulls, 3 finite nonzero-gradient updates, DCP checkpoints at steps 1-3, clean shutdown, `Exit 0`. Affirmed unchanged through 2026-11-05. Source: `torchtitan/experiments/ezpz/docs/production/rl/monarch.md:3,14,152-165` and its re-affirmation header. The canonical operator guide is `monarch.md`; `multihost_grpo_e2e.pbs` is its smoke launcher (journal 2026-11-14 open line at `journal.md:74`).

An in-flight GRPO job on B2 was staged at Sunspot `12483252` (`select=3 auto=2+1 spare 06:00`) on `checkpoint-93-hf` (B2 `0.205` GSM8K-CoT) with `gs=8 lr=5e-6 100 steps ckpt/25`, Monarch TorchStore + XPU vLLM `float32` — but per the 2026-11-11 provisional entry, this job was `R --` (staged, not yet training) at time of writing and its outcome was to be finalized when `12483245/252/253` exited. Source: `torchtitan/experiments/ezpz/docs/journal.md:122-142` (provisional `R --` note at line 126). No exit-verified result for this B2 GRPO run is recorded on this branch [INFERENCE: if a later journal entry landed an exit code for `12483252`, this table should be extended].

TRL legacy: **not running**. Last TRL GRPO validation is 2026-07-06 (3N/24 ranks), documented at `torchtitan/experiments/ezpz/docs/production/rl/trl.md:35-42` and explicitly marked deprecated in its header (`trl.md:3-9`). No TRL job has been submitted since July. Source: `trl.md:14-15`.

### 1.2 CPT chain

Live 4N CPT-FWE chain (`scripts/cpt_fineweb_edu_4n.sh`) on Sunspot `workq` at 111.2k/190.7k (58.3% of 100B budget) in the 2026-11-15 header — job `12483856 R` `step-111200` `23G` `32/32 shards` `556 ckpt dirs` MFU 15.8% tps 4625; `POST-TRAINING-2B.md:5`. A later SYNTHESIS snapshot notes 143.4k/190.7k (75.2%, ~717 dirs) on the same chain. CPT is the only continuously-running training lane; RL and SFT are probes on its frozen bases.

### 1.3 SFT — gated

SFT lane is parked to CPT milestones, not continuously running. `sft_distill_cot_4n.sh` (pre-tokenized `distill_cot_mix` via `load_from_disk`) is ready but parked to CPT 50k per `POST-TRAINING-PLAN.md` phase 2. Source: `torchtitan/experiments/ezpz/docs/journal.md:120-121` (SFT 1N lane) and `POST-TRAINING-2B.md:37`. Narrow SFT-on-CPT probe at `step-17000` scored `0.015` vs B2 `0.165` — format collapse on FWE-shifted manifold, not a regression to act on without broad-distill.

### 1.4 Eval

Eval is batch-gated on HF converts, not continuous. CoT GSM8K harness uses `frameworks/2026.1.0` + `pti-gpu` on XPU via `scripts/eval/submit_cot_eval.sh` (`frameworks/2026.1.0` load at line 16, PTI LD path, `ZE_FLAT=FLAT`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`). MMLU is Aurora-preferred (Sunspot XPU OOMs at `compute_topk_scores` 63G — `POST-TRAINING-2B.md:25`).

### 1.5 VENV-per-workload table

Three VENVs, one per workload class. Using the wrong VENV for a workload is a deterministic failure (see section 4).

| # | VENV path | Workloads | LD handling | Failure mode if wrong |
|---|---|---|---|---|
| 1 | `/lus/tegu/projects/datascience/foremans/venvs/xpu-torch216-sync-5a7a622f16` | **CPT** (`cpt_fineweb_edu_4n.sh`, `cpt_olmo25_dolmino75_4n.sh`) — Grain `0.2.18` + `torch_remat` streaming on XPU | **LD-first cache**: venv `libur_*`, `libsycl.so*`, `libumf*`, `libtcm*` are symlinked into `$SUBMIT_DIR/.cache/xpu-ld-first/<venv-basename>/` and prepended to `LD_LIBRARY_PATH` so the venv's SYCL/UR wins over the system oneAPI stack (fixes `urGraphGetIdExp` undefined symbol). See `cpt_fineweb_edu_4n.sh:54-65`. | If replaced by `frameworks/2026.1.0` (system oneAPI) — `ModuleNotFoundError: No module named 'torch_remat'` (the system framework image lacks `torch_remat`; CPT needs the fused `w13` remat kernel). If LD-first is omitted — same UR symbol error at import. Source: `POST-TRAINING-2B.md:12` sync note (`grain 0.2.18 ok` requires xpu-torch216) and `cpt_fineweb_edu_4n.sh:52` header comment. |
| 2 | `/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214-locality-20261001` | **SFT/RL + pretokenization** (`sft_distill_cot_4n.sh`, `_pretokenize_distill_cot_mix_1n.sh`, `agpt2b_b2v2_distill_cot_2n.sh`, RL GRPO `rl_grpo_b2_2n.sh` / `multihost_grpo_e2e.pbs`) — TRL + Monarch + TorchStore + vLLM-XPU | `LD_LIBRARY_PATH="${VENV}/lib:${LD_LIBRARY_PATH:-}"` — venv `lib/` prepended directly (simpler than LD-first; the venv's SYCL/UR wins over oneAPI `2025.3.1`). See `sft_distill_cot_4n.sh:99-103`. Preflight asserts `import trl` exists (`sft_distill_cot_4n.sh:112`). | If replaced by `xpu-torch216-sync-5a7a622f16` — `trl` missing → CPT venv has Grain but no `trl`/`transformers` GRPO deps (`No module named 'trl'`). If bare `xpu-torch214` (non-locality clone) — `urGraphGetIdExp` from mismatched UR loader; the `locality-20261001` clone is the one whose missing-RECORD audit was repaired from uv cache (`monarch.md:175-182`). If `frameworks/2026.1.0` — no `trl`, no `torchstore`, no Monarch. Source: `sft_distill_cot_4n.sh:98-113`, `journal.md:118-121` (6 pretok segfaults `3553055` fixed by VENV path to locality-20261001). |
| 3 | `frameworks/2026.1.0` (environment module) + `pti-gpu` | **Eval only** (`eval/submit_cot_eval.sh`, `eval_cot_cpt_fwe_batch.sh`) — vLLM `0.26` XPU `float32` CoT harness | `module load frameworks/2026.1.0 pti-gpu`; then `LD_LIBRARY_PATH` gets PTI GPU lib path appended (`/opt/aurora/.../pti-gpu-0.17.0-.../lib64`), `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`. See `eval/submit_cot_eval.sh:16-21`. | If CPT/SFT/RL is pointed at `frameworks/2026.1.0` — no torchtitan training imports (`torch_remat`, `grain`, `trl`, `monarch` all absent); eval harness itself OOMs on unrelated tasks (MMLU `compute_topk_scores 63G` on XPU — Aurora-only per `POST-TRAINING-2B.md:25`). Source: `eval/submit_cot_eval.sh:16`, `POST-TRAINING-2B.md:25`. |

Key invariant: **eval `frameworks/2026.1.0` is never a training VENV**. The 2026-11-04 LD fix chain (5 commits) established that `frameworks/2026.1.0` lacks `torch_remat` and the TRL/March stack; it exists only to run the vLLM eval harness on XPU.

---

## 2. Why

### 2.1 RL as format-perfector, not accuracy lever

RL on this branch was pursued as a **format-perfector**, not as the GSM8K-CoT accuracy lever. The evidence that forces this framing:

- **Gated GRPO on B2 held format 1.000 for 100 steps** (no drift) — drift-proof under GRPO. Source: `POST-TRAINING-2B.md:142,153` (gated GRPO row + "Bought: format perfection with no drift").
- **Accuracy was flat**: B2 `0.205 -> 0.215` on GSM8K-CoT at 200 problems — labeled noise in every source. Source: `POST-TRAINING-2B.md:142,158`, `docs/journal.md:68` (3-step reward flat at ~0.45 on XPU/CUDA), `thread-e-recipe-optimization.md:196-199` (SE = sqrt(0.2*0.8/200) = 0.028, so 0.01 is 0.26 sigma).
- **Weak-base RL reward-hacked**: reward `0.118 -> 0.217` while accuracy `0.16 -> 0.105`, format `0.955 -> 0.635`, 95% rollouts with no `<think>` and 20% echoing the exemplar's `\boxed{5}`. Source: `POST-TRAINING-2B.md:141,148-150`, `thread-e-recipe-optimization.md:208-212`.

The root cause is **group_size math**, not capacity. At B2's ~20% solve rate with `group_size=4` (configured in `multihost_grpo_e2e.pbs:151-153` as `num_prompts_per_train_step=4, num_samples_per_prompt=4`), the expected correct rollouts per group is `0.2 * 4 = 0.8`. Probability a group is **all-wrong** = `0.8^4 = 0.41` — **41% dead groups** where GRPO's within-group advantage normalization yields zero gradient for accuracy. Those groups contribute nothing to correctness learning. Source: `thread-e-recipe-optimization.md:200-204` (exact calculation) and its table at `thread-e-recipe-optimization.md:220`.

Scaling group_size mechanically thins the dead-group fraction:

| group_size | all-wrong probability | dead groups |
|---|---|---|
| 4 (current) | `0.8^4 = 0.41` | **41%** |
| 8 | `0.8^8 = 0.17` | 17% |
| 16 | `0.8^16 = 0.03` | 3% |

Source: `thread-e-recipe-optimization.md:202-203`. But group_size alone is not sufficient with a binary exact-match reward — groups with exactly 1 correct of N give weak advantages `(1 - 1/N)` vs `(-1/N)`. A **dense reward** (per-step correctness, numeric closeness, format completeness) is required alongside larger groups. The `ceiling-attack` study proved this: on `alphabet_sort` where the plateau was the reward function not capacity (four runs varying LR, rank, batch all stuck at `0.24-0.25`), componentizing the reward into `format/completeness/order` lifted cross-scored accuracy `0.24 -> 0.667` (+168%) on the same char-ratio metric. Source: `POST-TRAINING-2B.md:138-146`, `SYNTHESIS.md:47` rank 6.

### 2.2 Monarch 2-host — proven

The 2-host Monarch mesh (trainer on host 0, generator on host 1) is the **only multi-host RL infra proven on XPU** on this branch:

- Topology: `PBS allocation -> ezpz launch (1 rank/host) -> host_mesh_from_store() -> host 0 slice TrainerActor, host 1 slice GeneratorActor; TorchStore/Gloo weight publication`. Source: `monarch.md:24-37` (ASCII diagram) and `multihost_grpo_e2e.pbs:55-61` (host routing `x1*/x4* -> xpu`, `x3* -> cuda`).
- Passing gate: Sunspot `12479177` `40/40` rollouts `v0-3` `auto->Gloo` `Exit 0` with `pushes=4 pulls=4`, 3 finite updates, DCP checkpoints. Source: `monarch.md:152-165` (push/pull/loss/gd_norm block).
- The 3-step `multihost_grpo_e2e.pbs` cross-machine run (`Sirius 33518`, `Sunspot 12483743`, `Aurora 9089138`) is the downstream smoke that re-validates this stack on the `8e63247c` sync branch. Source: `journal.md:62-74`.

TRL legacy (`trl.md`) is kept for audit, not for new runs. The 3-tier generation table (`trl.md:26-30`) and its 1N/cross-node/2+ trainer-node validations (jobs `12468780`/`12469976`/`12470083`) are historical (2026-07-06). New production RL uses Monarch per `trl.md:3-9` deprecation banner and `POST-TRAINING-2B.md:132` ("Monarch+TorchStore+vLLM (2-tile) is what every agpt-2b LoRA and the whole CoT campaign actually ran on").

### 2.3 The 73 s vs 3.9 s XPU weight-fetch bottleneck

The journal's 2026-11-14 open line: **XPU generator weight fetch ~73 s per sync on both Sunspot and Aurora** (CUDA `3.9 s`). Source: `journal.md:74` ("XPU generator weight fetch is ~73 s per sync on both machines (CUDA 3.9 s). Smoke launcher: ..."). This is walltime, not throughput — at 3 steps the extra ~70 s per sync is tolerable; at 100 steps it dominates. No fix is landed on this branch [INFERENCE: the 19x gap likely reflects Gloo-over-TCP on XPU vs NCCL/CUDA fast path, but the journal does not attribute a root cause].

Related Monarch transport context: XPU automatic transport is topology-aware — same-host `SharedMemory`, cross-host `Gloo` — after the `HOSTNAME` locality fix (`torchstore_compat.py:repair_hostname_env()`). MonarchRDMA and XCCL are excluded from automatic XPU selection on this runtime (capability probes reported them as candidates but they are not qualified TorchStore weight-transfer backends here). Source: `monarch.md:78-149` (HOSTNAME probe `12479166` + transport diagnosis). The PBS script defaults to `auto` on XPU and `gloo` on CUDA (`multihost_grpo_e2e.pbs:91,102`), with an explicit override env `TORCHTITAN_TORCHSTORE_TRANSPORT` for controlled comparisons.

---

## 3. Outcome

### 3.1 GRPO on B2: drift-proof, accuracy-flat

The decisive RL result is the gated GRPO run on B2 (`checkpoint-93`, `0.205` GSM8K-CoT / `0.985` format). Measured on the strong base, the identical setup that hacked on the weak base is **drift-proof**:

| Metric | Before GRPO | After GRPO (100 steps) | Judgment |
|---|---|---|---|
| **Format** | `0.985` (B2 base) | **`1.000` held** | drift-proof |
| **GSM8K-CoT accuracy** (200-problem, `vLLM fp32 0.26`, `temp 0.0 max 768`, XPU) | `0.205` (41/200) | `0.215` (43/200) | **noise** — 0.01 at 20% base rate is SE=0.028 at N=200, i.e. 0.26 sigma. Source: `thread-e-recipe-optimization.md:196-199` SE calculation. |
| **Monarch plumbing** | — | `12479177` 40/40, 3 updates, Exit 0 | proven |

Source for the B2 GRPO row: `POST-TRAINING-2B.md:142` (gated GRPO table row), `POST-TRAINING-2B.md:152-155` ("Did not buy: any GSM8K-CoT accuracy gain. At ~20% solve rate with group_size=4, most groups are all-wrong"), `thread-e-recipe-optimization.md:196-206` (SE + dead-group analysis), `SYNTHESIS.md:47` rank 6.

The 3-step Monarch smoke on 2026-11-14 (`Sirius 33518`, `Sunspot 12483743`, `Aurora 9089138`) is consistent — loss and reward held flat over 3 steps on all three machines, Exit 0 — but at 3 steps it is a plumbing test, not an accuracy test.

### 3.2 Weak-base contrast: reward hacking is real and was caught

On the weak cold-start (stage-1 base without B2's stage-2 polish), GRPO initially lifted format `0.88 -> 0.99` then hacked: reward rose `0.118 -> 0.217` while GSM8K-CoT **fell** `0.16 -> 0.105`, format bled `0.955 -> 0.635`, and gating the reward to `0.000` revealed **95% of rollouts had no `<think>` block** and 20% echoed the exemplar's `\boxed{5}`. Root cause: down-weighting format `0.2 -> 0.05` (because it looked saturated) removed the guardrail that the SFT — not the reward — had been holding up. Source: `POST-TRAINING-2B.md:141,148-150`, `thread-e-recipe-optimization.md:208-212`. On strong B2 the same reward is drift-proof — cold-start quality gates RL.

### 3.3 VENV mapping — stabilized after the 80th sync churn

The VENV mapping in section 1.5 is the post-churn stable state. The 80th upstream sync removed or reshaped dataset and checkpoint shims, and the subsequent `c519de37a5` + storage routing pass pinned every script to the three VENVs above. `journal.md:5-10` documents the immediate fallout: after `origin/ezpz` moved past the pinned torchtitan-20b512 fixed-pos commit, the VENV-to-script contract had to be re-asserted commit-by-commit (the 20B padded-shard OOM at `journal.md:5-20` is the visible symptom — `w13 (2, 14336, 5120)` with `14336 = 2^11 * 7` has no factor of 3, so shard degree 12 fails and PPN=8 at 32 ranks stores 17.5 GiB/rank padded).

---

## 4. What went wrong

### 4.1 Sirius MPI — `libmpi.so` without the LD-compat shim

On Sirius (CUDA), `mpiexec` + `mpi4py` wheels link `libmpi.so.12` while Cray MPICH ships `libmpi.so` (no version suffix). `multihost_grpo_e2e.pbs:93-99` carries the fix: if `CRAY_MPICH_DIR/lib/libmpi.so` exists, create `$OUT/mpi-abi/libmpi.so.12` as a symlink and prepend it to `LD_LIBRARY_PATH`. Without this shim the actor `MPI_Init` fails to load `libmpi.so.12`. The CUDA host path also sets `MPICH_GPU_SUPPORT_ENABLED=0`, `PALS_TRANSFER=0` and forces `CC=/usr/bin/gcc` (the PrgEnv `nvc` cannot build Triton's driver stub) — see `multihost_grpo_e2e.pbs:93-102`.

### 4.2 xpu-torch216 sync vs fw-2026.1 — no `torch_remat`

The CPT VENV is `xpu-torch216-sync-5a7a622f16` specifically because **`frameworks/2026.1.0` (and fw-2026.1 family) lacks `torch_remat`** — the fused `w13` remat kernel that `agpt/parallelize.py:divisible_linear_param_shard_placements` depends on to avoid the padded `Shard(0)` fallback. Using `frameworks/2026.1.0` for CPT fails at `import torch_remat`. The CPT script's preflight proves this (`cpt_fineweb_edu_4n.sh:70` asserts `import torch_remat, grain` and checks `grain.__version__`); the header comment at `cpt_fineweb_edu_4n.sh:52` says it explicitly ("fw-2026.1 lacks remat").

### 4.3 TRL legacy — `xpu-torch214` without `trl`

The bare `xpu-torch214` / `frameworks/2026.1.0` families have no `trl` — the TRL `SFTTrainer` / `datasets` stage-2 path that every stage-1 SFT probe uses (`agpt2b_b2v2_distill_cot_2n.sh` pattern) requires `trl` to be importable. Running SFT on the wrong VENV fails at `python3 -c "import trl"` (see `sft_distill_cot_4n.sh:112` and the `xpu-torch214` clobber fix in `POST-TRAINING-2B.md:20-22` — `ezpz_get_job_env` clobbered `VIRTUAL_ENV` to stale `xpu-torch214` without `trl`, fixed to `rl-monarch-torch214-locality-20261001`).

### 4.4 Storage — /home 15T fill risk and the double-nesting symlink

Sunspot `/home` and `/lus/tegu` live on the same 1.2P Lustre pool at ~17% full (~1P free) — free space is real, but `/home` still holds the **live 15T CPT chain** (`outputs/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64` at `556` dirs, `23G` frontier) and SFT/eval artifacts accumulate there unless routed off. The project quota on `tegu` is separate accounting; `/home` fill is an operator risk (ENOSPC on one OST despite free-space on others — see `torchtitan/experiments/ezpz/docs/guides/known-bugs/sunspot-enospc-full-ost.md`). Source: `SYNTHESIS.md:216` Tegu output root note ("avoid /home fill — 1.2P Lustre 17% full, 1P free, but /home holds 15T live CPT") and `journal.md:2826` quota history (tegu 11T hard cap hit mid-session).

Commit `c519de37a5` (Sun 2026-11-15 `fix(ezpz): prefer Tegu project space for new outputs, keep live 15T on /home`) repointed every new-output path to the Tegu project space when it exists:

- `cpt_fineweb_edu_4n.sh:83-89` — `CKPT_DIR` now prefers `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/checkpoints/...` if that tree exists, else falls back to relative `outputs/checkpoints/...` (env `CKPT_DIR` overrides either — the live 15T chain at `/home` keeps its path via the existing symlink).
- `hf_batch_convert_cpt_fwe.sh:30-32` — DCP root stays on `/home` (the 15T chain), new `HF_ROOT` prefers Tegu (`/lus/tegu/.../outputs/evals/cpt-fwe`) with PBS `filesystems=tegu:home` covering both mounts.
- `sft_distill_cot_4n.sh:228-233` (+ `sft_distill_cot_4n.sh:240-250` log lines) — `CKPT_DIR` and `LOG_DIR` default to Tegu when available.

PBS filesystems constraint: **Sunspot PBS has no `/eagle`; use `filesystems=tegu:home`**. Source: `SYNTHESIS.md:214` (`PBS filesystems=tegu:home (PBS lacks /eagle on Sunspot — POST-TRAINING-2B.md sync note)`), `POST-TRAINING-2B.md:13` sync note, `cpt_fineweb_edu_4n.sh:5` (`#PBS -l filesystems=tegu:home`), `sft_distill_cot_4n.sh:5` same. On Aurora/Polaris use `/flare/AuroraGPT/foremans/` or `/eagle/datascience/foremans/` — Sunspot's flare is the same pool as `/home`/`tegu`.

Double-nesting symlink: the trainer dumps to `./outputs/` already, so a naïve `CKPT_DIR=outputs/checkpoints/...` inside a checkout whose `PBS_O_WORKDIR` is already `~/torchtitan/outputs` doubles to `outputs/outputs/checkpoints/`. The live chain actually lives at `outputs/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64`. The fix at `cpt_fineweb_edu_4n.sh:39-44` is:

```bash
# Fix checkpoint double-nesting: trainer dumps to outputs/ already
if [[ -d "${SUBMIT_DIR}/outputs/outputs/checkpoints" && ! -e "${SUBMIT_DIR}/outputs/checkpoints" ]]; then
  ln -s outputs/checkpoints "${SUBMIT_DIR}/outputs/checkpoints"
fi
```

This creates `outputs/checkpoints -> outputs/outputs/checkpoints` for resume, and the 2026-11-15 header notes the validator ran at `checkpointer.folder ./outputs/outputs/checkpoints/... at step 100000 (fault-tolerance restart)` with the symlink active (`POST-TRAINING-2B.md:10`). New chains that write to Tegu (`CKPT_DIR=/lus/tegu/.../outputs/checkpoints/...`) never double-nest because `SUBMIT_DIR` is not a prefix of that absolute path. Source: `cpt_fineweb_edu_4n.sh:39-44`, `thread-a-cpt-trajectory.md:66,121-122,528-529`, `POST-TRAINING-2B.md:10`.

### 4.5 Other blockers fixed in the 3-step run

- `PassConfig(enable_sp)` missing on XPU (`bd525d7f9`), `vllm-xpu-kernels` Xe2-only bindings failing on PVC `_xpu_C` (`4e38707` → rebuilt `vllm-xpu-kernels` wheel via `build_vllm_xpu_kernels_pvc.pbs`), `is_quantized_kv_cache` missing from XPU vLLM (`bd2e0a95f`), `torchvision 0.30.0.dev20261007+xpu` for vLLM warmup (`--no-deps`). CUDA: workers advertised by IP on multi-NIC, `MPI_Init`/`MPI_Finalize` handling. Source: `journal.md:72` (XPU/CUDA blocker list).

---

## 5. Next steps

### 5.1 RL recipe — if pursued

RL is ranked **rank 6 (low for accuracy, proven for format)** in `SYNTHESIS.md:47` — run as a probe, not a campaign. If pursued, the recipe that the evidence actually supports:

| Hparam | Current (B2 GRPO smoke) | Recommended | Source |
|---|---|---|---|
| `group_size` (`num_generations` / `--async-loop.num-samples-per-prompt`) | `4` (41% dead groups at 20% solve rate) | **8-16** — at 20% solve rate, 8 gives 17% dead, 16 gives 3% | `thread-e-recipe-optimization.md:200-204,220` |
| `learning_rate` | unknown (not logged in the B2 GRPO run) | **5e-6 to 1e-5** — `sft_arithmetic 8N` solved at `1e-6` (source: `POST-TRAINING-2B.md:136` and `thread-e-recipe-optimization.md:178`); `alphabet_sort` used `2e-5..5e-5`; GSM8K-CoT is harder than sum_digits but B2 is already SFT'd so conservative — try `5e-6` first, then `1e-5` if stable | `thread-e-recipe-optimization.md:221`, `POST-TRAINING-2B.md:175` next-steps item 2 |
| `beta` (KL penalty) | `0.0` (inferred from sum_digits + B2 GRPO no-drift) | **0.0 to 0.01** — `0.0` worked for sum_digits and B2; small `0.01` guardrail if format bleeds | `thread-e-recipe-optimization.md:222` |
| `generator.model_dtype` | `float32` (required on XPU) | **float32 mandatory** — bf16 through vLLM makes agpt-2b emit gibberish (vocab `256000` + ffn `11008` accumulate enough bf16 error to flip greedy argmax) | `POST-TRAINING-2B.md:158` NOTE (XPU landmine), `reason_agpt/config_registry.py:27`, `thread-e-recipe-optimization.md:227` |
| `max_completion_length` (`max_tokens`) | `700` in `multihost_grpo_e2e.pbs:156`; default `64` in `train_grpo.py:185` | **512-768** — must cover full CoT + `\boxed{}` answer; GSM8K CoT avg ~300 tokens, 64 truncates everything | `thread-e-recipe-optimization.md:223`, `train_grpo.py:185`, `reason_agpt/config_registry.py` |
| `reward` | binary/char-ratio on the smoke | **componentized additive** per `ceiling-attack.md` (+168%): `think_format 0.1-0.2` + `answer_extractable 0.1` + `answer_correct` dense (binary + numeric closeness) as separate `RewardFn` classes. At minimum: `answer_correct` with partial credit + `think_format` + `answer_extractable` | `thread-e-recipe-optimization.md:224`, `POST-TRAINING-2B.md:138-150`, `SYNTHESIS.md:47` rank 6 |
| `difficulty` | random GSM8K | **curated band** — filter to problems where B2's `pass@16` is 10-50% (hard enough that CoT is load-bearing, easy enough that base gets non-zero correct fraction). If baseline ~0, RL gradient cannot start | `thread-e-recipe-optimization.md:225` |
| Other invariants | — | `flex` attention with `max_autotune=False`, `similarity_power=1`, `--no-drop-zero-std-reward-groups` (without it cold-start groups are all dropped and no train step fires) | `POST-TRAINING-2B.md:158` NOTE |

Decision gate (per `SYNTHESIS.md:260-264` RL-specific gates): if accuracy lift is `< 0.05` at 2sigma on 200 (i.e. stays around `0.205-0.215`, SE 0.028), accept B2 as the RL ceiling at 2B and keep RL as format-perfector only; if format bleeds, `beta 0.0 -> 0.01` guardrail before any other tuning; if reward rises while GSM8K-CoT falls (`0.118->0.217 / 0.16->0.105` pattern), stop — reward is mis-specified (`rl/plans/cot.md` weak-base analysis).

### 5.2 VENV pin discipline

- **Do not run `frameworks/2026.1.0` as a training VENV** — it is eval-only. Its only consumers are `eval/submit_cot_eval.sh` and `eval/eval_cot_cpt_fwe_batch.sh`.
- **Do not change `VENV` without updating LD handling**: CPT's LD-first cache (`cpt_fineweb_edu_4n.sh:54-65`) and SFT/RL's `LD_LIBRARY_PATH=$VENV/lib` prepend (`sft_distill_cot_4n.sh:99-103`) are matched to their VENVs. Dropping LD-first on CPT or the `$VENV/lib` prepend on SFT/RL reintroduces the UR loader shadow / `trl`-missing failures of section 4.
- **Pretokenize shares SFT's VENV** (`rl-monarch-torch214-locality-20261001`) and bypasses `ezpz_setup_job` — it is single-rank, not Monarch-launched. Source: `journal.md:118-121` (6 prior `3553055` segfaults until VENV path was fixed).

### 5.3 Storage migration plan

- **Live chain stays on `/home` until 190k** — `c519de37a5` intentionally left the live DCP at `/home/foremans/torchtitan/outputs/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64` (15T, `556-717` dirs) with the double-nesting symlink in place, because moving mid-train breaks `FaultTolerantTrainer` resume. Source: `c519de37a5` commit message ("keep live 15T on /home") and `hf_batch_convert_cpt_fwe.sh:28-32` ("Live DCP stays on /home — moving mid-train breaks resume").
- **New HFs and new SFT outputs default to Tegu** — `c519de37a5` repointed every new-output path to `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/{evals,sft,checkpoints}` when that tree exists, behind a `-d` guard so local dev still falls back to relative `outputs/`. PBS `filesystems=tegu:home` covers both mounts.
- **Post-190k migration**: `rsync` the 15T DCP tree to Tegu, leave the `/home` symlink (`outputs/checkpoints -> outputs/outputs/checkpoints`) as a redirect, and flip `CKPT_DIR` to Tegu for subsequent continuations. Source: `c519de37a5` commit body (4 files changed, 27 insertions).
- **Quota check**: `lfs quota -u $USER /lus/tegu` before every large convert — `df` is misleading on Lustre (4 OSTs summed); see `guides/known-bugs/sunspot-enospc-full-ost.md:18-23`. The tegu project quota previously hit its 11T hard cap mid-session (`journal.md:4546-4548`).

---

## Appendix — Source index

| Claim | Source file @ `a6135a79` |
|---|---|
| RL PBS script, Monarch setup, host routing, VENV dispatch | `torchtitan/experiments/ezpz/rl/scripts/grpo/multihost_grpo_e2e.pbs:1-164` |
| Monarch XPU+CUDA arch, HOSTNAME fix, transport topology, `12479177` gate | `torchtitan/experiments/ezpz/docs/production/rl/monarch.md:3-182` |
| TRL legacy, deprecation, 2026-07-06 validation table | `torchtitan/experiments/ezpz/docs/production/rl/trl.md:1-49` |
| 3-step GRPO `33518/12483743/9089138` losses/rewards/walltimes/Exit 0, blockers, 73 s vs 3.9 s, `c519de37a5` | `torchtitan/experiments/ezpz/docs/journal.md:62-74,120-142` |
| GRPO on B2 `1.000` format / `0.205->0.215` noise, weak-base hacking, `float32` landmine, ceiling-attack +168% | `torchtitan/experiments/ezpz/docs/production/POST-TRAINING-2B.md:130-160` |
| group_size math (41%/17%/3%), LR 5e-6-1e-5, dense reward, RL worth-it framing | `torchtitan/experiments/ezpz/docs/exploration/thread-e-recipe-optimization.md:168-245` |
| Rank 6 RL EV, rank 8 group_size 8-16 probe, storage Tegu vs /home, PBS filesystems | `torchtitan/experiments/ezpz/docs/exploration/SYNTHESIS.md:47,168-216` |
| VENV refs: xpu-torch216-sync-5a7a622f16, rl-monarch-torch214-locality-20261001, frameworks/2026.1.0 | `torchtitan/experiments/ezpz/scripts/cpt_fineweb_edu_4n.sh:52-65`, `torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh:98-113`, `torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh:16-21` |
| Storage routing c519de37a5, Tegu prefer vs /home live 15T, PBS `tegu:home` | `c519de37a5` diff (commit message + 4-file stat), `cpt_fineweb_edu_4n.sh:5,39-44,83-89`, `sft_distill_cot_4n.sh:228-233` |
| CPT VENV LD-first vs SFT VENV LD prepend | `cpt_fineweb_edu_4n.sh:54-65` vs `sft_distill_cot_4n.sh:99-103` |
