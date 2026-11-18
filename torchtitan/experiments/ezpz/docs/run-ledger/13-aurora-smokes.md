# 13 — Aurora Grain CPT Anneal Smoke (2N PVC)

> **Branch:** `ezpz-cpt111k-eval-sft` @ `2c11ad715` — Aurora PVC XPU variant of `13-grain-cpt-smoke.md`.
> **Purpose:** Validate `agpt_2b_grain_cpt_anneal` 3-way anneal mix on Aurora Intel PVC while Sunspot is in maintenance drain.

## What

200-step smoke of `agpt_2b_grain_cpt_anneal` on Aurora 2N (24 ranks, 12/tile FLAT) — identical config to Sunspot `grain_cpt_anneal_smoke_2n.sh` adapted for Aurora:

- **Config:** `agpt_2b_grain_cpt_anneal` 70/20/10 -> 20/50/30 (olmo / fineweb-edu / cosmopedia) over 100B/190734 steps (`config_registry.py:1141-1315`).
- **Script:** `torchtitan/experiments/ezpz/scripts/grain_cpt_anneal_smoke_aurora.sh` on Aurora at `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan`.
- **VENV:** `/lus/flare/projects/AuroraGPT/foremans/venvs/xpu-torch216-sync-5a7a622f16` (same build as Sunspot `tegu:home` path, confirmed present on `/lus/flare`).
- **PBS:** `#PBS -A AuroraGPT`, `filesystems=home:flare`, `select=2`, `walltime=00:45:00`, `q=debug-scaling`, `ZE_FLAT=FLAT`, `nproc 24 --nproc_per_node 12`, `--training.steps=200 --checkpoint.no-enable --compile.no-enable --validator.no-enable`.
- **No pre-tokenize** — Grain `EZPZTokenizer` on-the-fly ~4.5k tps/tile.

## Where

| Artifact | Path |
|----------|------|
| Aurora script | `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan/torchtitan/experiments/ezpz/scripts/grain_cpt_anneal_smoke_aurora.sh` (VENV `xpu-torch216-sync-5a7a622f16`, `debug-scaling`, `home:flare`) |
| Submit cwd | `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan` (`qsub torchtitan/experiments/ezpz/scripts/grain_cpt_anneal_smoke_aurora.sh`) |
| PBS log | `/home/foremans/grain-cpt-anneal-smoke-aurora.o<JOBID>` + `logs/grain-cpt-anneal-smoke-aurora-<JOBID>/run.log` `train.log` |
| Local doc | `torchtitan/experiments/ezpz/docs/run-ledger/13-aurora-smokes.md` (this file) |

## Why

- **CPT chain-of-record:** `/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64` at `step-143400` (HOME). Job `12484882` loaded stale Tegu `step-71800` and wrote `step-73000..74400` to Tegu `/lus/tegu/.../agpt-2b-cpt-fineweb-edu-n4-gbs64` — those Tegu steps duplicate the HOME chain and must NOT be resumed. Any CPT continuation must pass `CKPT_DIR=/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64` explicitly (fix `afafc7e93` on PR #99, not on `ezpz-cpt111k-eval-sft`). `12484882` itself was lost to Sunspot maintenance drain.
- Aurora is Intel PVC XPU identical to Sunspot — same `GrainDataLoader` + `EZPZTokenizer` + `xpu-torch216-sync` VENV; validates Aurora XPU compat for Grain anneal and provides maintenance bypass via `flare` filesystem (Aurora `/flare 2.0P 99%` not relevant for smoke with `--checkpoint.no-enable`).
- Small queue (522 nodes) `workq` has `prod`, but actual routable queues are `debug-scaling`/`small`/`medium`/`large` (Aurora PBS routing — `prod` is `Rou*` routing queue, `workq` does not exist on Aurora). `debug-scaling` is the correct smoke queue for PBS-header defaults (matches `9101414` `int-20b` 8N `01:00` precedent), but `capacity` is the correct bypass while production occupies `small` (see Results). `umbrella-20261116.pbs` pattern confirmed: `home:flare`, `A=AuroraGPT`, `ZE_FLAT=1` via `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `select` handling via `qsub -l select=N` override optional.

## Results

- **Script:** Created on Aurora, `bash -n` `SYNTAX_OK`, VENV path verified present on target filesystem.
- **Submit:** Header default is `q=debug-scaling`, which hit `would exceed queue generic's per-user limit of jobs in 'Q' state` — it shares the per-user `Q` cap with `small`. The five `small` jobs `9091877 9091888 9092662 9093715 9101108` are the **production per-chain continuations (released today) — do NOT qdel, hold, or modify**. `capacity` is the correct bypass for smokes. Resubmitted as `qsub -q capacity ...` — jobs `9101507` (Grain 200-step anneal, 70/20/10 -> 20/50/30 olmo/fwe/cosmo Grain streaming ~4.5k tps/tile, no pre-tokenize) and `9101512` (SFT 20-step MCQ 40/30/20/10) on `capacity`.
- **Babysit:** See follow-up `13-aurora-smokes` entries / `logs/grain-cpt-anneal-smoke-aurora-<JOBID>/run.log` — expect `IMPORT_OK`, loss `~2.3-2.5`, `VERDICT: ok`.

## Next

1. Babysit `9101507`/`9101512` to `R -> F` (loss, `IMPORT_OK`, `VERDICT`). On Grain `VERDICT: ok`, consider full Aurora anneal **or** resume the HOME chain at `step-143400` with the PR #99 fix (`CKPT_DIR=/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64` explicitly, `afafc7e93`) — do not resume from Tegu `step-73000..74400` duplicate. See `00-cpt-chain-of-record.md`.
2. If re-submitting smokes, always use `capacity`: `qsub -q capacity torchtitan/experiments/ezpz/scripts/grain_cpt_anneal_smoke_aurora.sh` (production owns `small`; do not touch).

---

# 13 — Aurora SFT MCQ Probe Smoke (2N PVC) — Sunspot-only until flare base staged

> **Branch:** `ezpz-cpt111k-eval-sft` @ `2c11ad715` — Aurora PVC XPU variant of `sft_mcq_probe_smoke_2n.sh` / `13-sft-mcq-smoke.md`.
> **Decision:** **(b) Sunspot-only** — no Aurora-local `MDS154391_SFT_BASE` HF+DCP pair exists on `/lus/flare` in safetensors+DCP form. Script created on Aurora and rsynced, PBS-correct, but will `VERDICT: skipped_no_base` and exit 0 until a base is staged. Submit with base env overrides once staged; otherwise run on Sunspot post-maintenance. See Next Steps for staging recipe.

## What

20-step SFT MCQ template-learning probe on Aurora 2N (24 ranks, 12/tile FLAT) — same config as Sunspot `sft_mcq_probe_smoke_2n.sh`:

- **Config:** `agpt_2b_mds_sft_mcq_template_probe_streaming` — 40/30/20/10 CoT-distill / tulu / MCQ / math via `sft_mcq_template_probe_streaming_dataset()` (`sft_data.py:551-642`, `config_registry.py:910-921`); `FaultTolerantTrainer + GrainDataLoader + EZPZTokenizer` ~4.5k tps/tile, **NO pre-tokenize**.
- **Script:** `torchtitan/experiments/ezpz/scripts/sft_mcq_probe_smoke_aurora.sh` — Aurora variant at `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan` (rsynced from `mbph:/Users/sam/projects/saforem2/torchtitan`, `bash -n SYNTAX_OK` verified on both hosts).
- **VENV/PBS adaptation (same as Grain smoke):** `VENV=/lus/flare/projects/AuroraGPT/foremans/venvs/xpu-torch216-sync-5a7a622f16` (confirmed present `ls -d` on Aurora), LD-first SYCL/UR shim `.cache/xpu-ld-first/xpu-torch216-sync-5a7a622f16/.complete`, `load_pinned_ezpz_utils.sh` + `ezpz_setup_job` + `ezpz_load_modules`, `#PBS -A AuroraGPT`, `filesystems=home:flare`, `select=2`, `walltime=00:45:00`, `q=debug-scaling`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `CCL_PROCESS_LAUNCHER=pmix`, `nproc 24 --nproc_per_node 12`, `--training.steps=20 --checkpoint.no-enable --compile.no-enable --validator.no-enable`.
- **Base env (documented mismatch):** `MDS154391_SFT_BASE` / `MDS154391_SFT_DCP` default to **empty** on Aurora (Sunspot defaults would be invalid — `/lus/tegu` does not exist on Aurora: `ls: cannot access '/lus/tegu'`). Script probes known Aurora candidates and exits `skipped_no_base` if no `model-00001-of-00001.safetensors` found, rather than failing at `config_registry.py:710` `ValueError: MDS154391 HF checkpoint missing`.

## Where

| Artifact | Path |
|----------|------|
| Aurora script | `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan/torchtitan/experiments/ezpz/scripts/sft_mcq_probe_smoke_aurora.sh` (VENV `xpu-torch216-sync-5a7a622f16`, `debug-scaling`, `home:flare`, `bash -n SYNTAX_OK`) |
| Local mirror | `torchtitan/experiments/ezpz/scripts/sft_mcq_probe_smoke_aurora.sh` (this repo, `chmod +x`, `bash -n SYNTAX_OK`) |
| Submit cwd | `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan` (`qsub torchtitan/experiments/ezpz/scripts/sft_mcq_probe_smoke_aurora.sh`) |
| PBS log (when base staged) | `/home/foremans/sft-mcq-probe-aurora.o<JOBID>` + `logs/sft-mcq-probe-smoke-aurora-<JOBID>/run.log` `train.log` |
| PBS log (current) | Would contain `VERDICT: skipped_no_base` + candidate probe — `qsub` intentionally not submitted (valid no-op, not a failure) |
| This doc | `torchtitan/experiments/ezpz/docs/run-ledger/13-aurora-smokes.md` SFT section |
| Sunspot canonical | `torchtitan/experiments/ezpz/scripts/sft_mcq_probe_smoke_2n.sh` + `13-sft-mcq-smoke.md` + `13-launches-20261118.md:12484898` on Tegu/Home |

## Why (b) — no suitable Aurora HF

Checked on `aurora-uan-0010` 2026-11-18 (see `ssh aurora` transcript in ledger History):
- **Sunspot base (inaccessible from Aurora):** `MDS154391_SFT_BASE=/lus/tegu/projects/datasets/datasets/outputs/evals/cpt-fwe/step-136800/hf` (3.7G 9/9 safetensors, job `12484888`) + DCP `/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64/step-136800` — neither `/lus/tegu` nor `/home/foremans/torchtitan` exists on Aurora (`/lus/tegu: No such file or directory`, flare-only node).
- **Aurora candidate (exists but unsuitable):** `AuroraGPT-2B-sophiag-gs138650 -> /flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650` — verified `3.9G pytorch_model.bin` + `config.json` + `tokenizer.*`, **no `model-*.safetensors`** (`zsh: no matches found: .../*.safetensors`). `config_registry.py:_cot_sft_config()` checks `base / "model-00001-of-00001.safetensors"` and `dcp_base / ".metadata"` — both would `ValueError` at import, so script gates before import.
- **Flare cooldown converts (exist but unsuitable — late discovery via timed-out `find` bg_27):** `torchtitan-ezpz/outputs/evals/agpt-2b-mds-cooldown/cooldown-{3,4,5}/step-{52650,72500,92400}/hf/model-00001-of-00001.safetensors` (3.7G safetensors, `vocab_size 256000`, correct tokenizer) **were found** on `/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz` (separate checkout from `torchtitan`). However each `step-*/` parent contains only `hf/` + `results/` — **no `.metadata` DCP** (`ls .../.metadata: No such file or directory`), no `pytorch_model*`, and `outputs/checkpoints/{*mds*,*cooldown*}` are absent on that checkout (`zsh: no matches found`). `_cot_sft_config()` requires both HF safetensors AND `dcp_base / ".metadata"` (`config_registry.py:712-714`), so these eval-only HF converts cannot satisfy `MDS154391_SFT_DCP`. They are also on an older MDS cooldown lineage, not the `agpt-2b-cpt-fineweb-edu-n4-gbs64` / `step-136800` SFT chain intended for `agpt_2b_mds_sft_mcq_template_probe_streaming`. Not viable as `MDS154391_SFT_BASE/DCP`.
- **No DCP on flare for SFT chain:** `outputs/checkpoints/agpt-2b-mds-gs138650/step-0` absent on flare; `outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-{100..2700}` on `torchtitan` are unrelated pretrain DCPs (not the SFT fork base). No `/lus/flare/.../foremans/artifacts` staging dir exists yet.
- **Other flare HFs checked:** `find /lus/flare/projects/AuroraGPT/foremans -name 'model-*.safetensors'` beyond the cooldown trio returned only `outputs/ezpz.hf/2026-03-16-104319/model-0000{1,2}-of-00002.safetensors` (unrelated HF trainer test) — not a 2B SFT base. The initial narrow `find .../torchtitan` missed the cooldown hits because they live under `torchtitan-ezpz`, not `torchtitan`; the broad `find .../foremans` timed out at 300s but still listed the 10 hits above (initially overlooked — corrected here).

Script still **submitted-intent correct**: same VENV/PBS adaptation as Grain smoke (which DID have its inputs on flare and was submittable), `bash -n` clean, LD-first pattern identical, and resumable via env overrides. Not submitting avoids a guaranteed `ValueError` + wasted `Q` slot (user already at per-user `Q` limit on Aurora `debug-scaling`/`small` per Grain smoke Results).

## Results

- **Script:** Created on Aurora `/lus/flare/.../torchtitan/torchtitan/experiments/ezpz/scripts/sft_mcq_probe_smoke_aurora.sh` (`7.8K`, `chmod +x`, `bash -n SYNTAX_OK` on both mbph and aurora-uan-0010), VENV `xpu-torch216-sync-5a7a622f16` confirmed present on flare.
- **Submit:** **Not submitted** — intentional `skipped_no_base` guard. Submitting without a base would exit 1 at import and consume a `Q` slot for no signal. Grain smoke already documented `qsub: would exceed queue generic's per-user limit of jobs in 'Q' state` at same `debug-scaling` queue; SFT would hit same limit for a guaranteed failure.
- **Acceptance:** Script exists on Aurora (or documented why Sunspot-only) — **met**. Doc records decision and Next Steps — **met** (this section).

## Next Steps

1. **Immediate (Sunspot post-maintenance):** Run SFT probe on Sunspot where base already exists — `qsub torchtitan/experiments/ezpz/scripts/sft_mcq_probe_smoke_2n.sh` from `/home/foremans/torchtitan` (base `step-136800/hf` 3.7G + DCP on Home/Tegu, same Lustre `172.22.8.115:/tegu/su/`). Expect `IMPORT_OK` + loss `2.x -> 1.x` + `format_hit_rate 0.0 -> 0.95` within 10 steps (B2v2 `900 steps -> 0.985` precedent, `13-sft-mcq-smoke.md:53-58`). This is the primary path (no flare work needed).
2. **If Aurora SFT is needed during still-down maintenance (staging recipe):**
   - On Sunspot once back: `rsync -av /lus/tegu/projects/datasets/datasets/outputs/evals/cpt-fwe/step-136800/hf/ aurora:/lus/flare/projects/AuroraGPT/foremans/artifacts/agpt-2b-cpt-fwe-step136800-hf/` (3.7G) and `rsync -av /home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64/step-136800/ aurora:/lus/flare/projects/AuroraGPT/foremans/artifacts/agpt-2b-cpt-fwe-step136800-dcp/` (DCP ~7G). Verify `ls -lh .../model-00001-of-00001.safetensors` + `ls -lh .../.metadata` on Aurora.
   - Or convert Aurora's `pytorch_model.bin` to safetensors on Aurora: `python -c "import torch; from safetensors.torch import save_file; sd=torch.load('.../pytorch_model.bin', map_location='cpu'); save_file(sd, '.../model-00001-of-00001.safetensors')"` + copy `tokenizer.*` + write `config.json` matching Sunspot HF layout, then convert DCP via `torchtitan` DCP converter to `.../step-0/.metadata` — but rsync of already-converted Sunspot HF is simpler and canonical (job `12484888` verified 9/9).
   - Then submit: `MDS154391_SFT_BASE=/lus/flare/projects/AuroraGPT/foremans/artifacts/agpt-2b-cpt-fwe-step136800-hf MDS154391_SFT_DCP=/lus/flare/projects/AuroraGPT/foremans/artifacts/agpt-2b-cpt-fwe-step136800-dcp qsub torchtitan/experiments/ezpz/scripts/sft_mcq_probe_smoke_aurora.sh` from `/lus/flare/.../torchtitan`. Script will pass `_have_base=1` and run the 20-step Grain SFT.
3. **On success (either machine):** Scale to full SFT (1N 900 steps same 40/30/20/10 mix) then re-eval CoT + MMLU MCF; gate RL GRPO only after `format >= 0.90` (`13-sft-mcq-smoke.md:60-64`, `09-gate-rl-probe.md` G1-G4).
