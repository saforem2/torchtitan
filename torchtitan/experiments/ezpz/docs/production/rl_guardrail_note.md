# RL guardrail — CoT-gated GRPO on agpt-2B

> **Status: GATED — do not launch RL jobs.** Verified 2026-11-15 on
> branch `ezpz-cpt111k-eval-sft` (`3f763b24a3`). No `qsub` issued.

## Gate

Source: `POST-TRAINING-PLAN.md` Phase 3 — "RL as guardrail (only on strong
cold-start)":

> Do not start RL on CPT-derived SFT until **CoT > 0.15** — avoids
> CoT-stage-2 reward-hacking (reward 0.118 -> 0.217 while acc 0.16 -> 0.105).

Ground truth numbers from `POST-TRAINING-2B.md`:

| Checkpoint | CoT (GSM8K 200, greedy) | Format | Verdict |
|---|---|---:|---|
| **B2 = checkpoint-93** (two-stage SFT via `checkpoint-900`) | **0.205** (current harness re-score 0.165, same gate) | 0.985 | **exceeds 0.15 gate** — RL *could* run |
| CPT-FWE raw (no SFT) at 73k/74k | 0.00 (0/200, `n_unclosed=200`) | 0.0 | below gate — expected pre-SFT signal, not a regression |
| B2v2 ep3/ep6 (narrow extra epochs) | 0.09 / 0.105 | 0.86 / 0.91 | below gate, refuted |

**Decision (W2 ordering):** B2 at 0.205 clears the numeric gate, but RL stays
**gated behind W2 verification** — broad-CoT (`distill_cot_mix`:
50% OpenThoughts-114k-decontam + 50% gsm8k-r1cot via pretokenized
`/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192`) and the B3
`tulu_math_uc_mix` re-verification on CPT-FWE HF. Rationale per plan: Phase 2
levers are independent and must be scored first; RL has never moved agpt-2B
held-out CoT accuracy (gated GRPO on B2: 0.205 -> 0.215 noise, format 1.000) and
weak cold-starts reward-hack, so running RL before the broad cold-start is
evaluated wastes the budget and risks destabilizing the only 0.205 base.

**Action:** No RL `qsub`, no training launch. Re-check this gate only after
`submit_cot_eval.sh` reports `final/` CoT on the broad-mix and B3 candidates.

## VENV / stack check

Contract in the assignment prompt: Sunspot `rl-actors` venv
`py3.13 + torch 2.12 + xpu + monarch 0.5 + vLLM 0.22.1` with `ezpz_setup`
(machine-agnostic, no hard-coded `oneapi/2025.3.1`, `ZE_FLAT`, `CCL_OP_SYNC`,
`ONEAPI_DEVICE_SELECTOR`, `PPN`, `LD` shims) and `group_size 8-16`,
`generator.model_dtype=float32`.

Audited files (read verbatim 2026-11-15):

| Script | VENV it actually uses | `ezpz_setup`? | Hard-coded XPU env? | `group_size` source | `model_dtype` handling |
|---|---|---|---|---|---|
| `rl/scripts/grpo/agpt2b_grpo.sh` | `venvs/rl-grpo-lora` (`py3.12` per `build_rl_grpo_lora_venv.sh`) | **no** — `module load oneapi/release/2025.3.1`, `CCL_ROOT`, `LD_LIBRARY_PATH`, `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, `ONEAPI_DEVICE_SELECTOR`, `ZE_AFFINITY_MASK=0,1` hard-coded | yes | from config registry (`rl_grpo_lora_agpt_2b*` -> `group_size=4`) | via `AGPT_RL_PRECISION=float32` in `model_configs.py` (correct, but not CLI-visible) |
| `rl/scripts/grpo/agpt2b_grpo_cot*.sh` (cot, gated, long, zeroshot) | same `rl-grpo-lora` | no — identical hard-coded block | yes | same `group_size=4` | same |
| `rl/scripts/grpo/aurora2b_gsm8k_reason_grpo_{smoke,fast,xnode}.sh` | `venvs/rl-vllm` + `.venv/bin/ezpz` | **yes** — `source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job` | partial — `module load oneapi/release/2025.3.1` + `ZE_FLAT=FLAT` + `ONEAPI_DEVICE_SELECTOR` remain above the setup line (legacy SFT-era shim) | `--num_generations 4` (smoke/xnode) or `6` (fast) on `train_grpo` CLI | **fp32 mandatory** — vLLM server launched `fp32` (comment: "agpt-2b emits gibberish in bf16 through vLLM"); trainer `--bf16` is trainer-only |
| `rl/scripts/grpo/agpt2b_multihost_torchstore_validate.pbs` (current production gate) | `/lus/tegu/.../venvs/rl-monarch-torch214-locality-20261001` (repaired RECORD, Sunspot job 12479177 40/40 Exit 0) | **yes** — `source ...ezpz-utils && ezpz_setup "$V"` + `[[ $ZE_FLAT == FLAT ]] || exit 14` | no (delegated to `ezpz_setup`) | `--async-loop.num-samples-per-prompt=4` via `rl_grpo_lora_agpt_2b_gsm8k_b2smoke` | asserts `precision.param_dtype==float32` preflight (line 75) |
| `rl/xpu_trial/run_grpo.sh` (Aurora `prod` / Sunspot `workq` forward path) | `env_torch214.sh` (`oneapi/2026.1` + `gcc/14.3.0`, `ZE_FLAT=FLAT`) + `conda monarch_pbs` | via `env_torch214.sh` (no `ezpz_setup`; PALS `mpiexec` path) | no legacy 2025.3; uses `CCL_OFI_LIBRARY_PATH` from Cray libfabric, `FI_PROVIDER=cxi` on Aurora | config recipes `rl_grpo_qwen3_0_6b_*` (Qwen3-0.6B trial) | fp32 via recipe (Aurora trial not agpt-2B) |

**Findings / mismatches:**

1. **VENV name mismatch (expected `rl-actors` vs actual `rl-grpo-lora` /
   `rl-monarch-torch214-locality-20261001` / `rl-vllm`):** `rl-actors` (`py3.13`
   + torch 2.12 + vLLM 0.22.1) exists in `docs/journal.md` and
   `docs/production/rl/history/vllm-xpu-current-status.md` as the June 2026
   side-by-side isolation venv. It is **not** what any checked-in GRPO launcher
   currently points at. The Sunspot-validated production launcher
   (`agpt2b_multihost_torchstore_validate.pbs`) uses
   `rl-monarch-torch214-locality-20261001` (torch 2.14-era, oneAPI 2026.1 on the
   xpu_trial path). The legacy `agpt2b_grpo.sh` family still points at
   `rl-grpo-lora` (`py3.12`). Treating `rl-actors` as the canonical Sunspot VENV
   is stale against HEAD — update the contract or re-pin the launchers before any
   future RL wave. **No change made here; doc-only.**

2. **`ezpz_setup` mismatch:** `agpt2b_grpo.sh` + all `agpt2b_grpo_cot*.sh`
   variants do **not** use `ezpz_setup`/`ezpz_setup_job` and hard-code
   `oneapi/2025.3.1`, `CCL_ROOT`, `LD_LIBRARY_PATH`, `ZE_FLAT`, affinity, and
   `ONEAPI_DEVICE_SELECTOR`. This violates Golden Rule #1
   (machine-agnostic jobs via `ezpz_setup`). The TRL `aurora2b_*` scripts and
   the multihost PBS **do** use `ezpz_setup_job`/`ezpz_setup` correctly. Legacy
   scripts should be modernized to `source <(curl -fsSL https://ezpz.cool/utils.sh)
   && ezpz_setup /path/to/venv` + `ezpz launch` when RL is re-enabled.

3. **`group_size` mismatch:** Phase 3 plan says raise `group_size 8-16` (LR
   5e-6–1e-5) for the next B2 RL. All audited configs still set
   `group_size=4` / `num_generations=4` (or 6) and `num_samples_per_prompt=4`.
   Raising to 8–16 is an **open tuning item**, not yet landed. Do not treat
   current `group_size=4` smoke configs as the W2 RL recipe.

4. **`generator.model_dtype=float32` — PASS:** `rl/model_configs.py`
   defines `AGPT_RL_PRECISION = PrecisionConfig(dtype="float32",
   param_dtype="float32")` with docstring explaining bf16 gibberish
   (256k vocab + 11008 FFN flips greedy argmax). Generator mirrors trainer
   `precision.param_dtype` (post-#5776/#5894). Multihost validate PBS asserts it;
   TRL xnode launchers document "fp32 MANDATORY" and launch `vllm-serve` in fp32.
   No mismatch; keep `float32` on any future RL launch.

## What stays gated

- Any `qsub` of `rl/scripts/grpo/*.sh`, `*.pbs`, or `rl/xpu_trial/run_grpo.sh`
  for agpt-2B — including `agpt2b_grpo_cot_*.sh` on `checkpoint-93-hf`.
- Any new `CKPT=<cpt-fwe-hf> GRPO` until CoT on that SFT line exceeds 0.15 on
  `submit_cot_eval.sh` (`fp32` vLLM, `temperature 0.0 max_tokens 768`, 200-prob
  split).
- VENV or dtype changes outside `torchtitan/experiments/ezpz/` (Golden Rule #1).

## Re-enable checklist (when broad-CoT + B3 verify)

1. Re-score `cpt-fwe-74000/hf` B3 or broad-mix SFT `final/` via
   `torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh` — require
   CoT > 0.15 on the shared 200-prob split.
2. Modernize `agpt2b_grpo.sh` family to `ezpz.cool` + `ezpz_setup` (drop
   `module load oneapi/2025.3.1` / `ZE_FLAT` / `CCL_ROOT` hard-codes) or launch
   via `agpt2b_multihost_torchstore_validate.pbs` / `xpu_trial` forward path.
3. Bump `group_size` to 8–16 and LR to 5e-6–1e-5 per Phase 3; keep
   `generator.model_dtype=float32` and `--no-drop-zero-std-reward-groups`.
4. Use `rl-actors` only if re-validated at `py3.13+torch2.12+xpu+monarch0.5+vLLM0.22.1`;
   otherwise run the validated `rl-monarch-torch214-locality-20261001`
   (Sunspot) / `env_torch214.sh` (Aurora) stack that passed `12479177`.

## References

- `docs/production/POST-TRAINING-PLAN.md` Phase 3 (gate 0.15, reward-hacking lineage).
- `docs/production/POST-TRAINING-2B.md` — B2 0.205, CPT raw 0/200, B2v2 0.09–0.105, GRPO 0.205->0.215 noise.
- `rl/model_configs.py:28` (`AGPT_RL_PRECISION` float32), `rl/reason_agpt/config_registry.py` (group_size=4), `rl/alphabet_sort_agpt/config_registry.py`.
- `rl/scripts/grpo/agpt2b_grpo.sh:18-32`, `agpt2b_multihost_torchstore_validate.pbs:12-18`, `rl/xpu_trial/env_torch214.sh`, `docs/production/rl/monarch.md`.
