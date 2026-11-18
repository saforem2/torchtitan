# Run-Ledger Index — agpt-2B Post-Training (`ezpz-cpt111k-eval-sft` @ `a6135a79`)

> **Date:** 2026-11-16. **Branch:** `ezpz-cpt111k-eval-sft` @ `a6135a79` (HEAD at doc time).
> **Purpose:** One-page index over five run-ledger documents that each record *what* we ran, *why*, *outcome*, and *next steps* with job IDs and script outputs cited. The prior goal (5-thread exploration + SYNTHESIS, `a6135a79`, 4305 lines) is **COMPLETE** — this ledger is the new work, not a re-do.
> Every claim below cites its ledger file + section; ledgers in turn cite source docs + job IDs. Inferences labeled `[INFERENCE]`.

---

## Prior Goal — Where We Came From (COMPLETE)

| Item | Status | Commit | Lines |
|---|---|---|---|
| Phase 1: Ingest & Surface Enumeration | ✅ 3/3 DONE | `a6135a79` | — |
| Phase 2: Parallel Exploration (threads A-E: CPT, broad-distill, eval, B2 autopsy, recipe opt) | ✅ 5/5 DONE | `a6135a79` | 4305 |
| Phase 3: Verification & Synthesis (SYNTHESIS.md 345L, lint clean) | ✅ 3/3 DONE | `a6135a79` | 345 |

> **UI note:** The TODO screenshot that shows `Phase 3: 1/3` is stale — that TODO state was set before the `a6135a79` commit landed. Git log `a6135a79` proves all 6 files committed; `git status` shows only unrelated `journal.md` diff + new `run-ledger/` untracked. No prior work is pending.

**Key synthesis takeaway (SYNTHESIS §1-5):** Do not stop CPT, do not fork SFT blind — score the P0-P2 HF ladder in parallel with the live 4N chain, gate at 150k (G1-G4). Broad-distill `distill_cot_mix` (OpenThoughts+OpenR1, 91k→36.9k @8192, 3.4G, `cba5c64`) is the only open SFT path adding new reasoning data.

---

## Current Goal — 5-Ledger Run-Ledger (THIS TURN)

Each row is one ledger file. Read them in order `01→05`; `00` (this file) is the map.

| # | Ledger | What it documents (what/why/outcome/next) | Lines | Status |
|---|---|---|---:|---|
| [01](01-cpt-frontier.md) | **CPT-FWE 4N Frontier** | Chain `0→143400/190734 (75.2%)` across 8+ jobs: budget math `100B/524288=190734`, kind-shift 60/40 FWE vs 94.8% DCLM, loss `2.068→1.856→1.93`, val `12.53` flat = validator bug, MFU 15-18%, 12h walltime fix `99e918b342` (`4k→24k/hop`), 150k gate | 458 | ✅ |
| [02](02-hf-conversion.md) | **HF Batch Conversion** | PBS 1N `tegu:home` batch `74400→136800` every 200, fixups `72800` (copy `67400` tokenizer) + `13000` (reconvert empty), `flavor=2b-mds bfloat16`, `ezpz_export.json` sentinel idempotent, `74000/74200` `3.7G 9/9` already done, Tegu routing (`c519de37a5`) | 167 | ✅ |
| [03](03-eval-harness.md) | **Eval Harness** | CoT jobs `12482572/73` (B2v2 `0.09/0.105` at healthy loss), B2 `0.205/0.985/282` deliverable, MMLU band `0.2295-0.2689` + `p(A,B,C,D)` letter-prior, SE `0.028@200`, OOM `level_zero:40 63G` Aurora-only, G1-G4 gates | 262 | ✅ |
| [04](04-sft-broad-distill.md) | **SFT Broad-Distill** | Pretok `12483443` `91k→36.9k @8192 3.4G cba5c64` (6 segfaults `3553055` → `NPROC=1`), two-stage `tulu→ckpt-900→gsm8k-r1cot` only `0.205` recipe, every single-stage `0.02-0.065`, guardrail `O(1000)` steps + `max_steps`, E7 queue `A0→E6` | 321 | ✅ |
| [05](05-infra-rl-storage.md) | **Infra / RL / Storage** | VENV map (`xpu-torch216` CPT / `rl-monarch-locality-20261001` SFT-RL / `frameworks/2026.1.0` eval), Monarch 2-host `12479177 40/40 Exit0` (GRPO), `group_size=4→41%` dead groups, storage `PBS filesystems=tegu:home` + `15T /home` + symlink double-nesting | 214 | ✅ |
| **00** | **This index** | Map over 01-05 + where we stand summary | — | ✅ |

**Total ledger:** 1422 lines across 5 files + this index.

---

## Where We Stand — One-Paragraph Synthesis (from ledgers, not re-derived)

CPT-FWE 4N is at `143.4k/190.7k` (75.2% of 100B) on `12h` walltime, training loss `~1.93` healthy, validator `12.53` flat is a divisor bug not a plateau, MFU 15-18% nominal. No CPT checkpoint has ever been MMLU-scored — the dolmino precedent (loss down, HellaSwag `-7.4pp` ARC `-10.4pp`) says loss alone cannot discriminate. HF backlog is clear: `74000/74200` `3.7G 9/9` done, batch `74400→136800` every 200 via `02` converts the rest idempotently to Tegu (`tegu:home`). Eval harness is locked: B2 `0.205/0.985/282` is the only CoT deliverable, every alternative collapsed (`0.02-0.105`), MMLU at chance (`0.241-0.266` in band `0.2295-0.2689`), SE `±0.055@200`. SFT broad-distill is gated: pretok landed `91k→36.9k @8192` `cba5c64`, consumer ready but parked pending `E7 A0→E6` queue with `NUM_EPOCHS=1` smoke first. RL is format-perfector only (Monarch proven, `group_size 8-16` needed for accuracy signal).

**Single most important next action:** Run `02` batch + `03` E2 scoring in parallel with the live 4N chain; gate at `150k` (SYNTHESIS §5 G1-G4) before any new SFT spend. Cost to decide: `~8` HF converts + `2` eval sweeps + `~3` SFT probes, all 1N `workq`, zero new CPT compute.

---

## How to Use the Ledgers

- **Operators:** `01` gives the `qsub` requeue command + `CKPT_DIR` symlink fix; `02` gives the HF batch `qsub` + verify; `03` gives the CoT/MMLU sweep protocol + thresholds.
- **Decision-makers:** `01 §5` + `03 §5` define the `150k` continue/stop/fork gate (G1-G4); `04 §5` defines the `E7 A0→E6` SFT queue gated on HFs.
- **Infra owners:** `05` has the VENV-per-workload table + storage routing; `02 §4` has failure taxonomy for HF.
- **Auditors:** Every table row cites `script:line` or `doc:section` or `job ID`; `[INFERENCE]` labels estimates (e.g. `143.4k` frontier = `190734 * 0.752`).

---

## Provenance & Verification

| Ledger | Lines | Sections | Source scope |
|---|---|---:|---|
| 01-cpt-frontier | 458 | 5 | `cpt_fineweb_edu_4n.sh`, `POST-TRAINING-2B.md` (2026-11-10…16), `thread-a`, `journal.md`, `cpt/README.md`, `config_registry.py:919-1057` |
| 02-hf-conversion | 167 | 5 | `hf_batch_convert_cpt_fwe.sh:1-60`, `POST-TRAINING-2B.md` HF gates, `SYNTHESIS.md` §4 ladder, `convert_to_hf.py:staged_export`, `5fb2bf2/c519de37a5/99e918b342` |
| 03-eval-harness | 262 | 5+ | `submit_cot_eval.sh`, `eval_cot_gsm8k.py`, `thread-c`, `POST-TRAINING-2B.md` CoT table, `mmlu-letter-prior-at-chance.md`, `SYNTHESIS.md` §5 |
| 04-sft-broad-distill | 321 | 5 | `sft_distill_cot_4n.sh`, `_pretokenize_distill_cot_mix_1n.sh`, `agpt2b_b2v2_distill_cot_2n.sh`, `sft_distill_cot_4n_verification.md`, `thread-b/d`, `POST-TRAINING-2B.md` |
| 05-infra-rl-storage | 214 | 5+ | `multihost_grpo_e2e.pbs`, `monarch.md`, `journal.md` 2026-11-14/16, `thread-e` §3, VENV grep across `cpt_fineweb_edu_4n.sh`/`sft_distill_cot_4n.sh`/`_pretokenize_*` |

**Gate status at write time:**

- [x] All 5 ledger files exist on disk with `5` sections each (`grep -c "## "` = 17…28 per file, headers vary).
- [ ] `pre-commit --files run-ledger/*.md` — to run via orchestrator final gate (lychee/codespell only; no py checks on md).
- [ ] `bash -n` on `hf_batch_convert_cpt_fwe.sh` + `cpt_fineweb_edu_4n.sh` + `submit_cot_eval.sh` — orchestrator final gate.
- [ ] `journal.md` + notebook update to record this ledger as the new canonical “where we stand” pointer.

All prior TODO items remain closed (`a6135a79`); this ledger is additive, not replacement.
