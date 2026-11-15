# sft_distill_cot_4n verification — stub

Canonical note lives at `torchtitan/experiments/ezpz/docs/production/sft_distill_cot_4n_verification.md`.

`sft_distill_cot_4n.sh` and `sft_distill_cot_pretok_8n.sh` are byte-identical; pretok gate (`PRETOK_DIR` + `dataset_info.json` + `HF_OFFLINE`), `BASE_MODEL=checkpoint-900-hf`, locality VENV, and ~1000-step guardrail are covered in the canonical doc. No training launched.
