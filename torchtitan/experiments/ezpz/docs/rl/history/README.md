# RL bring-up history

Superseded historical records from the 2026-06 RL/vLLM-XPU bring-up. Kept for
the landmine record; **not current status**. For where GRPO on XPU stands
today, see [`../grpo-on-xpu-status.md`](../grpo-on-xpu-status.md) and
[`../2026-07-06_multinode-grpo-root-cause.md`](../2026-07-06_multinode-grpo-root-cause.md).

| Doc | Date | What it captured |
|---|---|---|
| [`vllm-xpu-investigation.md`](vllm-xpu-investigation.md) | 2026-06-10 | Original vLLM-XPU feasibility survey + sibling-venv recipe. |
| [`vllm-xpu-wiring-plan.md`](vllm-xpu-wiring-plan.md) | 2026-06-13 | Pre-implementation architecture decision (TRL `vllm_mode="server"` vs Monarch+TorchStore). Now implemented. |
| [`vllm-xpu-current-status.md`](vllm-xpu-current-status.md) | 2026-06-13 | The 15-job bare-vLLM debug chain that led to the venv design. ("current" in the filename is historical.) |
| [`upstream-rl-port-status.md`](upstream-rl-port-status.md) | 2026-06-13 | Why running upstream `torchtitan.experiments.rl` directly (Monarch+TorchStore) is blocked on the Sunspot stack. |
| [`2026-06-14_monarch-torch213-deep-dive.md`](2026-06-14_monarch-torch213-deep-dive.md) | 2026-06-14 | torch 2.13 + monarch + vllm-xpu push; final wall was vLLM `profile_run` -> oneDNN `could not create a memory` on `F.linear`. |
