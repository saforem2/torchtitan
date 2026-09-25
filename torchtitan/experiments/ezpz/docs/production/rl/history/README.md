# RL bring-up history

Superseded historical records from the 2026-06 RL/vLLM-XPU bring-up. Kept for
the landmine record; **not current status**. For where GRPO on XPU stands
today, see
[`../monarch-torchstore-vllm.md`](../monarch-torchstore-vllm.md).

> [!WARNING]
> Every page in this directory is historical. Environment snippets here predate
> the current Aurora `next-eval` runtime. Start from
> [`../monarch-torchstore-vllm.md`](../monarch-torchstore-vllm.md), not from a
> command copied out of `history/`.

| Doc | Date | What it captured |
|---|---|---|
| [`2026-06-13-bringup-and-2026-07-01-desync.md`](2026-06-13-bringup-and-2026-07-01-desync.md) | 2026-06-13 / 07-01 | The full bring-up chronology (26-job debug chain, TCP-KVS fix, xpu_overrides shim, 1N smoke metrics) + the 2026-07-01 multi-node desync investigation (REFUTED 2026-07-06). Relocated from grpo-on-xpu-status.md. |
| [`vllm-xpu-investigation.md`](vllm-xpu-investigation.md) | 2026-06-10 | Original vLLM-XPU feasibility survey + sibling-venv recipe. |
| [`vllm-xpu-wiring-plan.md`](vllm-xpu-wiring-plan.md) | 2026-06-13 | Pre-implementation architecture decision (TRL `vllm_mode="server"` vs Monarch+TorchStore). Now implemented. |
| [`vllm-xpu-current-status.md`](vllm-xpu-current-status.md) | 2026-06-13 | The 15-job bare-vLLM debug chain that led to the venv design. ("current" in the filename is historical.) |
| [`upstream-rl-port-status.md`](upstream-rl-port-status.md) | 2026-06-13 | Why running upstream `torchtitan.experiments.rl` directly (Monarch+TorchStore) is blocked on the Sunspot stack. |
| [`2026-06-14_monarch-torch213-deep-dive.md`](2026-06-14_monarch-torch213-deep-dive.md) | 2026-06-14 | torch 2.13 + monarch + vllm-xpu push; final wall was vLLM `profile_run` -> oneDNN `could not create a memory` on `F.linear`. |
