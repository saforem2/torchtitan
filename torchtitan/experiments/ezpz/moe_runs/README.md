# ezpz MoE JSON runs

This directory contains current AGPT JSON launch configs plus older DeepSeek and
Polaris experiment records. The AGPT configs below have current importable
factories and are covered by `tests/test_json_launch_configs.py`. The older
DeepSeek/Polaris files are retained for provenance but still use legacy schema;
do not treat them as supported current launch inputs without a separate port.

## Retained AGPT runs

- `agpt_dense_2b_50k_hsdp256_50k.json` is loaded by
  `ezpz.agpt:agpt_2b_50k_from_json`.
- `agpt_moe_2b_50k_ep12_dp256_50k.json`,
  `agpt_2b_50k_moe_ep12_dp2_full_sonic_1100.json`, and
  `agpt_2b_50k_moe_ep12_1node_smoke.json` are loaded by
  `ezpz.moe:agpt_2b_50k_moe_sdpa_aurora_full_sonic_from_json`.

The dense factory uses the current AGPT `2B_50K` model registry entry. The MoE
factory uses the current `EzpzRoutedExperts` full-Sonic path, which forwards the
router outputs and EP mesh required by Sonic.

Activation checkpointing is selected by the factories, not by JSON: dense uses
`SelectiveAC`; MoE uses `MoeSelectiveAC`, whose policy does not save the EP
all-to-all output. JSON cannot replace a config-policy object with the removed
legacy `{ "mode": ... }` representation.

Canonical training JSONs set `checkpoint.keep_latest_k` to `0` (retain all).
Both AGPT and MoE JSON override paths reject nonzero values instead of silently
purging chain checkpoints.

## Legacy DeepSeek/Polaris records

The old `launch_deepseek_v3_moe_ep12.sh`, its 128-node PBS wrapper, and their
DeepSeek/Polaris JSON files are retained as experiment provenance. They select
`deepseek_v3_10b_2b_ep12_from_json`, but no such factory exists in the current
DeepSeek registry, and their payloads use removed schema (`training.seq_len`,
`training.local_batch_size`, an old nested `model_spec.model.layer` override,
and `{ "activation_checkpoint": { "mode": ... } }`). They are therefore not
current launch inputs; port them explicitly before reuse rather than silently
mapping them onto a different model or attention policy.
