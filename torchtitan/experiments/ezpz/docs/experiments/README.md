# Experiment Benchmark Reports

Benchmark results organized by experiment type and machine.

## Experiments

- [**AGPT-2B Inkling synthetic-data distillation**](2026-09-28-agpt2b-inkling-distillation.md) -- 5,000-prompt ALCF Minerva teacher-generation campaign, 4,115 accepted examples, bounded Stage-9 SFT, and checkpoint evaluation.
- [**AGPT-2B Stage-4 MetaMath distillation**](2026-09-26-agpt2b-stage4-metamath.md) -- MetaMath GSM distillation, capability-retention interpolation, and accepted checkpoint lineage.
- [**agpt/**](agpt/) -- Dense AuroraGPT models (2B, 7B, 20B, 80B)
- [**moe/**](moe/) -- Mixture of Experts models (500M--10B)
- [**lr-finder/**](lr-finder/) -- Learning rate finder sweeps across models and optimizers
- [**synthetic/**](synthetic/) -- Synthetic mid-training data (summarize olmo-mix-1124)

## Naming Convention

Report filenames follow: `YYYYMMDD-HHMMSS-<description>-n<nodes>.md`

| Field | Example | Meaning |
|-------|---------|---------|
| Date | `20260412` | Run date |
| Time | `002800` | Run start time (UTC-5) |
| Description | `smoke` | Test type (smoke, baseline, perf, prod-sim) |
| Nodes | `n2` | Number of compute nodes |
