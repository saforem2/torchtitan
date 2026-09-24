# AGPT learning-rate finder results

The AGPT LR-finder archive is split by model generation so results with
different architectures and scaling behavior are not mixed in one table.

| generation | model family | sizes | results |
|---|---|---|---|
| **[AGPT v1](agpt-v1/README.md)** | original dense AGPT | 2B, 20B, 80B | small-batch and production-batch ladders across Aurora, Sunspot, and Polaris |
| **[AGPT v2](agpt-v2/README.md)** | OLMo-3-vocab AGPT (`olmo2tok` historical config name) | 5B, 10B, 30B | current GBS=6144 coarse-to-fine campaign and prior 30B GBS=960 results |

For LR-finder methodology, configuration, and usage, see the
[methodology README](../README.md).
