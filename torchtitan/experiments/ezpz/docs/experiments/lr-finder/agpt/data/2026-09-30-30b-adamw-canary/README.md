# 30B AdamW bounded canary evidence

Job `12479108` ran the canonical 26.20B OLMo-2-tokenizer model on Sunspot with repaired Torch 2.15, 192 ranks on 16 nodes, HSDP `dp_replicate=3` × `dp_shard=64`, sequence length 4,096, and GBS 960 sequences (3,932,160 tokens). Source commit: `7cc86ebc750d8455e1b8a0f6dc6177797f0d010b`.

The sweep completed ten finite AdamW updates from `1e-7` through `1e-4`; PBS exit was 0. The sampled smoothed-loss minimum is `11.677259086400811` at `1.0e-5`. Loss rose at all three higher samples (`12.1063`, `12.7745`, `12.9396`), so the coarse canary brackets a basin. The runner's logged `6.70e-7` suggestion is a safety-scaled detector candidate derived from a `6.70e-6` crossing; it is not the empirical minimum and is not a production recommendation.

This is bounded coarse evidence only. A matched fixed-LR validation remains required before releasing an optimizer matrix or publishing a production LR.

| artifact | SHA-256 |
|---|---|
| `sunspot-12479108-30b-adamw-canary.csv` | `17bbfbd17ca9e9bdc175a9445b3afbe75b807cc46c8c013c57e8eefef0f19c48` |
| `sunspot-12479108-30b-adamw-canary.png` | `b99636c817f3e56d3210916fd4bbc91ceb38dbb6dd7da72ca4cc6950370e7ee7` |
