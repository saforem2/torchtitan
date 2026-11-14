# Zephon elastic dataloader examples

Weighted HF streaming at runtime with elastic checkpoint/resume.

See [`torchtitan_recipes/zephon/dataloader.py`](../../torchtitan_recipes/zephon/dataloader.py)
(ported from `datologyai/torchtitan-zephon`) and
[`torchtitan_recipes/zephon/recipes.py`](../../torchtitan_recipes/zephon/recipes.py).

## Quick start

```bash
pip install zephon          # or zephon[hf,cloud] for hf:// and s3:// sources
```

```python
from torchtitan_recipes.zephon.dataloader import ZephonDataLoader

config.dataloader = ZephonDataLoader.Config.from_toml(
    "examples/zephon/local_jsonl.toml",
    aggregate_store="torch://",   # or aggregate_dir="/shared/zephon"
    run_id="my-run-42",
)
```

## TOML recipes

| File | Purpose |
|------|---------|
| `local_jsonl.toml` | Two local sources, token-aware mixing (smoke) |
| `elastic_local_jsonl.toml` | Same, plus `canonical_replicas=2` for elastic resume |
| `validation_local_jsonl.toml` | Deterministic, non-repeating validation source |
| `mixture.toml` | Weighted `hf://` streaming mixture (replace with your own) |

Relative `path` values resolve from the TOML file's directory.
`weight` is a relative token proportion (token-aware mixing on by default
when there are multiple sources).

## Notes

- `aggregate_store="torch://"` uses torchrun's TCPStore; `aggregate_dir` uses a shared filesystem.
- Distributed runs require `aggregate_store`/`aggregate_dir` and `run_id`.
- The checkpoint is stored as opaque bytes in TorchTitan's DCP state.
