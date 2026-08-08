"""Test the flat-attention key remap against the REAL key sets on disk.

Checks the round-trip against both fork seeds:
  from9500 -- pre-refactor FLAT keys (the broken one)
  from9200 -- current NESTED keys (must be untouched)
"""
import sys
sys.path.insert(0, ".")
from torchtitan.experiments.ezpz.ckpt_key_compat import (  # noqa: E402
    to_flat, to_nested, needs_flat_attention_compat,
)
from torch.distributed.checkpoint import FileSystemReader  # noqa: E402

BASE = "outputs/checkpoints"
FLAT = f"{BASE}/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144-constlr-from9500/step-9500"
NESTED = f"{BASE}/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9200/step-9200"

fails = []


def check(name, cond, detail=""):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


print("=== unit: rename directions ===")
check("nested->flat model",
      to_flat("layers.0.attention.qkv_linear.wk.weight")
      == "layers.0.attention.wk.weight")
check("flat->nested model",
      to_nested("layers.0.attention.wk.weight")
      == "layers.0.attention.qkv_linear.wk.weight")
check("nested->flat optimizer",
      to_flat("optimizer.param_groups.layers.3.attention.qkv_linear.wq.weight.betas")
      == "optimizer.param_groups.layers.3.attention.wq.weight.betas")
check("wo is NOT touched (stayed outside the wrapper)",
      to_nested("layers.0.attention.wo.weight") == "layers.0.attention.wo.weight")
check("non-attention keys untouched",
      to_nested("layers.0.feed_forward.w1.weight")
      == "layers.0.feed_forward.w1.weight")
check("to_nested is idempotent",
      to_nested(to_nested("layers.0.attention.wk.weight"))
      == "layers.0.attention.qkv_linear.wk.weight")
check("round-trip flat->nested->flat",
      to_flat(to_nested("layers.7.attention.wv.weight"))
      == "layers.7.attention.wv.weight")

print()
print("=== detection on the real seeds ===")
check("from9500 detected as needing compat", needs_flat_attention_compat(FLAT))
check("from9200 detected as NOT needing compat",
      not needs_flat_attention_compat(NESTED))
check("missing path returns False (no raise)",
      not needs_flat_attention_compat("/nonexistent/step-1"))

print()
print("=== full key-set round-trip against the real checkpoints ===")
for label, path, expect_flat in (("from9500", FLAT, True), ("from9200", NESTED, False)):
    keys = set(FileSystemReader(path).read_metadata().state_dict_metadata.keys())
    # Model production behaviour: the shim is installed ONLY when detection
    # says the checkpoint needs it. Applying the remap to a checkpoint that
    # does not need it is not a scenario that occurs, so do not assert it.
    shim_on = needs_flat_attention_compat(path)
    # The current model always requests the nested spelling.
    requested = {to_nested(k) for k in keys}
    # With the shim on, requests are renamed down to disk spelling; without
    # it they go through as-is.
    sent = {to_flat(k) for k in requested} if shim_on else requested
    missing = sent - keys
    extra = keys - sent
    check(f"{label}: shim {'ON' if shim_on else 'OFF'} -> every request exists on disk",
          not missing, f"{len(keys)} keys" + (f", missing {list(missing)[:2]}" if missing else ""))
    check(f"{label}: no orphaned disk keys", not extra,
          f"orphans {list(extra)[:2]}" if extra else "")
    att = sorted(k for k in keys if "layers.0.attention" in k and k.endswith(".weight"))
    print(f"        layout: {att[:3]}")
    check(f"{label}: layout is {'FLAT' if expect_flat else 'NESTED'} as expected",
          expect_flat == (not any("qkv_linear" in k for k in keys)))

print()
if fails:
    print(f"FAILED: {fails}")
    sys.exit(1)
print("ALL PASS -- the remap covers both seeds' real key sets")
