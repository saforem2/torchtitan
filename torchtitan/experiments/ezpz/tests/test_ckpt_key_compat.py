"""Test the flat-attention key remap against the REAL key sets on disk.

Checks the round-trip against both fork seeds:
  from9500 -- pre-refactor FLAT keys (the broken one)
  from9200 -- current NESTED keys (must be untouched)
"""
import sys
sys.path.insert(0, ".")
from torchtitan.experiments.ezpz.ckpt_key_compat import (  # noqa: E402
    to_flat, to_nested, needs_flat_attention_compat,
    head_to_old, head_to_new, needs_output_head_compat,
)
from torch.distributed.checkpoint import FileSystemReader  # noqa: E402

BASE = "outputs/checkpoints"
FLAT = f"{BASE}/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144-constlr-from9500/step-9500"
NESTED = f"{BASE}/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9200/step-9200"
# Written by CURRENT code, so it needs neither rename -- the negative control
# proving the shims stay off when they should.
CURRENT = f"{BASE}/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9200/step-20600"

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
print("=== unit: head rename (output <-> lm_head) ===")
check("lm_head->output model", head_to_old("lm_head.weight") == "output.weight")
check("output->lm_head model", head_to_new("output.weight") == "lm_head.weight")
check("head rename is anchored: a nested .output. is NOT touched",
      head_to_new("layers.0.moe.output.weight")
      == "layers.0.moe.output.weight")
check("head rename reaches optimizer FQNs",
      head_to_new("optimizer.param_groups.output.weight.betas")
      == "optimizer.param_groups.lm_head.weight.betas")
check("head round-trip",
      head_to_old(head_to_new("output.weight")) == "output.weight")
check("a key with neither spelling is untouched",
      head_to_new("norm.weight") == "norm.weight")

print()
print("=== head detection on the real checkpoints ===")
check("from9500 needs the head remap", needs_output_head_compat(FLAT))
check("missing path returns False (no raise)",
      not needs_output_head_compat("/nonexistent/step-1"))

print()
print("=== the two renames are INDEPENDENT ===")
# from9500 predates both; a current-code checkpoint predates neither. If these
# ever collapse into one flag, an old-attention/new-head checkpoint silently
# gets the wrong remap.
import os  # noqa: E402
for label, path in (("from9500", FLAT), ("from9200-20600", CURRENT)):
    if not os.path.isdir(path):
        print(f"  [skip] {label}: {path} not on this host")
        continue
    a, h = needs_flat_attention_compat(path), needs_output_head_compat(path)
    print(f"        {label}: attention={a} head={h}")
    if label == "from9500":
        check("from9500 needs BOTH remaps", a and h)
    else:
        check("a current-code checkpoint needs NEITHER", not a and not h)

print()
if fails:
    print(f"FAILED: {fails}")
    sys.exit(1)
print("ALL PASS -- both remaps cover the real key sets")
