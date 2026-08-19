#!/usr/bin/env python3
"""Drive the optimizer-key swap against the REAL step-9500 seed's metadata.

The swap has unit tests, but they use a FAKE optimizer container. This
exercises the actual translation against the actual on-disk key set: build the
exact request set current code would emit, push it through the shim's rename,
and assert every requested key exists in the checkpoint.

That is precisely the check DCP's `create_default_local_load_plan` performs
before it raises `Missing key in checkpoint state_dict` -- so if this passes,
that specific failure cannot recur for this checkpoint.

What it does NOT prove: that a full 3072-rank resume succeeds. Only that the
key layer is correct. Run standalone; needs torch for FileSystemReader.
"""
import importlib.util
import os
import sys

CKPT = (
    "/flare/AuroraGPT/foremans/runs/agpt-2b-constlr-from9200/torchtitan-ezpz/"
    "outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144-constlr-from9500/"
    "step-9500"
)

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "ckc", os.path.join(_here, "..", "ckpt_key_compat.py")
)
ckc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ckc)

from torch.distributed.checkpoint import FileSystemReader  # noqa: E402

disk = set(FileSystemReader(CKPT).read_metadata().state_dict_metadata.keys())
print(f"checkpoint keys on disk: {len(disk)}")

attention = ckc.needs_flat_attention_compat(CKPT)
head = ckc.needs_output_head_compat(CKPT)
print(f"detector: attention={attention} head={head}")
if not (attention or head):
    print("FAIL: detector says no remap needed, but this seed predates both")
    sys.exit(1)


def down(key: str) -> str:
    k = ckc.to_flat(key) if attention else key
    return ckc.head_to_old(k) if head else k


# Current code asks for the NEW spelling of every key the checkpoint holds.
# Model keys arrive bare; optimizer keys arrive with an `optimizer.` prefix
# added by DCP's flattener, and the container itself emits them without it.
requested_model, requested_opt = [], []
for k in disk:
    up = ckc.to_nested(k) if attention else k
    up = ckc.head_to_new(up) if head else up
    (requested_opt if k.startswith("optimizer.") else requested_model).append(up)

print(f"requested model keys: {len(requested_model)}")
print(f"requested optimizer keys: {len(requested_opt)}")

# MODEL path: the wrapper renames these directly.
missing_model = sorted(k for k in (down(k) for k in requested_model) if k not in disk)

# OPTIMIZER path: the container emits CONTAINER-level keys (no `optimizer.`
# prefix); the swap renames those, then DCP re-adds the prefix.
missing_opt = []
for up in requested_opt:
    container_key = up[len("optimizer."):]
    renamed = "optimizer." + down(container_key)
    if renamed not in disk:
        missing_opt.append(renamed)

print(f"\nmissing after rename -- model: {len(missing_model)}  optimizer: {len(missing_opt)}")
for k in (missing_model + missing_opt)[:8]:
    print("   MISSING:", k)

if missing_model or missing_opt:
    print("\nFAIL: the shim would still raise 'Missing key in checkpoint state_dict'")
    sys.exit(1)
print("\nPASS: every requested key resolves to a key that exists on disk")
