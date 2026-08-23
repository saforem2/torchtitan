# hybridep on XPU: not a version floor, not portable

**Status:** WONTFIX -- upstream feature is NVIDIA-only by design.
**Config:** `moe_10b_2b_sdpa_hybridep`
**Job:** 12473367 (broad MoE sweep)

## What it looks like

```
ImportError: cannot import name 'CustomClassBase' from
             'torch._library.opaque_object'
  torchtitan/distributed/deepep/hybridep.py:27
```

That reads like a torch version floor -- our pinned torch
(`2.13.0a0+gitcf30153`) predates the `CustomClassBase` API. `register_opaque_type`
from the same module does exist; only `CustomClassBase` is missing. So the
tempting conclusion is "bump torch and hybridep works."

That conclusion is wrong.

## Why bumping torch would not help

Two further blockers sit behind the import error, and each is sufficient on
its own:

1. **`deep_ep` is not installed** -- `import deep_ep` raises
   `ModuleNotFoundError` in the RC4 env. hybridep is a thin wrapper over it.

2. **`deep_ep` is CUDA-only.** The module's own docstring is explicit:

   > HybridEP: Expert Parallel Communication for GB200 NVLink72 Systems.
   > Provides efficient token dispatch/combine for MoE training via
   > TMA-optimized all-to-all.

   and the config documentation describes calling `cudaStreamSynchronize`
   after dispatch. It targets NVLink72 fabric and TMA, neither of which
   exists on Intel Max 1550.

So `moe_10b_2b_sdpa_hybridep` is not a config we can fix -- it is a config
that cannot run on this hardware. The correct disposition is to exclude it
from XPU sweeps rather than carry it as an open bug.

## What to use instead

The working EP paths on XPU are the `standard` moe_comm_backend configs.
Note that the large EP configs have their own unrelated problem (an Intel
Unified Runtime abort in all-to-all); see
[moe-ep-a2a-degrades-with-size.md](./moe-ep-a2a-degrades-with-size.md).

## If someone revisits this

The order of blockers matters. Bumping torch clears only the import; you
would then hit the missing `deep_ep`, and building `deep_ep` would then hit
CUDA/NVSHMEM requirements. There is no partial win along that path, so do
not spend a torch bump on it.
