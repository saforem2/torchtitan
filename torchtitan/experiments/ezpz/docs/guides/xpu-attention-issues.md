# XPU Attention Issues

**Last updated:** 2026-08-31

Summary of SDPA, Flash Attention, and FlexAttention issues on Intel XPU
(Aurora / Sunspot) with `aurora_frameworks-2025.3.1` (PyTorch 2.10.0a0).

## SDPA Backend Selection

### Problem

Intel XPU supports multiple SDPA backends with vastly different performance:

| Backend | Speed | Notes |
|---------|-------|-------|
| `OVERRIDEABLE` | **23x faster** | XPU fused attention kernel |
| `CUDNN_ATTENTION` | N/A | Not available on XPU |
| `FLASH_ATTENTION` | N/A | Not available on XPU |
| `MATH` | 1x (baseline) | Always works, slowest |

The `OVERRIDEABLE` backend provides the XPU fused kernel, but there's a
PyTorch bug where `sdpa_kernel()` context manager is **not respected inside
FSDP-wrapped modules** — the MATH backend is always used regardless of
settings.

### Impact

For models where the N×N attention matrix exceeds device memory (e.g. 80B
with 72 heads at seq_len=8192 → 9 GiB per tile), the MATH backend's
materialization of the full attention matrix requires TP to reduce per-rank
head count.

### Fix

`XPUScaledDotProductAttention` in `agpt/__init__.py` prioritizes the
`OVERRIDEABLE` backend:

```python
sdpa_backends = [
    SDPBackend.OVERRIDEABLE,     # XPU fused kernel (23x faster)
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.MATH,
]
```

All agpt configs use this automatically via `_default_inner_attention()`,
which selects `XPUScaledDotProductAttention` when `torch.xpu.is_available()`.

---

## torch.compile + `set_priority=True` (PyTorch 2.11)

### Problem

`sdpa_kernel(..., set_priority=True)` crashes during `torch.compile` tracing:

```
RuntimeError('Invalid backend')
```

`torch._dynamo` creates FX graph nodes that pass proxy objects to `int()`
when `set_priority=True`, producing invalid backend IDs for
`_set_sdp_priority_order()`. Only triggers in the full model context
(activation checkpointing + DTensor + distributed), not in standalone tests.

### Fix

`EzpzScaledDotProductAttention` overrides `forward()` to call
`sdpa_kernel(self.sdpa_backends)` **without** `set_priority=True`:

```python
with sdpa_kernel(self.sdpa_backends):
    out = F.scaled_dot_product_attention(
        q, k, v, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa
    )
```

---

## XPU SDPA Head Dimension Mismatch

### Problem

XPU's MATH backend returns output with Q/K's head dimension instead of V's
when they differ. Standard CUDA backends correctly return `(B, H, L, Ev)`
when `E != Ev`, but XPU does not.

This affects MoE models which use MLA (Multi-head Latent Attention) with
different dimensions for Q/K vs V via LoRA-based projection:

- `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` (e.g. 192)
- `v_head_dim` (e.g. 128)

### Fix

`moe/model.py` pads V to match Q/K head dimension before SDPA, then strips
after:

```python
pad_v = self.qk_head_dim != self.v_head_dim
if pad_v:
    v = F.pad(v, (0, self.qk_head_dim - self.v_head_dim))

output = self.inner_attention(q, k, v, ...)

if pad_v:
    output = output[..., :self.v_head_dim]
```

---

## FlexAttention + `torch.autocast(float32)` on XPU

### Problem

The MoE router (`models/common/moe.py:231`) computes gate scores in fp32:

```python
with torch.autocast(device_type=x.device.type, dtype=torch.float32):
    scores = self.gate(x)
```

XPU's `torch.autocast()` only supports `bf16` and `fp16`, **not `fp32`**:

```
UserWarning: In XPU autocast, but the target dtype is not supported.
Disabling autocast. XPU Autocast only supports dtypes of
torch.bfloat16, torch.float16 currently.
```

For SDPA configs this is a **non-fatal warning** (autocast silently disables).

> [!IMPORTANT]
> **CORRECTED 2026-08-19: the fp32 autocast warning was never what broke
> FlexAttention on XPU.** The warning above is real and still fires, but it is
> cosmetic on both paths. The flex configs were failing for two unrelated
> reasons, both since fixed:
>
> 1. blendcorpus never emitted `positions`, so core never built the
>    `BlockMask` and the model got `None` (`AssertionError: attention_masks
>    must be instance of BlockMask, got <class 'NoneType'>`).
> 2. the EOD token id was read off `_bc_cfg`, a blendcorpus-owned object that
>    does not expose it, so `getattr(..., None)` silently returned `None`.
>
> Fixing those exposed a third issue -- FullAC re-runs the MoE router on the
> recompute pass and reassigns tokens (560 vs 559) -- which selective AC
> solves, because it keeps `aten.topk.default` as MUST_SAVE. See
> [known-bugs/moe-flex-attention-blockmask.md](known-bugs/moe-flex-attention-blockmask.md).

### Affected configs

| Config | Attention | XPU status |
|--------|-----------|-----------|
| moe debugmodel, 500M, 2B, 4B, 7B | SDPA | OK (warning only) |
| `moe_10b_2b` | FlexAttention | **OK** (fixed 2026-08-19, 5/5 at 79.95% mem) |
| `moe_10b_2b_sdpa` | SDPA | **OK** (control) |
| `moe_small` | FlexAttention | **OK** (fixed 2026-08-19, 5/5 at 72.64% mem) |
| `moe_16b`, `moe_671b` | FlexAttention | untested since the fix |

Both fixed flex configs run with **no flags** (job 12473388, 2N, 5 steps,
defaults only). `moe_16b` and `moe_671b` are also `attn_backend="flex"` but
do not set `dataloader.emit_positions = True`, and neither has been run since
the fix -- their status here is "unknown", not "working". (There is no
`moe_236b` in the registry; the old row named a flavor that is not a config.)

<details closed><summary>Previous (incorrect) status table, kept for search</summary>

| Config | Attention | XPU status |
|--------|-----------|-----------|
| moe debugmodel, 500M, 2B, 4B, 7B | SDPA | OK (warning only) |
| moe 10B_2B | FlexAttention | **Broken** |
| moe 10B_2B_sdpa | SDPA | **OK** (workaround) |
| moe small, 16B, 236B, 671B | FlexAttention | **Broken** |

</details>

### The `_sdpa` variants

These predate the flex fix and were the workaround at the time. They are
**still useful, now as controls** rather than as the required path: they
passed 5/5 before and after both fixes (74.05% and 94.54% memory, unchanged),
which is the evidence that the maskless path was not perturbed. Keep them in
the test matrix for that reason.

```python
def _10b_2b_sdpa() -> moeModel.Config:
    cfg = _10b_2b()
    sdpa_cfg = ScaledDotProductAttention.Config()
    for layer_cfg in cfg.layers:
        layer_cfg.attention.inner_attention = sdpa_cfg
        layer_cfg.attention.mask_type = "causal"  # was "block_causal"
    return cfg
```

- `mask_type="causal"` for SDPA (standard causal masking via `is_causal=True`)
- `mask_type="block_causal"` for FlexAttention (block-sparse `BlockMask`)

---

## FlexAttention + Triton LLVM on XPU

### Problem (resolved upstream)

FlexAttention uses Triton kernels compiled via LLVM. An LLVM optimization
pass caused FlexAttention failures, leading to a temporary workaround:

```python
os.environ.setdefault("DISABLE_LLVM_OPT", "1")
```

### Resolution

Upstream PR pytorch/pytorch#179586 fixed the Triton pin, and torchtitan
commit `878041cb` removed the `DISABLE_LLVM_OPT` workaround.

This section used to end "not relevant for ezpz XPU configs (we use SDPA, not
FlexAttention)". That is no longer true: `moe_small` and `moe_10b_2b` run
FlexAttention on XPU as of 2026-08-19. The Triton/LLVM issue itself stays
resolved -- neither config needs `DISABLE_LLVM_OPT`.

---

## MoE `fullgraph=True` Compile Failure

### Problem

After upstream `00b7f569` removed `maybe_enable_amp` from the trainer's
`forward_backward_step`, MoE models with `fullgraph=True` hit the dynamo
recompilation limit:

```
FailOnRecompileLimitHit
```

The `maybe_enable_amp` context manager previously provided a graph boundary
that prevented recompilation across the dynamic MoE routing dispatch. Without
it, the router's top-k selection produces different tensor shapes on each
forward pass, triggering recompilation.

### Fix

`moe/parallelize.py` applies compile per-block without `fullgraph=True`:

```python
torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
for layer_id, block in model.layers.named_children():
    block.compile(backend=compile_config.backend)
    model.layers.register_module(layer_id, block)
```

### Note

Even with per-block compile, **torch.compile hurts MoE throughput by ~35%**
on XPU due to periodic recompilations from dynamic routing. The recommendation
is to **not use compile for MoE models**.

---

## Summary: What to Use on XPU

| Model type | Attention | Compile | Notes |
|-----------|-----------|---------|-------|
| agpt (dense) | `XPUScaledDotProductAttention` | **Yes** (+8-31%) | Auto-selected on XPU |
| MoE (SDPA variants) | `ScaledDotProductAttention` | **No** (-35%) | Dynamic routing breaks compile |
| MoE (FlexAttention) | `FlexAttention` | not separately measured | Works since 2026-08-19; needs `emit_positions` + selective AC, both already the config default |

FlexAttention MoE is no longer "not supported on XPU" -- that row said so
until 2026-08-19. `moe_small` and `moe_10b_2b` pass 5/5 with no flags
(job 12473388). Note that "defaults only" there includes `compile=True`
(the `moe()` default), so the flex path passes *with* compile on; the -35%
figure in the row above was measured on the SDPA variants and has not been
re-measured for flex. `moe_16b` and `moe_671b` are also flex but have not
been run since the fix.

### Code locations

| File | What |
|------|------|
| `agpt/__init__.py:36-96` | `EzpzScaledDotProductAttention`, `XPUScaledDotProductAttention` |
| `agpt/__init__.py:137-141` | `_default_inner_attention()` device selection |
| `moe/__init__.py:934-946` | `_10b_2b_sdpa()` FlexAttention→SDPA workaround |
| `moe/model.py:130-145` | V-padding for asymmetric head dims |
| `moe/parallelize.py:187-197` | Per-block compile without fullgraph |
| `models/common/moe.py:231` | fp32 autocast in router (XPU warning source) |
| `models/common/attention.py:267-283` | FlexAttention inductor configs |
