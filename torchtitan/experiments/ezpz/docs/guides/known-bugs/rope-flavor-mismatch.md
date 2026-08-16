# RoPE flavor mismatch at convert time silently corrupts an export

> **Canonical record** for this failure mode. Found 2026-08-16 while
> investigating a different question ([exp04](../../production/agpt/30b-exp/exp04-fp32-inference-investigation.md));
> mitigated the same day. **No production eval was affected** -- see
> [Blast radius](#blast-radius).

## The bug

`convert_to_hf.py` decides the RoPE convention from the `--model_flavor` you
pass, **not from the checkpoint**. It cannot do otherwise: both rope caches are
registered `persistent=False` (`torchtitan/models/common/rope.py:115`), so
nothing about the convention reaches disk. A checkpoint trained with cos_sin
RoPE and a checkpoint trained with complex RoPE are byte-indistinguishable.

`AgptStateDictAdapter` (`agpt/state_dict_adapter.py:39-62`) branches on it:

```python
rope = model_config.layers[0].attention.rope     # from the FLAVOR you passed
self._is_cos_sin = isinstance(rope, CosSinRoPE.Config)
...
def to_hf(self, state_dict):
    if not self._is_cos_sin:
        return super().to_hf(state_dict)   # applies the Q/K permute
    # cos_sin: HF-native rotate_half layout, NO permute
```

Pass the wrong flavor and the permute is applied when it should be skipped (or
vice versa). The export **loads without error** -- shapes and key names are
identical -- and only fails as gibberish at generation time. The adapter's own
header comment already warned about this; nothing enforced it.

## Why it went live

Production 2B/20B switched to the cos_sin (`_real`) flavor in `5ffb850a1`
(2026-06-25), which set `CONFIG_SUFFIX="${CONFIG_SUFFIX-_real}"` in the
autoretry submit scripts. `eval-2b-v2.sh:69` kept defaulting to
`MODEL_FLAVOR=2b` -- the **complex** flavor. Nobody noticed because the two
halves live in different files and the failure is silent.

`polaris_20b_eval_{sweep,step}.sh` got this right by hardcoding `20b_real`.

## Blast radius

**MEASURED -- no completed 2B eval was corrupted.** The clone that produced
both finished 4.674T chains, `runs/agpt-2b-v2/torchtitan-ezpz`, is pinned at
`f319e3fa`, which **predates** the `_real` default. It carries only the legacy
`*_aurora_venv_failover.sh` scripts, and its line 182 is
`--config="agpt_${MODEL}${CONFIG_SUFFIX:-}"` with **no assignment to
`CONFIG_SUFFIX` anywhere in the file** -- so it launched `--config=agpt_2b`:
complex RoPE, exactly what `MODEL_FLAVOR=2b` converts.

Independently corroborated by the eval curves themselves: HellaSwag rises
monotonically **0.405 -> 0.561** across the completed chain. Scrambled Q/K
pairing cannot produce a clean monotone learning curve. This also closes off an
attractive-but-wrong explanation for the near-chance MMLU -- **it is not a
conversion artifact.**

**Exposed:** every chain trained after 2026-06-25.

| clone | `CONFIG_SUFFIX` | trained RoPE | old eval default |
|---|---|---|---|
| `runs/agpt-2b-v2` | *absent* | complex | correct |
| `runs/agpt-20b-v2` | `_real` | cos_sin | **WRONG** |
| `runs/agpt-2b-constlr-from9200` | `_real` | cos_sin | **WRONG** |

The 20B chains and both constant-LR forks would have converted incorrectly on
the next eval sweep, and it would have read as a capability regression rather
than a bug.

## Mitigation (2026-08-16)

The checkpoint cannot be interrogated, so the fix is to make a wrong flavor
**loud and auditable** instead of silent.

**1. `eval-2b-v2.sh` infers the flavor from the checkpoint name** rather than
defaulting to a constant, and prints which it chose:

```bash
case "$V2_CKPT_NAME" in
    agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144|agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288)
        MODEL_FLAVOR="2b" ;;      # pre-_real clone: complex
    *)
        MODEL_FLAVOR="2b_real" ;; # everything newer: cos_sin
esac
```

**2. `convert_to_hf.py` announces the convention it is about to use**, and
warns when a non-`_real` flavor is used (the usually-wrong case post-2026-06-25):

```
[convert_to_hf] flavor='2b' -> RoPE=complex (Q/K permute APPLIED). If this
does not match how the checkpoint was TRAINED, the export is silently corrupt
```

**3. Every export now writes `ezpz_export.json`** beside the weights recording
`source_dcp`, `model_flavor`, `rope`, and `export_dtype` -- so an existing HF
directory can be audited after the fact, which was previously impossible.

## What would fix this properly

Record the RoPE backend **in the checkpoint** -- either a non-persistent-free
marker buffer or a field in the training config saved alongside the shards --
and have `convert_to_hf.py` read it and *reject* a contradicting
`--model_flavor`. The mitigation above narrows the window; it does not close
it, because a caller can still pass an explicit wrong flavor.

Requires a training-side change, so it is not free, and every existing
checkpoint would still need the name-based fallback.

## Detecting a bad export you already have

Generate ~50 greedy tokens. A RoPE mismatch produces fluent-looking token
salad -- correct vocabulary, no coherence -- not a crash and not repetition.
If `ezpz_export.json` is absent the export predates 2026-08-16; check the
producing chain against the table above.

## Related

- [`exp04-fp32-inference-investigation.md`](../../production/agpt/30b-exp/exp04-fp32-inference-investigation.md)
  -- where this was found (H3), including the separate and still-unisolated
  vLLM bf16 gibberish, which is **not** this bug.
- [`state_dict_adapter.py`](../../../agpt/state_dict_adapter.py) -- the branch.
- `agpt/__init__.py:773-774` -- `2b_real` / `20b_real` registration.
