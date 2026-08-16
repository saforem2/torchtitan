# RoPE flavor mismatch at convert time silently corrupts an export

> **Which chain used which RoPE? -> [Registry](#registry-which-chain-trained-with-which-rope).**

> **Canonical record** for this failure mode. Found 2026-08-16 while
> investigating a different question ([exp04](../../production/agpt/30b-exp/exp04-fp32-inference-investigation.md));
> mitigated the same day, and the mitigation itself was **corrected** the same
> day after it shipped backwards -- see [Correction](#correction-2026-08-16).
> **No production eval is known to be affected**; see
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

## Registry: which chain trained with which RoPE

**This table is the canonical answer.** Derived from actual `--config=` launch
lines in the per-trainer console logs -- **not** from clone script defaults,
which is how the first version of this page got it exactly backwards (see
[Correction](#correction-2026-08-16)).

| checkpoint dir | launched config | RoPE | convert with |
|---|---|---|---|
| `agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288` **(completed 4.674T)** | `agpt_2b_real` | **cos_sin** | `2b_real` |
| `agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144` **(completed 4.674T)** | `agpt_2b_real` | **cos_sin** | `2b_real` |
| `agpt-2b-stage2-dolmino-n512-gbs12288` | `agpt_2b_real` | **cos_sin** | `2b_real` |
| `agpt-2b-...-n512-gbs12288-constlr-from9200` | `agpt_2b_real` | **cos_sin** | `2b_real` |
| `agpt-2b-...-n256-gbs6144-constlr-from9500` | `agpt_2b_real` | **cos_sin** | `2b_real` |
| `agpt-2b-stage2-olmo50dolmino50-const2e6-n256-gbs6144` | `agpt_2b_real` | **cos_sin** | `2b_real` |
| `agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288` | `agpt_20b_real` | **cos_sin** | `20b_real` |
| `agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144` | `agpt_20b_real` | **cos_sin** | `20b_real` |
| agpt **80B** | `agpt_80b` | **complex** | `80b` |
| MDS / Megatron-DeepSpeed bases | n/a | n/a -- different codebase | `2b-mds` |

**Short version: everything agpt on Aurora is cos_sin except 80B.** Verified
consistent across all six umbrellas back to `8663177` (2026-07-17), which is
every umbrella whose logs still exist.

80B is complex because it runs `compile` OFF -- the `_real` flavor exists to
give inductor something it can lower, and that win is moot without compile
(`5ffb850a1`).

### How to check a chain yourself

```bash
grep -ohm1 -- '--config=agpt_[a-z0-9_]*' \
  logs/multi-autoretry-<JOBID>/trainer-<N>-*.console.log
```

That is ground truth. **Do not infer from a clone's `CONFIG_SUFFIX`** -- a
clone can carry several submit scripts with different defaults, and the one you
grep may not be the one that ran.

## Correction (2026-08-16)

The first version of this page claimed the two completed 4.674T 2B chains
trained **complex**, and the first version of the `eval-2b-v2.sh` fix encoded
that as a name-based special case. **Both were wrong.**

The error: I read `CONFIG_SUFFIX` out of `runs/agpt-2b-v2`'s *legacy*
`submit_agpt_2b_aurora_venv_failover.sh` -- which indeed has no assignment --
and concluded the chain ran complex. But those chains ran under the **umbrella**
script (`submit_agpt_multi_autoretry.sh`), which has defaulted `_real` since
`5ffb850a1`. I inferred from the wrong file instead of checking a launch line.

The fix as first shipped would have converted both completed chains with the
**wrong** flavor -- the same corruption this page documents, freshly introduced
while purporting to prevent it. Corrected in the same commit series.

What still holds from the original analysis: HellaSwag rising monotonically
**0.405 -> 0.561** across the completed chain means those evals were **not**
corrupt, so near-chance MMLU is **not** a conversion artifact. That conclusion
was right; the reasoning offered for it was not.

## Blast radius

**No completed eval is known to be corrupt** -- the HellaSwag trajectory above
rules it out for the 2B chains, and the Polaris 20B scripts already hardcode
`20b_real`.

The exposure was the *default*: `eval-2b-v2.sh` defaulted to `2b` (complex)
while every chain it evaluates is cos_sin. Any sweep that did not pass
`MODEL_FLAVOR` explicitly would have produced a corrupt export that loads
cleanly and reads as a capability regression.

## Mitigation (2026-08-16)

The checkpoint cannot be interrogated, so the fix is to make a wrong flavor
**loud and auditable** instead of silent.

**1. `eval-2b-v2.sh` defaults to `2b_real`** -- correct for every agpt 2B chain
on Aurora -- and prints the flavor plus what would justify overriding it:

```bash
MODEL_FLAVOR="${MODEL_FLAVOR:-2b_real}"
echo "[eval-2b-v2] MODEL_FLAVOR='${MODEL_FLAVOR}' -- cos_sin RoPE is correct for"
echo "[eval-2b-v2]   every agpt 2B chain trained on Aurora. Override only if you"
echo "[eval-2b-v2]   have a launch line showing --config=agpt_2b (complex)."
```

(An earlier version of this fix used a name-based `case` that special-cased the
completed chains to complex. That was wrong -- see
[Correction](#correction-2026-08-16).)

**2. `convert_to_hf.py` announces the convention it is about to use**, and
warns on any non-80B complex conversion:

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
