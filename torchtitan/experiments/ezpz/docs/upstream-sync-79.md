# 79th upstream sync -- 26 commits, four stacked defects, all from one PR

> **2026-08-19, Sunspot, frameworks RC.** Merge `210282fcd`; verification jobs
> `12473338` (PRE reference) through `12473368` (POST7), plus MoE sweep
> `12473367`.

> Last updated: 2026-08-31 (one correction added below: the `full_dtensor`
> pin was reversed on 2026-08-20 and that value is now illegal).

## Result

| arm | PRE (`12473338`) | POST7 (`12473368`) |
|---|---|---|
| agpt2b | 20/20 | **20/20** |
| agpt2b-tp2 | 20/20 | **20/20** |
| moe-prod (EP=12 forced) | 20/20 | 2 steps -- see below |
| moe-2b-det | VOID (memory) | **20/20 -- better than PRE** |

The dense path -- everything the 30B and 80B production configs use -- is
verified unchanged across four independent POST runs. `moe-2b-det` now passes
where it VOIDed before the merge.

## One upstream PR, four separate breakages

Every failure traced to **#4085, "spmd_types as default backend"**, which was
flagged as the highest-risk commit before merging. Each fix revealed the next:

| # | symptom | cause | fix |
|---|---|---|---|
| 1 | `ValueError: Invalid mesh dim: 'fsdp'` | under spmd_types the dense mesh is `['pp','dp','cp','tp']`; no flattened `fsdp` axis | `resolve_fsdp_mesh()`, mirroring `llama3/parallelize.py:71` (`7bd902b5c`) |
| 2 | `all parameters must be DTensors ... Got plain tensor` | `model.parallelize()` was gated on `tp_enabled`; under the new backend it must run unconditionally (it is what makes params DTensors) | unconditional + `validate_config` (`60009b04d`) |
| 3 | `Cannot concatenate overlapping meshes` | dense mesh resolved by the new API, sparse still on the legacy `efsdp` lookup; and `FSDPMeshInfo` built by hand does not flatten the DP submesh | `resolve_sparse_fsdp_mesh` + `_get_mesh_info` (`d0c02753c`, `ae180b3e7`) |
| 4 | `aten.mul.Tensor got mixed torch.Tensor and DTensor` (rope.py:263) | the RoPE `cache` buffer had no `state_shardings`; the MLA path replaces `attention.sharding_config` wholesale, so nothing annotated it | annotate `attention.rope` (`c23a544bc`) |

Plus one unrelated-to-#4085 break: **#4172 deleted
`components/lr_scheduler.py`** (it had become a shim when #4140 packaged the
optimizer components), breaking four ezpz registries (`be5749ee0`).

`spmd_backend` is pinned to `full_dtensor` (`621ac2406`) because #4 was found
after the pin; with the rope annotation in place, revisiting spmd_types is now
a reasonable follow-up rather than a blocker.

> [!WARNING]
> **Superseded the next day; do not set `full_dtensor` today.** `b2ff09632`
> (2026-08-20) repinned both `agpt` and `moe` to `partial_dtensor` because
> upstream #4217 deletes the `full_dtensor` backend. It is no longer a legal
> value of the field -- `parallelism.spmd_backend` is
> `Literal["partial_dtensor", "spmd_types"]` on this tree and anything else
> raises. The rest of this page is unaffected.

## What defect #4 says about SPMD migration

Upstream's own note at `protocols/module.py:279` is
`TODO(fegin): Change to assert once ALL Models are migrated` -- a module with
no `sharding_config` **silently returns** and its state stays a plain tensor.
That silence is the hazard: the failure surfaces layers away, in a multiply
inside core's rope, with nothing pointing at the missing annotation.

Two details worth carrying forward:

- The buffer is named **`cache`** (`rope.py:114`), not `freqs_cis`. The MoE
  sharding file already had `freqs_cis` entries in `in_src_shardings` /
  `in_dst_shardings` -- those are the *argument* name at the attention
  boundary, a different thing from the module's own state. Their presence made
  the file look annotated when it was not.
- `set_decoder_sharding_config` covers **root-level configs only**
  (embeddings, norm, lm_head) and says so; per-layer attention is the caller's
  job. For MLA-based MoE nothing was doing that job.

## MoE sweep (`12473367`), post-fix

| config | result |
|---|---|
| moe_debugmodel | 5/5 |
| moe_500m | 5/5 |
| moe_2b | 5/5 |
| moe_4b | 5/5 |
| moe_10b_2b_sdpa | 5/5 |
| moe_7b | 1/5 -- `UR_RESULT_ERROR_OUT_OF_RESOURCES` at 95.4% memory |
| moe_small, moe_10b_2b | flex-attention `BlockMask` assertion, unrelated |

**`moe_10b_2b_sdpa` passes at its own default EP but fails at forced EP=12**
-- the smoke arm's `--parallelism.expert-parallel-degree=12` is the
difference, so that 2-step death is a flag of mine, not a config defect.

## Method notes

- **The paired PRE/POST design earned its cost.** POST alone showed 0/4 and
  would have read as "MoE is flaky again"; a same-script PRE reference on a
  worktree made "the merge broke this" unambiguous, and made each fix's
  progress legible as the error moved rather than looking like thrashing.
- **The pre-smoke caught a bug of mine before the merge could hide it** -- a
  helper inserted between a `@dataclass` and its class. Without the import
  gate that would have surfaced post-merge and been blamed on upstream.
- Several rounds were spent patching one error at a time. The turn came from
  stopping to ask *does upstream's own path work here?* (job `12473350`),
  which converted guesswork into a two-line answer.
