# Sync 84 numerics: fused QKV / gate-up vs the pre-merge tree

**Date:** 2026-09-15 · **Host:** Perlmutter (`login14`, A100, `pytorch/2.13.0` = `2.13.0+cu130`)
**Question:** #4526 made QKV fusion mandatory (one GEMM instead of three) and
#4535 made gate-up fusion the default. Does that change the model's output?

**Answer: no, beyond float regrouping. Cleared to land on numerics.**

## Why Perlmutter

The comparison is pure linear algebra, so it is accelerator-independent, and
Perlmutter has the torch version HEAD requires (2.13, for
`DataParallelMeshDims`). It is NOT a stand-in for the XPU smoke -- see the
`spmd_types` note at the bottom.

## The trap: the naive A/B looks like a large disagreement

First run, both trees seeded identically, each building its own model:

```
              logits_sum     logits_std    loss        grad_norm
pre-merge     -3716.146      0.98647370    0.97313046  0.12344086
merged        +1090.280      0.98669304    0.97356308  0.12347840
```

`logits_sum` does not even share a SIGN. But every distributional statistic
agrees to ~4 digits, which is the signature of *the same distribution, a
different draw* -- not of broken arithmetic.

Confirmed directly by comparing init:

```
tok_embeddings.weight   -188.3066622664744   IDENTICAL on both trees
lm_head.weight          -85.079 vs -66.638   differs, std 0.0616680 / 0.0616727
w2.weight                10.422 vs  -6.377   differs, std 0.0279494 / 0.0279322
```

`tok_embeddings` is initialized BEFORE the first fused layer and is bit-exact.
Everything initialized after it differs in value while matching in std. The
fused `w13` draws from the RNG stream in a different order than separate
`w1`/`w3`, so the two trees get different weights from the same seed.

**A seeded A/B that rebuilds the model on each side does not isolate the
math.** It measures init order too.

## The decisive test: identical weights

Build on the pre-merge tree, save the state dict, load it into the merged
tree, compare the forward.

```
LOADSTATUS {"missing": [], "unexpected": []}      <- hooks preserve logical w1/w3

              logits_sum          logits_std       logits_absmax
pre-merge     -3716.1463866537    0.98647369557    5.161099434
merged        -3716.1466573014    0.98647369409    5.161098480
```

Per-element, over all 256 x 32000 logits:

```
max |delta|     1.907e-06        (float32 eps = 1.192e-07)
mean |delta|    1.433e-07
exact matches   15.09%
```

The headline `max relative delta` is 4.2e-01, which looks alarming and is an
artifact: the worst element has magnitude **8.8e-07**. Stratified by magnitude
the picture is unambiguous:

| threshold | n | max relative error |
|---|---:|---|
| \|a\| > 0.001 | 8,185,359 | 9.653e-04 |
| \|a\| > 0.01 | 8,125,827 | 7.806e-05 |
| \|a\| > 0.1 | 7,530,899 | 9.222e-06 |
| \|a\| > 1.0 | 2,544,664 | **1.275e-06** |

Monotone decreasing -- exactly what float regrouping looks like, and the
opposite of what a wrong layout produces (which would be large and
magnitude-independent; see
[bad attn layout is a silent no-op](../guides/known-bugs/)).

**And the decision-relevant metric is exact:**

```
argmax agreement  100.00%
top-5 agreement   100.00%
```

## Conclusion

Fusion changes the arithmetic grouping, not the arithmetic. Deltas sit at
1-20x float32 eps, shrink monotonically with magnitude, and never change which
token the model would pick. Checkpoints load across the merge with zero key
mismatches.

**This clears the numerics question. It does NOT clear:**

- **The XPU 2N smoke.** Different accelerator, different kernels, and the
  `spmd_types` question is a TORCH VERSION FLOOR, not an XPU bug -- Perlmutter's
  `pytorch/2.13.0` has `_fsdp_param.py` at 1095 lines with ZERO occurrences of
  `_resolve_spmd_types_for_storage` / `self.is_spmd_types`, the exact same gap
  as the ALCF build. It reproduces the problem rather than answering it.
- **Training dynamics.** This is one forward and one backward on `debugmodel`.

## Reproducing

Harness in `experiments/ezpz/tests/numerics/`:

```bash
module load pytorch/2.13.0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MPICH_GPU_SUPPORT_ENABLED=0
python same_weights.py $PWD/tt-premerge save $PWD/w.pt
python same_weights.py $PWD/tt-merged  load $PWD/w.pt
```

`run_ab.sbatch` runs the naive (own-init) A/B on a compute node for both
`debugmodel` and `2b`. Note `ezpz` initializes MPI at import, so
`MPICH_GPU_SUPPORT_ENABLED=0` is required on a login node.
