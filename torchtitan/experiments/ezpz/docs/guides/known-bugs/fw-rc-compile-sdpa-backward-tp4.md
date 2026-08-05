# frameworks-RC torch: `torch.compile` at TP=4 crashes in the SDPA flash-backward

> [!IMPORTANT]
> **FIXED in frameworks RC4 (verified 2026-08-05, job 12472578).** The patched
> build `torch 2.13.0a0+gitcf30153` / `pytorch_2.13.0_patched_08_02_2026`
> (conda env `RC4_..._rel_one_2026.1.0_python_3.12.12`) trains **clean at TP=4**:
> 10/10 steps, loss 12.98 -> 8.47, finite grad_norms, and **zero**
> `assert_size_stride` occurrences in any rung. The full ladder on that build
> (2N, agpt-2b, seq 4096, LBS 1):
>
> | rung | loss step 1 -> 10 | MFU | assert_size_stride |
> | --- | --- | --- | --- |
> | eager | 12.98 -> 8.38 | 19.36% | 0 |
> | compile TP=1 | 12.92 -> 9.30 | 21.31% | 0 |
> | compile TP=2 | 12.95 -> 8.65 | 13.31% | 0 |
> | compile TP=4 | 12.98 -> 8.47 | 7.83% | 0 |
>
> Note the base git hash is UNCHANGED (`cf30153`) -- only the patch level
> differs, so identify the build by `patched_08_02_2026`, not by the hash.
>
> Caveats: verified at 2N / 10 steps / agpt-2b only. That is enough to falsify
> the assert (the original fired within ~30s at this scale) but not to certify
> production; confirm at scale before relying on it. Separately, MFU degrades
> sharply with TP on this build (21% -> 13% -> 7.8%) -- do not adopt TP=4 for
> throughput without measuring.
>
> Everything below documents the ORIGINAL (pre-RC4) failure and remains the
> reference for the older fw-RC conda stack.

**Status (2026-07-26): reproduced + scoped.** On the "frameworks" RC conda torch
(`2.13.0a0+gitcf30153`, oneAPI 2026.1.0, XPU) any `torch.compile` agpt run at
**tensor-parallel degree 4** aborts in the compiled backward with an
`assert_size_stride` on `aten._scaled_dot_product_flash_attention_backward`.
TP=1 and TP=2 compile and train fine on the same stack; the `.venv` stack
(torch `2.13.0.dev20260519+xpu`, oneAPI 2025.3.1) trains fine at all TP degrees.
Workaround: run on the `.venv` stack for compiled TP=4, or run eager
(`--compile.no-enable`) on the RC stack, or stay at TP<=2. No upstream fix is
merged.

## Symptom

All ranks abort (SIGTERM 143) inside the compiled backward
(`loss.backward()` -> aot_autograd compiled bwd -> inductor generated `call` ->
`self.partitions[0](...)`):

```
assert_size_stride(buf29, (4, 4, 4096, 128), (2097152, 524288, 128, 1),
  'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
AssertionError: expected size 4==4, stride 128==524288 at dim=1;
                expected size 4096==4096, stride 512==128 at dim=2
Error in op: torch.ops.aten._scaled_dot_product_flash_attention_backward.default
This error most often comes from an incorrect fake (aka meta) kernel for a custom op.
```

(The exact buf shape/strides scale with seq_len and batch; the dim1/dim2 stride
swap is the invariant.)

## Reproduction

```bash
# frameworks RC stack (torch 2.13.0a0+gitcf30153, oneAPI 2026.1.0), 4 tiles, TP=4:
ezpz launch --nhosts=1 -- python3 -m ezpz.examples.fsdp_tp \
  --model=agpt-2b --seq-len=4096 --tokenizer_name=google/gemma-7b \
  --tp=4 --batch-size=4 --dp-shard=1 --dp-replicate=3 \
  --loss-impl=loss-parallel --compile
```

`--compile-mode=max-autotune` hits the identical assert; autotune is NOT required.
The torchtitan trainer path crashes the same way
(`python3 -m torchtitan.experiments.ezpz.train --module=ezpz.agpt
--config=agpt_2b_real --compile.enable --parallelism.tensor-parallel-degree=4 ...`).

## Scope (bisected, small config seq2048/bs1 so memory is not a confound)

| stack | TP=1 | TP=2 | TP=4 |
| ----- | ---- | ---- | ---- |
| RC (torch gitcf30153 + oneAPI 2026.1.0, libsycl.so.9) | trains | trains | **13x assert, crash 143** |
| .venv (torch dev20260519 + oneAPI 2025.3.1, libsycl.so.8) | trains | trains | trains |

- **TP=4-gated, not size-gated.** TP=1 and TP=2 train clean at the SAME small
  config where TP=4 asserts. Shrinking seq_len/batch does not help TP=4.
- **compile-general, not autotune-specific.** Plain `--compile` (default mode) and
  `--compile-mode=max-autotune` both hit it.
- **Build-gated.** Crashes only on the RC stack; the `.venv` stack trains compiled
  at every TP degree tested (proven: TP=4 seq8192 loss 13.01->6.53).

## Why the two builds differ, and why "torch vs oneAPI" is not separable

The RC torch and its oneAPI are ABI-welded and cannot be mixed (confirmed by
`readelf -d libtorch_xpu.so`):

- RC torch NEEDs `libsycl.so.9` (shipped by oneAPI 2026.1.0) and its attention
  kernels are the CONSOLIDATED `libtorch-xpu-ops-sycl-*Kernels.so` set.
- .venv torch NEEDs `libsycl.so.8` (oneAPI 2025.3.1) and the SPLIT
  `libtorch-xpu-ops-sycltla-mha_bwd.so` / `-mha_fwd.so`.

Loading RC torch under oneAPI 2025.3.1 fails at import with
`ImportError: libsycl.so.9: cannot open shared object file` (2025.3.1 ships
`libsycl.so.8`). So the SDPA-flash-backward kernel was rewritten AND the SYCL
runtime SONAME bumped together between the two builds; you cannot hold torch
constant while varying oneAPI. Treat the RC stack as one indivisible unit.

## Mechanism (high-confidence, inferred from strides + build diff; not fully bisected to a commit)

The compiled backward emits an `assert_size_stride` guard around the SDPA-flash
backward extern op from the op's Python meta/fake kernel, which reports a
contiguous `(B,H,S,D)` grad_query. Under TP=4 the real RC XPU kernel returns
grad_query in a transposed / seq-major layout (dim1 and dim2 strides swapped),
so the runtime strides violate the guard. The meta kernel and inductor lowering
are byte-identical between the RC and .venv torch source trees; the difference is
in the rebuilt XPU SYCL attention kernel (the consolidated
`libtorch-xpu-ops-sycl-AttentionKernels.so` vs the old split mha_bwd lib). Why it
manifests only at TP=4 (and not TP<=2) is not fully explained -- the TP=4 mesh +
loss-parallel sharding produces a different inductor graph partition
(`self.partitions[0]`), which is where the guard fires. A candidate upstream
change is pytorch/pytorch #181559 "[XPU] Enable _flash_attention_forward/
_flash_attention_backward" (OPEN, no fix), but this is NOT bisected/confirmed.
intel/torch-xpu-ops #3093 is unrelated (NestedTensor forward "no viable backend").

## Workaround

- **Compiled TP=4:** use the `.venv` stack (torch `dev20260519+xpu` + oneAPI
  2025.3.1, via `source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup .venv`).
  Proven to train compiled at TP=4.
- **On the RC stack:** run eager (`--compile.no-enable`) -- proven to train; or keep
  TP<=2 if compile is required.
- **Do NOT** `TORCHINDUCTOR_SIZE_ASSERTS=0` to silence the guard: downstream inductor
  indexes the grad buffer with the wrong (contiguous) strides, silently swapping the
  head and sequence axes of grad_query -> silently wrong gradients, no crash.

## Notes / gotchas for reproducing

- Bisecting this over non-interactive SSH is error-prone: TP=1/TP=2 at seq4096/bs4
  OOM (looks like a different failure -- shrink to seq2048/bs1 so they fit);
  `module swap` of oneAPI under an activated conda+venv cascade-deactivates the
  venv; and RC torch simply won't import under 2025.3.1 (libsycl.so.9). Grab a node
  with `qsub ~/test.sh` and ssh in; keep the RC env's ordering intact.
