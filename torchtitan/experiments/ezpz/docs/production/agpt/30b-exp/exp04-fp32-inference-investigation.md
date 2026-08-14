# exp04 -- Does fp32-master training force fp32 inference?

> **Date:** 2026-08-14. **Status:** investigation complete for the HF path;
> the vLLM leg could not be re-run (see [Caveats](#caveats)).
>
> Written to settle a specific objection to
> [`README.md` Section 6](README.md#6-what-here-is-judgment-not-measurement):
> the 30B-exp proposal recommends fp32 master weights, and the counter-argument
> was that this forces fp32 at serving time and doubles deployment cost.
>
> Every claim below is tagged **MEASURED** (I ran it, this session),
> **INFERRED** (reasoning from code/config I read), or **RECALLED-FROM-DOCS**
> (an earlier run's writeup that I did not reproduce).

## Verdict

**No. The fp32 master weight choice during training does not force fp32 at
inference, and it is not the cause of the vLLM garbage output.** The two are
unrelated: training forward/backward compute is bf16 either way (the fp32
master is a separate optimizer-side copy), and the exported HF checkpoint is
written in bf16 regardless of `training.dtype`. **MEASURED:** the completed 2B
production checkpoint (step-92,859) generates fluent, correct text in HF
transformers at `dtype=bfloat16`, bit-identical to fp32 over a 200-token greedy
decode -- so the *model* is not bf16-fragile.

**What the real cause is: not established with certainty.** I refuted the
documented explanation but could not run the one test that would replace it.
The recorded root cause -- "vocab 256000 + ffn 11008 accumulate enough bf16
error to flip the greedy argmax" -- is **refuted as stated**: **MEASURED**, the
same vocab-256000 model family generates coherently in bf16 through HF on the
identical weights. Whatever breaks, breaks *inside vLLM*, not in the model's
bf16 numerics. That makes **H5 (vLLM-XPU-specific bf16 path) the leading
hypothesis and H4 (genuine model fragility) effectively ruled out**, but H5 is
**UNTESTED** -- I could not obtain a vLLM XPU device this session. Treat "vLLM
bf16 is broken for this model on XPU" as a strong inference, not a measurement.

## Evidence

### 1. Training compute is bf16 regardless of the fp32 master (INFERRED, from code)

The fp32 master and the bf16 compute cast are separate mechanisms:

- `torchtitan/config/configs.py:55` -- `dtype: Literal["bfloat16","float32"] = "float32"`.
  Its own docstring says this controls whether there is "an extra copy of fp32
  weights", i.e. the *master* copy.
- `torchtitan/config/configs.py:62` -- `mixed_precision_param = "bfloat16"`.
- `torchtitan/config/configs.py:70` -- `mixed_precision_reduce = "float32"`.
- `torchtitan/experiments/ezpz/agpt/parallelize.py:177` -- the FSDP policy is built
  with `param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param]`, which flows
  into `MixedPrecisionPolicy(param_dtype=..., reduce_dtype=...)` at
  `torchtitan/distributed/fsdp.py:97,165`.

So `training.dtype=float32` adds an fp32 master copy while
`mixed_precision_param=bfloat16` keeps every forward/backward GEMM in bf16.
The activations the model was trained under are bf16 activations. This confirms
the analysis in the task framing and the comment at
`torchtitan/experiments/ezpz/agpt/config_registry.py:200-206`.

### 2. The exported checkpoint is bf16 on disk anyway (MEASURED)

`convert_to_hf.py` defaults to `--export_dtype bfloat16`
(`torchtitan/experiments/ezpz/eval/convert_to_hf.py:99`) and the production eval
script passes it explicitly (`scripts/eval/eval-2b-v2.sh:160`). Verified on the
actual artifact:

```
outputs/evals/agpt-2b-v2-256n/step-1000/hf/model-00001-of-00001.safetensors
DTYPES {'BF16': 111}     # all 111 tensors
```

**The fp32 master never reaches the served checkpoint.** Even a fully
fp32-master-trained run ships bf16 weights. This alone decouples the training
choice from the serving choice.

### 3. The production 2B serves correctly in bf16 through HF (MEASURED -- the key result)

Model: `outputs/evals/agpt-2b-v2-256n/step-92859/hf` (the completed 4.674T-token
chain), on Aurora, `venvs/aurora/tt-lm-eval` (transformers 4.57.6), CPU, greedy.

200-token greedy decode, bf16 vs fp32 -- **byte-identical output**:

```
=== IDENTICAL: True
```

Short prompts, 40 tokens, bf16:

```
"The capital of France is" -> " Paris.\n\nParis is the capital of France. It is
 the most populous city in France and the second most populous city in the
 European Union..."
"Water boils at a temperature of" -> " 100 degrees Celsius..."
"In 1969, humans first landed on" -> " the moon.\n\nThe Apollo 11 mission was
 the first manned mission to land on the moon..."
```

Four more prompts at 100 tokens: 2/4 byte-identical to fp32; the 2 that diverged
produced *equally fluent* text on both sides (bf16: "the executive, the
legislative, and the judicial branches. The executive branch is headed by the
president..."). Divergence without degradation is the expected signature of
benign floating-point nondeterminism, not corruption.

### 4. The vocab-256000 model the docs blame is ALSO fine in bf16 (MEASURED -- the refutation)

The recorded root cause attributes the gibberish to agpt-2b's large vocab
(256000) and FFN (11008) accumulating bf16 error
(`docs/production/rl/history/grpo-lora-agpt2b-repro.md:113`). I tested that exact
family: the MDS base `/home/foremans/global_step138650` on Sunspot
(`vocab_size=256000`, hidden 2048, ffn 11008, 12L -- the same architecture and
the direct ancestor of the ckpt-900 that gibberished), venv `pt2.12`:

```
=== dtype=torch.bfloat16 param0=torch.bfloat16 vocab=256000 ===
[bf16] ' Paris.\n\nThe capital of France is Paris...'
[bf16] ' sort(arr):\n    return arr.sort()\n\ndef main():...'
=== dtype=torch.float32 param0=torch.float32 vocab=256000 ===
[fp32] ' Paris.\n\nThe capital of France is Paris...'
[fp32] ' sort_list(list):\n    """Sort a list of integers"""...'
```

Coherent in bf16. **The "large vocab/FFN makes bf16 unusable" explanation does
not survive contact with the architecture it was written about.** The property
blamed (vocab 256000 + ffn 11008) is present here and produces no gibberish.

### 5. How much bf16 headroom the logits actually have (MEASURED)

Teacher-forced, 59 positions, step-92859, bf16 vs fp32 logits:

| quantity | value |
|---|---|
| argmax agreement | **94.92%** (56/59) |
| max abs logit diff | 2.75 |
| top1-top2 gap (median / mean / min) | 0.82 / 1.24 / 0.0010 |
| median abs logit | 10.89 (bf16 ULP there ~0.0625) |

So bf16 *does* flip ~5% of argmaxes -- the distribution is genuinely flat enough
for rounding to matter. The docs' mechanism is real in the small. But **MEASURED**
above: it does not compound into gibberish in HF, even over 200 tokens. A 5%
argmax flip rate produces *a different fluent continuation*, not word salad. The
observed vLLM failure ("5.ciptakan,. Đóeld. Đóeld...") is categorically worse
than what this error budget can explain. Something in the vLLM path is wrong
beyond precision.

### 6. lm-eval's bf16 scores are real (MEASURED -- rules out a silently-fp32 eval)

I initially suspected the eval suite was secretly running fp32, which would have
meant we had no bf16 evidence at all. Checked: our exported `config.json` has
**no `torch_dtype`/`dtype` field** (0 of 40 eval configs carry one -- see
`eval/configs/agpt_2b_config.json`, which indeed has no dtype key), and lm-eval's
`HFLM` default is `dtype="auto"`. But transformers' `"auto"` falls back to the
*checkpoint's* dtype, and I verified the resolution empirically:

```
config dtype attr: None | torch_dtype: None
LOADED-WITH-AUTO param dtypes: {'torch.bfloat16': 111}
```

So lm-eval loaded **bf16**, and the published numbers (2B step-46429: HellaSwag
0.4753, ARC-Easy 0.6006, PIQA 0.6997 -- README Section 1.2) are bf16 numbers, far
above chance. A model that were bf16-broken could not score these. (One
exception: the IFEval runs recorded `model_dtype = torch.float32` -- that base
loaded from a `pytorch_model.bin` rather than bf16 safetensors. IFEval is
therefore *not* bf16 evidence; the lm-eval suite above is.)

### 7. Double-BOS / swapped bos-eos ruled out as the bf16 trigger (MEASURED)

The repro doc lists a secondary finding: config.json has bos=1/eos=2 while the
gemma tokenizer has bos=2/eos=1, plus a double-BOS from the chat template. I
confirmed the mismatch is real on the MDS base:

```
tokenizer bos_token_id = 2 eos_token_id = 1
config bos_token_id   = 1 eos_token_id = 2
```

But it does not interact with dtype -- single-BOS and double-BOS give the same
coherent output in *both* bf16 and fp32 (4/4 coherent). It is a genuine latent
bug (`scripts/eval/fix_ckpt_eos.py` exists to patch it) but not this one.

## Hypotheses tested

| # | Hypothesis | Verdict | Reason |
|---|---|---|---|
| **H1** | Converter writes wrong dtype / mis-casts | **RULED OUT** | **MEASURED:** exported safetensors are uniformly BF16 (111/111); the cast at `convert_to_hf.py:62-63` is a plain `.to(bfloat16)` with no reordering. Round-trip HF->TT->HF was previously verified bit-exact (RECALLED-FROM-DOCS). Real but *separate* gap: `config.json` carries **no `torch_dtype`** -- see Recommended fix. |
| **H2** | HF config.json disagrees with trained architecture | **RULED OUT for the served model** | **MEASURED:** `eval/configs/agpt_2b_config.json` (vocab 256128, theta 50000, eps 1e-5, 16H/4KV, untied) matches the `"2B"` flavor at `agpt/__init__.py:435-442` field-for-field, and the model generates correctly under it. The 256128-vs-256000 gap is 128-alignment padding, and each base is evaluated with its own vocab (`eval-2b-v2.sh:64-70`). **Except:** bos/eos are swapped vs the tokenizer (item 7) -- real, but dtype-independent. |
| **H3** | RoPE flavor mismatch (complex vs cos_sin) | **RULED OUT as the cause; but a LIVE LATENT RISK** | Not the cause: **MEASURED**, the served checkpoint generates correctly, which a scrambled Q/K pairing could not. `AgptStateDictAdapter` (`agpt/state_dict_adapter.py:39-62`) detects the RoPE type and skips the permute for cos_sin. **However (INFERRED, unresolved):** production 2B/20B default to `CONFIG_SUFFIX=_real` (cos_sin) since commit `5ffb850a1`, 2026-06-25 (`submit_agpt_2b_autoretry.sh:129`, `submit_agpt_multi_autoretry.sh:204`), while `eval-2b-v2.sh:69` defaults to `MODEL_FLAVOR=2b` (**complex**). The adapter reads RoPE from the *flavor you pass*, not from the checkpoint -- so converting a `_real` chain with flavor `2b` applies the permute and corrupts the export. See Recommended fix. |
| **H4** | Genuine bf16 fragility of this architecture | **RULED OUT** | **MEASURED:** 200-token bf16 greedy decode is byte-identical to fp32 on the production 2B; the blamed vocab-256000 family is coherent in bf16 (item 4). The flat-logit mechanism is real (5% argmax flips, item 5) but demonstrably does not compound into gibberish. |
| **H5** | vLLM-XPU-specific bf16 path is broken | **LEADING HYPOTHESIS -- UNTESTED** | By elimination: the failure is reproducible only through vLLM, on weights that are provably fine in bf16 through HF. Prior art (RECALLED-FROM-DOCS) is a controlled A/B -- same prompt, weights, and engine, only dtype differing (`grpo-lora-agpt2b-repro.md:97-104`). I could not re-run it: the ckpt-900 safetensors have since been deleted, and vLLM aborts on login nodes (`RuntimeError: Device string must not be empty`, no XPU). **Not confirmed.** |

## Deployment implication

**If H5 holds, the model can serve in bf16 and the fp32-master recommendation
costs nothing at inference.** Section 6's argument survives intact: the compute
dtype is bf16 either way, and the served checkpoint is bf16 on disk (**MEASURED**,
item 2). The training-time and serving-time dtype decisions are independent.

Rough cost of serving fp32 instead of bf16 -- **GENERAL EXPECTATION, NOT MEASURED
here**:

- **Weights/memory: ~2x.** 2B at bf16 ~4.0 GB (the measured safetensors file is
  3.97 GB) vs ~8 GB at fp32. Scaled to 30B: ~60 GB vs ~120 GB, which changes how
  many tiles a replica needs and cuts KV-cache headroom at fixed memory.
- **Decode throughput: roughly ~2x slower.** Autoregressive decode is
  memory-bandwidth-bound at low batch, so time per token tracks bytes of weights
  streamed per step; doubling the weight bytes roughly halves the rate. Prefill
  is compute-bound and would instead be limited by fp32 vs bf16 matmul
  throughput, typically a larger penalty on accelerators with bf16-specific
  units.
- The one **MEASURED-elsewhere** datapoint in our own docs is milder than 2x:
  "fp32 generation is ~30% slower than bf16"
  (`grpo-lora-agpt2b-repro.md:180`, RECALLED-FROM-DOCS) -- an RL-loop measurement
  at 2B on XPU, not a serving benchmark, and not necessarily bandwidth-bound.
  Prefer measuring before quoting any number in the proposal.

Either way this is a **serving-stack** cost, not a consequence of how the model
was trained -- it should not be charged against the fp32-master decision.

## Recommended fix

1. **Write `torch_dtype` into the exported `config.json`.** (highest value, ~1
   line) Every consumer -- vLLM, HF, lm-eval -- infers load dtype from it, and
   ours is absent in 40/40 eval checkpoints. vLLM's resolver
   (`ModelArchConfigConvertorBase.get_torch_dtype`) falls back to safetensors
   metadata and then to **`torch.float32`**, and `_resolve_auto_dtype` then
   down-casts fp32 to the platform-preferred dtype -- so what a served
   checkpoint runs as is decided by fallback logic rather than by us. Set it
   explicitly to match `--export_dtype` in `convert_to_hf.py`, and add it to
   `eval/configs/agpt_*_config.json`. This removes a whole class of
   silent-dtype ambiguity and makes any future bf16/fp32 report interpretable.
2. **Re-run the vLLM A/B to actually confirm H5** -- the one missing measurement.
   On a compute node with XPUs: same converted checkpoint (e.g.
   `outputs/evals/agpt-2b-v2-256n/step-92859/hf`), greedy, `dtype=bfloat16` vs
   `dtype=float32`, nothing else varying. HF bf16 output on that exact
   checkpoint is already recorded above as the reference. If bf16 gibberishes
   there while HF bf16 does not, H5 is confirmed and it is a vLLM-XPU bug to
   file upstream -- **not** a property of our model, and **not** a reason to
   train or ship in fp32.
3. **Fix the RoPE flavor default mismatch (H3).** Make `eval-2b-v2.sh` /
   `convert_and_eval.sh` default `MODEL_FLAVOR` to the `_real` flavor for chains
   trained after `5ffb850a1`, or better, record the RoPE backend in the
   checkpoint and have `convert_to_hf.py` read it instead of trusting a
   hand-passed flavor. Today a correct-looking command silently corrupts the
   export. The Polaris 20B eval scripts already default to `20b_real`
   (`polaris_20b_eval_sweep.sh:55`); the 2B path does not.
4. **Patch bos/eos at conversion time**, not after the fact -- fold
   `scripts/eval/fix_ckpt_eos.py` into the converter so every exported
   checkpoint gets `bos=2`, `eos=[1,107]`.
5. **Correct the docs.** `POST-TRAINING-2B.md:112-115`, `rl/monarch.md:92`,
   `plans/cot.md:217`, and `grpo-lora-agpt2b-repro.md:113` all state the cause as
   the model's vocab/FFN size making bf16 unusable. That is refuted (item 4).
   Keep `--generator.model-dtype=float32` as the operational workaround -- it
   works -- but re-label it "workaround for a vLLM-path failure, cause not yet
   isolated" so nobody generalizes it into "agpt models need fp32", which would
   wrongly tax every future deployment.

## H3 exposure, resolved (2026-08-14)

The investigation left H3's *live* exposure inferred from script defaults. I
resolved it by reading the **pinned production clones** rather than the
main-repo scripts, which is where the answer actually lives. **The fleet is
split, and the split is why nothing has broken yet.**

| clone | `CONFIG_SUFFIX` default | trained RoPE | eval default `MODEL_FLAVOR=2b` |
|---|---|---|---|
| `runs/agpt-2b-v2` (the completed 4.674T chains) | **absent** -- `${CONFIG_SUFFIX:-}` expands empty | **complex** | **correct** |
| `runs/agpt-20b-v2` | `${CONFIG_SUFFIX-_real}` | **cos_sin** | **WRONG** |
| `runs/agpt-2b-constlr-from9200` | `${CONFIG_SUFFIX-_real}` | **cos_sin** | **WRONG** |

**MEASURED.** `runs/agpt-2b-v2/torchtitan-ezpz` is pinned at `f319e3fa`,
predating `5ffb850a1` (2026-06-25, the commit that made `_real` the default).
It carries only the *legacy* `*_aurora_venv_failover.sh` scripts -- no
`*_autoretry.sh` exists in it -- and its line 182 is
`--config="agpt_${MODEL}${CONFIG_SUFFIX:-}"` with **no assignment anywhere in
the file**. So the completed 2B chains launched as `--config=agpt_2b`:
**complex RoPE, which is exactly what `MODEL_FLAVOR=2b` converts.**

**Every 2B eval in the campaign was therefore converted correctly.** This is
independently corroborated by the eval curves themselves: HellaSwag rises
monotonically 0.405 -> 0.561 across the completed chain. Scrambled Q/K pairing
does not produce a clean monotone learning curve -- it produces gibberish. The
near-chance MMLU is **not** a conversion artifact, which closes off an
attractive but wrong explanation for Section 1.1.

**The trap is live for everything newer.** Both clones pinned after
`5ffb850a1` train cos_sin while `eval-2b-v2.sh:69` still defaults to the
complex flavor. The adapter reads RoPE from the flavor you pass, not from the
checkpoint (`state_dict_adapter.py:47-49`; the file's own header comment
already warns "flavor still must match how the model was trained"). A
correct-looking convert command on the constant-LR fork or the 20B chain will
silently apply the permute and produce a corrupted export -- which will read as
a capability regression, not as a bug.

Tracked as task #73. The right fix is to record the RoPE backend in the
checkpoint and have `convert_to_hf.py` read it, rather than trusting a
hand-passed flavor; `polaris_20b_eval_sweep.sh:55` gets this right today only
by hardcoding `20b_real`.

## Caveats

What I could **not** verify:

- **H5 is unconfirmed.** I never reproduced the vLLM gibberish. The original
  ckpt-900 safetensors are gone (`checkpoint-900-hf/model.safetensors` is a
  dangling symlink; the `checkpoint-*/` dirs retain only `trainer_state.json`),
  and vLLM cannot initialize on a login node. **The root cause of the vLLM
  garbage output is therefore NOT established** -- I refuted the recorded
  explanation and narrowed the location to inside vLLM, no more. Fix #2 is the
  missing experiment.
- **All my generation tests ran on CPU**, not XPU. If the failure needs the XPU
  kernels (plausible for an H5-class bug -- attention backend, kernel dtype
  handling), CPU bf16 being clean does not exonerate XPU bf16. It does still
  exonerate the *weights* and the *architecture*, which is what H1-H4 were about.
- **Different checkpoints.** The gibberish was seen on SFT ckpt-900 (vocab
  256000). I measured the pretrain base of that lineage (`global_step138650`,
  same arch/vocab) and the olmo-mix production 2B (vocab 256128). I did not test
  ckpt-900 itself. An SFT-introduced pathology specific to that checkpoint cannot
  be excluded, though the fp32-fixes-it behavior argues against it.
- **Transformers version skew:** Aurora tests ran 4.57.6, Sunspot 5.9.0. Both
  produced coherent bf16 output, so the conclusion is not version-specific.
- **The throughput numbers in Deployment implication are expectations, not
  measurements.** I ran no serving benchmark. The only empirical figure (~30%)
  is from an RL loop, quoted from docs.
- ~~**H3's live exposure is inferred from script defaults**~~ -- **RESOLVED
  2026-08-14, see [H3 exposure, resolved](#h3-exposure-resolved-2026-08-14)
  below.** The mechanism is real, but no completed 2B eval was affected.
- No jobs were submitted and none were killed; the queued production jobs
  (8756070, 8756957, 8756071, 8756072, 8752939, 8752824) were left untouched --
  observed via `qstat` only, all `Q`/`H`. Nothing was committed to git.
