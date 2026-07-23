# Polaris 20B "eval gibberish" root cause: training-data / model tokenizer mismatch

> **Status:** ROOT CAUSE CONFIRMED 2026-07-23. The Polaris 20B dolma run
> (`agpt-20b-sophiag-dolma-n128-gbs1024`) trained on **Llama2-tokenized**
> data but the model config + HF export + lm-eval all used the **gemma-7b**
> tokenizer. The model is coherent in Llama2 id-space (training loss ~2.05
> is real); it only produces "gibberish" at eval because eval speaks a
> different tokenizer than the one the data was encoded with.
>
> **This is NOT a RoPE bug, NOT a conversion bug, NOT an eval-backend bug.**
> Those were all ruled out (see "Hypotheses refuted" below).
>
> **Fix (eval-only, run kept):** evaluate with the Llama2 tokenizer
> (`bos=1`, `eos=2`), not gemma. The training run itself is untouched and
> valid.

## Symptom

Every Polaris 20B checkpoint scored at chance on every lm-eval task, and a
generation probe emitted fluent-subword salad, e.g.:

```
PROMPT: 'The capital of France is'
GEN:    'KB<unused6><unused6> really specialized vowgresKBgres histó histó histó ...'
```

Real gemma subwords (`histó`, `gres`, `KB`, `abundance`, `attitudes`) in
degenerate loops -- not `<unused>` noise, not a crash. Training loss was
healthy (12.9 -> ~2.05) the whole time.

## Root cause

The Polaris data-list points at Llama2-tokenized dolma:

```
$ head -1 torchtitan/experiments/ezpz/data-lists/polaris/dolma.txt
0.00185... /eagle/datasets/dolma/data_v1.7_Llama2Tokenizer/algebraic-stack-train-0000_text_document algebraic-stack-train
```

The raw Megatron `.bin` is `uint16`, and over a 5,000,000-token sample the
ids are `min=1, max=31017`, with **zero** ids `>= 32000`. That is
physically Llama2 (vocab 32000, fits in uint16); it cannot be gemma
(vocab 256128 needs uint32 and routinely exceeds 65535).

Decoding the exact training ids two ways is the smoking gun:

```
raw ids[:20]: [1, 320, 2042, 29912, 25898, 8117, 13, 3047, 278, 20389, ...]

LLAMA2 DECODE: '\section{Introduction}\r\nWith the explosive growth of
                Internet of Things (IoT) devices, wireless communication
                networks (WCNs) are increasingly facing the challenge of
                allocating finite transmit power and bandwidth...'

GEMMA  DECODE: 'gvityheitsskin instal<unused6>aring= ... vegetation attribute
                abundance ... gres attitudesgres histó ...'
```

The gemma decode of the *training data* is the same salad the *model*
emits at eval -- because the model faithfully learned P(next Llama2-id |
Llama2 context), and eval then encodes prompts and decodes outputs with
gemma. Every id means different text across the two tokenizers.

Meanwhile the model config declares `vocab_size=256128` (gemma), so only
the first ~32000 of 256128 embedding / lm_head rows were ever exercised
by Llama2 data. The rest are untrained.

## Why the Aurora 20B evals were fine

The Aurora v2 chains train on `olmo-mix-1124` = `data_fused_gemma_eod`
(gemma-tokenized), matching the gemma tokenizer used at eval. The Polaris
data-list happened to point at the Llama2 dolma tree. Same model code,
different data tokenizer -- that is the whole difference.

## Hypotheses refuted (in order, with the evidence that killed each)

1. **RoPE Q/K permute (complex vs cos_sin).** A DCP-vs-HF weight compare
   showed every exported tensor (`tok_embeddings`, `w1`, `wq`, `norm`)
   matches the DCP master to bf16 precision (maxdiff = one bf16 ULP;
   mean/std agree to 4 sig figs). A permute would preserve mean/std but
   blow up maxdiff on `wq` -- it didn't. The export is bit-faithful, and
   the run trained cos_sin (`--config=agpt_20b_real`) which is HF-native
   (`CosSinRoPE._rotate_half` == HF `rotate_half`), so skip-permute is
   already correct. Both permute directions were tried; both gibbered.
2. **Missing / random `lm_head` (weight tying).** The export contains
   `lm_head.weight` (579 tensors total). agpt is untied
   (`Decoder.Config.enable_weight_tying=False`) with a separately trained
   `lm_head`, and `tie_word_embeddings:false` in the HF config is correct.
3. **Non-vanilla architecture.** `agpt_20b_real` is plain Llama3: no
   qk_norm, no logit softcap, no relu^2. `rope_theta=500000`,
   `rms_norm_eps=1e-5`, `head_dim=128` all match the HF config.
4. **eval backend.** Both vLLM and plain `transformers` generation produce
   the same salad -- not a vLLM issue.
5. **TP loss-reporting bug.** The run is `tp=1` (`dp_shard=512`), so the
   historical TP>1 `loss/dp_world_size` bug does not apply.

Every structural hypothesis was refuted because the model is genuinely
correct -- in Llama2 id-space. Only a tokenizer mismatch explains a model
that is simultaneously (a) bit-faithful in weights, (b) coherent in
training loss, and (c) salad at eval.

## Fix: evaluate with the Llama2 tokenizer

The training run is kept as-is (Llama2 is its true tokenizer). Only the
eval/export path changes:

1. Build a Llama2 HF tokenizer dir from the SentencePiece model the dolma
   bins were built with:

   ```python
   from transformers import LlamaTokenizer
   tok = LlamaTokenizer(vocab_file="/eagle/datasets/dolma/utils/tokenizer.model")
   tok.save_pretrained("assets/hf/llama2-dolma-tokenizer")  # vocab 32000, bos=1, eos=2
   ```

2. Point the HF export at that tokenizer and correct the special-token ids
   in `config.json` to Llama2's convention (`bos_token_id=1`,
   `eos_token_id=2`; the gemma export had them swapped as `bos=2`/`eos=1`).
   Keep `vocab_size=256128` -- it must match the weight tensor shape; the
   rows `>= 32000` are simply never produced by a 32000-vocab tokenizer.

3. Run lm-eval with `tokenizer=<llama2 dir>` alongside `pretrained=`. All
   the standard MC tasks (hellaswag, arc_*, winogrande, piqa, openbookqa,
   boolq) are **loglikelihood** tasks -- they score gold continuations
   token-by-token and never free-generate into the untrained id range, so
   the vocab-size overhang is harmless.

Reference implementation:
`scripts/eval/polaris_20b_eval_llama2tok_verify.sh` (reuses the existing
faithful safetensors via a `hf-llama2/` sibling dir with symlinked weights
+ Llama2 tokenizer + corrected config).

## Note on the `AgptStateDictAdapter`

The RoPE-aware `AgptStateDictAdapter` (skip the Q/K permute for cos_sin
`_real` flavors) is **correct and necessary** for a bit-faithful cos_sin
export -- it is why the weight compare came back clean. It was not the
gibberish cause, and its earlier docstring wrongly claimed it was. Keep
the code; the causal claim belongs to this tokenizer mismatch, not the
permute.

## How to avoid this next time

- The tokenizer a Megatron `.bin` was built with is encoded in its path
  (`.../data_v1.7_Llama2Tokenizer/...`) and provable from the bin: `uint16`
  + `max(id) < 32000` == Llama2. Check it against the model's declared
  `vocab_size` and the eval tokenizer before trusting eval numbers.
- A model that trains to a healthy loss but evals at chance with
  fluent-subword salad is the fingerprint of a tokenizer id-space
  mismatch, not a weights/RoPE bug. Decode a raw training bin with the
  eval tokenizer first -- if that is salad, the eval tokenizer is wrong.
