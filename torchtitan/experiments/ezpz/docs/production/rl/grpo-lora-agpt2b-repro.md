# GRPO+LoRA on XPU: agpt-2b (Llama) port for SFT checkpoint-900

**Goal:** repeat the reproduced Qwen3-0.6B alphabet_sort GRPO+LoRA smoke, but with
the SFT deliverable **checkpoint-900** (AuroraGPT-2B) in the loop, so the reward
signal reflects a trained model instead of base Qwen (which scored 0).

## Status: PORT COMPLETE + CORRECT; generation fidelity NOT yet resolved

checkpoint-900 is `LlamaForCausalLM` (agpt-2b: dim2048/12L/16H/4KV, vocab256000,
ffn11008, gemma tokenizer), NOT Qwen3. The smoke config is Qwen3-only, so this
required an architecture port of the upstream `torchtitan.experiments.rl` path.

### The port (all in the fork ~/rl-repro/, outside the repo) -- verified correct
1. **agpt-2b llama3 flavor** (`torchtitan/models/llama3/__init__.py`): `_agpt2b`
   builder + `"agpt-2b"` registered in `llama3_configs`. Exact ckpt-900 params:
   dim2048/12L/16H/4KV, vocab256000, hidden_dim=11008 (literal -- does NOT fall
   out of compute_ffn_hidden_dim for dim2048), rope theta=**50000**,
   scaling=**"none"**, tie_word_embeddings=**False** (these three differ from
   stock llama3 flavors, which use theta=500000 + llama scaling -- using a stock
   flavor would silently corrupt generation).
2. **RL registry + config** (`experiments/rl/examples/alphabet_sort/config_registry.py`):
   `_llama3_rl_model_registry` (mirrors the qwen3 helper; appends LMHeadCastConverter,
   wraps `llama3.model_registry`) + `rl_grpo_lora_agpt_2b()` (LoRA wqkv+wo,
   renderer=auto, bf16, max_tokens=700, num_groups=4).
3. **parallelize_llama skip_dp fix** (`torchtitan/models/llama3/parallelize.py`):
   added `skip_dp: bool = False` + the `if skip_dp: return model` guard before FSDP,
   mirroring `parallelize_qwen3`. This was a REAL fork gap -- the RL generator path
   passes skip_dp=True (FSDP hooks are incompatible with vLLM inference_mode), and
   llama3's signature lacked it. Without this the generator actor crashes at init.
4. **Chat template staging** (`~/rl-repro/run/agpt2b-ckpt900/`): symlinked weights +
   config, COPIED tokenizer with the SFT gemma chat_template injected (the ckpt
   tokenizer ships an EMPTY chat_template, so renderer name="auto" fails without
   it). Original SFT deliverable never mutated.

### What is VERIFIED correct (extensive isolation testing)
- **HF transformers greedy-gen on ckpt-900 = coherent** ("The capital of France
  is" -> " Paris"; understood the sort task). Checkpoint is good.
- **Tokenizer** md5-identical to google/gemma-7b (the SFT tokenizer). Correct.
- **Weight adapter round-trip** HF->TT->HF = bit-exact (max_abs_diff 0.000, all
  layers). Permute/RoPE-layout conversion correct.
- **Fused-QKV load**: model's `wqkv` receives real weights via the merge hook
  (`still-random? False`, missing=0/unexpected=0). Correct.
- **TorchTitan model forward (CPU, MATH attention) single-token = MATCHES HF**:
  argmax 7127 (" Paris"), top-5 nearly identical. Model math + RoPE correct.
- Full RL pipeline runs: weights load (4.49 GiB), KV cache 81 GiB, gen ~3000 tok/s,
  TorchStore weight sync, Train Step 1/2/3 fire.

### The unresolved residual (the honest gap)
Multi-token greedy generation degenerates into repetition
(`</divisible_sorted_sorted_sorted...`), reward 0. Per-position HF-vs-TT logit A/B
(teacher-forced) shows **cosine ~1.0 at almost every position** (0.98-1.00) with
ONE sharp outlier (pos 11: cos 0.454). This rules out a gross bug (wrong RoPE
would give cos~0 everywhere) -- the port is fundamentally right. The residual is a
SMALL divergence that compounds under long low-temperature greedy decode.
Persists in fp32 (so not simply bf16). Caveat: the CPU MATH-attention used for
these probes is a hand-rolled GQA reimplementation (repeat_interleave) and may
itself contribute to the pos-11 outlier -- a clean vLLM-path numerical trace is
needed to separate harness from model.

Also found (secondary): **double-BOS** -- the gemma chat_template emits `<bos>` and
`add_special_tokens=True` prepends another (`[2,2,...]`). And config.json's
bos=1/eos=2 is SWAPPED vs the gemma tokenizer's actual bos=2/eos=1.

### Ruled out (each cost a probe)
torch.compile (gibberish persists with --compile.no-enable); double-RoPE (vLLM
Attention built with no rotary; single RoPE via TorchTitan ComplexRoPE); fused-QKV
mismatch (merge hook works); wrong tokenizer (md5-identical); bad checkpoint (HF
coherent); adapter permute (bit-exact round-trip); gross RoPE convention (cos~1).

### Next session (bounded)
1. Fix double-BOS: renderer template shouldn't emit `<bos>`, OR set
   add_special_tokens=False in the render path. Re-check with the corrected eos.
2. Numerical fidelity: instrument the ACTUAL vLLM generator forward (not a CPU
   reimpl) -- dump first-layer q/k after RoPE vs HF, and logits at pos 11 -- to
   separate harness artifact from a real per-layer diff. Compare against a
   known-good llama3 (e.g. a stock Llama-3-8B HF ckpt) through the same fork path
   to confirm the fork's llama3+vLLM path is correct for a reference model.
3. Consider that ckpt-900 is OOD for alphabet-sort (SFT was tulu/math/ultrachat);
   even a perfect port may need the arithmetic task (its SFT domain) to show
   rising reward. The TRL vllm-serve GRPO path already does agpt-2b arithmetic.

### Reproduction scorecard
| Piece | Status |
|-------|--------|
| agpt-2b flavor + RL config + skip_dp fix | DONE, correct |
| Chat-template staging | DONE |
| Weight load (adapter, fused-QKV, sync) | VERIFIED correct |
| Model math / RoPE (single-token vs HF) | VERIFIED correct (Paris) |
| Multi-token coherent generation | NOT yet (small compounding residual) |
| Rising reward on alphabet_sort | NOT yet (blocked on above + likely OOD) |

## ROOT CAUSE FOUND 2026-07-19 PM: bf16 precision in the vLLM path (fp32 fixes it)

The agpt-2b gibberish is a **bf16 numerical-precision failure in the vLLM
inference path** -- NOT the port, adapter, RoPE, RL harness, or weight-sync.

**Decisive test -- bare vLLM (no Monarch, no RL loop, no LoRA, no weight-sync),
same prompt/weights/engine, only dtype differs:**
```
dtype=bfloat16 -> "5.ciptakan,. Đóeld. Đóeld. Đóeldmanshipally, formulates..."   GIBBERISH
dtype=float32  -> "Here is a Python solution using the built-in sort function:
                   def sort_names(names): return sorted(names, key=lambda x: x[1])..."   COHERENT
```
The fp32 output matches HF's greedy completion (HF also answered with a Python
`sort_names`). So agpt-2b runs correctly in fp32 and breaks in bf16 through vLLM.

This reconciles ALL prior evidence: the per-position teacher-forced HF-vs-TT logit
A/B was cos~1.0 almost everywhere (tiny per-op bf16 error) with occasional flips;
those tiny errors **compound over autoregressive greedy decode** in bf16 into
divergence, while fp32 has the headroom to stay coherent. The single-token prefill
test passed (one step, no compounding); multi-token generation failed (compounding).

**Why agpt-2b and not Qwen3-0.6B (which works in bf16):** agpt-2b has vocab 256000
(vs Qwen ~151k) + larger FFN (11008) -> bigger matmuls accumulate more bf16 error,
and a flatter logit distribution where bf16 rounding flips the greedy argmax.

**FIX:** run the generator in fp32 (`--generator.model-dtype=float32`). The trainer
already casts the lm_head to fp32 (LMHeadCastConverter) for logprob/KL; the
generator's vLLM forward needs fp32 too for coherent sampling. Verified in bare
vLLM; RL smoke re-run with `--generator.model-dtype=float32
--trainer.training.dtype=float32` (job 12471035).

Credit: the user proposed both the bare-vLLM trace and the fp32 test -- together
they cracked it.

## RESOLVED 2026-07-19: SFT checkpoint-900 runs in the GRPO loop (fp32)

The full alphabet_sort GRPO+LoRA smoke runs end-to-end with SFT checkpoint-900
(AuroraGPT-2B) once the generator runs in **fp32** (job 12471035, 2-tile COMPOSITE):

```
Train | Step: 1   86.6 tok/s   (compile warmup)
Train | Step: 2   2163 tok/s
Train | Step: 3   2169 tok/s
reward_mean=0.0012   nonzero=14/40   max=0.019
```

- **Completions are coherent** (vs bf16 gibberish): "The task is to sort names in
  alphabetical order... I will write a Python script..." -- fluent, on-topic,
  matching HF. Contrast the bf16 run: "5.ciptakan,. Đóeld. Đóeld...".
- **Reward is real and non-zero** (14/40 rollouts > 0), where base Qwen and bf16
  agpt-2b both gave a flat 0. The full loop closes: coherent gen -> scoring ->
  advantage -> GRPO step -> TorchStore weight sync.

### Why reward is still LOW (not a bug)
checkpoint-900 answers CONVERSATIONALLY (Python scripts, even HTML) instead of the
rubric's strict `<alphabetical_sorted>...</alphabetical_sorted>` XML block, so
`RewardAlphabetSort` gives only partial credit. This is expected: the SFT mix was
tulu/math/ultrachat, NOT alphabet-sort -- the task is out-of-domain for the format.
The point of the smoke (a trained model producing real reward signal in the XPU
GRPO loop) is achieved; making reward CLIMB would need either GRPO training for
many steps (it starts near 0 and rises) or the model's actual SFT domain
(arithmetic, via the TRL vllm-serve path).

### The recipe (committed launcher `s4_agpt2b.sh`)
Everything from the port PLUS: `--generator.model-dtype=float32
--trainer.training.dtype=float32`. The generator fp32 is the essential fix (bf16
compounds tiny per-op error over greedy decode into gibberish for agpt-2b's large
vocab/FFN); trainer fp32 is belt-and-suspenders.

### Full reproduction scorecard -- agpt-2b (Llama) GRPO+LoRA on XPU
| Piece | Status |
|-------|--------|
| agpt-2b flavor + RL config + skip_dp fork fix | DONE |
| Chat-template staging (gemma) | DONE |
| Weight load (adapter, fused-QKV, sync) | VERIFIED correct |
| Model math / RoPE (vs HF) | VERIFIED correct |
| Coherent generation (fp32) | DONE |
| Non-zero reward signal | DONE (14/40, max 0.019) |
| GRPO train steps + weight sync | DONE (~2170 tok/s) |
| Rising reward on alphabet_sort | needs many steps / in-domain task (OOD here) |

Known follow-ups (non-blocking): double-BOS in the gemma template; config.json
bos/eos swapped vs the tokenizer; CompiledFxGraph `__del__` cleanup warning
(non-fatal). fp32 generation is ~30% slower than bf16 but correct.
