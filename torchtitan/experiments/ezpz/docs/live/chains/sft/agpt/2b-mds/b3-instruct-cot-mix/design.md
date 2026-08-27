# B3 cold-start SFT design: combined instruct + CoT rebuild from gs138650

Date: 2026-07-22
Status: approved (brainstorming), pending implementation plan

## Motivation

The 100-step B2 GRPO run (job 12471432) proved RL is NOT the accuracy lever for
agpt-2b at ~20% GSM8K: merged step-100 eval'd 0.205 -> 0.215 on the 200-problem
GSM8K CoT metric (+1pp, noise), while format went 0.985 -> 1.000. GRPO cleanly
perfects format but cannot raise accuracy when the base's correct-rollout density
is thin. Per docs/live/chains/rl/plans/cot.md, the #1 next lever is a
stronger/longer cold-start SFT. This spec designs that rebuild.

## Goal

ONE SFT run that rebuilds the cold-start from the stage-3 pretrain base, learning
instruction-following AND chain-of-thought reasoning together -- replacing the
current two-stage lineage (gs138650 -> tulu-math SFT -> gsm8k-r1cot SFT = B2).
Call the output B3. Success = beat B2's 200-problem GSM8K CoT accuracy (0.205)
while holding format >= 0.95 and not regressing general instruction-following.

## Key context corrections (from cluster recon)

- "End-of-stage-3 MDS checkpoint" is not a distinct artifact. The MDS 2B is one
  continuous 140k-step run; its "stages" are data-mix transitions. gs138650
  (staged HF at /home/foremans/global_step138650) sits inside stage-3 (~1350 steps
  before the 140k end) and IS the practical stage-3 base already in use.
- Stage-3 pretraining DATA is not SFT-consumable: OLMo/dolmino/nvidia-math
  pretrain corpora in blendcorpus .bin on /flare (Aurora), unreachable from
  Sunspot, and the TRL SFTTrainer takes HF chat datasets, not raw pretrain
  corpora. So "SFT on stage-3 data" is reinterpreted as "SFT on instruction + CoT
  HF datasets" (the reachable, intended version).
- Pretrain seq_len was 8192, so max_length up to 8192 is positionally SAFE
  (learned RoPE range). gs138650 config: max_position_embeddings 131072 (nominal),
  rope_theta 50000, rope_scaling null, hidden 2048, 12 layers, vocab 256000.

## Architecture

    gs138650 (HF, /home/foremans/global_step138650)   # stage-3 pretrain base
       | single TRL SFT (train_sft.py, SFTTrainer, FSDP1 full_shard)
       | assistant_only_loss=True, gemma chat template, packing, max_length 8192
       | data = balanced instruct U CoT mix (below), 1 epoch
       v
    B3 cold-start  ->  consolidate  ->  200-problem GSM8K CoT eval vs B2 0.205

No trainer changes. Reuses train_sft.py (module
torchtitan.experiments.ezpz.rl.train_sft), the existing mix-spec parser, offline
pretokenize path (--pretokenize_to / --pretokenized_dataset), and gemma template.

## Data mix (Approach B, balanced)

datasets_sft.py mix-spec, all_exhausted interleave. Registered as
b3_instruct_cot_mix:

| Component            | Weight | Role                          | Status       |
| -------------------- | ------ | ----------------------------- | ------------ |
| tulu-3-sft-mixture   | 0.30   | general instruction-following | wired        |
| OpenR1-Math-220k     | 0.25   | rich long-form R1 CoT         | NEW loader   |
| gsm8k-r1cot          | 0.15   | in-distribution CoT (eval fmt)| wired        |
| ultrachat-200k       | 0.15   | multi-turn chat               | wired        |
| OpenMathInstruct-2   | 0.15   | math breadth                  | wired        |

= 45% instruction/chat, 55% math/CoT (40% true CoT-envelope). Skeleton mirrors
the proven tulu_math_uc_mix but folds the CoT envelope in rather than bolting a
second SFT stage on after. Weights are a starting point; the 2N smoke validates
them (format learns, loss sane) before the full run.

### OpenR1-Math-220k loader (only new code)

Add to datasets_sft.py. open-r1/OpenR1-Math-220k (HF; needs compute/interactive
proxy for first download). Steps:
1. Load the dataset; inspect actual schema on-cluster during build (generations /
   answer / problem columns).
2. Extract (problem, R1 reasoning trace, final answer) per row.
3. Reformat to our EXACT envelope: <think>{trace}</think>\n<answer>\boxed{ans}
   </answer> -- byte-consistent with gsm8k-r1cot and what eval_cot_gsm8k.py scores.
4. Filter rows whose trace does not yield a clean boxed answer (do not train
   malformed envelopes).

## Sequence length: 8192

Pretrain was 8192 -> 8192 is the positionally safe max. OpenR1 traces are long
(R1 reasoning routinely 2k-10k+ tokens); at shorter lengths a large fraction
truncate mid-reasoning, breaking the envelope. 8192 keeps essentially all traces
intact = maximal reasoning-depth signal. Cost: XPU has no flash-attention, so
attention is O(L^2); 8192 is the highest attention/memory cost. The 2N smoke
tunes micro-batch/grad-accum to fit. max_length is baked into the offline
pretokenize, so it is fixed once per staged dataset.

## Offline data staging

Stage to /tegu/datasets/datasets/ (real path
/lus/tegu/projects/datasets/datasets/, datasets group, separate quota from
datascience; already holds fineweb-edu-100BT). Done ONCE on an interactive node
with proxy, so the 8N job reads locally with HF_HUB_OFFLINE=1 (dodges HF rate
limits + the offline-compute-node problem; avoids 96 ranks racing runtime
tokenize -- a known SIGTERM bottleneck).

1. Raw HF datasets -> /tegu/datasets/datasets/hf/ (HF_HOME/HF_DATASETS_CACHE
   there): OpenR1-Math-220k, tulu-3-sft-mixture, ultrachat-200k,
   OpenMathInstruct-2, gsm8k.
2. Pretokenized, pre-mixed, envelope-formatted SFT dataset @ 8192 ->
   /tegu/datasets/datasets/agpt2b-b3-instruct-cot-mix/ via
   train_sft.py --sft_dataset b3_instruct_cot_mix --pretokenize_to <path>
   (prep only, no training).

Flow:
    [interactive, proxy]  HF_HOME=/tegu/.../hf  download 5 datasets
                          build OpenR1 loader + b3 mix
                          train_sft.py ... --pretokenize_to /tegu/.../agpt2b-b3-instruct-cot-mix
    [8N PBS, offline]     train_sft.py ... --pretokenized_dataset /tegu/.../agpt2b-b3-instruct-cot-mix

On-disk size reported after build (datasets-group quota).

## Training config

- Base: gs138650 (HF). Length: 8192 (baked at pretokenize).
- Batch fit: tulu-math fit micro-batch=2 @ 1024 (2048 tok/device). At 8192 start
  conservative (micro-batch=1 @ 8192 = 8192 tok/device); smoke confirms; hold
  GBS ~ 6144-equivalent via grad-accum + node count.
- Epochs/LR: 1 epoch over the mix; LR 2e-5; gemma template; assistant_only_loss;
  packing.
- Scale: <= 8N (plan pins gs138650 lineage <= 8N; v2-256n base has a 32N oneCCL
  crash). 2N smoke first.
- Checkpointing: interval sized so partial progress saves.
- Naming: output B3, dir outputs/sft/agpt2b-gs138650-instruct-cot-mix-8n/.

## Eval (the verdict)

Consolidate B3 -> 200-problem GSM8K CoT via eval_cot_gsm8k.py, compare to B2
(acc 0.205, format 0.985). Success = accuracy up, format held >= 0.95. Same
trustworthy metric as the RL verdict -- NOT a 20-sample in-loop number. Optionally
spot-check general instruction-following did not regress.

## Scope (what gets built)

1. OpenR1-Math-220k loader in datasets_sft.py (only new code).
2. b3_instruct_cot_mix mix-spec registered in datasets_sft.py.
3. Staging script: download 5 datasets + pretokenize @ 8192 to /tegu/datasets/
   (interactive+proxy, once).
4. 2N smoke launcher: mix loads, batch fits @ 8192, format learns, loss sane.
5. 8N full launcher: reads the pretokenized copy offline.
6. Eval: reuse the eval half (submit_cot_eval.sh / merge_and_eval_cot_lora.sh eval).

All under experiments/ezpz/. No core/upstream edits.

## Risks / open items

- Batch fit at 8192 on XPU (no flash-attn) is unproven for this model; smoke MUST
  confirm before the 8N run. If OOM at micro-batch=1 @ 8192, fall back to 4096 or
  length-bucket (short data low-len, OpenR1 8192).
- OpenR1 schema unknown until inspected; loader answer-extraction may need
  per-source handling. Filter rate reported after build.
- Staged pretokenized size @ 8192 could be large; report vs datasets-group quota.
- gs138650 base has bos 1 / eos 2 (raw); the gemma chat template + fix_ckpt_eos
  [1,107] convention applies at consolidation for eval, as with B2.
