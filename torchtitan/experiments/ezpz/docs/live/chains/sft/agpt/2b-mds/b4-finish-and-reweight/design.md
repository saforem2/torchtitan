# B4 cold-start SFT fix: recover + beat B2 after the B3 dilution regression

Date: 2026-07-24
Status: approved (brainstorming), pending implementation plan

## Motivation

B3 (single combined SFT from gs138650 on a broad instruct+CoT mix) REGRESSED the
GSM8K CoT metric: cot_accuracy 0.05 vs B2's 0.205, format 0.86 vs 0.985. A 3-probe
diagnosis (workflow wf_70eef54d) proved this is REAL and a RECIPE problem, not a
bug:
- Merge clean (re-consolidated 3605: 111 tensors, 7.94GB fp32, 0 nan/inf, correct
  arch); eval clean (extractor pulls the model's own boxed number); NOT
  late-collapse (acc flat-low across ckpt 900/1800/3605 = 0.035/0.070/0.050).
- Raw generations = coherent English, well-formed <think>/<answer>, but WRONG
  arithmetic + run-on chains (mean gen_len 601 vs B2 282; 26/200 never close
  </answer>, hit max_tokens).
- Root cause: B3's mix had gsm8k-r1cot at only 0.15, diluted by 0.25 OpenR1 long
  DeepSeek-R1 traces + 0.15 OpenMathInstruct + 0.45 instruction/chat. The
  long-form R1 traces taught a VERBOSE style that spent the 2B's limited capacity
  on style over arithmetic and diluted the eval-matching short-CoT signal. B2's
  0.205 came from a TWO-STAGE lineage whose FINAL stage was short in-distribution
  gsm8k-r1cot.

The OpenR1 loader is NOT emitting wrong answers (an earlier hypothesis) -- the
data is sound; mixing long-form CoT dilutes a 2B's short-GSM8K competence.

## Goal

Beat B2's 0.205 on the 200-problem GSM8K CoT metric by restoring the short-CoT
finishing signal B3 diluted, while using OpenR1's rich reasoning in a form a 2B
can actually use. Two independent paths run in PARALLEL (cheap relative to the
information; they test different hypotheses).

## Path A (B4a) -- finishing stage on the B3 base

Take the B3 checkpoint (already has broad instruction+math+chat ability) and run a
SHORT 2nd SFT stage on gsm8k-r1cot ONLY. Mirrors B2's winning two-stage structure
but on a stronger, broader base -> tests whether the broad B3 pretraining HELPED
and only the finishing signal was missing (best-of-both-worlds hypothesis).

- Base: outputs/evals/cot/b3-3605-hf-diag/ (the B3-3605 re-consolidation the probe
  left on disk -- 7.9G, complete HF: config.json eos=[1,107], tokenizer, lm_head).
  No re-consolidation needed.
- Data: gsm8k-r1cot ONLY (openai/gsm8k main, 7473 short CoT examples). Small enough
  to tokenize inline -- NO offline pretokenize job needed.
- Epochs: 3, but SAVE + EVAL a checkpoint PER EPOCH (B2's 0.205 was 3 epochs, but
  B4a's base is more capable so 3 may overfit the 7473-example set; per-epoch eval
  finds the sweet spot instead of betting on one number).
- Scale: 2N, ~30-60 min. lr 2e-5, gemma template, assistant_only_loss, packing.
- Success: any epoch's checkpoint beats B2's 0.205; watch gen_len falls toward
  ~282 (B2) from B3's 601, and the never-closed-</answer> count drops.

## Path B (B4b) -- reweighted single-stage from gs138650

Fresh single SFT from gs138650 with the mix rebalanced to fix the dilution AND
length-filter OpenR1 so its rich reasoning stays but the run-on failure mode is
removed. Tests whether a single well-balanced SFT beats the two-stage.

- Base: gs138650 (HF, /home/foremans/global_step138650).
- New mix b4_reweight_mix (register in datasets_sft.py):
    gsm8k-r1cot        0.40   (was 0.15 -- restore in-distribution short-CoT weight)
    OpenR1-Math-220k*  0.15   (LENGTH-FILTERED, see below)
    tulu-3-sft-mixture 0.30   (general instruction-following)
    ultrachat-200k     0.15   (multi-turn chat)
  (OpenMathInstruct-2 DROPPED -- it added math breadth the 2B couldn't convert to
  accuracy and inflated the mix; gsm8k-r1cot + short OpenR1 cover math CoT.)
- OpenR1 length filter (the run-on culprit fix): in _openr1_format_row /
  _build_openr1_math_cot, DROP rows whose <think> trace exceeds ~1200 chars (the
  probe's run-on threshold). Keep the short, useful traces. Add a maxlen param
  (env or arg) so the cutoff is tunable; default ~1200.
- max_length: 4096 (no long traces to preserve after the filter -> half the
  attention cost of B3's 8192, and the eval only needs short chains). Re-pretokenize
  the smaller filtered mix offline to /tegu/datasets (cheaper than B3: fewer rows,
  shorter len -> ~1-2h tokenize, small packed size).
- Scale: 8N, 1 epoch, lr 2e-5. Same offline-pretokenized path as B3.
- Success: beats 0.205; gen_len ~short; low never-closed count.

## Shared

- Eval: identical consolidate_and_eval_cot.sh -> 200-problem GSM8K CoT (fp32 vLLM),
  BASE=aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf config,
  TOK_SRC=~/rl-repro/run/agpt2b-ckpt900, eos [1,107]. Compare B2 0.205 / B3 0.05 /
  B4a-per-epoch / B4b on the shared metric.
- Guardrail metrics (from the B3 failure): gen_len (target ~282, not 601) and the
  count of generations that never close </answer> (target near 0, not 26/200) --
  accuracy alone hid the run-on style problem in-loop.
- Quota: after B3, freed to ~14T; B4b's filtered/4096 pretok is far smaller than
  B3's 422G. Watch but low risk.

## Scope (what gets built)

Path A: 1 launcher (agpt2b_b4a_gsm8k_finish_2n.sh) -- 2nd-stage SFT on
b3-3605-hf-diag over gsm8k-r1cot, save_strategy=epoch (3 ckpts). No new data code.
Path B: (1) OpenR1 length-filter param in datasets_sft.py; (2) b4_reweight_mix
registration; (3) pretokenize script (@4096, filtered mix); (4) 8N launcher.
Eval: reuse consolidate_and_eval_cot.sh (+ add gen_len / unclosed-answer reporting
to eval_cot_gsm8k.py so the guardrail is visible).

All under experiments/ezpz/. No core edits.

## Risks / open items

- Path A overfit on 7473 examples at 3 epochs -> mitigated by per-epoch eval.
- Path B length filter drop-rate unknown -> report after build; if it drops too
  much OpenR1, lower the weight rather than the cutoff.
- 2B ceiling: probes note ~0.2 may be near the 2B GSM8K ceiling; if neither path
  clears ~0.22-0.25, the lever is a bigger/math base, not more SFT mixing --
  decide after B4a/B4b numbers land.
