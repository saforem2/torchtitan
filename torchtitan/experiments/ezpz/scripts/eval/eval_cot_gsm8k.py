# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Generation-based, chat-templated GSM8K chain-of-thought eval (Stage 0).

Measures the two things the CoT plan gates on, SEPARATELY:
  - format hit-rate: fraction of generations matching
    <think>...</think> <answer>...</answer> with a parseable answer.
  - CoT accuracy: exact-match on the answer extracted ONLY from the
    <answer>/\\boxed{} span AFTER </think> (never from the reasoning text, and
    NOT via a last-standalone-number fallback -- that grabs CoT scratch numbers).

Uses vLLM (the validated agpt-2b inference path). agpt-2b through vLLM must run in
fp32 or it emits gibberish, so default dtype=float32 (override with --dtype).

Run on a compute node in the rl-vllm venv, e.g.:
  venvs/rl-vllm/bin/python torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py \\
    --model <staged-ckpt-900-dir> --limit 200
"""

from __future__ import annotations

import argparse
import json
import re

# gsm8k gold answers end in "#### N"; strip commas and $.
_GOLD_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
# require a <think>...</think> then <answer>...</answer> (answer AFTER think).
_FORMAT_RE = re.compile(
    r"<think>.*?</think>\s*<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _norm_num(s: str) -> str | None:
    """Normalize a numeric string for exact-match (drop commas/$/spaces)."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    m = _NUM_RE.search(s)
    if not m:
        return None
    n = m.group(0).replace(",", "")
    # canonicalize 5.0 -> 5
    try:
        f = float(n)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def gold_answer(gsm8k_answer: str) -> str | None:
    m = _GOLD_RE.search(gsm8k_answer)
    return _norm_num(m.group(1)) if m else None


def extract_cot_answer(text: str) -> tuple[bool, str | None]:
    """(format_ok, extracted_answer). Only reads the <answer> span after
    </think>; inside it prefers \\boxed{}, else the first number in the span.
    Returns format_ok=False (and still tries a fallback answer) when the
    strict envelope is absent."""
    m = _FORMAT_RE.search(text)
    if m:
        span = m.group(1)
        boxed = _BOXED_RE.search(span)
        ans = _norm_num(boxed.group(1)) if boxed else _norm_num(span)
        return (ans is not None), ans
    # Not well-formed: strip any <think> block, then look for boxed / #### only
    # (NO last-number fallback -- that would score CoT scratch numbers).
    stripped = _THINK_RE.sub("", text)
    boxed = _BOXED_RE.search(stripped)
    if boxed:
        return False, _norm_num(boxed.group(1))
    hard = re.search(r"####\s*(-?[\d,]+)", stripped)
    if hard:
        return False, _norm_num(hard.group(1))
    return False, None


_PROMPT_SUFFIX = (
    "\nReason step by step inside <think></think>, then give the final answer "
    "inside <answer>\\boxed{}</answer>."
)


def build_prompts(tokenizer, questions: list[str]) -> list[str]:
    out = []
    for q in questions:
        msgs = [{"role": "user", "content": q + _PROMPT_SUFFIX}]
        out.append(
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        )
    return out


def summarize(texts, finish_reasons, fmts, corrects):
    n = len(texts)
    n_format = sum(int(x) for x in fmts)
    n_correct = sum(int(x) for x in corrects)
    n_unclosed = sum(1 for fr in finish_reasons if fr == "length")
    mean_gen_len = round(sum(len(t) for t in texts) / n, 1) if n else 0.0
    return {
        "n": n,
        "format_hit_rate": round(n_format / n, 4) if n else 0.0,
        "cot_accuracy": round(n_correct / n, 4) if n else 0.0,
        "mean_gen_len": mean_gen_len,
        "n_unclosed": n_unclosed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF checkpoint dir (with chat template)")
    ap.add_argument("--limit", type=int, default=200, help="num test examples")
    ap.add_argument("--dtype", default="float32", help="vLLM dtype (fp32 for agpt-2b)")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy headline")
    ap.add_argument("--gpu-mem", type=float, default=0.70)
    ap.add_argument("--out", default=None, help="write per-example jsonl here")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = load_dataset("gsm8k", "main", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    questions = [r["question"] for r in ds]
    golds = [gold_answer(r["answer"]) for r in ds]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(tokenizer, questions)

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=True,
        max_model_len=2048,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sp)

    texts = []
    finish_reasons = []
    fmts = []
    corrects = []
    rows = []
    for i, o in enumerate(outputs):
        text = o.outputs[0].text
        finish_reason = o.outputs[0].finish_reason
        fmt_ok, ans = extract_cot_answer(text)
        correct = ans is not None and golds[i] is not None and ans == golds[i]
        texts.append(text)
        finish_reasons.append(finish_reason)
        fmts.append(fmt_ok)
        corrects.append(correct)
        rows.append(
            {"idx": i, "format_ok": fmt_ok, "pred": ans, "gold": golds[i],
             "correct": correct, "gen_len": len(text),
             "finish_reason": finish_reason}
        )

    summary = summarize(
        texts=texts, finish_reasons=finish_reasons, fmts=fmts, corrects=corrects
    )
    summary.update({
        "model": args.model,
        "dtype": args.dtype,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    })
    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("wrote per-example ->", args.out)


if __name__ == "__main__":
    main()
