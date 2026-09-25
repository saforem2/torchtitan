#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate and strictly filter teacher-free GSM8K STaR traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from torchtitan.experiments.ezpz.rl.decontam_traces import GSM8KDecontaminator
from torchtitan.experiments.ezpz.scripts.eval.eval_cot_gsm8k import (
    build_prompts,
    extract_cot_answer,
)

PROMPT_SUFFIX = (
    "\nReason step by step inside <think></think>, then give the final "
    "answer inside <answer>\\boxed{}</answer>."
)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_STEP_SPLIT_RE = re.compile(r"(?:\n+|(?<=[.!?])\s+)")
_GOLD_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def _norm_num(value: str) -> str | None:
    value = value.strip().replace(",", "").replace("$", "").rstrip(".")
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)
    if not match:
        return None
    number = float(match.group(0).replace(",", ""))
    return str(int(number)) if number == int(number) else str(number)


def reasoning_step_count(text: str) -> int:
    match = _THINK_RE.search(text)
    if not match:
        return 0
    return sum(bool(part.strip()) for part in _STEP_SPLIT_RE.split(match.group(1)))


def assess_trace(text: str, finish_reason: str | None, gold: str) -> tuple[bool, str]:
    if finish_reason != "stop":
        return False, "not_stopped"
    format_ok, answer = extract_cot_answer(text)
    if not format_ok:
        return False, "invalid_envelope"
    if answer != gold:
        return False, "incorrect"
    if reasoning_step_count(text) < 2:
        return False, "fewer_than_two_reasoning_steps"
    return True, "accepted"


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.building")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--seed", type=int, default=154391)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--minimum-accepted", type=int, default=1)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    if args.limit:
        order = order[: min(args.limit, len(order))]
    rows = [train[i] for i in order]
    questions = [str(row["question"]) for row in rows]
    golds = []
    for row in rows:
        match = _GOLD_RE.search(str(row["answer"]))
        gold = _norm_num(match.group(1)) if match else None
        if gold is None:
            raise RuntimeError("selected GSM8K row has no parseable gold answer")
        golds.append(gold)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(tokenizer, questions)
    engine = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="float32",
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.70,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.samples_per_prompt,
        seed=args.seed,
        stop_token_ids=[1, 107],
    )
    try:
        outputs = engine.generate(prompts, sampling)
    finally:
        engine.llm_engine.engine_core.shutdown()

    detector = GSM8KDecontaminator(str(row["question"]) for row in test)
    raw_rows: list[dict] = []
    accepted_rows: list[dict] = []
    reasons: Counter[str] = Counter()
    accepted_hashes: set[str] = set()
    contaminated = 0

    for source_index, question, gold, request in zip(order, questions, golds, outputs):
        accepted_for_problem = False
        collision, collision_reason = detector.is_contaminated(question)
        if collision:
            contaminated += 1
        for sample_index, item in enumerate(request.outputs):
            text = item.text
            accepted, reason = assess_trace(text, item.finish_reason, gold)
            digest = hashlib.sha256(text.encode()).hexdigest()
            if collision:
                accepted, reason = False, f"contaminated_{collision_reason}"
            elif digest in accepted_hashes:
                accepted, reason = False, "duplicate_trace"
            elif accepted and accepted_for_problem:
                accepted, reason = False, "extra_correct_trace"
            raw_rows.append(
                {
                    "source_index": source_index,
                    "sample_index": sample_index,
                    "question": question,
                    "prompt": prompts[len(raw_rows) // args.samples_per_prompt],
                    "generation": text,
                    "gold": gold,
                    "finish_reason": item.finish_reason,
                    "reasoning_steps": reasoning_step_count(text),
                    "accepted": accepted,
                    "reason": reason,
                    "sha256": digest,
                }
            )
            reasons[reason] += 1
            if accepted:
                accepted_for_problem = True
                accepted_hashes.add(digest)
                accepted_rows.append(
                    {
                        "source_index": source_index,
                        "prompt": [
                            {"role": "user", "content": question + PROMPT_SUFFIX}
                        ],
                        "completion": [{"role": "assistant", "content": text}],
                        "generation_sha256": digest,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_generations.jsonl"
    accepted_path = args.output_dir / "accepted_sft.jsonl"
    _write_jsonl_atomic(raw_path, raw_rows)
    _write_jsonl_atomic(accepted_path, accepted_rows)
    report = {
        "model": str(args.model),
        "seed": args.seed,
        "problems": len(questions),
        "samples_per_prompt": args.samples_per_prompt,
        "raw_generations": len(raw_rows),
        "accepted_problems": len(accepted_rows),
        "acceptance_rate": len(accepted_rows) / len(questions) if questions else 0.0,
        "test_collisions": contaminated,
        "reasons": dict(sorted(reasons.items())),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "accepted_sha256": hashlib.sha256(accepted_path.read_bytes()).hexdigest(),
    }
    report_tmp = args.output_dir / ".report.json.building"
    report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report_tmp.replace(args.output_dir / "report.json")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if contaminated:
        raise RuntimeError(f"GSM8K test contamination detected in {contaminated} rows")
    if len(accepted_rows) < args.minimum_accepted:
        raise RuntimeError(
            f"accepted {len(accepted_rows)} problems; required {args.minimum_accepted}"
        )
    print("STAGE3_STAR_GENERATION_OK", flush=True)


if __name__ == "__main__":
    main()
