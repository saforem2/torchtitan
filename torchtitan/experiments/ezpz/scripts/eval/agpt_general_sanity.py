#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small deterministic direct-generation sanity suite for AGPT HF artifacts."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROMPTS = [
    "Answer in one short sentence: What planet do humans live on?",
    "Extract only the city name: The conference will take place in Chicago on Monday.",
    "Summarize in one sentence: Plants use sunlight, water, and carbon dioxide to make sugars and release oxygen.",
    "If a book costs $7 and you buy 3 books, state the total cost in one sentence.",
    "Name the three primary colors, separated by commas.",
    "Rewrite politely: Send me the report now.",
    "Which is larger, 17 or 12? Answer with only the larger number.",
    "Complete the analogy in one word: bird is to sky as fish is to ___.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rendered = [
        "<start_of_turn>user\n"
        + prompt.lstrip("\n")
        + "<end_of_turn>\n<start_of_turn>model\n"
        for prompt in PROMPTS
    ]
    ids = tokenizer.encode(rendered[0], add_special_tokens=False)
    if not ids or ids[0] != 106 or ids[-1] != 108 or 107 not in ids:
        raise ValueError("AGPT tokenizer or prompt serialization mismatch")
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="float32",
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=0.7,
        attention_config={"backend": "TRITON_ATTN"},
    )
    params = SamplingParams(temperature=0.0, max_tokens=128, stop_token_ids=[1, 107])
    try:
        outputs = llm.generate(rendered, params)
    finally:
        llm.llm_engine.engine_core.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for prompt, result in zip(PROMPTS, outputs):
            item = result.outputs[0]
            row = {
                "prompt": prompt,
                "text": item.text,
                "finish_reason": item.finish_reason,
                "token_ids": item.token_ids,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False))
    print("AGPT_GENERAL_SANITY_DONE", args.output)


if __name__ == "__main__":
    main()
