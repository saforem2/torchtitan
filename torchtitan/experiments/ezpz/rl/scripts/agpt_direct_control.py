#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Direct vLLM semantic check for repaired AGPT HF artifacts on XPU."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROMPTS = [
    "Sort these names in alphabetical order by FIRST name: BethMillar\n\n"
    "Reply with ONLY this block, one name per line, nothing before or after:\n"
    "<alphabetical_sorted>\nName1\nName2\n...\n</alphabetical_sorted>\n\n"
    "Formatting example (DIFFERENT names -- do not reuse these; sort the names "
    "given above instead):\n<alphabetical_sorted>\nQuinnRivera\nOmarSaito\n"
    "PiaValdez\n</alphabetical_sorted>",
    "Sort these names in alphabetical order by FIRST name: "
    "ShahramKhosravi, AliceZimmer, BobYoung\n\n"
    "Reply with ONLY this block, one name per line, nothing before or after:\n"
    "<alphabetical_sorted>\nName1\nName2\n...\n</alphabetical_sorted>\n\n"
    "Formatting example (DIFFERENT names -- do not reuse these; sort the names "
    "given above instead):\n<alphabetical_sorted>\nQuinnRivera\nOmarSaito\n"
    "PiaValdez\n</alphabetical_sorted>",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in PROMPTS
    ]
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="float32",
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.7,
        attention_config={"backend": "TRITON_ATTN"},
    )
    params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=128,
        n=4,
        seed=0,
        stop_token_ids=[1, 107],
    )
    outputs = llm.generate(rendered, params)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for prompt, rendered_prompt, result in zip(PROMPTS, rendered, outputs):
            for item in result.outputs:
                row = {
                    "prompt": prompt,
                    "rendered": rendered_prompt,
                    "text": item.text,
                    "finish_reason": item.finish_reason,
                    "token_ids": item.token_ids,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps(row, ensure_ascii=False))
    print("AGPT_DIRECT_DONE", args.output)


if __name__ == "__main__":
    main()
