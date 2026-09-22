#!/usr/bin/env python3
"""Direct greedy generation smoke for AGPT MDS stage3-mix HF exports."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    prompts = [
        "The capital of France is",
        "Water boils at a temperature of",
        "2 + 2 =",
        "def sort_list(xs):",
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("xpu")
    model.eval()
    rows = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        inputs = {key: value.to("xpu") for key, value in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=64,
                eos_token_id=1,
                pad_token_id=0,
            )
        continuation = generated[0, inputs["input_ids"].shape[1] :].cpu().tolist()
        row = {
            "prompt": prompt,
            "text": tokenizer.decode(continuation, skip_special_tokens=False),
            "token_ids": continuation,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    print("MDS_STAGE3_DIRECT_DONE", args.output)


if __name__ == "__main__":
    main()
