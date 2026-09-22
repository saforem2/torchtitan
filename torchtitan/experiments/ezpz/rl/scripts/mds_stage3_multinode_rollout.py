#!/usr/bin/env python3
"""Generate identical deterministic MDS prompts on each launched node."""

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "2 + 2 =",
    "def sort_list(xs):",
]


def rank() -> int:
    for name in ("PALS_RANKID", "PMI_RANK", "PMIX_RANK", "RANK"):
        if name in os.environ:
            return int(os.environ[name])
    return 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("expected_sha256")
    args = parser.parse_args()

    current_rank = rank()
    hostname = socket.gethostname()
    weight = args.model / "model-00001-of-00001.safetensors"
    actual_sha256 = sha256(weight)
    if actual_sha256 != args.expected_sha256:
        raise RuntimeError(
            f"rank={current_rank} model hash {actual_sha256} != {args.expected_sha256}"
        )

    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="bfloat16",
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=512,
        gpu_memory_utilization=0.7,
        attention_config={"backend": "TRITON_ATTN"},
    )
    outputs = llm.generate(
        PROMPTS,
        SamplingParams(temperature=0.0, max_tokens=64, seed=0),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"rank-{current_rank}-{hostname}.jsonl"
    with output_path.open("w") as stream:
        for prompt, result in zip(PROMPTS, outputs):
            item = result.outputs[0]
            row = {
                "rank": current_rank,
                "hostname": hostname,
                "model_sha256": actual_sha256,
                "prompt": prompt,
                "text": item.text,
                "finish_reason": item.finish_reason,
                "token_ids": item.token_ids,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"MDS_MULTINODE_DONE rank={current_rank} host={hostname} out={output_path}")


if __name__ == "__main__":
    main()
