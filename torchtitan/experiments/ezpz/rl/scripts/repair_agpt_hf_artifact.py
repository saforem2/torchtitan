#!/usr/bin/env python3
"""Repair and validate the HF interface for Gemma-tokenized AGPT SFT models.

AGPT's SFT jobs installed their chat template at runtime, so FSDP checkpoint
consolidation does not preserve it.  This tool makes the reconstructed HF
artifact self-contained and reproduces the exact SFT serialization contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


AGPT_GEMMA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' or message['role'] == 'user' %}"
    "<start_of_turn>user\n{{ message['content'].lstrip('\\n') }}<end_of_turn>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<start_of_turn>model\n"
    "{% generation %}{{ message['content'].lstrip('\\n') }}<end_of_turn>\n{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)

EXPECTED_TOKEN_IDS = {
    "<pad>": 0,
    "<eos>": 1,
    "<bos>": 2,
    "<start_of_turn>": 106,
    "<end_of_turn>": 107,
}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def repair_artifact(checkpoint_dir: Path) -> None:
    """Install the training-time template and correct generation metadata."""
    config_path = checkpoint_dir / "config.json"
    tokenizer_config_path = checkpoint_dir / "tokenizer_config.json"

    config = _read_json(config_path)
    tokenizer_config = _read_json(tokenizer_config_path)

    config.update(bos_token_id=2, eos_token_id=[1, 107], pad_token_id=0)
    tokenizer_config["chat_template"] = AGPT_GEMMA_CHAT_TEMPLATE
    generation_config = {
        "bos_token_id": 2,
        "eos_token_id": [1, 107],
        "pad_token_id": 0,
        "do_sample": True,
    }

    _write_json(config_path, config)
    _write_json(tokenizer_config_path, tokenizer_config)
    _write_json(checkpoint_dir / "generation_config.json", generation_config)
    (checkpoint_dir / "chat_template.jinja").write_text(
        AGPT_GEMMA_CHAT_TEMPLATE + "\n"
    )


def validate_artifact(checkpoint_dir: Path) -> None:
    """Validate metadata, token IDs, and exact inference serialization."""
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)

    for token, expected in EXPECTED_TOKEN_IDS.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != expected:
            raise ValueError(f"{token} has id {actual}, expected {expected}")

    if config.bos_token_id != 2:
        raise ValueError(f"bos_token_id={config.bos_token_id}, expected 2")
    eos_ids = (
        list(config.eos_token_id)
        if isinstance(config.eos_token_id, (list, tuple))
        else [config.eos_token_id]
    )
    if eos_ids != [1, 107]:
        raise ValueError(f"eos_token_id={eos_ids}, expected [1, 107]")
    if config.vocab_size != 256000:
        raise ValueError(f"vocab_size={config.vocab_size}, expected 256000")
    if max(tokenizer.get_vocab().values()) >= config.vocab_size:
        raise ValueError("tokenizer contains an id outside the model vocabulary")

    messages = [
        {"role": "system", "content": "\nSYS  "},
        {"role": "user", "content": "\nQuestion?  "},
    ]
    expected = (
        "<start_of_turn>user\nSYS  <end_of_turn>\n"
        "<start_of_turn>user\nQuestion?  <end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if rendered != expected:
        raise ValueError(
            "chat serialization mismatch:\n"
            f"expected={expected!r}\nactual={rendered!r}"
        )
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if not ids or ids[0] != EXPECTED_TOKEN_IDS["<start_of_turn>"]:
        raise ValueError(f"rendered ids start with {ids[:4]}, expected 106")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    repair_artifact(args.checkpoint_dir)
    if not args.no_validate:
        validate_artifact(args.checkpoint_dir)
    print(f"[agpt-artifact] repaired and validated {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
