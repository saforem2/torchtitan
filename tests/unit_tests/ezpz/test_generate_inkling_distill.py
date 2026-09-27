# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
from pathlib import Path


_SCRIPT = (
    Path(__file__).parents[3]
    / "torchtitan/experiments/ezpz/scripts/data/generate_inkling_distill.py"
)
_spec = importlib.util.spec_from_file_location("generate_inkling_distill", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_prompt_hash_normalizes_case_and_whitespace():
    assert _mod.prompt_hash("  Hello\n WORLD ") == _mod.prompt_hash("hello world")


def test_sample_unique_rows_backfills_duplicates():
    seen = {_mod.prompt_hash("duplicate")}
    rows = [
        {"prompt": "duplicate"},
        {"prompt": "unique one"},
        {"prompt": "unique two"},
        {"prompt": "unique three"},
    ]
    selected = _mod._sample_unique_rows(rows, 3, 154391, seen)
    assert len(selected) == 3
    assert all(row["prompt"] != "duplicate" for row in selected)
    assert len({row["prompt_hash"] for row in selected}) == 3


def test_parse_json_object_accepts_plain_and_fenced_json():
    assert _mod.parse_json_object('{"response":"ok"}') == {"response": "ok"}
    assert _mod.parse_json_object('```json\n{"response":"ok"}\n```') == {
        "response": "ok"
    }


def test_request_payload_pins_json_mode_and_low_reasoning():
    payload = _mod._request_payload(
        {"category": "instruction_flan", "prompt": "Say hello."}, 512
    )
    assert payload["model"] == "inkling-bf16"
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 512
    assert payload["reasoning_effort"] == "low"
    assert payload["response_format"] == {"type": "json_object"}


def test_validate_math_requires_exact_gold_answer():
    base = {
        "id": "x",
        "category": "math_gsm8k",
        "source": "test",
        "prompt_hash": "abc",
        "prompt": "What is 6 times 7?",
        "expected_answer": "42",
        "model": "inkling-bf16",
    }
    good, reason = _mod.validate_generation(
        {**base, "content": '{"rationale":"6 times 7 is 42.","final_answer":"42"}'}
    )
    assert reason == "accepted"
    assert good is not None
    assert good["completion"][0]["content"] == (
        "<think>6 times 7 is 42.</think>\n<answer>\\boxed{42}</answer>"
    )
    bad, reason = _mod.validate_generation(
        {**base, "content": '{"rationale":"wrong","final_answer":"41"}'}
    )
    assert bad is None
    assert reason == "answer_mismatch"


def test_validate_general_rejects_missing_and_repetitive_response():
    base = {
        "id": "x",
        "category": "instruction_flan",
        "source": "test",
        "prompt_hash": "abc",
        "prompt": "Explain photosynthesis.",
    }
    missing, reason = _mod.validate_generation({**base, "content": "{}"})
    assert missing is None
    assert reason == "missing_response"

    repeated = " ".join(["repeat this phrase again now forever"] * 8)
    rejected, reason = _mod.validate_generation(
        {**base, "content": '{"response": ' + repr(repeated).replace("'", '"') + "}"}
    )
    assert rejected is None
    assert reason == "repetition"


def test_validate_rejects_api_errors_and_long_outputs():
    base = {
        "id": "x",
        "category": "code",
        "source": "test",
        "prompt_hash": "abc",
        "prompt": "Write a function.",
    }
    assert _mod.validate_generation({**base, "error": "503"})[1] == "api_error"
    record = {**base, "content": '{"response":"' + "x" * 2001 + '"}'}
    assert _mod.validate_generation(record)[1] == "response_too_long"
