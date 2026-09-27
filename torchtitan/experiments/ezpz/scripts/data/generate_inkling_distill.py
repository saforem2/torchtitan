# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Build and generate a small, auditable Inkling distillation corpus.

The pipeline deliberately separates prompt selection from generation:

  prepare  -> immutable JSONL prompt manifest + metadata
  generate -> append-only raw API responses (safe to resume)
  validate -> accepted/rejected JSONL + summary + SFT prompt/completion JSONL

The ALCF Minerva endpoint is OpenAI-compatible. Authentication is obtained by
running ``uvx --from alcf-ai alcf-ai auth get-access-token``; tokens are never
written to output artifacts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import threading
import time
from typing import Any
import urllib.error
import urllib.request

BASE_URL = "https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1"
MODEL = "inkling-bf16"
SEED = 154391
PROMPT_LIMIT = 1_600
RESPONSE_LIMIT = 2_000
SOURCE_QUOTAS = {
    "math_gsm8k": 1_000,
    "math_openmath": 1_000,
    "instruction_ifdata": 500,
    "instruction_no_robots": 500,
    "instruction_flan": 500,
    "instruction_aya": 500,
    "code": 500,
    "science": 500,
}
TULU_SOURCE_MAP = {
    "instruction_ifdata": "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
    "instruction_no_robots": "ai2-adapt-dev/no_robots_converted",
    "instruction_flan": "ai2-adapt-dev/flan_v2_converted",
    "instruction_aya": "ai2-adapt-dev/tulu_v3.9_aya_100k",
    "code": "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
    "science": "ai2-adapt-dev/tulu_v3.9_sciriff_10k",
}
MMLU_SUBJECTS = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
)


def normalize_text(text: str) -> str:
    return " ".join(str(text).casefold().split())


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def normalize_answer(text: str) -> str:
    text = str(text).strip().replace(",", "").replace("$", "")
    text = re.sub(r"^\\boxed\{(.*)\}$", r"\1", text)
    return re.sub(r"\s+", "", text).rstrip(".")


def extract_gsm_answer(text: str) -> str | None:
    matches = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", str(text))
    return normalize_answer(matches[-1]) if matches else None


def first_user_message(messages: list[dict[str, Any]]) -> str:
    for message in messages or []:
        if message.get("role") == "user" and str(message.get("content", "")).strip():
            return str(message["content"]).strip()
    return ""


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = str(text).strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("teacher output must be a JSON object")
    return value


def has_pathological_repetition(text: str) -> bool:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if len(lines) >= 4 and len(set(lines)) <= len(lines) // 2:
        return True
    words = re.findall(r"\S+", text.casefold())
    if len(words) < 24:
        return False
    grams = [tuple(words[i : i + 6]) for i in range(len(words) - 5)]
    return bool(grams) and (len(grams) - len(set(grams))) / len(grams) > 0.25


def validate_generation(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if record.get("error"):
        return None, "api_error"
    try:
        parsed = parse_json_object(record.get("content", ""))
    except (json.JSONDecodeError, ValueError):
        return None, "invalid_json"
    category = str(record["category"])
    if category.startswith("math_"):
        rationale = str(parsed.get("rationale", "")).strip()
        answer = str(parsed.get("final_answer", "")).strip()
        if not rationale or not answer:
            return None, "missing_math_fields"
        if normalize_answer(answer) != normalize_answer(record.get("expected_answer", "")):
            return None, "answer_mismatch"
        completion = f"<think>{rationale}</think>\n<answer>\\boxed{{{answer}}}</answer>"
    else:
        completion = str(parsed.get("response", "")).strip()
        if not completion:
            return None, "missing_response"
    if len(completion) > RESPONSE_LIMIT:
        return None, "response_too_long"
    if has_pathological_repetition(completion):
        return None, "repetition"
    accepted = {
        "id": record["id"],
        "category": category,
        "source": record["source"],
        "prompt_hash": record["prompt_hash"],
        "prompt": [{"role": "user", "content": record["prompt"]}],
        "completion": [{"role": "assistant", "content": completion}],
        "teacher": record.get("model", MODEL),
        "request_id": record.get("request_id"),
        "usage": record.get("usage"),
    }
    return accepted, "accepted"


def _held_out_hashes() -> set[str]:
    from datasets import load_dataset

    specs = [
        ("openai/gsm8k", "main", "test", ("question",)),
        ("DigitalLearningGmbH/MATH-lighteval", None, "test", ("problem",)),
        ("HuggingFaceH4/MATH-500", None, "test", ("problem",)),
        ("google/IFEval", None, "train", ("prompt",)),
    ]
    hashes: set[str] = set()
    for name, config, split, fields in specs:
        ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
        for row in ds:
            for field in fields:
                value = row.get(field)
                if value:
                    hashes.add(prompt_hash(value))
    for subject in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        hashes.update(prompt_hash(question) for question in ds["question"] if question)
    return hashes


def _sample_unique_rows(
    rows: list[dict[str, Any]], count: int, seed: int, seen: set[str]
) -> list[dict[str, Any]]:
    """Deterministically sample a full quota while skipping global duplicates."""
    candidates = list(rows)
    random.Random(seed).shuffle(candidates)
    selected = []
    for row in candidates:
        digest = prompt_hash(row["prompt"])
        if digest in seen:
            continue
        seen.add(digest)
        row = dict(row)
        row["prompt_hash"] = digest
        selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError(
        f"source has only {len(selected)} unique eligible rows after deduplication; "
        f"needs {count}"
    )


def prepare(output: Path) -> None:
    from datasets import load_dataset

    held_out = _held_out_hashes()
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    gsm_rows = []
    for row in gsm:
        prompt = str(row["question"]).strip()
        answer = extract_gsm_answer(row["answer"])
        if prompt and answer and len(prompt) <= PROMPT_LIMIT and prompt_hash(prompt) not in held_out:
            gsm_rows.append({"prompt": prompt, "expected_answer": answer})
    for row in _sample_unique_rows(
        gsm_rows, SOURCE_QUOTAS["math_gsm8k"], SEED + 1, seen
    ):
        selected.append({"category": "math_gsm8k", "source": "openai/gsm8k", **row})

    om = load_dataset("nvidia/OpenMathInstruct-2", split="train")
    om_rows = []
    for row in om:
        if row.get("problem_source") not in {"augmented_math", "original_math"}:
            continue
        prompt = str(row.get("problem", "")).strip()
        answer = str(row.get("expected_answer", "")).strip()
        if prompt and answer and len(prompt) <= PROMPT_LIMIT and prompt_hash(prompt) not in held_out:
            om_rows.append({"prompt": prompt, "expected_answer": answer})
    for row in _sample_unique_rows(
        om_rows, SOURCE_QUOTAS["math_openmath"], SEED + 2, seen
    ):
        selected.append({"category": "math_openmath", "source": "nvidia/OpenMathInstruct-2", **row})

    tulu = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    by_source: dict[str, list[dict[str, Any]]] = {k: [] for k in TULU_SOURCE_MAP}
    reverse = {v: k for k, v in TULU_SOURCE_MAP.items()}
    for row in tulu:
        category = reverse.get(row.get("source"))
        if not category:
            continue
        prompt = first_user_message(row.get("messages") or [])
        if not prompt or len(prompt) > PROMPT_LIMIT or prompt_hash(prompt) in held_out:
            continue
        by_source[category].append({"prompt": prompt})
    for offset, (category, source) in enumerate(TULU_SOURCE_MAP.items(), start=10):
        for row in _sample_unique_rows(
            by_source[category], SOURCE_QUOTAS[category], SEED + offset, seen
        ):
            selected.append({"category": category, "source": source, **row})

    if len(selected) != sum(SOURCE_QUOTAS.values()):
        raise AssertionError((len(selected), SOURCE_QUOTAS))
    for row in selected:
        digest = row["prompt_hash"]
        row["id"] = f"inkling-{row['category']}-{digest[:16]}"
    # Deterministically mix categories so any prefix (for example a 100-row
    # canary) samples the whole recipe rather than one alphabetic category.
    random.Random(SEED + 100).shuffle(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    meta = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "seed": SEED,
        "quotas": SOURCE_QUOTAS,
        "rows": len(unique),
        "sha256": digest,
        "held_out_hash_count": len(held_out),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True))


def _teacher_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    if row["category"].startswith("math_"):
        instruction = (
            "Solve the problem correctly. Return only one JSON object with string keys "
            '"rationale" and "final_answer". Keep the rationale concise and self-contained; '
            "do not use markdown fences."
        )
    else:
        instruction = (
            "Answer the user request accurately, directly, and concisely. Return only one JSON "
            'object with the string key "response"; do not use markdown fences.'
        )
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": row["prompt"]},
    ]


def _token() -> str:
    return subprocess.check_output(
        ["uvx", "--from", "alcf-ai", "alcf-ai", "auth", "get-access-token"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _request(row: dict[str, Any], token: str, max_tokens: int, retries: int) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": _teacher_messages(row),
    }
    data = json.dumps(payload).encode()
    last_error = ""
    for attempt in range(retries + 1):
        started = time.monotonic()
        try:
            req = urllib.request.Request(
                BASE_URL + "/chat/completions",
                data=data,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.load(response)
            choice = result["choices"][0]
            return {
                **row,
                "model": result.get("model", MODEL),
                "request_id": result.get("id"),
                "content": choice.get("message", {}).get("content", ""),
                "finish_reason": choice.get("finish_reason"),
                "usage": result.get("usage"),
                "latency_s": round(time.monotonic() - started, 3),
                "attempts": attempt + 1,
                "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            }
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(30, 2**attempt))
    return {**row, "model": MODEL, "error": last_error, "attempts": retries + 1}


def generate(manifest: Path, output: Path, limit: int | None, concurrency: int, max_tokens: int, retries: int) -> None:
    rows = [json.loads(line) for line in manifest.open() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    done: set[str] = set()
    if output.exists():
        for line in output.open():
            if line.strip():
                done.add(json.loads(line)["id"])
    pending = [row for row in rows if row["id"] not in done]
    token = _token()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    started = time.monotonic()
    with output.open("a") as f, concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_request, row, token, max_tokens, retries): row["id"] for row in pending}
        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            record = future.result()
            with lock:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"{len(done)+i}/{len(rows)} {record['id']} attempts={record.get('attempts')} error={bool(record.get('error'))}", flush=True)
    print(json.dumps({"requested": len(rows), "already_done": len(done), "generated": len(pending), "elapsed_s": round(time.monotonic()-started, 3)}))


def validate(raw: Path, accepted: Path, rejected: Path) -> None:
    rows = [json.loads(line) for line in raw.open() if line.strip()]
    good, bad = [], []
    reasons: dict[str, int] = {}
    for row in rows:
        item, reason = validate_generation(row)
        reasons[reason] = reasons.get(reason, 0) + 1
        if item is None:
            bad.append({"id": row.get("id"), "category": row.get("category"), "reason": reason})
        else:
            good.append(item)
    good.sort(key=lambda x: x["id"])
    bad.sort(key=lambda x: str(x["id"]))
    accepted.parent.mkdir(parents=True, exist_ok=True)
    for path, data in ((accepted, good), (rejected, bad)):
        with path.open("w") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "raw_rows": len(rows),
        "accepted_rows": len(good),
        "rejected_rows": len(bad),
        "reasons": reasons,
        "accepted_sha256": hashlib.sha256(accepted.read_bytes()).hexdigest(),
    }
    accepted.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    g = sub.add_parser("generate")
    g.add_argument("--manifest", type=Path, required=True)
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--limit", type=int)
    g.add_argument("--concurrency", type=int, default=4)
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--retries", type=int, default=5)
    v = sub.add_parser("validate")
    v.add_argument("--raw", type=Path, required=True)
    v.add_argument("--accepted", type=Path, required=True)
    v.add_argument("--rejected", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "generate":
        generate(args.manifest, args.output, args.limit, args.concurrency, args.max_tokens, args.retries)
    else:
        validate(args.raw, args.accepted, args.rejected)


if __name__ == "__main__":
    main()
