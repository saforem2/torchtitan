#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Re-verify a Stage-3 STaR corpus from its artifacts, without regenerating."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(output_dir: Path, minimum_accepted: int) -> dict:
    raw_path = output_dir / "raw_generations.jsonl"
    accepted_path = output_dir / "accepted_sft.jsonl"
    report_path = output_dir / "report.json"
    for path in (raw_path, accepted_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing required artifact: {path}")

    report = json.loads(report_path.read_text())
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    accepted = [
        json.loads(line) for line in accepted_path.read_text().splitlines() if line
    ]

    contaminated = {
        row["source_index"] for row in raw if str(row["reason"]).startswith("contam")
    }
    accepted_indices = {row["source_index"] for row in accepted}
    leaked = sorted(contaminated & accepted_indices)
    accepted_in_raw = sum(1 for row in raw if row["accepted"])

    verified = {
        "output_dir": str(output_dir),
        "raw_generations": len(raw),
        "accepted_rows": len(accepted),
        "accepted_unique_problems": len(accepted_indices),
        "contaminated_problems_detected": len(contaminated),
        "contaminated_leaked_into_corpus": len(leaked),
        "accepted_sha256": hashlib.sha256(accepted_path.read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }

    failures = []
    if leaked:
        failures.append(f"contaminated problems leaked into corpus: {leaked}")
    if len(accepted) != len(accepted_indices):
        failures.append("corpus contains more than one trace per problem")
    if accepted_in_raw != len(accepted):
        failures.append(
            f"raw file marks {accepted_in_raw} accepted but corpus has {len(accepted)}"
        )
    if len(accepted) < minimum_accepted:
        failures.append(f"{len(accepted)} accepted below required {minimum_accepted}")
    if report.get("accepted_sha256") != verified["accepted_sha256"]:
        failures.append("accepted_sft.jsonl hash does not match report.json")
    if report.get("raw_sha256") != verified["raw_sha256"]:
        failures.append("raw_generations.jsonl hash does not match report.json")

    verified["failures"] = failures
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-accepted", type=int, default=2000)
    args = parser.parse_args()

    verified = verify(args.output_dir, args.minimum_accepted)
    print(json.dumps(verified, indent=2, sort_keys=True), flush=True)
    if verified["failures"]:
        raise SystemExit(1)
    print("STAGE3_STAR_CORPUS_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
