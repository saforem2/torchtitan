# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json

from torchtitan.experiments.ezpz.rl.scripts.stage3_star_verify import verify


def _corpus(tmp_path, *, leak: bool):
    raw_rows = [
        {
            "source_index": 1,
            "reason": "accepted",
            "accepted": True,
            # Real generations embed newlines; the verifier must not split on them.
            "generation": "<think>step one.\nstep two.</think>\n<answer>4</answer>",
        },
        {"source_index": 2, "reason": "contaminated_ngram", "accepted": leak},
        {"source_index": 3, "reason": "incorrect", "accepted": False},
    ]
    accepted_rows = [{"source_index": 1}]
    if leak:
        accepted_rows.append({"source_index": 2})

    raw_path = tmp_path / "raw_generations.jsonl"
    accepted_path = tmp_path / "accepted_sft.jsonl"
    raw_path.write_text("".join(json.dumps(r) + "\n" for r in raw_rows))
    accepted_path.write_text("".join(json.dumps(r) + "\n" for r in accepted_rows))
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "accepted_sha256": hashlib.sha256(
                    accepted_path.read_bytes()
                ).hexdigest(),
                "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            }
        )
    )
    return tmp_path


def test_detected_but_excluded_contamination_passes(tmp_path):
    result = verify(_corpus(tmp_path, leak=False), minimum_accepted=1)
    assert result["contaminated_problems_detected"] == 1
    assert result["contaminated_leaked_into_corpus"] == 0
    assert result["failures"] == []


def test_leaked_contamination_fails(tmp_path):
    result = verify(_corpus(tmp_path, leak=True), minimum_accepted=1)
    assert result["contaminated_leaked_into_corpus"] == 1
    assert any("leaked" in f for f in result["failures"])


def test_minimum_accepted_is_enforced(tmp_path):
    result = verify(_corpus(tmp_path, leak=False), minimum_accepted=5000)
    assert any("below required" in f for f in result["failures"])
