# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

EZPZ = Path(__file__).parents[1] / "torchtitan" / "experiments" / "ezpz"
ONEOFF = EZPZ / "scripts" / "eval" / "oneoff"
REEVAL = ONEOFF / "reeval-ropefix-sweep.sh"
STATUS = ONEOFF / "ropefix_sweep_status.sh"


def _inline_eval_driver() -> str:
    text = REEVAL.read_text()
    match = re.search(r"python3 <<'PYEOF'\n(.*?)\nPYEOF", text, re.DOTALL)
    assert match, "reeval script must contain its inline lm-eval driver"
    return match.group(1)


def test_reeval_driver_merges_new_results_into_existing_file(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    results_path = results_dir / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "winogrande": {"acc,none": 0.61},
                "winogrande@0shot": {"acc,none": 0.61},
                "n-shot": {"winogrande": 0, "winogrande@0shot": 0},
            }
        )
    )

    stubs = tmp_path / "stubs"
    (stubs / "transformers").mkdir(parents=True)
    (stubs / "transformers" / "__init__.py").write_text("")
    (stubs / "transformers" / "modeling_utils.py").write_text("")
    (stubs / "lm_eval").mkdir()
    (stubs / "lm_eval" / "__init__.py").write_text(
        "class Evaluator:\n"
        "    @staticmethod\n"
        "    def simple_evaluate(**kwargs):\n"
        "        shots = kwargs['num_fewshot']\n"
        "        return {\n"
        "            'results': {task: {'acc,none': shots / 100} for task in kwargs['tasks']},\n"
        "            'n-shot': {task: shots for task in kwargs['tasks']},\n"
        "        }\n"
        "evaluator = Evaluator()\n"
    )
    env = os.environ | {
        "PYTHONPATH": str(stubs),
        "HF_DIR": str(tmp_path / "hf"),
        "RES_DIR": str(results_dir),
        "TASKS": "arc_easy",
        "SHOTS_SPEC": "5:mmlu;25:arc_challenge",
    }

    subprocess.run([sys.executable, "-c", _inline_eval_driver()], env=env, check=True)

    payload = json.loads(results_path.read_text())
    assert payload["winogrande"]["acc,none"] == 0.61
    assert payload["winogrande@0shot"]["acc,none"] == 0.61
    assert payload["arc_easy@0shot"]["acc,none"] == 0
    assert payload["mmlu@5shot"]["acc,none"] == 0.05
    assert payload["arc_challenge@25shot"]["acc,none"] == 0.25
    assert payload["n-shot"]["winogrande"] == 0
    assert payload["n-shot"]["mmlu"] == 5
    assert payload["n-shot"]["arc_challenge"] == 25


def _write_results(root: Path, arm: str, step: int, tasks: list[str]) -> None:
    path = root / "outputs" / "evals" / f"agpt-{arm}" / f"step-{step}" / "results"
    path.mkdir(parents=True)
    (path / "results.json").write_text(json.dumps({task: {} for task in tasks}))


def test_status_requires_task_coverage_not_matching_step_counts(tmp_path: Path) -> None:
    tasks = {
        "20b-v2-512n": ["arc_challenge", "hellaswag", "arc_easy"],
        "20b-v2-256n": ["arc_challenge", "hellaswag", "arc_easy"],
        "2b-v2-512n": [
            "arc_easy",
            "hellaswag",
            "winogrande",
            "piqa",
            "openbookqa",
            "boolq",
            "mmlu",
            "arc_challenge",
        ],
    }
    switches = {"20b-v2-512n": 4401, "20b-v2-256n": 3101, "2b-v2-512n": 30401}
    for arm, required in tasks.items():
        step = switches[arm]
        _write_results(tmp_path, arm, step, required)
        _write_results(tmp_path, f"{arm}-ropefix", step, required[:-1])

    run = subprocess.run(
        ["bash", str(STATUS)],
        env=os.environ | {"R": str(tmp_path)},
        text=True,
        capture_output=True,
        check=True,
    )

    assert run.stdout.count("pending (1 task result left)") == 3
    assert "COMPLETE" not in run.stdout


def test_status_reports_complete_only_with_all_task_results(tmp_path: Path) -> None:
    tasks = {
        "20b-v2-512n": ["arc_challenge", "hellaswag", "arc_easy"],
        "20b-v2-256n": ["arc_challenge", "hellaswag", "arc_easy"],
        "2b-v2-512n": [
            "arc_easy",
            "hellaswag",
            "winogrande",
            "piqa",
            "openbookqa",
            "boolq",
            "mmlu",
            "arc_challenge",
        ],
    }
    switches = {"20b-v2-512n": 4401, "20b-v2-256n": 3101, "2b-v2-512n": 30401}
    for arm, required in tasks.items():
        step = switches[arm]
        _write_results(tmp_path, arm, step, required)
        _write_results(tmp_path, f"{arm}-ropefix", step, required)

    run = subprocess.run(
        ["bash", str(STATUS)],
        env=os.environ | {"R": str(tmp_path)},
        text=True,
        capture_output=True,
        check=True,
    )

    assert run.stdout.count("COMPLETE") == 3
    assert "pending" not in run.stdout


def test_status_accepts_shot_qualified_task_results(tmp_path: Path) -> None:
    tasks = {
        "20b-v2-512n": ["arc_challenge", "hellaswag", "arc_easy"],
        "20b-v2-256n": ["arc_challenge", "hellaswag", "arc_easy"],
        "2b-v2-512n": [
            "arc_easy",
            "hellaswag",
            "winogrande",
            "piqa",
            "openbookqa",
            "boolq",
            "mmlu",
            "arc_challenge",
        ],
    }
    switches = {"20b-v2-512n": 4401, "20b-v2-256n": 3101, "2b-v2-512n": 30401}
    for arm, required in tasks.items():
        step = switches[arm]
        _write_results(tmp_path, arm, step, required)
        qualified = [f"{task}@5shot" for task in required]
        _write_results(tmp_path, f"{arm}-ropefix", step, qualified)

    run = subprocess.run(
        ["bash", str(STATUS)],
        env=os.environ | {"R": str(tmp_path)},
        text=True,
        capture_output=True,
        check=True,
    )

    assert run.stdout.count("COMPLETE") == 3
    assert "pending" not in run.stdout
