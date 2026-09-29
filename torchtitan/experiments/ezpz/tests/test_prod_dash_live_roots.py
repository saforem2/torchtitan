#!/usr/bin/env python3
"""Regression tests for active-job checkout discovery in prod_dash."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PROD_DASH = os.path.join(HERE, "..", "utils", "prod_dash.py")


def _load_aggregator_symbols(*names):
    with open(PROD_DASH) as stream:
        tree = ast.parse(stream.read())
    template = None
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "_AGG"):
            template = ast.literal_eval(node.value)
            break
    assert template is not None, "_AGG not found in prod_dash.py"
    script = template % {
        "repo": "/repo",
        "ttl": 3600,
        "fresh": 0,
        "show_all": 0,
        "exp_max_age": 604800,
        "live_window": 300.0,
    }
    embedded = ast.parse(script)
    namespace = {
        "json": json,
        "os": os,
        "re": re,
        "subprocess": subprocess,
        "CKPT_RE": re.compile(r"checkpoints/(agpt-[A-Za-z0-9._-]+?)(?:/|\s|$)"),
    }
    found = {}
    for node in embedded.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), "<prod-dash-agg>", "exec"), namespace)
            found[node.name] = namespace[node.name]
    missing = set(names) - found.keys()
    assert not missing, "%s not found in _AGG" % ", ".join(sorted(missing))
    return found


class ActiveJobRootsTest(unittest.TestCase):
    def test_discovers_checkout_from_pbs_output_path(self):
        job_id = "8870515"
        payload = {
            "Jobs": {
                job_id + ".aurora-pbs": {
                    "Output_Path": (
                        "aurora-uan-0009:/lus/flare/projects/AuroraGPT/foremans/"
                        "projects/saforem2/torchtitan-aurora-diag-550d2c670/"
                        "agpt-multi-cont-3seat.o8870515"
                    )
                }
            }
        }
        completed = types.SimpleNamespace(stdout=json.dumps(payload))
        original_run = subprocess.run
        subprocess.run = lambda *args, **kwargs: completed
        try:
            roots = _load_aggregator_symbols("_active_job_roots")[
                "_active_job_roots"
            ]([{"id": job_id, "state": "R"}])
        finally:
            subprocess.run = original_run

        self.assertEqual(roots, [
            "/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/"
            "torchtitan-aurora-diag-550d2c670"
        ])

    def test_trainer_log_ignores_initial_checkpoint_path(self):
        checkpoint_bases = _load_aggregator_symbols("_checkpoint_bases")[
            "_checkpoint_bases"
        ]
        current = "agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9000"
        initial = "agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288"
        text = (
            "--checkpoint.folder=checkpoints/%s "
            "--checkpoint.initial-load-path=/runs/outputs/checkpoints/%s/step-9000"
        ) % (current, initial)

        self.assertEqual(
            checkpoint_bases("trainer-0-20b-n512.console.log", text),
            [current],
        )

    def test_umbrella_log_keeps_every_checkpoint_path(self):
        checkpoint_bases = _load_aggregator_symbols("_checkpoint_bases")[
            "_checkpoint_bases"
        ]
        text = " ".join([
            "checkpoints/agpt-20b-current",
            "checkpoints/agpt-20b-other",
        ])

        self.assertEqual(
            checkpoint_bases("agpt-multi-cont-3seat.o8870515", text),
            ["agpt-20b-current", "agpt-20b-other"],
        )

    def test_detached_refresh_uses_running_interpreter(self):
        with open(PROD_DASH) as stream:
            tree = ast.parse(stream.read())
        template = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "_AGG"
        )

        self.assertNotIn(".venv/bin/python3", template)
        self.assertIn("sys.executable", template)


if __name__ == "__main__":
    unittest.main()
