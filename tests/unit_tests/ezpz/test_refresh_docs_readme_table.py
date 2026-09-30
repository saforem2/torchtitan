# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[3]
    / "torchtitan/experiments/ezpz/utils/refresh_docs_readme_table.py"
)
SPEC = importlib.util.spec_from_file_location("refresh_docs_readme_table", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_refresh_recently_updated_rewrites_tracked_doc_count(
    tmp_path: Path, monkeypatch
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Docs\n\n"
        "The 25 most-recently-changed docs by git commit date (across all 264\n"
        "docs, not just the curated tables below). Auto-generated.\n\n"
        "<!-- BEGIN recently-updated (auto-generated) -->\n"
        "old\n"
        "<!-- END recently-updated (auto-generated) -->\n"
    )
    rows = [
        ("2026-09-30", f"Doc {index}", f"./doc-{index}.md")
        for index in range(303)
    ]
    monkeypatch.setattr(MODULE, "_collect_doc_rows", lambda _: rows)

    assert MODULE.refresh_recently_updated(readme)

    updated = readme.read_text()
    assert "across all 303\ndocs" in updated
    assert "across all 264\ndocs" not in updated
    assert "[Doc 0](./doc-0.md)" in updated
