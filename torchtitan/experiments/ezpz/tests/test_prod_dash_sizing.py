#!/usr/bin/env python3
"""prod_dash figure sizing must never overflow the terminal or kitcat.

The dashboard's inline plot is laid out by kitty unicode placeholders, which
address at most 297 cells per axis; exceeding that is a hard ValueError, not a
clipped image:

    ValueError: image too large for unicode placeholders:
                needs 76x402 cells, max is 297x297

This test re-derives the cell grid the way kitcat does -- render the figure at
dpi * device_pixel_ratio, then ceil(image_px / cell_px) -- and asserts the
result fits both the protocol limit and the visible window, across a spread of
real terminal geometries.

It caught two bugs in the fix itself: a max(4.0, w) figsize floor and
max(20, ...) cell floors, both of which silently re-overflowed an ordinary
80x24 SSH window. Any minimum-size clamp reintroduces that class of bug -- if
a floor comes back, this test should fail.

Pure stdlib: size_figure is extracted by AST so the module's matplotlib /
kitcat / wandb imports are never touched.

Run: python3 torchtitan/experiments/ezpz/tests/test_prod_dash_sizing.py
"""
from __future__ import annotations

import ast
import builtins
import math
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PROD_DASH = os.path.join(HERE, "..", "utils", "prod_dash.py")

# (cols, rows, cell_w, cell_h, dpi_scale, reserved_rows, description)
GEOMETRIES = [
    (402, 90, 8, 16, 2.0, 10, "wide HiDPI (the reported failure family)"),
    (213, 56, 14, 30, 2.0, 10, "kitty retina, typical"),
    (80, 24, 8, 16, 1.0, 10, "plain 80x24 ssh/tmux"),
    (400, 100, 9, 18, 2.0, 12, "very wide"),
    (500, 300, 8, 16, 1.0, 10, "wider than the 297-cell cap"),
    (40, 12, 8, 16, 1.0, 10, "tiny window"),
    (120, 20, 8, 16, 1.0, 18, "board nearly fills the window"),
    (120, 15, 8, 16, 1.0, 20, "board over-fills the window"),
    (200, 50, 16, 32, 2.0, 11, "retina, large cells"),
]


def _load_size_figure():
    """Extract KITCAT_MAX_CELLS + size_figure from prod_dash without importing it."""
    tree = ast.parse(open(PROD_DASH).read())
    ns = {"sys": sys, "os": os}
    found = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(
            node.targets[0], "id", ""
        ) == "KITCAT_MAX_CELLS":
            exec(compile(ast.Module([node], []), "<pd>", "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name == "size_figure":
            exec(compile(ast.Module([node], []), "<pd>", "exec"), ns)
            found = True
    assert found, "size_figure not found in prod_dash.py"
    assert ns.get("KITCAT_MAX_CELLS") == 297, "cell cap changed; update this test"
    return ns


def _cells_for(ns, cols, rows, cw, ch, scale, reserved, dpi=100):
    """figsize -> the cell grid kitcat would request."""
    ns["terminal_cells"] = lambda: (cols, rows, cw, ch)
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "kitcat.terminal_query":
            mod = types.ModuleType(name)
            mod.get_dpi_scale = lambda: scale
            return mod
        return real_import(name, *a, **k)

    builtins.__import__ = fake_import
    try:
        w_in, h_in = ns["size_figure"](reserved_rows=reserved, dpi=dpi)
    finally:
        builtins.__import__ = real_import
    # kitcat renders at dpi*scale, then lays out ceil(px/cell) cells.
    return (math.ceil(w_in * dpi * scale / cw),
            math.ceil(h_in * dpi * scale / ch),
            w_in, h_in)


def main():
    ns = _load_size_figure()
    cap = ns["KITCAT_MAX_CELLS"]
    failures = []
    for cols, rows, cw, ch, scale, reserved, desc in GEOMETRIES:
        c, r, w_in, h_in = _cells_for(ns, cols, rows, cw, ch, scale, reserved)
        budget_rows = max(1, min(rows - reserved - 1, cap))
        problems = []
        if c > cap or r > cap:
            problems.append("exceeds kitcat %dx%d cap" % (cap, cap))
        if c > cols:
            problems.append("wider than window (%d > %d cols)" % (c, cols))
        if r > budget_rows:
            problems.append("taller than free space (%d > %d rows)" % (r, budget_rows))
        status = "FAIL" if problems else "ok"
        print("[%-4s] %-42s %6.1fx%-5.1fin -> %3dx%-3d cells"
              % (status, desc, w_in, h_in, c, r))
        for p in problems:
            print("         %s" % p)
            failures.append((desc, p))
    print()
    if failures:
        print("FAILED: %d geometry check(s)" % len(failures))
        return 1
    print("PASS: all %d geometries fit the window and the cell cap" % len(GEOMETRIES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
