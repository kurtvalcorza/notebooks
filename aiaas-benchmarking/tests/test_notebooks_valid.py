"""Deterministic artifact check: every notebook is valid JSON and each Python
code cell parses. Catches corrupt/truncated notebooks and syntax errors a
reviewer would otherwise have to spot by eye.

Whole-cell magics (`%%bash`, `%%writefile`, …) make the entire cell non-Python,
so those cells are skipped. For ordinary cells, individual line magics / shell
escapes (`%timeit`, `!pip ...`) are stripped and the remaining Python is parsed —
so a syntax error elsewhere in such a cell is still caught.
"""
import ast
import glob
import json
import os

import pytest

NB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = sorted(glob.glob(os.path.join(NB_DIR, "*.ipynb")))


def test_notebooks_discovered():
    assert NOTEBOOKS, f"no .ipynb found under {NB_DIR}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[os.path.basename(p) for p in NOTEBOOKS])
def test_notebook_valid_json_and_cells_parse(path):
    with open(path) as f:
        nb = json.load(f)  # raises -> fails on invalid JSON

    assert isinstance(nb.get("cells"), list) and nb["cells"], "notebook has no cells"

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        lines = src.splitlines()
        first = next((ln for ln in lines if ln.strip()), "")
        if first.lstrip().startswith("%%"):
            continue  # whole-cell magic (e.g. %%bash) — not Python
        # Strip line magics / shell escapes but keep the surrounding Python, so a
        # syntax error in the real code of a magic-containing cell still fails.
        py = "\n".join(ln for ln in lines if not ln.lstrip().startswith(("!", "%")))
        try:
            ast.parse(py)
        except SyntaxError as e:
            raise AssertionError(
                f"{os.path.basename(path)} cell {i} does not parse: {e}"
            ) from e
