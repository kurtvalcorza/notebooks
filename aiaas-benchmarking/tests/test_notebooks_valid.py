"""Deterministic artifact check: every notebook is valid JSON and each Python
code cell parses. Catches corrupt/truncated notebooks and syntax errors a
reviewer would otherwise have to spot by eye.

Cells whose source uses Jupyter line/cell magics (`!`, `%`) are skipped — they
aren't plain Python and don't parse with `ast`.
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
        if any(line.lstrip().startswith(("!", "%")) for line in src.splitlines()):
            continue  # IPython magic / shell line, not plain Python
        try:
            ast.parse(src)
        except SyntaxError as e:
            raise AssertionError(
                f"{os.path.basename(path)} cell {i} does not parse: {e}"
            ) from e
