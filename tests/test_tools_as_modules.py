"""Tools import each other as `tools.<name>`, so the justfile must run them with `python -m tools.<name>`."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = sorted(p.stem for p in (ROOT / "tools").glob("*.py") if p.stem not in {"__init__", "kaggle_assets"})


@pytest.mark.parametrize("name", TOOLS)
def test_tool_loads_as_a_module(name):
    result = subprocess.run([sys.executable, "-m", f"tools.{name}", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_justfile_runs_tools_as_modules():
    justfile = (ROOT / "justfile").read_text()
    assert not re.search(r"python tools/", justfile), "use `uv run python -m tools.<name>` so sibling imports resolve"
