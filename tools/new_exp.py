"""Create experiments/<name>/ from a base experiment, rewriting the docstring header.

Usage: uv run python tools/new_exp.py exp001_more_trees [--base exp000_baseline]
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import tyro
from tyro.conf import Positional

NAME_RE = re.compile(r"^exp\d{3}_[a-z0-9_]+$")
SCRIPTS = ("run.py", "infer.py")


def validate_name(name: str) -> None:
    if not NAME_RE.match(name):
        raise ValueError(
            f"experiment name must look like exp012_lgbm_te (exp + 3 digits + _ + snake_case), got {name!r}"
        )


def create(experiments_dir: Path, name: str, base: str) -> Path:
    validate_name(name)
    src = experiments_dir / base
    dst = experiments_dir / name
    if not src.is_dir():
        raise FileNotFoundError(f"base experiment not found: {src}")
    if dst.exists():
        raise FileExistsError(f"{dst} already exists")
    dst.mkdir(parents=True)
    for script in SCRIPTS:
        shutil.copy2(src / script, dst / script)
    run_py = dst / "run.py"
    run_py.write_text(_rewrite_header(run_py.read_text(), name, base))
    return dst


def _rewrite_header(text: str, name: str, base: str) -> str:
    text = re.sub(r'^"""exp\d{3}_[a-z0-9_]+:', f'"""{name}:', text, count=1)
    return re.sub(r"^base: .*$", f"base: {base}", text, count=1, flags=re.M)


@dataclass
class Args:
    name: Positional[str]
    base: str = "exp000_baseline"


def main(args: Args) -> None:
    root = Path(__file__).resolve().parents[1]
    created = create(root / "experiments", args.name, args.base)
    print(created)


if __name__ == "__main__":
    main(tyro.cli(Args))
