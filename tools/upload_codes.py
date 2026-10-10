"""Stage src/kgl, one experiment, and pyproject.toml, then create or version the <comp>-codes Dataset.

Usage: uv run python tools/upload_codes.py exp000_baseline [--notes "why"]
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import tyro
from tyro.conf import Positional

from tools import kaggle_assets as ka


def stage(root: Path, exp: str, *, user: str, comp: str) -> Path:
    """Fresh staging directory under output/codes/ with only the files the kernel needs."""
    stage_dir = root / "output" / "codes"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    ka.copy_tree(root / "src" / "kgl", stage_dir / "src" / "kgl")
    ka.copy_tree(root / "experiments" / exp, stage_dir / "experiments" / exp)
    shutil.copy2(root / "pyproject.toml", stage_dir / "pyproject.toml")
    ka.write_dataset_metadata(stage_dir, user, f"{comp}-codes")
    return stage_dir


@dataclass
class Args:
    exp: Positional[str]
    notes: str = "update"
    comp: str = os.environ.get("COMP", "")
    user: str = os.environ.get("KAGGLE_USERNAME", "")


def main(args: Args) -> None:
    if not args.comp or not args.user:
        raise SystemExit("COMP and KAGGLE_USERNAME must be set (see .env)")
    root = Path(__file__).resolve().parents[1]
    stage_dir = stage(root, args.exp, user=args.user, comp=args.comp)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    dataset = f"{args.user}/{ka.slugify(args.comp + '-codes')}"
    ka.create_or_version(api, stage_dir, dataset, args.notes)
    print(f"uploaded {dataset}; waiting for processing ...")
    ka.wait_ready(api, dataset, interval=15)
    print(f"{dataset} ready")


if __name__ == "__main__":
    main(tyro.cli(Args))
