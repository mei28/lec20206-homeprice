"""Stage one run's model files, config.json, and metrics.json, then create or version the model Dataset.

Usage: uv run python tools/upload_model.py exp000_baseline [--run default] [--title short-name]
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import tyro
from tyro.conf import Positional

from tools import kaggle_assets as ka


def default_title(comp: str, exp: str) -> str:
    return f"{comp}-{exp}-model"


def stage(root: Path, exp: str, run: str, *, user: str, comp: str, title: str | None = None) -> Path:
    run_dir = root / "output" / exp / run
    stage_dir = root / "output" / "model_upload"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    ka.copy_tree(run_dir / "model", stage_dir / "model")
    for name in ("config.json", "metrics.json"):
        shutil.copy2(run_dir / name, stage_dir / name)
    ka.write_dataset_metadata(stage_dir, user, title or default_title(comp, exp))
    return stage_dir


@dataclass
class Args:
    exp: Positional[str]
    run: str = "default"
    title: str | None = None
    notes: str = "update"
    comp: str = os.environ.get("COMP", "")
    user: str = os.environ.get("KAGGLE_USERNAME", "")


def main(args: Args) -> None:
    if not args.comp or not args.user:
        raise SystemExit("COMP and KAGGLE_USERNAME must be set (see .env)")
    root = Path(__file__).resolve().parents[1]
    stage_dir = stage(root, args.exp, args.run, user=args.user, comp=args.comp, title=args.title)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    dataset = f"{args.user}/{ka.slugify(args.title or default_title(args.comp, args.exp))}"
    ka.create_or_version(api, stage_dir, dataset, f"{args.exp}/{args.run}: {args.notes}")
    print(f"uploaded {dataset}; waiting for processing ...")
    ka.wait_ready(api, dataset, interval=15)
    print(f"{dataset} ready")


if __name__ == "__main__":
    main(tyro.cli(Args))
