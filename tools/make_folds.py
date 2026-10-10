"""Build a versioned fold file under folds/ from the competition's training table.

Usage: uv run python tools/make_folds.py skf5_seed42 --strategy StratifiedKFold --target Survived \
           --id-col PassengerId [--group col] [--time-col col] [--stratify-bins 10] [--n-splits 5] [--seed 42]
"""

import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import tyro
from tyro.conf import Positional

from kgl import cv
from kgl.env import comp_dir, detect


def run(
    train: Path,
    out: Path,
    *,
    id_col: str,
    strategy: str,
    n_splits: int,
    seed: int,
    target: str | None = None,
    group: str | None = None,
    time_col: str | None = None,
    stratify_bins: int | None = None,
) -> Path:
    df = pl.read_csv(train)
    folds = cv.make_folds(
        df,
        strategy=strategy,
        n_splits=n_splits,
        seed=seed,
        target=target,
        group=group,
        time_col=time_col,
        stratify_bins=stratify_bins,
    )
    meta = {
        "strategy": strategy,
        "n_splits": n_splits,
        "seed": seed,
        "target": target,
        "group": group,
        "time_col": time_col,
        "stratify_bins": stratify_bins,
        "id_col": id_col,
        "source": str(train),
    }
    cv.save_folds(df[id_col], folds, out, meta)
    return out


@dataclass
class Args:
    name: Positional[str]
    strategy: str
    id_col: str
    target: str | None = None
    group: str | None = None
    time_col: str | None = None
    stratify_bins: int | None = None
    n_splits: int = 5
    seed: int = 42
    comp: str = os.environ.get("COMP", "")
    train_file: str = "train.csv"


def main(args: Args) -> None:
    env = detect()
    if not args.comp:
        raise SystemExit("--comp is required (or set COMP in .env)")
    out = run(
        comp_dir(env, args.comp) / args.train_file,
        env.root / "folds" / f"{args.name}.parquet",
        id_col=args.id_col,
        strategy=args.strategy,
        n_splits=args.n_splits,
        seed=args.seed,
        target=args.target,
        group=args.group,
        time_col=args.time_col,
        stratify_bins=args.stratify_bins,
    )
    print(out)


if __name__ == "__main__":
    main(tyro.cli(Args))
