"""Fold assignment: build once, save under folds/, and share across experiments."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kgl import registry

TIME_STRATEGIES = {"TimeSeriesSplit"}


def make_folds(
    df: pl.DataFrame,
    *,
    strategy: str,
    n_splits: int,
    seed: int,
    target: str | None = None,
    group: str | None = None,
    time_col: str | None = None,
    stratify_bins: int | None = None,
) -> pl.Series:
    """Return a fold index per row, aligned with `df`.

    `strategy` is a class name in sklearn.model_selection or a dotted path. Time strategies sort by
    `time_col`, never shuffle, and mark the warm-up rows that are never validated with -1. A continuous
    target is binned into `stratify_bins` quantiles so stratified splitters can use it.
    """
    is_time = strategy.rsplit(".", 1)[-1] in TIME_STRATEGIES
    if is_time and time_col is None:
        raise ValueError(f"{strategy} needs time_col")
    spec = strategy if "." in strategy else f"sklearn.model_selection.{strategy}"
    if is_time:
        splitter = registry.build(spec, n_splits=n_splits)
        order = np.argsort(df[time_col].to_numpy(), kind="stable")
    else:
        splitter = registry.build(spec, n_splits=n_splits, shuffle=True, random_state=seed)
        order = np.arange(df.height)

    y = _stratify_target(df, target, stratify_bins) if target is not None else None
    groups = df[group].to_numpy()[order] if group is not None else None
    if y is not None:
        y = y[order]

    folds = np.full(df.height, -1, dtype=np.int64)
    for k, (_, val_idx) in enumerate(splitter.split(np.zeros(df.height), y, groups)):
        folds[order[val_idx]] = k
    return pl.Series("fold", folds)


def _stratify_target(df: pl.DataFrame, target: str, bins: int | None) -> np.ndarray:
    values = df[target].to_numpy()
    if bins is None:
        return values
    edges = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
    return np.searchsorted(edges, values, side="right")


def split(folds: pl.Series | np.ndarray, k: int, *, time: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Row indices for training and validating fold `k`. Time folds train only on earlier folds (and warm-up)."""
    f = np.asarray(folds)
    val_idx = np.flatnonzero(f == k)
    train_idx = np.flatnonzero(f < k) if time else np.flatnonzero(f != k)
    return train_idx, val_idx


def save_folds(ids: pl.Series, folds: pl.Series, path: Path, meta: dict[str, Any]) -> None:
    """Write `(id, fold)` as parquet plus a JSON sidecar describing how the folds were made. Never overwrites."""
    if path.exists():
        raise FileExistsError(f"{path} exists; fold files are versioned by name, so pick a new name instead")
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"id": ids, "fold": folds}).write_parquet(path)
    sidecar = {**meta, "n_rows": len(folds), "n_folds": int(folds.max()) + 1}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")


def load_folds(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def read_fold_meta(path: Path) -> dict[str, Any]:
    return json.loads(path.with_suffix(".json").read_text())


def attach_folds(df: pl.DataFrame, folds_df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    """Join the saved fold column onto `df` by id; every row must receive a fold."""
    out = df.join(folds_df.rename({"id": id_col}), on=id_col, how="left")
    if out.height != df.height:
        raise ValueError(f"join changed the row count ({df.height} -> {out.height}); ids are not unique")
    missing = out["fold"].null_count()
    if missing:
        raise ValueError(f"{missing} rows have no fold; regenerate folds for this data version")
    return out


def check_folds(df: pl.DataFrame, fold_col: str, *, group: str | None = None, time_col: str | None = None) -> None:
    """Assert the properties the CV design relies on: groups stay within one fold, time moves forward across folds."""
    if group is not None:
        spans = df.group_by(group).agg(pl.col(fold_col).n_unique().alias("n"))
        leaking = spans.filter(pl.col("n") > 1)
        assert leaking.height == 0, (
            f"{leaking.height} groups appear in more than one fold: {leaking[group].head(5).to_list()}"
        )
    if time_col is not None:
        bounds = (
            df.group_by(fold_col)
            .agg(pl.col(time_col).min().alias("t_min"), pl.col(time_col).max().alias("t_max"))
            .sort(fold_col)
        )
        t_min, t_max = bounds["t_min"].to_list(), bounds["t_max"].to_list()
        for i in range(1, len(t_min)):
            k, prev = bounds[fold_col][i], bounds[fold_col][i - 1]
            assert t_min[i] > t_max[i - 1], f"fold {k} starts at {t_min[i]} before fold {prev} ends at {t_max[i - 1]}"
