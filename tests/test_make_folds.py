import polars as pl

from kgl import cv
from tools import make_folds


def test_run_writes_versioned_fold_files(frame, tmp_path):
    train = tmp_path / "train.csv"
    frame.write_csv(train)
    out = tmp_path / "folds" / "skf5_seed0.parquet"
    make_folds.run(train, out, id_col="id", strategy="StratifiedKFold", n_splits=5, seed=0, target="target")
    folds = cv.load_folds(out)
    assert folds.columns == ["id", "fold"]
    assert folds.height == frame.height
    meta = cv.read_fold_meta(out)
    assert meta["strategy"] == "StratifiedKFold" and meta["n_splits"] == 5 and meta["seed"] == 0
    assert meta["target"] == "target" and meta["id_col"] == "id"
    assert pl.read_parquet(out)["fold"].n_unique() == 5


def test_run_records_the_holdout_cutoff(frame, tmp_path):
    train = tmp_path / "train.csv"
    frame.write_csv(train)
    out = tmp_path / "folds" / "holdout_t150.parquet"
    make_folds.run(
        train, out, id_col="id", strategy="TimeHoldout", n_splits=1, seed=0, time_col="time", holdout_from=150
    )
    assert cv.read_fold_meta(out)["holdout_from"] == 150
    assert cv.load_folds(out).filter(pl.col("fold") == 0).height == 50


def test_run_reads_columns_whose_late_rows_are_not_numeric(frame, tmp_path):
    # Raw tables like MLIT's mix numbers and labels ("2,000㎡以上") in one column, far below the first rows.
    mixed = frame.with_columns(pl.Series("area", [str(i) for i in range(199)] + ["2,000㎡以上"]))
    train = tmp_path / "train.csv"
    mixed.write_csv(train)
    out = tmp_path / "folds" / "skf5_seed0.parquet"
    make_folds.run(train, out, id_col="id", strategy="KFold", n_splits=5, seed=0)
    assert cv.load_folds(out).height == frame.height
