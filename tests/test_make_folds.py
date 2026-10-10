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
