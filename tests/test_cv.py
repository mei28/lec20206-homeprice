import polars as pl

from kgl import cv


def test_kfold_partitions_rows_evenly(frame):
    folds = cv.make_folds(frame, strategy="KFold", n_splits=5, seed=0)
    assert len(folds) == frame.height
    assert sorted(folds.unique().to_list()) == [0, 1, 2, 3, 4]
    assert folds.value_counts()["count"].to_list() == [40] * 5


def test_stratified_kfold_keeps_class_ratio(frame):
    folds = cv.make_folds(frame, strategy="StratifiedKFold", n_splits=5, seed=0, target="target")
    overall = frame["target"].mean()
    per_fold = frame.with_columns(fold=folds).group_by("fold").agg(pl.col("target").mean())["target"]
    assert all(abs(m - overall) < 0.05 for m in per_fold.to_list())


def test_stratified_group_kfold_never_splits_a_group(frame):
    folds = cv.make_folds(frame, strategy="StratifiedGroupKFold", n_splits=5, seed=0, target="target", group="group")
    n_folds_per_group = frame.with_columns(fold=folds).group_by("group").agg(pl.col("fold").n_unique())["fold"]
    assert n_folds_per_group.to_list() == [1] * 20


def test_time_series_split_orders_blocks_and_marks_warmup(frame):
    folds = cv.make_folds(frame, strategy="TimeSeriesSplit", n_splits=4, seed=0, time_col="time")
    with_fold = frame.with_columns(fold=folds)
    assert with_fold.filter(pl.col("fold") == -1)["time"].max() < with_fold.filter(pl.col("fold") == 0)["time"].min()
    for k in range(1, 4):
        assert (
            with_fold.filter(pl.col("fold") == k)["time"].min()
            > with_fold.filter(pl.col("fold") == k - 1)["time"].max()
        )


def test_time_series_split_is_not_shuffled_even_if_rows_are(frame):
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=1)
    folds = cv.make_folds(shuffled, strategy="TimeSeriesSplit", n_splits=4, seed=0, time_col="time")
    with_fold = shuffled.with_columns(fold=folds)
    assert with_fold.filter(pl.col("fold") == 3)["time"].min() > with_fold.filter(pl.col("fold") == 2)["time"].max()


def test_regression_target_is_binned_for_stratification(frame):
    folds = cv.make_folds(frame, strategy="StratifiedKFold", n_splits=5, seed=0, target="y_cont", stratify_bins=10)
    with_fold = frame.with_columns(fold=folds)
    assert folds.value_counts()["count"].to_list() == [40] * 5
    overall = frame["y_cont"].mean()
    per_fold = with_fold.group_by("fold").agg(pl.col("y_cont").mean())["y_cont"]
    assert all(abs(m - overall) < 0.25 for m in per_fold.to_list())


def test_split_plain_uses_all_other_folds_for_training():
    folds = pl.Series("fold", [0, 0, 1, 1, 2, 2])
    train_idx, val_idx = cv.split(folds, 1)
    assert val_idx.tolist() == [2, 3]
    assert train_idx.tolist() == [0, 1, 4, 5]


def test_split_time_uses_only_earlier_folds_for_training():
    folds = pl.Series("fold", [-1, -1, 0, 0, 1, 1, 2, 2])
    train_idx, val_idx = cv.split(folds, 1, time=True)
    assert val_idx.tolist() == [4, 5]
    assert train_idx.tolist() == [0, 1, 2, 3]


def test_save_and_load_folds_roundtrip_with_sidecar(frame, tmp_path):
    folds = cv.make_folds(frame, strategy="KFold", n_splits=5, seed=0)
    path = tmp_path / "kf5_seed0.parquet"
    cv.save_folds(frame["id"], folds, path, meta={"strategy": "KFold", "n_splits": 5, "seed": 0})
    loaded = cv.load_folds(path)
    assert loaded.columns == ["id", "fold"]
    assert loaded["fold"].to_list() == folds.to_list()
    sidecar = cv.read_fold_meta(path)
    assert sidecar["strategy"] == "KFold"
    assert sidecar["n_rows"] == 200


def test_save_folds_refuses_to_overwrite(frame, tmp_path):
    folds = cv.make_folds(frame, strategy="KFold", n_splits=5, seed=0)
    path = tmp_path / "kf5_seed0.parquet"
    cv.save_folds(frame["id"], folds, path, meta={})
    import pytest

    with pytest.raises(FileExistsError):
        cv.save_folds(frame["id"], folds, path, meta={})


def test_attach_folds_adds_a_fold_for_every_row(frame):
    folds_df = pl.DataFrame({"id": frame["id"], "fold": cv.make_folds(frame, strategy="KFold", n_splits=5, seed=0)})
    out = cv.attach_folds(frame.sample(fraction=1.0, shuffle=True, seed=3), folds_df, "id")
    assert out.height == frame.height
    assert out["fold"].null_count() == 0
    assert out.filter(pl.col("id") == 7)["fold"].item() == folds_df.filter(pl.col("id") == 7)["fold"].item()


def test_attach_folds_raises_when_an_id_has_no_fold(frame):
    import pytest

    folds_df = pl.DataFrame({"id": frame["id"][:-1], "fold": [0] * (frame.height - 1)})
    with pytest.raises(ValueError):
        cv.attach_folds(frame, folds_df, "id")


def test_check_folds_detects_group_overlap(frame):
    import pytest

    bad = frame.with_columns(fold=pl.Series(cv.make_folds(frame, strategy="KFold", n_splits=5, seed=0)))
    with pytest.raises(AssertionError):
        cv.check_folds(bad, "fold", group="group")
    good = frame.with_columns(
        fold=cv.make_folds(frame, strategy="StratifiedGroupKFold", n_splits=5, seed=0, target="target", group="group")
    )
    cv.check_folds(good, "fold", group="group")


def test_check_folds_detects_time_order_violation(frame):
    import pytest

    good = frame.with_columns(
        fold=cv.make_folds(frame, strategy="TimeSeriesSplit", n_splits=4, seed=0, time_col="time")
    )
    cv.check_folds(good, "fold", time_col="time")
    bad = frame.with_columns(fold=cv.make_folds(frame, strategy="KFold", n_splits=4, seed=0))
    with pytest.raises(AssertionError):
        cv.check_folds(bad, "fold", time_col="time")
