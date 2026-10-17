import numpy as np
import pandas as pd
import polars as pl
import pytest

from kgl import classic


@pytest.fixture
def feature_rows():
    """Rows shaped like kgl.feature_set's X: small and large categories, numbers, and count columns."""
    rng = np.random.default_rng(0)
    n = 400
    city = rng.choice(["千代田区", "港区", "新宿区"], size=n).astype(object)
    city[:5] = None
    X = pd.DataFrame(
        {
            "市区町村名": pd.Categorical(city),
            "地区名": pd.Categorical(rng.choice(["丸の内", "六本木"], size=n)),
            "最寄駅：名称": pd.Categorical(rng.choice(["東京", "六本木"], size=n)),
            "建築年": pd.Categorical(rng.choice(["2001年", "2015年"], size=n)),
            "area": rng.uniform(20, 120, size=n),
            "te_district": rng.normal(size=n),
            "count_地区名": rng.integers(1, 5000, size=n),
        }
    )
    extras = pd.DataFrame({"build_year_num": rng.uniform(1970, 2024, size=n), "log_area": np.log(X["area"])})
    y = 0.01 * X["area"].to_numpy() + X["te_district"].to_numpy() + rng.normal(scale=0.1, size=n)
    return X, extras, y


def test_numeric_extras_are_build_year_age_and_log_area():
    raw = pl.DataFrame(
        {"建築年": ["2013年", "1990年"], "取引時期": ["2024年第3四半期", "2020年第1四半期"], "面積（㎡）": ["70", "20"]}
    )
    extras = classic.numeric_extras(raw)
    assert extras.columns.tolist() == ["build_year_num", "age", "log_area"]
    assert extras["build_year_num"].tolist() == [2013.0, 1990.0]
    assert extras["age"].tolist() == [11.0, 30.0]
    assert extras["log_area"].to_numpy() == pytest.approx(np.log([70.0, 20.0]))


def test_linear_frame_keeps_numbers_logs_counts_and_spells_out_small_categories(feature_rows):
    X, extras, _ = feature_rows
    frame = classic.linear_frame(X, extras)
    assert frame.columns.tolist() == [
        "area",
        "te_district",
        "build_year_num",
        "log_area",
        "log_count_地区名",
        "市区町村名",
    ]
    assert frame["log_count_地区名"].to_numpy() == pytest.approx(np.log1p(X["count_地区名"].to_numpy()))
    assert frame["市区町村名"].iloc[0] == "NA" and frame["市区町村名"].iloc[5] == str(X["市区町村名"].iloc[5])
    assert "log_count_地区名" not in classic.linear_frame(X, extras, with_counts=False).columns


def test_forest_frame_turns_every_category_into_its_code_with_missing_as_nan(feature_rows):
    X, extras, _ = feature_rows
    frame = classic.forest_frame(X, extras)
    assert all(pd.api.types.is_float_dtype(t) for t in frame.dtypes)
    assert np.isnan(frame["市区町村名"].iloc[0])
    assert frame["地区名"].to_numpy() == pytest.approx(X["地区名"].cat.codes.to_numpy().astype(float))
    assert frame.columns.tolist()[:4] == ["area", "te_district", "count_地区名", "build_year_num"]


def test_mlp_and_ridge_fit_their_preprocessing_on_the_training_rows_only(feature_rows):
    X, extras, y = feature_rows
    frame = classic.linear_frame(X, extras)
    train, unseen = frame.iloc[:300], frame.iloc[300:].copy()
    unseen["市区町村名"] = "渋谷区"  # never seen while fitting
    ridge = classic.make_ridge(train, alphas=(0.1, 1.0, 10.0)).fit(train, y[:300])
    mlp_params = {"hidden_layer_sizes": (16,), "max_iter": 50, "early_stopping": True, "random_state": 0}
    mlp = classic.make_mlp(train, mlp_params).fit(train, y[:300])
    for model in (ridge, mlp):
        pred = model.predict(unseen)
        assert pred.shape == (100,) and np.isfinite(pred).all()
    assert np.sqrt(np.mean((ridge.predict(frame.iloc[300:]) - y[300:]) ** 2)) < 0.2


def test_forest_is_a_random_forest_with_the_given_settings(feature_rows):
    X, extras, y = feature_rows
    forest = classic.make_forest({"n_estimators": 20, "min_samples_leaf": 5, "random_state": 0, "n_jobs": 1})
    forest.fit(classic.forest_frame(X, extras), y)
    assert forest.n_estimators == 20 and forest.min_samples_leaf == 5
