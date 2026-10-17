"""The S7 members that are neither GBDT nor pretrained (Ridge, the MLP, RandomForest) and their inputs.

The inputs start from kgl.feature_set's frame. Ridge and the MLP take numbers (counts as log1p) and the small
categories spelled out for one-hot encoding; the large categories (district, station) are represented by their
target encodings, and the build year by its number. RandomForest takes every category as its integer code.
Every preprocessing step sits inside the model, so it is fit on the training rows only.
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kgl import mlit

LARGE_CATEGORIES = ("地区名", "最寄駅：名称", "建築年")


def numeric_extras(df: pl.DataFrame) -> pd.DataFrame:
    """Columns a linear model needs as numbers: build year and age instead of the 建築年 category, and log area."""
    return df.select(
        mlit.build_year("建築年").cast(pl.Float64).alias("build_year_num"),
        (mlit.trade_period("取引時期").floor() - mlit.build_year("建築年")).cast(pl.Float64).alias("age"),
        mlit.area_m2("面積（㎡）").log().alias("log_area"),
    ).to_pandas()


def _categorical(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def _one_hot(X: pd.DataFrame) -> list[str]:
    return [c for c in _categorical(X) if c not in LARGE_CATEGORIES]


def _counts(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c.startswith("count_")]


def linear_frame(X: pd.DataFrame, extras: pd.DataFrame, *, with_counts: bool = True) -> pd.DataFrame:
    """Inputs for Ridge and the MLP: numbers (counts as log1p), extras, and the small categories as strings."""
    categorical, counts = _categorical(X), _counts(X)
    numeric = [c for c in X.columns if c not in categorical and c not in counts]
    out = pd.concat([X[numeric].reset_index(drop=True), extras.reset_index(drop=True)], axis=1)
    if with_counts:
        for c in counts:
            out[f"log_{c}"] = np.log1p(X[c].to_numpy().astype(float))
    for c in _one_hot(X):
        out[c] = X[c].cat.add_categories("NA").fillna("NA").astype(str).to_numpy()
    return out


def forest_frame(X: pd.DataFrame, extras: pd.DataFrame) -> pd.DataFrame:
    """Inputs for RandomForest: numbers with NaN kept, and every category as its integer code (NaN if missing)."""
    categorical = _categorical(X)
    out = pd.concat([X.drop(columns=categorical).reset_index(drop=True), extras.reset_index(drop=True)], axis=1)
    for c in categorical:
        out[c] = X[c].cat.codes.astype(float).where(lambda codes: codes >= 0).to_numpy()
    return out.astype(float)


def linear_preprocess(frame: pd.DataFrame) -> ColumnTransformer:
    """Median imputation (with missing indicators) and scaling for numbers, one-hot for the spelled-out categories."""
    one_hot = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    numeric = [c for c in frame.columns if c not in one_hot]
    return ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
            ("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), one_hot),
        ]
    )


def make_ridge(frame: pd.DataFrame, *, alphas: tuple[float, ...]) -> Pipeline:
    return make_pipeline(linear_preprocess(frame), RidgeCV(alphas=alphas))


def make_mlp(frame: pd.DataFrame, params: dict[str, Any]) -> TransformedTargetRegressor:
    """The MLP with the target standardized too, so the network starts near the right scale."""
    return TransformedTargetRegressor(
        regressor=make_pipeline(linear_preprocess(frame), MLPRegressor(**params)),
        transformer=StandardScaler(),
    )


def make_forest(params: dict[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(**params)
