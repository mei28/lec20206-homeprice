"""The feature set adopted through S5, built for the holdout setting, the refit setting, and the test.

- Holdout setting: everything that uses prices is fit on the training block (`tr_idx`) only, so the holdout
  rows' features and index offsets do not depend on holdout prices.
- Refit setting: fit on every training row; used to train the submission model and to build the test rows.

Columns: the S3 features (#12, #9), target encodings of district and station (#18), location aggregates
(#19), and counts (#33). The target is the log price minus the offset (S4's hedonic index, #16).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from kgl import encoding, homeprice, mlit, price_index

TARGET = "取引価格（総額）"
STEPS = {"type", "period", "notation"}
TE_KEYS = {"te_district": "地区名", "te_station": "最寄駅：名称"}
COUNT_COLUMNS = ["間取り", "地区名", "最寄駅：名称"]


@dataclass(frozen=True)
class FeatureSet:
    X: pd.DataFrame
    offset: np.ndarray
    X_refit: pd.DataFrame
    offset_refit: np.ndarray
    X_test: pd.DataFrame
    offset_test: np.ndarray
    y_log: np.ndarray


def _trade_period(df: pl.DataFrame) -> np.ndarray:
    return df.select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy()


def _target_encodings(fit_df, fit_target, apply_df, *, m, folds, seed):
    """Out-of-fold encodings for the fitting rows and plain encodings for the rows to predict."""
    fold_of_row = np.random.default_rng(seed).integers(0, folds, fit_df.height)
    prior = float(fit_target.mean())
    fit_columns, apply_columns = {}, {}
    for name, key in TE_KEYS.items():
        fit_columns[name] = encoding.oof_smoothed_mean(fit_df[key], fit_target, fold_of_row, m=m)
        apply_columns[name] = encoding.smoothed_mean(fit_df[key], fit_target, apply_df[key], prior=prior, m=m)
    return pd.DataFrame(fit_columns), pd.DataFrame(apply_columns)


def _target_free(train_df: pl.DataFrame, apply_df: pl.DataFrame) -> pd.DataFrame:
    aggregates = homeprice.location_aggregates(train_df, apply_df, "地区名", prefix="district").hstack(
        homeprice.location_aggregates(train_df, apply_df, "最寄駅：名称", prefix="station")
    )
    counts = homeprice.count_features(train_df, apply_df, COUNT_COLUMNS)
    return aggregates.hstack(counts).to_pandas()


def build(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    *,
    te_smoothing: float,
    te_folds: int,
    seed: int,
    trend_quarters: int,
) -> FeatureSet:
    y_log = train_df[TARGET].cast(pl.Float64).log1p().to_numpy()
    t_train, t_test = _trade_period(train_df), _trade_period(test_df)
    log_area = train_df.select(mlit.area_m2("面積（㎡）").log().alias("a"))["a"].to_numpy()
    log_ppm2 = y_log - log_area

    vocab = homeprice.make_vocab([train_df, test_df])
    base = homeprice.features(train_df, STEPS, vocab)
    base_test = homeprice.features(test_df, STEPS, vocab)
    target_free = _target_free(train_df, train_df)
    target_free_test = _target_free(train_df, test_df)

    # Holdout setting: the index and the encodings see the training block only.
    periods, index = price_index.hedonic_index(
        t_train[tr_idx], log_ppm2[tr_idx], homeprice.hedonic_controls(train_df[tr_idx])
    )
    offset = price_index.offset(periods, index, t_train, k=trend_quarters)
    fit_te, apply_te = _target_encodings(
        train_df[tr_idx], (log_ppm2 - offset)[tr_idx], train_df[va_idx], m=te_smoothing, folds=te_folds, seed=seed
    )
    te = pd.DataFrame(index=range(train_df.height), columns=fit_te.columns, dtype=float)
    te.iloc[tr_idx] = fit_te.to_numpy()
    te.iloc[va_idx] = apply_te.to_numpy()

    # Refit setting: every training row; the test rows are encoded from all of them.
    all_rows = np.arange(train_df.height)
    periods_all, index_all = price_index.hedonic_index(
        t_train, log_ppm2, homeprice.hedonic_controls(train_df[all_rows])
    )
    offset_refit = price_index.offset(periods_all, index_all, t_train, k=trend_quarters)
    offset_test = price_index.offset(periods_all, index_all, t_test, k=trend_quarters)
    te_refit, te_test = _target_encodings(
        train_df, log_ppm2 - offset_refit, test_df, m=te_smoothing, folds=te_folds, seed=seed
    )

    return FeatureSet(
        X=pd.concat([base, te, target_free], axis=1),
        offset=offset,
        X_refit=pd.concat([base, te_refit, target_free], axis=1),
        offset_refit=offset_refit,
        X_test=pd.concat([base_test, te_test, target_free_test], axis=1),
        offset_test=offset_test,
        y_log=y_log,
    )
