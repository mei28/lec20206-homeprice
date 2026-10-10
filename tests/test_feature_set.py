import numpy as np
import polars as pl
import pytest

from kgl import feature_set


def _raw(n_per_quarter: int, quarters: list[str], seed: int) -> pl.DataFrame:
    """Rows in the competition's raw notation over the given quarters."""
    rng = np.random.default_rng(seed)
    n = n_per_quarter * len(quarters)
    pick = lambda values: [values[i] for i in rng.integers(0, len(values), n)]  # noqa: E731
    return pl.DataFrame(
        {
            "id": np.arange(n),
            "価格情報区分": pick(["成約価格情報", "不動産取引価格情報"]),
            "市区町村名": pick(["港区", "青梅市"]),
            "地区名": pick(["赤坂", "青山", "河辺", None]),
            "最寄駅：名称": pick(["赤坂", "青梅"]),
            "最寄駅：距離（分）": pick(["5", "12", "30分～60分"]),
            "間取り": pick(["３ＬＤＫ", "１Ｒ"]),
            "面積（㎡）": pick(["25", "70"]),
            "建築年": pick(["2013年", "1990年", None]),
            "建物の構造": pick(["ＲＣ", "ＳＲＣ"]),
            "用途": pick([None, "住宅"]),
            "今後の利用目的": pick([None, "住宅"]),
            "都市計画": pick(["商業", None]),
            "建ぺい率（％）": pick([None, "60"]),
            "容積率（％）": pick([None, "200"]),
            "取引時期": [q for q in quarters for _ in range(n_per_quarter)],
            "改装": pick([None, "未改装"]),
            "取引の事情等": [None] * n,
            "取引価格（総額）": [str(v) for v in rng.integers(10, 90, n) * 1_000_000],
        }
    )


TRAIN_QUARTERS = [f"{y}年第{q}四半期" for y in (2023, 2024) for q in (1, 2, 3, 4)] + [
    "2025年第1四半期",
    "2025年第2四半期",
]


@pytest.fixture
def frames():
    train = _raw(20, TRAIN_QUARTERS, seed=0)
    test = _raw(10, ["2025年第3四半期"], seed=1).drop("取引価格（総額）").with_columns(pl.col("id") + 10_000)
    t = np.repeat(np.arange(len(TRAIN_QUARTERS)), 20)
    tr_idx, va_idx = np.flatnonzero(t < 8), np.flatnonzero(t >= 8)
    return train, test, tr_idx, va_idx


def _build(train, test, tr_idx, va_idx):
    return feature_set.build(train, test, tr_idx, va_idx, te_smoothing=20.0, te_folds=5, seed=42, trend_quarters=8)


def test_build_returns_aligned_frames_with_the_s5_columns(frames):
    train, test, tr_idx, va_idx = frames
    fs = _build(train, test, tr_idx, va_idx)
    assert len(fs.X) == len(fs.X_refit) == len(fs.offset) == len(fs.offset_refit) == train.height
    assert len(fs.X_test) == len(fs.offset_test) == test.height
    assert list(fs.X.columns) == list(fs.X_refit.columns) == list(fs.X_test.columns)
    for column in ("trade_period", "te_district", "te_station", "district_median_area", "count_間取り"):
        assert column in fs.X.columns


def test_holdout_setting_does_not_depend_on_holdout_prices(frames):
    train, test, tr_idx, va_idx = frames
    changed = train.with_columns(
        pl.when(pl.int_range(pl.len()).is_in(va_idx.tolist()))
        .then(pl.lit("1"))
        .otherwise(pl.col("取引価格（総額）"))
        .alias("取引価格（総額）")
    )
    a, b = _build(train, test, tr_idx, va_idx), _build(changed, test, tr_idx, va_idx)
    assert a.X.equals(b.X)
    assert np.array_equal(a.offset, b.offset)
