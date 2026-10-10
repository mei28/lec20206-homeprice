import numpy as np
import polars as pl
import pytest

from kgl import homeprice


@pytest.fixture
def raw() -> pl.DataFrame:
    """Two rows in the competition's raw notation (every column a string)."""
    return pl.DataFrame(
        {
            "価格情報区分": ["成約価格情報", "不動産取引価格情報"],
            "市区町村名": ["港区", "青梅市"],
            "地区名": ["赤坂", None],
            "最寄駅：名称": ["赤坂", "青梅"],
            "最寄駅：距離（分）": ["5", "30分～60分"],
            "間取り": ["３ＬＤＫ", "１Ｒ"],
            "面積（㎡）": ["70", "2,000㎡以上"],
            "建築年": ["2013年", "戦前"],
            "建物の構造": ["ＲＣ", "ＳＲＣ"],
            "用途": [None, "住宅"],
            "今後の利用目的": [None, "住宅"],
            "都市計画": ["商業", None],
            "建ぺい率（％）": [None, "60"],
            "容積率（％）": [None, "200"],
            "取引時期": ["2024年第4四半期", "2025年第2四半期"],
            "改装": [None, "未改装"],
            "取引の事情等": [None, None],
        }
    )


ADOPTED = {"type", "period", "notation"}


def test_adopted_steps_parse_period_minutes_and_area(raw):
    frame, categorical = homeprice.build_frame(raw, ADOPTED)
    assert frame["trade_period"].to_list() == [2024.75, 2025.25]
    assert frame["station_minutes"].to_list() == [5.0, 45.0]
    assert frame["area"].to_list() == [70.0, 2000.0]
    assert "取引時期" not in categorical and "価格情報区分" in categorical and "建築年" in categorical


def test_without_steps_the_s1_conversion_is_kept(raw):
    frame, categorical = homeprice.build_frame(raw, set())
    assert frame["station_minutes"].to_list() == [5.0, None]
    assert frame["area"].to_list() == [70.0, None]
    assert "取引時期" in categorical and "価格情報区分" not in categorical


def test_age_and_layout_steps_add_their_columns(raw):
    frame, categorical = homeprice.build_frame(raw, ADOPTED | {"age", "layout"})
    assert frame["build_year"].to_list() == [2013, 1945]
    assert frame["age"].to_list() == [11.0, 80.0]
    assert frame["rooms"].to_list() == [3, 1]
    assert frame["layout_R"].to_list() == [False, True]
    assert "建築年" not in categorical


def test_to_pandas_uses_the_vocabulary_and_location_codes(raw):
    vocab = homeprice.make_vocab([raw])
    X = homeprice.features(raw, ADOPTED, vocab)
    assert list(X["市区町村名"].cat.categories) == ["港区", "青梅市"]
    codes = homeprice.features(raw, ADOPTED, vocab, location_codes=True)
    assert codes["市区町村名"].tolist() == [0.0, 1.0]
    assert np.isnan(codes["地区名"].iloc[1])


def test_hedonic_controls_one_hot_the_ward_and_fill_missing_numbers(raw):
    controls = homeprice.hedonic_controls(raw.with_columns(pl.Series("建築年", ["2013年", None])))
    dense = controls.toarray()
    # columns: wards (港区, 青梅市), contract flag, log area, age, age^2/100, minutes, age missing, minutes missing
    assert dense.shape == (2, 9)
    assert dense[:, :2].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert dense[:, 2].tolist() == [1.0, 0.0]
    assert dense[:, 3] == pytest.approx(np.log([70.0, 2000.0]))
    # the missing age takes the median of the known ages (11), and its flag is set
    assert dense[:, 4].tolist() == [11.0, 11.0] and dense[:, 5].tolist() == [1.21, 1.21]
    assert dense[:, 7].tolist() == [0.0, 1.0]
    assert dense[:, 6].tolist() == [5.0, 45.0] and dense[:, 8].tolist() == [0.0, 0.0]


def test_hedonic_controls_zero_a_column_with_no_known_value(raw):
    # The missing-age segment of S4's per-segment index has no known age at all.
    dense = homeprice.hedonic_controls(raw.with_columns(pl.lit(None, dtype=pl.String).alias("建築年"))).toarray()
    assert dense[:, 4].tolist() == [0.0, 0.0] and dense[:, 5].tolist() == [0.0, 0.0]
    assert dense[:, 7].tolist() == [1.0, 1.0]


def test_count_features_count_rows_per_value_in_the_fitting_rows(raw):
    fit = pl.concat([raw, raw.head(1)])
    out = homeprice.count_features(
        fit, raw.with_columns(pl.Series("間取り", ["３ＬＤＫ", "２ＤＫ"])), ["間取り", "地区名"]
    )
    assert out.columns == ["count_間取り", "count_地区名"]
    assert out["count_間取り"].to_list() == [2, 0]
    assert out["count_地区名"].to_list() == [2, 1]  # the missing 地区名 counts the fitting rows that miss it


def test_location_aggregates_summarize_the_fitting_rows_per_key(raw):
    fit = pl.concat([raw, raw.with_columns(pl.Series("面積（㎡）", ["30", "50"]))])
    out = homeprice.location_aggregates(fit, raw, "市区町村名", prefix="ward")
    assert out.columns == ["ward_median_area", "ward_median_build_year", "ward_mean_floor_ratio", "ward_contract_share"]
    assert out["ward_median_area"].to_list() == [50.0, 1025.0]
    assert out["ward_median_build_year"].to_list() == [2013.0, 1945.0]
    assert out["ward_mean_floor_ratio"].to_list() == [None, 200.0]
    assert out["ward_contract_share"].to_list() == [1.0, 0.0]
