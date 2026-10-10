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
