import polars as pl
import pytest

from kgl import mlit


def apply(expr: pl.Expr, values: list) -> list:
    return pl.DataFrame({"x": pl.Series(values, dtype=pl.String)}).select(expr.alias("out"))["out"].to_list()


def test_trade_period_is_year_plus_quarter_fraction():
    assert apply(mlit.trade_period("x"), ["2019年第1四半期", "2025年第2四半期", "2026年第1四半期"]) == [
        2019.0,
        2025.25,
        2026.0,
    ]


def test_station_minutes_reads_numbers_and_takes_range_midpoints():
    values = ["6", "0", "29", "30分～60分", "1H～1H30", "1H30～2H", "2H～", None]
    assert apply(mlit.station_minutes("x"), values) == [6.0, 0.0, 29.0, 45.0, 75.0, 105.0, 120.0, None]


def test_station_minutes_rejects_unknown_labels():
    with pytest.raises(pl.exceptions.InvalidOperationError):
        apply(mlit.station_minutes("x"), ["3H～"])


def test_build_year_reads_the_year_and_maps_prewar_to_1945():
    assert apply(mlit.build_year("x"), ["2013年", "戦前", None]) == [2013, 1945, None]


def test_build_year_rejects_unknown_labels():
    with pytest.raises(pl.exceptions.InvalidOperationError):
        apply(mlit.build_year("x"), ["平成5年"])


def test_area_reads_numbers_and_the_open_ended_label():
    assert apply(mlit.area_m2("x"), ["45", "1400", "2,000㎡以上"]) == [45.0, 1400.0, 2000.0]


def test_area_rejects_unknown_labels():
    with pytest.raises(pl.exceptions.InvalidOperationError):
        apply(mlit.area_m2("x"), ["3,000㎡以上"])


def test_layout_rooms_reads_the_leading_digit_of_full_width_text():
    assert apply(mlit.layout_rooms("x"), ["３ＬＤＫ＋Ｓ", "１Ｒ", "スタジオ", None]) == [3, 1, None, None]


def test_layout_flags_mark_each_letter():
    df = pl.DataFrame({"x": ["３ＬＤＫ＋Ｓ", "１Ｒ", "２ＤＫ", "メゾネット"]}).select(mlit.layout_flags("x"))
    assert df.to_dict(as_series=False) == {
        "layout_L": [True, False, False, False],
        "layout_D": [True, False, True, False],
        "layout_K": [True, False, True, False],
        "layout_S": [True, False, False, False],
        "layout_R": [False, True, False, False],
    }
