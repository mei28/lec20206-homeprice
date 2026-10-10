"""Feature frames for utokyo-mlapp-2026-ex1, built from the raw columns (all read as strings).

`steps` switches the stage changes on: "type" (keep 価格情報区分), "period" (trade period as a number,
#12), "notation" (parsed station minutes and area, #9), "age" (#10), "layout" (#11). Without a step the
column keeps S1's conversion. Column order is fixed, because LightGBM's column sampling depends on it.
"""

from collections.abc import Iterable

import pandas as pd
import polars as pl

from kgl import mlit

LOCATION = ["市区町村名", "地区名", "最寄駅：名称"]
OTHER_CATEGORICAL = ["間取り", "建物の構造", "用途", "今後の利用目的", "都市計画", "改装", "取引の事情等"]
ALL_CATEGORICAL = LOCATION + OTHER_CATEGORICAL + ["価格情報区分", "取引時期", "建築年"]


def build_frame(df: pl.DataFrame, steps: set[str]) -> tuple[pl.DataFrame, list[str]]:
    """Return the feature frame and the names of its categorical columns."""
    exprs = [pl.col("建ぺい率（％）").cast(pl.Float64), pl.col("容積率（％）").cast(pl.Float64)]
    categorical = LOCATION + OTHER_CATEGORICAL
    if "type" in steps:
        categorical = categorical + ["価格情報区分"]
    if "period" in steps:
        exprs.append(mlit.trade_period("取引時期").alias("trade_period"))
    else:
        categorical = categorical + ["取引時期"]
    if "notation" in steps:
        exprs += [
            mlit.station_minutes("最寄駅：距離（分）").alias("station_minutes"),
            mlit.area_m2("面積（㎡）").alias("area"),
        ]
    else:
        # S1's conversion: anything that is not a plain number becomes missing.
        exprs += [
            pl.col("最寄駅：距離（分）").cast(pl.Float64, strict=False).alias("station_minutes"),
            pl.col("面積（㎡）").cast(pl.Float64, strict=False).alias("area"),
        ]
    if "age" in steps:
        year = mlit.build_year("建築年")
        exprs += [year.alias("build_year"), (mlit.trade_period("取引時期").floor() - year).alias("age")]
    else:
        categorical = categorical + ["建築年"]
    if "layout" in steps:
        exprs += [mlit.layout_rooms("間取り").alias("rooms"), *mlit.layout_flags("間取り")]
    return df.select(exprs + [pl.col(c) for c in categorical]), categorical


def make_vocab(frames: Iterable[pl.DataFrame]) -> dict[str, list[str]]:
    """Sorted category values per categorical column over the given frames (no target is used)."""
    frames = list(frames)
    return {c: sorted({v for f in frames for v in f[c].drop_nulls().to_list()}) for c in ALL_CATEGORICAL}


def to_pandas(
    frame: pl.DataFrame, categorical: list[str], vocab: dict[str, list[str]], *, location_codes: bool = False
) -> pd.DataFrame:
    """Categories with a fixed vocabulary; with `location_codes`, the location columns become integer codes."""
    out = frame.to_pandas()
    for c in categorical:
        out[c] = pd.Categorical(out[c], categories=vocab[c])
        if location_codes and c in LOCATION:
            out[c] = out[c].cat.codes.astype(float).where(lambda codes: codes >= 0)
    return out


def features(
    df: pl.DataFrame, steps: set[str], vocab: dict[str, list[str]], *, location_codes: bool = False
) -> pd.DataFrame:
    frame, categorical = build_frame(df, steps)
    return to_pandas(frame, categorical, vocab, location_codes=location_codes)
