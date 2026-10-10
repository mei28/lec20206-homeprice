"""External location data (S8): station coordinates and ridership, and district representative points.

Sources (public, free, CC BY 4.0; downloaded by `just external`):
- 国土数値情報 駅別乗降客数 S12 (2024 data, file S12-25), MLIT. Station geometry and daily riders per line.
- Geolonia japanese-addresses (latest.csv), built from MLIT 位置参照情報. 町丁目 representative points.
Neither contains real estate transaction data.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

# Rough box around Tokyo and its neighbours: same-named stations elsewhere in Japan are left out.
TOKYO_BOX = {"lon": (138.9, 140.2), "lat": (35.3, 36.0)}
TOKYO_STATION = (35.681236, 139.767125)
EARTH_RADIUS_KM = 6371.0


def _station_key(expr: pl.Expr) -> pl.Expr:
    """'大森(東京)' -> '大森', 'ヶ' -> 'ケ': the competition data and S12 spell some names differently."""
    return expr.str.replace(r"\(.*\)$", "").str.replace_all("ヶ", "ケ").str.replace_all("ヵ", "カ").str.strip_chars()


def load_stations(geojson_path: Path) -> pl.DataFrame:
    """One row per station name inside the Tokyo box: mean coordinates over its lines and summed 2024 riders."""
    rows = []
    for feature in json.loads(Path(geojson_path).read_text())["features"]:
        coords = np.asarray(feature["geometry"]["coordinates"], dtype=float).reshape(-1, 2)
        rows.append(
            {
                "name": feature["properties"]["S12_001"],
                "lon": float(coords[:, 0].mean()),
                "lat": float(coords[:, 1].mean()),
                "riders": feature["properties"]["S12_061"],
            }
        )
    return (
        pl.DataFrame(rows)
        .filter(pl.col("lon").is_between(*TOKYO_BOX["lon"]) & pl.col("lat").is_between(*TOKYO_BOX["lat"]))
        .with_columns(station_key=_station_key(pl.col("name")))
        .group_by("station_key")
        .agg(
            pl.col("lat").mean().alias("station_lat"),
            pl.col("lon").mean().alias("station_lon"),
            pl.col("riders").sum().alias("station_riders"),
        )
    )


def station_features(df: pl.DataFrame, stations: pl.DataFrame) -> pl.DataFrame:
    """Station latitude, longitude, and riders for each row's nearest station (null when it does not match)."""
    keys = df.select(_station_key(pl.col("最寄駅：名称")).alias("station_key"))
    joined = keys.join(stations, on="station_key", how="left", maintain_order="left")
    return joined.select("station_lat", "station_lon", "station_riders")


def load_towns(csv_path: Path) -> pl.DataFrame:
    """Tokyo 町 (the 丁目 dropped) with the mean of their 丁目 representative points."""
    return (
        pl.read_csv(csv_path, infer_schema_length=0)
        .filter(pl.col("都道府県名") == "東京都")
        .with_columns(
            pl.col("大字町丁目名").str.replace(r"[一二三四五六七八九十]+丁目$", "").alias("地区名"),
            pl.col("緯度").cast(pl.Float64),
            pl.col("経度").cast(pl.Float64),
        )
        .group_by("市区町村名", "地区名")
        .agg(pl.col("緯度").mean().alias("district_lat"), pl.col("経度").mean().alias("district_lon"))
    )


def district_features(df: pl.DataFrame, towns: pl.DataFrame) -> pl.DataFrame:
    """District latitude and longitude, and the great-circle distance to Tokyo Station in km."""
    joined = df.select("市区町村名", "地区名").join(
        towns, on=["市区町村名", "地区名"], how="left", maintain_order="left"
    )
    lat1, lon1 = np.radians(TOKYO_STATION[0]), np.radians(TOKYO_STATION[1])
    lat2, lon2 = pl.col("district_lat").radians(), pl.col("district_lon").radians()
    haversine = ((lat2 - lat1) / 2).sin() ** 2 + np.cos(lat1) * lat2.cos() * ((lon2 - lon1) / 2).sin() ** 2
    return joined.select(
        "district_lat",
        "district_lon",
        (2 * EARTH_RADIUS_KM * haversine.sqrt().arcsin()).alias("km_to_tokyo_station"),
    )
