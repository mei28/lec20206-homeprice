"""Parse the raw notation of MLIT 不動産取引価格情報 columns into numbers.

Each function takes a column name and returns a polars expression. Labels outside the known set raise
instead of becoming null, so a new notation in the data shows up as an error, not as silent missing values.
"""

import polars as pl

# Station walking time is a number up to 29 minutes, then a range label.
# Bounded ranges map to their midpoint; the open-ended "2H～" maps to its lower bound.
_STATION_MINUTES = {str(m): float(m) for m in range(30)} | {
    "30分～60分": 45.0,
    "1H～1H30": 75.0,
    "1H30～2H": 105.0,
    "2H～": 120.0,
}

# "戦前" (before the war) has no year; 1945 is its upper bound.
_PREWAR_YEAR = "1945年"

# The only non-numeric area label in the data; it is the lower bound of an open range.
_AREA_LABELS = {"2,000㎡以上": "2000"}


def trade_period(col: str) -> pl.Expr:
    """'2025年第2四半期' -> 2025.25 (year + (quarter - 1) / 4)."""
    parts = pl.col(col).str.extract_groups(r"^(\d{4})年第([1-4])四半期$")
    year = parts.struct.field("1").cast(pl.Float64, strict=True)
    quarter = parts.struct.field("2").cast(pl.Float64, strict=True)
    return year + (quarter - 1) / 4


def station_minutes(col: str) -> pl.Expr:
    """Walking minutes to the nearest station; ranges become their midpoint."""
    return pl.col(col).replace_strict(_STATION_MINUTES, return_dtype=pl.Float64)


def build_year(col: str) -> pl.Expr:
    """'2013年' -> 2013; '戦前' -> 1945."""
    return pl.col(col).replace("戦前", _PREWAR_YEAR).str.strip_suffix("年").cast(pl.Int32, strict=True)


def area_m2(col: str) -> pl.Expr:
    """'45' -> 45.0; '2,000㎡以上' -> 2000.0."""
    return pl.col(col).replace(_AREA_LABELS).cast(pl.Float64, strict=True)


def layout_rooms(col: str) -> pl.Expr:
    """Leading room count of a layout such as '３ＬＤＫ＋Ｓ' (full-width); null for 'スタジオ' and the like."""
    return pl.col(col).str.normalize("NFKC").str.extract(r"^(\d+)", 1).cast(pl.Int32, strict=True)


def layout_flags(col: str) -> list[pl.Expr]:
    """One boolean per layout letter: L (living), D (dining), K (kitchen), S (storeroom), R (one room)."""
    text = pl.col(col).str.normalize("NFKC").fill_null("")
    return [text.str.contains(letter, literal=True).alias(f"layout_{letter}") for letter in "LDKSR"]
