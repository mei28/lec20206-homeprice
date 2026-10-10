import json

import polars as pl
import pytest

from kgl import external


@pytest.fixture
def s12_geojson(tmp_path):
    """Two lines serving 新宿 inside Tokyo, one 幡ヶ谷, and a 府中 far outside the Tokyo box."""

    def feature(name, coords, riders):
        return {
            "type": "Feature",
            "properties": {"S12_001": name, "S12_061": riders},
            "geometry": {"type": "LineString", "coordinates": coords},
        }

    path = tmp_path / "s12.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    feature("新宿", [[139.70, 35.69], [139.70, 35.69]], 100),
                    feature("新宿", [[139.71, 35.70], [139.71, 35.70]], 300),
                    feature("幡ヶ谷", [[139.67, 35.68], [139.67, 35.68]], 50),
                    feature("府中", [[133.24, 34.57], [133.24, 34.57]], 10),
                ],
            },
            ensure_ascii=False,
        )
    )
    return path


def test_load_stations_merges_lines_per_normalized_name_inside_tokyo(s12_geojson):
    stations = external.load_stations(s12_geojson).sort("station_key")
    assert stations["station_key"].to_list() == ["幡ケ谷", "新宿"]
    shinjuku = stations.row(1, named=True)
    assert shinjuku["station_lon"] == pytest.approx(139.705) and shinjuku["station_lat"] == pytest.approx(35.695)
    assert shinjuku["station_riders"] == 400


def test_station_features_normalize_our_names(s12_geojson):
    df = pl.DataFrame({"最寄駅：名称": ["新宿", "幡ケ谷", "府中(東京)", None]})
    out = external.station_features(df, external.load_stations(s12_geojson))
    assert out["station_riders"].to_list() == [400, 50, None, None]
    assert out.columns == ["station_lat", "station_lon", "station_riders"]


def test_load_towns_drops_the_chome_and_averages_points(tmp_path):
    path = tmp_path / "latest.csv"
    pl.DataFrame(
        {
            "都道府県名": ["東京都", "東京都", "東京都", "北海道"],
            "市区町村名": ["港区", "港区", "港区", "札幌市中央区"],
            "大字町丁目名": ["赤坂一丁目", "赤坂二丁目", "青山", "旭ケ丘一丁目"],
            "緯度": ["35.60", "35.70", "35.66", "43.0"],
            "経度": ["139.70", "139.80", "139.71", "141.3"],
        }
    ).write_csv(path)
    towns = external.load_towns(path).sort("地区名")
    assert towns["地区名"].to_list() == ["赤坂", "青山"]
    assert towns["district_lat"].to_list() == pytest.approx([35.65, 35.66])


def test_district_features_add_the_distance_to_tokyo_station():
    towns = pl.DataFrame(
        {"市区町村名": ["千代田区"], "地区名": ["丸の内"], "district_lat": [35.681236], "district_lon": [139.767125]}
    )
    df = pl.DataFrame({"市区町村名": ["千代田区", "港区"], "地区名": ["丸の内", "赤坂"]})
    out = external.district_features(df, towns)
    assert out.columns == ["district_lat", "district_lon", "km_to_tokyo_station"]
    assert out["km_to_tokyo_station"].to_list()[0] == pytest.approx(0.0, abs=1e-6)
    assert out["km_to_tokyo_station"].to_list()[1] is None
