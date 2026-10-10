import polars as pl

from kgl import env as kenv
from kgl import features


def _env(tmp_path):
    return kenv.Env(kind="local", root=tmp_path, input_dir=tmp_path / "input", output_dir=tmp_path / "output")


def test_cache_path_is_deterministic_and_depends_on_params(tmp_path):
    e = _env(tmp_path)
    a = features.cache_path(e, "basic", {"n": 1})
    b = features.cache_path(e, "basic", {"n": 1})
    c = features.cache_path(e, "basic", {"n": 2})
    assert a == b and a != c
    assert a.parent == tmp_path / "output" / "features"
    assert a.name.startswith("basic_") and a.suffix == ".parquet"


def test_cached_returns_the_stored_frame_on_second_call(tmp_path):
    e = _env(tmp_path)
    calls = []

    def build() -> pl.DataFrame:
        calls.append(1)
        return pl.DataFrame({"x": [1, 2, 3]})

    first = features.cached(e, "basic", {"n": 1}, build)
    second = features.cached(e, "basic", {"n": 1}, build)
    assert first.equals(second)
    assert features.cache_path(e, "basic", {"n": 1}).exists()
    assert len(calls) == 1
