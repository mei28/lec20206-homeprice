import numpy as np
import polars as pl
import pytest


@pytest.fixture
def frame() -> pl.DataFrame:
    """200 rows: binary target (30% positives), continuous target, 20 groups of 10, increasing time."""
    rng = np.random.default_rng(0)
    n = 200
    return pl.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "target": (rng.random(n) < 0.3).astype(int),
            "y_cont": rng.normal(size=n),
            "group": np.repeat(np.arange(20), 10),
            "time": np.arange(n),
        }
    )
