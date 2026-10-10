"""Feature caching: expensive frames are stored under output/features/ keyed by name and parameters."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from kgl.env import Env


def cache_path(env: Env, name: str, params: dict[str, Any]) -> Path:
    digest = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return env.output_dir / "features" / f"{name}_{digest}.parquet"


def cached(env: Env, name: str, params: dict[str, Any], build: Callable[[], pl.DataFrame]) -> pl.DataFrame:
    path = cache_path(env, name, params)
    if path.exists():
        return pl.read_parquet(path)
    frame = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return frame
