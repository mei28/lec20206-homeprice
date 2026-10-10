"""Run artifacts: metrics.json, config.json, OOF predictions, and the submission file."""

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class Metrics:
    exp: str
    base: str
    run: str
    cv: float
    cv_folds: list[float]
    metric: str
    config: dict[str, Any]
    argv: list[str]
    git_commit: str
    started_at: str
    finished_at: str
    notes: str = ""


def write_metrics(run_dir: Path, metrics: Metrics) -> Path:
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(asdict(metrics), indent=2, default=str) + "\n")
    return path


def read_metrics(path: Path) -> Metrics:
    return Metrics(**json.loads(path.read_text()))


def write_config(run_dir: Path, config: dict[str, Any], argv: list[str]) -> Path:
    path = run_dir / "config.json"
    path.write_text(json.dumps({"config": config, "argv": argv}, indent=2, default=str) + "\n")
    return path


def write_oof(run_dir: Path, oof: pl.DataFrame) -> Path:
    path = run_dir / "oof.parquet"
    oof.write_parquet(path)
    return path


def write_submission(run_dir: Path, sub: pl.DataFrame) -> Path:
    path = run_dir / "submission.csv"
    sub.write_csv(path)
    return path


def validate_submission(sub: pl.DataFrame, sample: pl.DataFrame, id_col: str) -> None:
    """Assert the submission has the sample's shape: same rows, columns, id order, and no nulls."""
    assert sub.height == sample.height, f"rows: got {sub.height}, sample has {sample.height}"
    assert sub.columns == sample.columns, f"columns: got {sub.columns}, sample has {sample.columns}"
    assert sub[id_col].to_list() == sample[id_col].to_list(), f"id order differs from the sample ({id_col})"
    nulls = {c: n for c, n in zip(sub.columns, sub.null_count().row(0), strict=True) if n}
    assert not nulls, f"null values in submission: {nulls}"


def git_commit(repo: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"not a git repository with commits: {repo} ({result.stderr.strip()})")
    return result.stdout.strip()
