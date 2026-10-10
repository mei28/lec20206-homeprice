import polars as pl
import pytest

from kgl import io


def _metrics() -> io.Metrics:
    return io.Metrics(
        exp="exp000_baseline",
        base="none",
        run="default",
        cv=0.8123,
        cv_folds=[0.81, 0.82, 0.80, 0.83, 0.80],
        metric="accuracy",
        config={"seed": 42, "model": "lightgbm.LGBMClassifier"},
        argv=[],
        git_commit="abc123",
        started_at="2026-09-20T10:00:00",
        finished_at="2026-09-20T10:01:00",
        notes="",
    )


def test_metrics_roundtrip(tmp_path):
    io.write_metrics(tmp_path, _metrics())
    assert (tmp_path / "metrics.json").exists()
    assert io.read_metrics(tmp_path / "metrics.json") == _metrics()


def _sample() -> pl.DataFrame:
    return pl.DataFrame({"PassengerId": [892, 893, 894], "Survived": [0, 1, 0]})


def test_validate_submission_accepts_sample_shaped_frame():
    sub = _sample().with_columns(pl.Series("Survived", [1, 1, 0]))
    io.validate_submission(sub, _sample(), "PassengerId")


def test_validate_submission_rejects_row_count():
    with pytest.raises(AssertionError, match="rows"):
        io.validate_submission(_sample().head(2), _sample(), "PassengerId")


def test_validate_submission_rejects_column_names():
    with pytest.raises(AssertionError, match="columns"):
        io.validate_submission(_sample().rename({"Survived": "pred"}), _sample(), "PassengerId")


def test_validate_submission_rejects_id_order():
    with pytest.raises(AssertionError, match="id"):
        io.validate_submission(_sample().reverse(), _sample(), "PassengerId")


def test_validate_submission_rejects_nulls():
    sub = _sample().with_columns(pl.Series("Survived", [0, None, 1]))
    with pytest.raises(AssertionError, match="null"):
        io.validate_submission(sub, _sample(), "PassengerId")


def test_git_commit_is_the_head_sha_of_the_given_repo(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x"],
        cwd=tmp_path,
        check=True,
    )
    sha = io.git_commit(tmp_path)
    assert len(sha) == 40
    assert set(sha) <= set("0123456789abcdef")


def test_git_commit_raises_outside_a_repo(tmp_path):
    with pytest.raises(RuntimeError):
        io.git_commit(tmp_path)
