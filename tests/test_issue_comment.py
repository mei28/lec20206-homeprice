from kgl.io import Metrics
from tools import issue_comment


def _m() -> Metrics:
    return Metrics(
        exp="exp001_more_trees",
        base="exp000_baseline",
        run="1a2b3c4d",
        cv=0.8123,
        cv_folds=[0.81, 0.82, 0.80],
        metric="accuracy",
        config={"seed": 42},
        argv=["--model-params.n-estimators", "500"],
        git_commit="abcdef1234567890",
        started_at="2026-09-20T10:00:00",
        finished_at="2026-09-20T10:01:00",
        notes="",
    )


def test_compose_includes_run_identity_scores_and_path():
    body = issue_comment.compose(_m(), lb=None)
    assert "exp001_more_trees" in body and "1a2b3c4d" in body
    assert "0.8123" in body
    assert "0.81" in body and "0.82" in body and "0.80" in body
    assert "output/exp001_more_trees/1a2b3c4d" in body
    assert "base: exp000_baseline" in body
    assert "abcdef1" in body
    assert "--model-params.n-estimators 500" in body
    assert "LB" not in body


def test_compose_adds_lb_line_when_known():
    body = issue_comment.compose(_m(), lb="0.7799")
    assert "public LB: 0.7799" in body
