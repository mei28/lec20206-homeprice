import datetime as dt

import pytest

from tools import cv_lb_plot

TABLE = (
    "# Submissions\n\n| date | exp | run | cv | public_lb | message | kernel_version | file |\n"
    "|---|---|---|---|---|---|---|---|\n"
    "| 2026-09-20 | exp000_base | default | 0.8000 | 0.7900 | baseline |  | submission.csv |\n"
    "| 2026-09-22 | exp001_more | 1a2b3c4d | 0.8100 | 0.7800 | more trees | 3 | submission.csv |\n"
)


def test_read_points_keeps_submission_order():
    points = cv_lb_plot.read_points(TABLE)
    assert points == [
        cv_lb_plot.Point(order=1, date=dt.date(2026, 9, 20), exp="exp000_base", cv=0.80, lb=0.79),
        cv_lb_plot.Point(order=2, date=dt.date(2026, 9, 22), exp="exp001_more", cv=0.81, lb=0.78),
    ]


def test_read_points_fails_on_a_table_without_submissions():
    empty = TABLE.split("| 2026-09-20")[0]
    with pytest.raises(ValueError, match="no submissions"):
        cv_lb_plot.read_points(empty)


def test_label_is_the_experiment_number():
    assert cv_lb_plot.label("exp012_fpfix") == "012"


def test_plot_writes_a_png(tmp_path):
    out = tmp_path / "figures" / "cv_lb.png"
    cv_lb_plot.plot(cv_lb_plot.read_points(TABLE), out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_draws_the_first_submission_alone(tmp_path):
    out = tmp_path / "cv_lb.png"
    cv_lb_plot.plot(cv_lb_plot.read_points(TABLE)[:1], out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
