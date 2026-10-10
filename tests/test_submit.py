import pytest

from tools import submit


def test_final_command_for_csv_competition():
    cmd = submit.final_command(
        "csv", "titanic", file="output/exp000/default/submission.csv", message="baseline cv=0.81"
    )
    assert cmd == [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        "titanic",
        "-f",
        "output/exp000/default/submission.csv",
        "-m",
        "baseline cv=0.81",
    ]


def test_final_command_for_code_competition_names_kernel_and_version():
    cmd = submit.final_command("code", "cmi", file="submission.csv", message="m", kernel="mei28/cmi-sub", version=3)
    assert cmd == [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        "cmi",
        "-k",
        "mei28/cmi-sub",
        "-v",
        "3",
        "-f",
        "submission.csv",
        "-m",
        "m",
    ]


def test_final_command_for_code_competition_requires_kernel_and_version():
    with pytest.raises(ValueError):
        submit.final_command("code", "cmi", file="submission.csv", message="m")


def test_final_command_rejects_unknown_mode():
    with pytest.raises(ValueError):
        submit.final_command("docker", "cmi", file="submission.csv", message="m")


def test_submission_file_missing_points_at_infer(tmp_path):
    with pytest.raises(FileNotFoundError, match="just infer exp000_baseline"):
        submit.submission_file(tmp_path, "exp000_baseline", "default")


def test_record_row_is_a_markdown_table_row():
    row = submit.record_row(
        date="2026-09-20",
        exp="exp000_baseline",
        run="default",
        cv=0.8123,
        public_lb="0.7799",
        message="baseline",
        kernel_version="",
        file="submission.csv",
    )
    assert row == "| 2026-09-20 | exp000_baseline | default | 0.8123 | 0.7799 | baseline |  | submission.csv |"


def test_append_submission_creates_header_once(tmp_path):
    path = tmp_path / "submissions.md"
    submit.append_submission(path, "| a | b | c | d | e | f | g | h |")
    submit.append_submission(path, "| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |")
    text = path.read_text()
    assert text.count("| date | exp | run | cv | public_lb | message | kernel_version | file |") == 1
    assert text.rstrip().splitlines()[-2:] == ["| a | b | c | d | e | f | g | h |", "| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |"]


def test_shell_line_quotes_arguments_with_spaces():
    cmd = submit.final_command("csv", "titanic", file="out/submission.csv", message="exp000/default cv=0.69 baseline")
    line = submit.shell_line(cmd)
    assert line == "kaggle competitions submit -c titanic -f out/submission.csv -m 'exp000/default cv=0.69 baseline'"
