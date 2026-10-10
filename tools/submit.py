"""Validate a submission file, print the exact `kaggle competitions submit` command, and run it on --now.

Usage: uv run python tools/submit.py exp000_baseline "baseline" [--run default] [--now]
Reads comp, sample file, and id column from the run's metrics.json (the resolved Config).
"""

import datetime as dt
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import tyro
from tyro.conf import Positional

from kgl.env import comp_dir, detect
from kgl.io import read_metrics, validate_submission

SUBMISSIONS_HEADER = (
    "# Submissions\n\n"
    "| date | exp | run | cv | public_lb | message | kernel_version | file |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def final_command(
    mode: str, comp: str, *, file: str, message: str, kernel: str | None = None, version: int | None = None
) -> list[str]:
    base = ["kaggle", "competitions", "submit", "-c", comp]
    if mode == "csv":
        return [*base, "-f", file, "-m", message]
    if mode == "code":
        if kernel is None or version is None:
            raise ValueError("code competitions need the kernel slug and the kernel version to submit")
        return [*base, "-k", kernel, "-v", str(version), "-f", file, "-m", message]
    raise ValueError(f"SUBMIT_MODE must be csv or code, got {mode!r}")


def shell_line(cmd: list[str]) -> str:
    """The command as one line the user can paste into a shell."""
    return shlex.join(cmd)


def submission_file(root: Path, exp: str, run: str) -> Path:
    path = root / "output" / exp / run / "submission.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `just infer {exp}` first")
    return path


def record_row(
    *, date: str, exp: str, run: str, cv: float, public_lb: str, message: str, kernel_version: str, file: str
) -> str:
    return f"| {date} | {exp} | {run} | {cv:.4f} | {public_lb} | {message} | {kernel_version} | {file} |"


def append_submission(path: Path, row: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SUBMISSIONS_HEADER)
    with path.open("a") as f:
        f.write(row + "\n")


@dataclass
class Args:
    exp: Positional[str]
    message: Positional[str]
    run: str = "default"
    now: bool = False
    mode: str = os.environ.get("SUBMIT_MODE", "csv")
    kernel: str | None = None
    version: int | None = None


def main(args: Args) -> None:
    env = detect()
    root = env.root
    metrics = read_metrics(root / "output" / args.exp / args.run / "metrics.json")
    cfg = metrics.config
    file = submission_file(root, args.exp, args.run)
    sample = pl.read_csv(comp_dir(env, cfg["comp"]) / cfg["sample_sub"])
    validate_submission(pl.read_csv(file), sample, cfg["id_col"])
    print(f"submission ok: {file} ({sample.height} rows, cv={metrics.cv:.4f})")

    message = f"{args.exp}/{args.run} cv={metrics.cv:.4f} {args.message}"
    cmd = final_command(
        args.mode, cfg["comp"], file=str(file), message=message, kernel=args.kernel, version=args.version
    )
    if not args.now:
        print(
            "to submit, run:\n  " + shell_line(cmd) + f"\nor: just submit {args.exp} {shlex.quote(args.message)} --now"
        )
        return

    limits = subprocess.run(
        ["kaggle", "competitions", "submission-limits", "-c", cfg["comp"]], capture_output=True, text=True
    )
    print(limits.stdout.strip())
    subprocess.run(cmd, check=True)

    from kaggle.api.kaggle_api_extended import KaggleApi

    from tools.check_submission import latest, wait_until_complete

    api = KaggleApi()
    api.authenticate()
    done = wait_until_complete(api, cfg["comp"], latest(api, cfg["comp"]).ref, interval=30)
    row = record_row(
        date=dt.date.today().isoformat(),
        exp=args.exp,
        run=args.run,
        cv=metrics.cv,
        public_lb=str(done.public_score),
        message=args.message,
        kernel_version=str(args.version or ""),
        file=file.name,
    )
    append_submission(root / "docs" / "submissions.md", row)
    print(f"public LB: {done.public_score} (recorded in docs/submissions.md)")


if __name__ == "__main__":
    main(tyro.cli(Args))
