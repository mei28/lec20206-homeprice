"""Wait for the latest Kaggle submission to finish scoring, print the public LB, and optionally record it.

Usage: uv run python -m tools.check_submission [--interval 30] [--record --exp exp000_baseline --run default]
The --record form appends the row to docs/submissions.md for a submission whose `just submit --now`
run was interrupted before recording.
"""

import datetime as dt
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
from kagglesdk.competitions.types.submission_status import SubmissionStatus

from kgl.io import read_metrics
from tools.submit import append_submission, record_row


def latest(api: Any, comp: str) -> Any:
    subs = api.competition_submissions(comp)
    if not subs:
        raise RuntimeError(f"no submissions found for {comp}")
    return subs[0]


def wait_until_complete(api: Any, comp: str, ref: Any, interval: float) -> Any:
    """Poll until submission `ref` leaves PENDING. Raise when Kaggle reports an error."""
    while True:
        matches = [s for s in api.competition_submissions(comp) if s.ref == ref]
        if not matches:
            raise RuntimeError(f"submission {ref} disappeared from {comp}")
        sub = matches[0]
        if sub.status == SubmissionStatus.COMPLETE:
            return sub
        if sub.status == SubmissionStatus.ERROR:
            raise RuntimeError(f"submission {ref} failed: {sub.error_description}")
        time.sleep(interval)


def row_for(sub: Any, *, exp: str, run: str, cv: float, date: str) -> str:
    """A docs/submissions.md row for a scored submission; the free-text part of the description is the message."""
    prefix = f"{exp}/{run} cv={cv:.4f} "
    message = sub.description[len(prefix) :] if sub.description.startswith(prefix) else sub.description
    return record_row(
        date=date,
        exp=exp,
        run=run,
        cv=cv,
        public_lb=str(sub.public_score),
        message=message,
        kernel_version="",
        file=sub.file_name,
    )


@dataclass
class Args:
    comp: str = os.environ.get("COMP", "")
    interval: float = 30.0
    record: bool = False
    exp: str = ""
    run: str = "default"


def main(args: Args) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    if not args.comp:
        raise SystemExit("--comp is required (or set COMP in .env)")
    if args.record and not args.exp:
        raise SystemExit("--record needs --exp (and --run when not default)")
    api = KaggleApi()
    api.authenticate()
    sub = latest(api, args.comp)
    print(f"waiting for submission {sub.ref} ({sub.status.name}) ...")
    done = wait_until_complete(api, args.comp, sub.ref, args.interval)
    print(f"public LB: {done.public_score}")
    if args.record:
        root = Path(__file__).resolve().parents[1]
        metrics = read_metrics(root / "output" / args.exp / args.run / "metrics.json")
        row = row_for(done, exp=args.exp, run=args.run, cv=metrics.cv, date=dt.date.today().isoformat())
        append_submission(root / "docs" / "submissions.md", row)
        print("recorded in docs/submissions.md")


if __name__ == "__main__":
    main(tyro.cli(Args))
