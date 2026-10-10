"""Post an experiment's result as a comment on its idea Issue.

Usage: uv run python tools/issue_comment.py exp001_more_trees 12 [--run default] [--lb 0.7799]
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import tyro
from tyro.conf import Positional

from kgl.io import Metrics, read_metrics


def compose(metrics: Metrics, lb: str | None) -> str:
    folds = ", ".join(f"{s:.4f}" for s in metrics.cv_folds)
    argv = " ".join(metrics.argv) if metrics.argv else "(defaults)"
    lines = [
        f"### {metrics.exp} / {metrics.run}",
        "",
        f"- CV ({metrics.metric}): **{metrics.cv:.4f}** (folds: {folds})",
    ]
    if lb is not None:
        lines.append(f"- public LB: {lb}")
    lines += [
        f"- base: {metrics.base}",
        f"- overrides: `{argv}`",
        f"- output: `output/{metrics.exp}/{metrics.run}`",
        f"- commit: {metrics.git_commit[:7]}",
    ]
    if metrics.notes:
        lines += ["", metrics.notes]
    return "\n".join(lines) + "\n"


@dataclass
class Args:
    exp: Positional[str]
    issue: Positional[int]
    run: str = "default"
    lb: str | None = None


def main(args: Args) -> None:
    root = Path(__file__).resolve().parents[1]
    metrics = read_metrics(root / "output" / args.exp / args.run / "metrics.json")
    body = compose(metrics, args.lb)
    subprocess.run(["gh", "issue", "comment", str(args.issue), "--body-file", "-"], input=body, text=True, check=True)
    print(f"commented on #{args.issue}")


if __name__ == "__main__":
    main(tyro.cli(Args))
