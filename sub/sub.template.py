import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import subprocess
    import sys
    from pathlib import Path

    CODES = "/kaggle/input/$CODES"
    MODEL = "/kaggle/input/$MODEL"
    EXP = "$EXP"
    DEPS = "$DEPS"
    return CODES, DEPS, EXP, MODEL, Path, os, subprocess, sys


@app.cell
def _(DEPS, subprocess, sys):
    if DEPS:
        wheels = sorted(__import__("glob").glob(f"/kaggle/input/{DEPS}/*.whl"))
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *wheels], check=True)
    return


@app.cell
def _(CODES, EXP, MODEL, os, subprocess, sys):
    subprocess.run(
        [sys.executable, f"{CODES}/experiments/{EXP}/infer.py", "--model-dir", f"{MODEL}/model"],
        env={**os.environ, "PYTHONPATH": f"{CODES}/src"},
        check=True,
    )
    return


@app.cell
def _(Path):
    import polars as pl

    sub = pl.read_csv(Path("/kaggle/working/submission.csv"))
    print(sub.shape)
    print(sub.head())
    return


if __name__ == "__main__":
    app.run()
