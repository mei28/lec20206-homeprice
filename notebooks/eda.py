import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from kgl.env import comp_dir, detect

    return comp_dir, detect, mo, pl


@app.cell
def _(mo):
    mo.md(
        """
        # EDA

        Questions to answer before modelling: how was the test set built, which columns shift between
        train and test, what does the target look like. Conclusions go to `docs/eda.md`.
        """
    )
    return


@app.cell
def _(comp_dir, detect, mo, pl):
    import os

    comp = os.environ.get("COMP", "titanic")
    data_dir = comp_dir(detect(), comp)
    train = pl.read_csv(data_dir / "train.csv")
    test = pl.read_csv(data_dir / "test.csv")
    mo.md(f"`{comp}`: train {train.shape}, test {test.shape}")
    return test, train


@app.cell
def _(mo, train):
    mo.vstack([mo.md("## Train: schema and nulls"), train.null_count(), train.describe()])
    return


@app.cell
def _(mo, pl, test, train):
    numeric = [c for c, t in train.schema.items() if t.is_numeric() and c in test.columns]
    shift = pl.DataFrame(
        {
            "column": numeric,
            "train_mean": [train[c].mean() for c in numeric],
            "test_mean": [test[c].mean() for c in numeric],
            "train_null_pct": [train[c].null_count() / train.height for c in numeric],
            "test_null_pct": [test[c].null_count() / test.height for c in numeric],
        }
    )
    mo.vstack([mo.md("## Train vs test on shared numeric columns"), shift])
    return


if __name__ == "__main__":
    app.run()
