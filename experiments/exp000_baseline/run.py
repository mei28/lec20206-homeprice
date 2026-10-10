"""exp000_baseline: LightGBM on raw Titanic columns.

base: none
hypothesis: a GBDT on the raw numeric and categorical columns gives a sane first CV to compare against
issue: none
"""

import datetime as dt
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import tyro

from kgl import cv
from kgl.env import Env, comp_dir, detect, run_dir, run_id
from kgl.io import Metrics, git_commit, write_config, write_metrics, write_oof
from kgl.log import get_logger, seed_everything, trace
from kgl.metrics import score_folds
from kgl.registry import build, get, register
from kgl.tracker import make_tracker

EXP = Path(__file__).resolve().parent.name


@dataclass
class ModelParams:
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 20
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8


@dataclass
class Config:
    comp: str = "titanic"
    target: str = "Survived"
    id_col: str = "PassengerId"
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    sample_sub: str = "gender_submission.csv"
    folds: str = "skf5_seed42"
    metric: str = "accuracy"
    model: str = "lightgbm.LGBMClassifier"
    model_params: ModelParams = field(default_factory=ModelParams)
    features: tuple[str, ...] = ("basic",)
    seed: int = 42
    tracker: str = "none"
    debug: bool = False
    run: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        d = dict(d)
        d["model_params"] = ModelParams(**d["model_params"])
        d["features"] = tuple(d["features"])
        return cls(**d)


# --- features -----------------------------------------------------------------------------------

SEX = {"male": 0, "female": 1}
EMBARKED = {"S": 0, "C": 1, "Q": 2}


@register("basic")
def f_basic(df: pl.DataFrame) -> pl.DataFrame:
    """Raw columns as numbers. Category codes are fixed maps, so train and test are encoded identically."""
    return df.select(
        pl.col("Pclass").cast(pl.Float64),
        pl.col("Sex").replace_strict(SEX, return_dtype=pl.Float64).alias("Sex"),
        pl.col("Age").cast(pl.Float64),
        pl.col("SibSp").cast(pl.Float64),
        pl.col("Parch").cast(pl.Float64),
        pl.col("Fare").cast(pl.Float64),
        pl.col("Embarked").replace_strict(EMBARKED, default=None, return_dtype=pl.Float64).alias("Embarked"),
    )


def build_features(df: pl.DataFrame, cfg: Config) -> np.ndarray:
    blocks = [get(name)(df) for name in cfg.features]
    return pl.concat(blocks, how="horizontal").to_numpy()


def load_train(env: Env, cfg: Config) -> pl.DataFrame:
    return pl.read_csv(comp_dir(env, cfg.comp) / cfg.train_file)


def load_test(env: Env, cfg: Config) -> pl.DataFrame:
    return pl.read_csv(comp_dir(env, cfg.comp) / cfg.test_file)


# --- model --------------------------------------------------------------------------------------


def fit_fold(X_tr: np.ndarray, y_tr: np.ndarray, cfg: Config) -> lgb.Booster:
    """The validation fold is not passed to fit: with a fixed tree count it would only shape the model
    if early stopping were enabled, and that would make the CV optimistic."""
    model = build(cfg.model, **asdict(cfg.model_params), random_state=cfg.seed, verbose=-1)
    model.fit(X_tr, y_tr)
    return model.booster_


def predict(models: list[lgb.Booster], X: np.ndarray) -> np.ndarray:
    """Mean positive-class probability over fold models. Shared with infer.py."""
    return np.mean([m.predict(X) for m in models], axis=0)


def postprocess(prob: np.ndarray) -> np.ndarray:
    return (prob >= 0.5).astype(int)


# --- main ---------------------------------------------------------------------------------------


def main(cfg: Config) -> None:
    started = dt.datetime.now().isoformat(timespec="seconds")
    env = detect()
    run = cfg.run or run_id(sys.argv[1:])
    out = run_dir(env, EXP, run)
    logger = get_logger(f"{EXP}/{run}", out)
    seed_everything(cfg.seed)
    write_config(out, asdict(cfg), sys.argv[1:])
    logger.info("config: %s", asdict(cfg))

    tracker = make_tracker(cfg.tracker, project=cfg.comp)
    tracker.start(run_name=f"{EXP}/{run}", config=asdict(cfg))

    fold_path = env.root / "folds" / f"{cfg.folds}.parquet"
    is_time = cv.read_fold_meta(fold_path)["strategy"].rsplit(".", 1)[-1] in cv.TIME_STRATEGIES
    train = cv.attach_folds(load_train(env, cfg), cv.load_folds(fold_path), cfg.id_col)
    if cfg.debug:
        train = train.head(200)
    with trace("features"):
        X = build_features(train, cfg)
    y = train[cfg.target].to_numpy()
    folds = train["fold"].to_numpy()

    run_folds = [k for k in sorted(set(folds.tolist())) if k >= 0]
    if cfg.debug:
        run_folds = run_folds[:1]
    (out / "model").mkdir(exist_ok=True)
    oof = np.full(len(y), np.nan)
    for k in run_folds:
        tr, va = cv.split(folds, k, time=is_time)
        with trace(f"fold {k}"):
            booster = fit_fold(X[tr], y[tr], cfg)
        booster.save_model(str(out / "model" / f"fold{k}.txt"))
        oof[va] = booster.predict(X[va])
        score = score_folds(y[va], postprocess(oof[va]), np.zeros(len(va)), cfg.metric)[0]
        logger.info("fold %d %s=%.4f", k, cfg.metric, score)
        tracker.log({f"fold_{cfg.metric}": score}, step=k)

    scored = np.isin(folds, run_folds)
    cv_score, per_fold = score_folds(y[scored], postprocess(oof[scored]), folds[scored], cfg.metric)
    logger.info("cv %s=%.4f folds=%s", cfg.metric, cv_score, [round(s, 4) for s in per_fold])
    tracker.log({f"cv_{cfg.metric}": cv_score})
    tracker.finish()

    write_oof(
        out, train.select(cfg.id_col, cfg.target, "fold").with_columns(pl.Series("pred", oof)).filter(pl.Series(scored))
    )
    write_metrics(
        out,
        Metrics(
            exp=EXP,
            base=_base_from_docstring(),
            run=run,
            cv=cv_score,
            cv_folds=per_fold,
            metric=cfg.metric,
            config=asdict(cfg),
            argv=sys.argv[1:],
            git_commit=git_commit(env.root) if env.kind == "local" else "kaggle",
            started_at=started,
            finished_at=dt.datetime.now().isoformat(timespec="seconds"),
            notes=cfg.notes,
        ),
    )
    print(f"cv={cv_score:.4f} -> {out}")


def _base_from_docstring() -> str:
    for line in (__doc__ or "").splitlines():
        if line.startswith("base:"):
            return line.split(":", 1)[1].strip()
    raise ValueError("run.py docstring must contain a 'base:' line")


if __name__ == "__main__":
    main(tyro.cli(Config))
