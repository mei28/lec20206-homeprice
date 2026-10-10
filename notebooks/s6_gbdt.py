import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import time
    from dataclasses import asdict, dataclass, field

    import catboost as cb
    import marimo as mo
    import numpy as np
    import polars as pl
    import psutil
    import xgboost as xgb

    from kgl import cv, feature_set, lesson, mlit
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, validate_submission, write_submission
    from kgl.metrics import get_metric

    return (
        asdict,
        cb,
        comp_dir,
        cv,
        dataclass,
        detect,
        feature_set,
        field,
        get_metric,
        git_commit,
        json,
        lesson,
        mlit,
        mo,
        np,
        pl,
        psutil,
        time,
        validate_submission,
        write_submission,
        xgb,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S6. GBDT を比べる（LightGBM・XGBoost・CatBoost）

    ここまでのモデルはすべて LightGBM でした。GBDT（勾配ブースティング決定木）には、ほかにもよく使われるライブラリがあります。
    同じ特徴量・同じ目的変数・同じ holdout で学習し、スコアと「間違え方」を比べます。

    | ライブラリ | 木の育て方 | カテゴリの扱い | Issue |
    |---|---|---|---|
    | LightGBM | 誤差が最も減る葉から1枚ずつ分ける（葉ごと） | カテゴリを並べ替えて2つに分ける | S5 の `all_three` |
    | XGBoost | 深さを揃えて1段ずつ分ける（深さごと） | カテゴリを2つの集合に分ける（`enable_categorical`） | #27 |
    | CatBoost | 各段で全部の葉に同じ条件を使う（対称な木） | 行の順番をずらした Target encoding を内部で作る | #24 |

    特徴量と目的変数は S5 までに採用したものです（`kgl.feature_set`：S3 の列、地区と駅の TE、集約特徴、件数。目的変数は log 価格からヘドニック指数を引いた実質価格）。
    LightGBM の結果は S5 の `all_three` をそのまま使います。

    この段階の結果は、S9 のアンサンブルの材料にもなります。単独のスコアが少し悪くても、LightGBM と違う間違え方をするモデルは、平均を取ったときに役に立ちます。
    """)
    return


@app.cell
def _(dataclass, field, psutil):
    THREADS = psutil.cpu_count(logical=False)

    @dataclass
    class XgbParams:
        objective: str = "reg:squarederror"
        eta: float = 0.05
        max_depth: int = 8
        min_child_weight: float = 20.0
        subsample: float = 0.8
        colsample_bytree: float = 0.8
        tree_method: str = "hist"
        # Partition splits for every categorical column (one-hot only for columns with a single category).
        max_cat_to_onehot: int = 1
        seed: int = 42
        nthread: int = THREADS

    @dataclass
    class CatParams:
        learning_rate: float = 0.08
        depth: int = 8
        loss_function: str = "RMSE"
        random_seed: int = 42
        thread_count: int = THREADS

    @dataclass
    class Config:
        comp: str = "utokyo-mlapp-2026-ex1"
        target: str = "取引価格（総額）"
        id_col: str = "id"
        test_file: str = "test.csv"
        train_file: str = "train.csv"
        sample_sub: str = "sample_submission.csv"
        folds: str = "holdout_2025q1"
        metric: str = "rmsle"
        max_rounds: int = 5000
        early_stopping_rounds: int = 100
        trend_quarters: int = 8
        te_smoothing: float = 20.0
        te_folds: int = 5
        seed: int = 42
        xgb: XgbParams = field(default_factory=XgbParams)
        cat: CatParams = field(default_factory=CatParams)

    cfg = Config()
    EXP = "s6_gbdt"
    BASE = "s5_te"
    return BASE, EXP, cfg


@app.cell
def _(cfg, comp_dir, cv, detect, feature_set, pl):
    env = detect()
    data_dir = comp_dir(env, cfg.comp)
    train_raw = pl.read_csv(data_dir / cfg.train_file, infer_schema_length=0).with_columns(
        pl.col(cfg.id_col).cast(pl.Int64)
    )
    test_raw = pl.read_csv(data_dir / cfg.test_file, infer_schema_length=0).with_columns(
        pl.col(cfg.id_col).cast(pl.Int64)
    )
    sample = pl.read_csv(data_dir / cfg.sample_sub)
    train_folded = cv.attach_folds(train_raw, cv.load_folds(env.root / "folds" / f"{cfg.folds}.parquet"), cfg.id_col)
    tr_idx, va_idx = cv.split(train_folded["fold"].to_numpy(), 0, time=True)
    fs = feature_set.build(
        train_folded,
        test_raw,
        tr_idx,
        va_idx,
        te_smoothing=cfg.te_smoothing,
        te_folds=cfg.te_folds,
        seed=cfg.seed,
        trend_quarters=cfg.trend_quarters,
    )
    y_real = fs.y_log - fs.offset
    y_real_refit = fs.y_log - fs.offset_refit
    return env, fs, sample, test_raw, tr_idx, train_folded, va_idx, y_real, y_real_refit


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 学習する関数

    どのライブラリも、holdout で early stopping し、本数を S1 と同じ規則（holdout での本数 × 行数の比）で増やして全期間で学習し直します。
    CatBoost はカテゴリの欠損をそのまま受け取れないので、欠損を `"NA"` という文字列にします。
    """)
    return


@app.cell
def _(asdict, cb, cfg, fs, lesson, tr_idx, va_idx, xgb):
    CATEGORICAL = [c for c in fs.X.columns if str(fs.X[c].dtype) == "category"]

    def fit_xgb(y_real):
        train = xgb.DMatrix(fs.X.iloc[tr_idx], label=y_real[tr_idx], enable_categorical=True)
        valid = xgb.DMatrix(fs.X.iloc[va_idx], label=y_real[va_idx], enable_categorical=True)
        booster = xgb.train(
            asdict(cfg.xgb),
            train,
            cfg.max_rounds,
            evals=[(valid, "holdout")],
            early_stopping_rounds=cfg.early_stopping_rounds,
            verbose_eval=False,
        )
        pred = booster.predict(valid, iteration_range=(0, booster.best_iteration + 1))
        return booster.best_iteration + 1, pred

    def refit_xgb(rounds, y_real_refit):
        full = xgb.DMatrix(fs.X_refit, label=y_real_refit, enable_categorical=True)
        booster = xgb.train(asdict(cfg.xgb), full, rounds)
        return booster.predict(xgb.DMatrix(fs.X_test, enable_categorical=True))

    def for_catboost(X):
        out = X.copy()
        for c in CATEGORICAL:
            out[c] = out[c].cat.add_categories("NA").fillna("NA").astype(str)
        return out

    def fit_cat(y_real):
        model = cb.CatBoostRegressor(
            iterations=cfg.max_rounds, od_type="Iter", od_wait=cfg.early_stopping_rounds, verbose=0, **asdict(cfg.cat)
        )
        X_cat = for_catboost(fs.X)
        model.fit(
            X_cat.iloc[tr_idx],
            y_real[tr_idx],
            cat_features=CATEGORICAL,
            eval_set=(X_cat.iloc[va_idx], y_real[va_idx]),
            use_best_model=True,
        )
        return model.get_best_iteration() + 1, model.predict(X_cat.iloc[va_idx])

    def refit_cat(rounds, y_real_refit):
        model = cb.CatBoostRegressor(iterations=rounds, verbose=0, **asdict(cfg.cat))
        model.fit(for_catboost(fs.X_refit), y_real_refit, cat_features=CATEGORICAL)
        return model.predict(for_catboost(fs.X_test))

    def scaled_rounds(best):
        return lesson.refit_rounds(best, n_all=len(fs.y_log), n_train=len(tr_idx))

    MODELS = {"xgboost": (fit_xgb, refit_xgb, "#27"), "catboost": (fit_cat, refit_cat, "#24")}
    return MODELS, scaled_rounds


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 学習して比べる

    XGBoost と CatBoost を holdout で学習します。所要時間も記録します。LightGBM（S5 の `all_three`）は、保存済みの holdout の予測を読み込みます。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    MODELS,
    asdict,
    cfg,
    env,
    fs,
    get_metric,
    git_commit,
    lesson,
    np,
    pl,
    scaled_rounds,
    time,
    train_folded,
    va_idx,
    y_real,
):
    rmsle = get_metric("rmsle")
    commit = git_commit(env.root)
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    results = {}
    holdout_logs = {}
    best_rounds = {}
    for _name, (_fit, _refit, _issue) in MODELS.items():
        _start = time.time()
        _best, _pred_real = _fit(y_real)
        _seconds = time.time() - _start
        _pred_log = _pred_real + fs.offset[va_idx]
        _score = rmsle(np.expm1(fs.y_log[va_idx]), np.expm1(_pred_log))
        results[_name] = {
            "run": _name,
            "issue": _issue,
            "holdout_rmsle": _score,
            "bias": float((_pred_log - fs.y_log[va_idx]).mean()),
            "best_rounds": _best,
            "fit_seconds": round(_seconds),
        }
        holdout_logs[_name] = _pred_log
        best_rounds[_name] = _best
        lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=_name,
            score=_score,
            metric=cfg.metric,
            config=asdict(cfg)
            | {
                "model": _name,
                "features": "kgl.feature_set (S5 all_three)",
                "best_rounds": _best,
                "refit_rounds": scaled_rounds(_best),
            },
            holdout=holdout_ids,
            pred=np.expm1(_pred_log),
            notes=f"{_name} on the S5 feature set ({_issue})",
            commit=commit,
        )
    return best_rounds, holdout_ids, holdout_logs, results


@app.cell
def _(env, fs, holdout_ids, holdout_logs, json, np, pl, results, va_idx):
    lgbm_oof = pl.read_parquet(env.output_dir / "s5_te" / "all_three" / "oof.parquet")
    assert lgbm_oof["id"].to_list() == holdout_ids["id"].to_list()
    lgbm_metrics = json.loads((env.output_dir / "s5_te" / "all_three" / "metrics.json").read_text())
    all_logs = {"lightgbm": np.log1p(lgbm_oof["pred"].to_numpy())} | holdout_logs
    errors = {name: pred - fs.y_log[va_idx] for name, pred in all_logs.items()}
    boot_idx = np.random.default_rng(0).integers(0, len(va_idx), size=(2000, len(va_idx)))

    def paired_diff(sq_a, sq_b):
        """RMSLE(a) - RMSLE(b) on the holdout and its 95% bootstrap range, resampling the same rows for both."""
        diffs = np.sqrt(sq_a[boot_idx].mean(axis=1)) - np.sqrt(sq_b[boot_idx].mean(axis=1))
        low, high = np.percentile(diffs, [2.5, 97.5])
        return float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean())), float(low), float(high)

    summary = pl.DataFrame(
        [
            {
                "run": "lightgbm",
                "issue": "S5",
                "holdout_rmsle": float(np.sqrt((errors["lightgbm"] ** 2).mean())),
                "bias": float(errors["lightgbm"].mean()),
                "best_rounds": lgbm_metrics["config"]["best_iteration"],
            }
        ]
        + [
            {k: r[k] for k in ("run", "issue", "holdout_rmsle", "bias", "best_rounds", "fit_seconds")}
            | dict(
                zip(
                    ("change_vs_lightgbm", "ci_low", "ci_high"),
                    paired_diff(errors[r["run"]] ** 2, errors["lightgbm"] ** 2),
                    strict=True,
                )
            )
            for r in results.values()
        ],
        infer_schema_length=None,
    )
    summary
    return errors, summary


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 間違え方は似ているか

    holdout の各行の誤差（予測の log − 正解の log）の相関を見ます。相関が 1 に近いほど、同じ行で同じように外しています。
    相関が低いモデルどうしは、平均を取ると誤差が打ち消し合います（S9）。
    """)
    return


@app.cell
def _(errors, np, pl):
    names = list(errors)
    correlation = pl.DataFrame(
        {"model": names} | {b: [float(np.corrcoef(errors[a], errors[b])[0, 1]) for a in names] for b in names}
    )
    correlation
    return (correlation,)


@app.cell
def _(errors, mlit, pl, train_folded, va_idx):
    age = train_folded[va_idx].select((mlit.trade_period("取引時期").floor() - mlit.build_year("建築年")).alias("age"))[
        "age"
    ]
    age_frame = pl.DataFrame({"age": age} | {name: err for name, err in errors.items()})
    bias_by_age = (
        age_frame.drop_nulls("age")
        .filter(pl.col("age").is_between(0, 59))
        .with_columns(age_bin=pl.col("age") // 10 * 10)
        .group_by("age_bin")
        .agg([pl.col(name).mean().round(3) for name in errors])
        .sort("age_bin")
    )
    bias_by_age
    return (bias_by_age,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 提出ファイルを作る

    XGBoost と CatBoost を全期間で学習し直し、test を予測します。S9 のアンサンブルでは、これらの test 予測と holdout の予測を材料にします。
    """)
    return


@app.cell
def _(
    EXP,
    MODELS,
    best_rounds,
    cfg,
    env,
    fs,
    np,
    pl,
    sample,
    scaled_rounds,
    test_raw,
    validate_submission,
    write_submission,
    y_real_refit,
):
    submission_paths = {}
    for _name, (_fit, _refit, _issue) in MODELS.items():
        _test_real = _refit(scaled_rounds(best_rounds[_name]), y_real_refit)
        _submission = pl.DataFrame({cfg.id_col: test_raw[cfg.id_col], "price": np.expm1(_test_real + fs.offset_test)})
        validate_submission(_submission, sample, cfg.id_col)
        submission_paths[_name] = str(write_submission(env.output_dir / EXP / _name, _submission))
    submission_paths
    return


if __name__ == "__main__":
    app.run()
