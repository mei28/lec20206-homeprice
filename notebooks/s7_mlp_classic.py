import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    from dataclasses import asdict, dataclass, field

    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import psutil
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from kgl import cv, feature_set, lesson, mlit
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, validate_submission, write_submission
    from kgl.metrics import get_metric

    return (
        ColumnTransformer,
        MLPRegressor,
        OneHotEncoder,
        RandomForestRegressor,
        RidgeCV,
        SimpleImputer,
        StandardScaler,
        TransformedTargetRegressor,
        asdict,
        comp_dir,
        cv,
        dataclass,
        detect,
        feature_set,
        field,
        get_metric,
        git_commit,
        lesson,
        make_pipeline,
        mlit,
        mo,
        np,
        pd,
        pl,
        psutil,
        time,
        validate_submission,
        write_submission,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S7. MLP と古典的なモデル（Ridge・RandomForest）

    S6 までは決定木のブースティング（GBDT）だけでした。この段階では、仕組みのまったく違うモデルを3つ作ります。

    | モデル | 仕組み | Issue |
    |---|---|---|
    | Ridge | 特徴量の重み付き和（線形回帰）に、重みを小さく保つ罰則を付けたもの | #29 |
    | RandomForest | 深い木を独立にたくさん育てて平均する（バギング） | #28 |
    | MLP | 全結合の層を重ねた素朴なニューラルネット | #20 |

    単独のスコアは GBDT に届かない見込みです。この段階の狙いは、GBDT と違う間違え方をするモデルを用意することで、
    その価値は S9 のアンサンブルで確かめます。

    特徴量と目的変数は S5 までと同じです（`kgl.feature_set`、目的変数は log 価格からヘドニック指数を引いた実質価格）。
    ただし Ridge と MLP は、カテゴリをそのまま扱えず、数値の大きさもそろえる必要があるので、前処理を加えます。
    """)
    return


@app.cell
def _(dataclass, field, psutil):
    THREADS = psutil.cpu_count(logical=False)

    @dataclass
    class MlpParams:
        hidden_layer_sizes: tuple[int, ...] = (256, 128)
        alpha: float = 1e-4
        batch_size: int = 512
        learning_rate_init: float = 1e-3
        max_iter: int = 60
        early_stopping: bool = True
        n_iter_no_change: int = 5
        random_state: int = 42

    @dataclass
    class ForestParams:
        n_estimators: int = 150
        min_samples_leaf: int = 10
        max_features: float = 0.33
        max_samples: float = 0.5
        random_state: int = 42
        n_jobs: int = THREADS

    @dataclass
    class Config:
        comp: str = "utokyo-mlapp-2026-ex1"
        target: str = "取引価格（総額）"
        id_col: str = "id"
        train_file: str = "train.csv"
        test_file: str = "test.csv"
        sample_sub: str = "sample_submission.csv"
        folds: str = "holdout_2025q1"
        metric: str = "rmsle"
        trend_quarters: int = 8
        te_smoothing: float = 20.0
        te_folds: int = 5
        seed: int = 42
        ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
        mlp: MlpParams = field(default_factory=MlpParams)
        forest: ForestParams = field(default_factory=ForestParams)

    cfg = Config()
    EXP = "s7_mlp_classic"
    BASE = "s5_te"
    return BASE, EXP, cfg


@app.cell
def _(cfg, comp_dir, cv, detect, feature_set, mlit, np, pl):
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

    def numeric_extras(df):
        """Columns a linear model needs as numbers: build year and age instead of the 建築年 category, and log area."""
        return df.select(
            mlit.build_year("建築年").cast(pl.Float64).alias("build_year_num"),
            (mlit.trade_period("取引時期").floor() - mlit.build_year("建築年")).cast(pl.Float64).alias("age"),
            mlit.area_m2("面積（㎡）").log().alias("log_area"),
        ).to_pandas()

    extras_train, extras_test = numeric_extras(train_folded), numeric_extras(test_raw)
    y_real = fs.y_log - fs.offset
    y_real_refit = fs.y_log - fs.offset_refit
    assert np.isfinite(y_real).all()
    return (
        env,
        extras_test,
        extras_train,
        fs,
        sample,
        test_raw,
        tr_idx,
        train_folded,
        va_idx,
        y_real,
        y_real_refit,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 前処理

    - 数値の列: 欠損を学習データの中央値で埋め（欠損だったことを示す列も足す）、平均 0・標準偏差 1 にそろえる（Ridge と MLP）。
      件数の列は桁が大きく違うので、log をとってから使います。
    - 種類の少ないカテゴリ（市区町村、間取り、構造など）: one-hot（値ごとに 0/1 の列を作る）。20行未満の値はまとめます。
    - 種類の多いカテゴリ（地区、駅）: S5 の Target encoding の列で代わりにし、カテゴリそのものは使いません。
    - 建築年: カテゴリではなく、数値の建築年と築年数にします（S3 では GBDT に効きませんでしたが、線形モデルには年の大小の順序が必要です）。
    - RandomForest: 木なので大きさをそろえる必要はなく、カテゴリは順序のコード（0, 1, 2, …）で渡します。

    前処理の中央値・平均・標準偏差・one-hot の語彙は、学習に使う行だけで計算します（scikit-learn の Pipeline に入れて、学習のたびに fit し直す）。
    holdout の行の値で前処理を作ると、それもリークになります。
    """)
    return


@app.cell
def _(extras_test, extras_train, fs, np, pd):
    CATEGORICAL = [c for c in fs.X.columns if str(fs.X[c].dtype) == "category"]
    ONE_HOT = [c for c in CATEGORICAL if c not in ("地区名", "最寄駅：名称", "建築年")]
    COUNTS = [c for c in fs.X.columns if c.startswith("count_")]
    NUMERIC = [c for c in fs.X.columns if c not in CATEGORICAL and c not in COUNTS]

    def linear_frame(X, extras, *, with_counts=True):
        """Inputs for Ridge and the MLP: numbers (counts as log1p), extras, and the small categories as strings."""
        out = pd.concat([X[NUMERIC].reset_index(drop=True), extras.reset_index(drop=True)], axis=1)
        if with_counts:
            for c in COUNTS:
                out[f"log_{c}"] = np.log1p(X[c].to_numpy().astype(float))
        for c in ONE_HOT:
            out[c] = X[c].cat.add_categories("NA").fillna("NA").astype(str).to_numpy()
        return out

    def forest_frame(X, extras):
        """Inputs for RandomForest: numbers with NaN kept, and every category as its integer code (NaN if missing)."""
        out = pd.concat([X.drop(columns=CATEGORICAL).reset_index(drop=True), extras.reset_index(drop=True)], axis=1)
        for c in CATEGORICAL:
            out[c] = X[c].cat.codes.astype(float).where(lambda codes: codes >= 0).to_numpy()
        return out.astype(float)

    frames = {
        "linear": (
            linear_frame(fs.X, extras_train),
            linear_frame(fs.X_refit, extras_train),
            linear_frame(fs.X_test, extras_test),
        ),
        "linear_no_counts": (
            linear_frame(fs.X, extras_train, with_counts=False),
            linear_frame(fs.X_refit, extras_train, with_counts=False),
            linear_frame(fs.X_test, extras_test, with_counts=False),
        ),
        "forest": (
            forest_frame(fs.X, extras_train),
            forest_frame(fs.X_refit, extras_train),
            forest_frame(fs.X_test, extras_test),
        ),
    }
    return ONE_HOT, frames


@app.cell
def _(
    ColumnTransformer,
    MLPRegressor,
    ONE_HOT,
    OneHotEncoder,
    RandomForestRegressor,
    RidgeCV,
    SimpleImputer,
    StandardScaler,
    TransformedTargetRegressor,
    asdict,
    cfg,
    make_pipeline,
):
    def linear_preprocess(frame):
        numeric = [c for c in frame.columns if c not in ONE_HOT]
        return ColumnTransformer(
            [
                (
                    "numeric",
                    make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
                    numeric,
                ),
                ("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), ONE_HOT),
            ]
        )

    def make_model(name, frame):
        """A fresh, unfitted model; every preprocessing step is inside, so it is fit on the training rows only."""
        if name in ("ridge", "ridge_no_counts"):
            return make_pipeline(linear_preprocess(frame), RidgeCV(alphas=cfg.ridge_alphas))
        if name == "mlp":
            # The target is standardized too, so the network starts near the right scale.
            return TransformedTargetRegressor(
                regressor=make_pipeline(linear_preprocess(frame), MLPRegressor(**asdict(cfg.mlp))),
                transformer=StandardScaler(),
            )
        if name == "random_forest":
            return RandomForestRegressor(**asdict(cfg.forest))
        raise KeyError(name)

    MODELS = {
        "ridge": ("linear", "#29"),
        "ridge_no_counts": ("linear_no_counts", "#33"),
        "random_forest": ("forest", "#28"),
        "mlp": ("linear", "#20"),
    }
    return MODELS, make_model


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 学習して比べる

    holdout の設定（学習期間の行で fit、2025年第1〜第2四半期で採点）で4つ学習します。`ridge_no_counts` は、件数の列（#33）を抜いた Ridge です。
    件数が線形モデルに効くかを確かめるために比べます。
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
    frames,
    fs,
    get_metric,
    git_commit,
    lesson,
    make_model,
    np,
    pl,
    time,
    tr_idx,
    train_folded,
    va_idx,
    y_real,
):
    rmsle = get_metric("rmsle")
    commit = git_commit(env.root)
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    results = {}
    holdout_logs = {}
    for _name, (_frame_key, _issue) in MODELS.items():
        _X = frames[_frame_key][0]
        _model = make_model(_name, _X)
        _start = time.time()
        _model.fit(_X.iloc[tr_idx], y_real[tr_idx])
        _seconds = time.time() - _start
        _pred_log = _model.predict(_X.iloc[va_idx]) + fs.offset[va_idx]
        _score = rmsle(np.expm1(fs.y_log[va_idx]), np.expm1(_pred_log))
        results[_name] = {
            "run": _name,
            "issue": _issue,
            "holdout_rmsle": _score,
            "bias": float((_pred_log - fs.y_log[va_idx]).mean()),
            "fit_seconds": round(_seconds),
        }
        holdout_logs[_name] = _pred_log
        lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=_name,
            score=_score,
            metric=cfg.metric,
            config=asdict(cfg) | {"model": _name, "inputs": _frame_key, "features": "kgl.feature_set (S5 all_three)"},
            holdout=holdout_ids,
            pred=np.expm1(_pred_log),
            notes=f"{_name} on the S5 feature set ({_issue})",
            commit=commit,
        )
    return holdout_ids, holdout_logs, results


@app.cell
def _(env, fs, holdout_ids, holdout_logs, np, pl, results, va_idx):
    def load_log(stage, run):
        oof = pl.read_parquet(env.output_dir / stage / run / "oof.parquet")
        assert oof["id"].to_list() == holdout_ids["id"].to_list()
        return np.log1p(oof["pred"].to_numpy())

    gbdt_logs = {
        "lightgbm": load_log("s5_te", "all_three"),
        "xgboost": load_log("s6_gbdt", "xgboost"),
        "catboost": load_log("s6_gbdt", "catboost"),
    }
    errors = {name: pred - fs.y_log[va_idx] for name, pred in (gbdt_logs | holdout_logs).items()}
    boot_idx = np.random.default_rng(0).integers(0, len(va_idx), size=(2000, len(va_idx)))

    def paired_diff(sq_a, sq_b):
        """RMSLE(a) - RMSLE(b) on the holdout and its 95% bootstrap range, resampling the same rows for both."""
        diffs = np.sqrt(sq_a[boot_idx].mean(axis=1)) - np.sqrt(sq_b[boot_idx].mean(axis=1))
        low, high = np.percentile(diffs, [2.5, 97.5])
        return float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean())), float(low), float(high)

    summary = pl.DataFrame(
        [
            {
                "run": name,
                "holdout_rmsle": float(np.sqrt((errors[name] ** 2).mean())),
                "bias": float(errors[name].mean()),
            }
            for name in gbdt_logs
        ]
        + [
            {k: r[k] for k in ("run", "issue", "holdout_rmsle", "bias", "fit_seconds")}
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
    counts_effect = paired_diff(errors["ridge"] ** 2, errors["ridge_no_counts"] ** 2)
    summary
    return counts_effect, errors, summary


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 間違え方は GBDT と違うか

    S6 と同じく、holdout の各行の誤差（予測の log − 正解の log）の相関を、GBDT 3つと今回の4つで並べます。
    """)
    return


@app.cell
def _(errors, np, pl):
    names = list(errors)
    correlation = pl.DataFrame(
        {"model": names} | {b: [round(float(np.corrcoef(errors[a], errors[b])[0, 1]), 3) for a in names] for b in names}
    )
    correlation
    return (correlation,)


@app.cell
def _(correlation, counts_effect, mo, pl, summary):
    def row(run):
        return summary.filter(pl.col("run") == run).row(0, named=True)

    def corr(a, b):
        return correlation.filter(pl.col("model") == a)[b][0]

    mo.md(f"""
    ## 結果を読む

    - MLP は {row("mlp")["holdout_rmsle"]:.4f}、RandomForest は {row("random_forest")["holdout_rmsle"]:.4f}、Ridge は {row("ridge")["holdout_rmsle"]:.4f} で、
      どれも単独では GBDT（LightGBM {row("lightgbm")["holdout_rmsle"]:.4f}、CatBoost {row("catboost")["holdout_rmsle"]:.4f}）に届きませんでした。予想どおりです。
    - 間違え方は違います。LightGBM との誤差の相関は、Ridge が {corr("ridge", "lightgbm"):.3f}、MLP と RandomForest が {corr("mlp", "lightgbm"):.3f} と {corr("random_forest", "lightgbm"):.3f} で、
      GBDT どうし（LightGBM と XGBoost で {corr("xgboost", "lightgbm"):.3f}）より低くなっています。これが S9 で役に立つかを確かめます。
    - Ridge と MLP は平均のずれ（{row("ridge")["bias"]:+.3f}、{row("mlp")["bias"]:+.3f}）が GBDT より大きく、低めに予測しています。
    - 件数の列（#33）は Ridge で {counts_effect[0]:+.4f}（95% の範囲 {counts_effect[1]:+.4f}〜{counts_effect[2]:+.4f}）効きました。
      LightGBM では偶然の範囲だった特徴量が、カテゴリを細かく扱えない線形モデルでは効く、という #33 の予想どおりの結果です。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 提出ファイルを作る

    4つのモデルを全期間の行で fit し直し（前処理も全期間の行で作り直す）、test を予測します。S9 のアンサンブルの材料にします。
    """)
    return


@app.cell
def _(
    EXP,
    MODELS,
    cfg,
    env,
    frames,
    fs,
    make_model,
    np,
    pl,
    sample,
    test_raw,
    validate_submission,
    write_submission,
    y_real_refit,
):
    submission_paths = {}
    for _name, (_frame_key, _issue) in MODELS.items():
        _, _X_refit, _X_test = frames[_frame_key]
        _model = make_model(_name, _X_refit)
        _model.fit(_X_refit, y_real_refit)
        _submission = pl.DataFrame(
            {cfg.id_col: test_raw[cfg.id_col], "price": np.expm1(_model.predict(_X_test) + fs.offset_test)}
        )
        validate_submission(_submission, sample, cfg.id_col)
        submission_paths[_name] = str(write_submission(env.output_dir / EXP / _name, _submission))
    submission_paths
    return


if __name__ == "__main__":
    app.run()
