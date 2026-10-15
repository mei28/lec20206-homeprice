import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from dataclasses import asdict, dataclass, field

    import lightgbm as lgb
    import marimo as mo
    import numpy as np
    import polars as pl
    import psutil
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from kgl import cv, ensemble, feature_set, lesson, mlit
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, validate_submission, write_submission
    from kgl.metrics import get_metric

    return (
        MLPRegressor,
        Ridge,
        SimpleImputer,
        StandardScaler,
        asdict,
        comp_dir,
        cv,
        dataclass,
        detect,
        ensemble,
        feature_set,
        field,
        get_metric,
        git_commit,
        lesson,
        lgb,
        make_pipeline,
        mlit,
        mo,
        np,
        pl,
        psutil,
        validate_submission,
        write_submission,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S12. 黒魔術（非線形のスタッキングと、スタッキングの平均）

    参考にするのは、俵さんの「[黒魔術への招待：Neural Network Stacking の探求](https://tawara.hatenablog.com/entry/2020/12/16/132415)」（Kaggle Advent Calendar 2020）です（#35）。
    MoA というコンペ（206クラスの多ラベル分類）で、5つのモデルの予測を NN（MLP、1D-CNN、2D-CNN、GCN）でスタッキングしました。
    さらに、作ったスタッキングをすべて単純に平均すると、Private の順位が34位から1〜2位相当まで上がった、という記事です。

    この課題にそのまま持ち込めるものと、持ち込めないものがあります。

    | 記事の手法 | この課題では |
    |---|---|
    | 1D-CNN、2D-CNN、GCN | 使わない。206クラス × 5モデルという「クラスの軸」の構造を使う方法で、目的変数が1つ・メンバーが6つのこの課題には使える構造がない |
    | MLP のスタッキング、seed 平均 | 使う |
    | スタッキングどうしの平均（Take Me Higher） | 使う |

    もう1つ、この課題向けの工夫を足します。S10 の Ridge のスタッキングは、全部の行に同じ重みを使います。
    スタッキングに行の特徴（成約価格か取引価格か、面積、駅からの分数、地区の価格水準）も見せれば、物件によって頼るメンバーを変えられるかもしれません。

    記事が強調している注意点もそのまま当てはまります。スタッキングは過学習しやすく、良い単独モデルがあってこその手法です。
    ここでスタッキングを学習できるのは holdout の約2.2万行（四半期ごとに約1.1万行）だけなので、なおさらです。
    """)
    return


@app.cell
def _(dataclass, field, psutil):
    @dataclass
    class StackerLgbParams:
        objective: str = "regression"
        learning_rate: float = 0.03
        num_leaves: int = 15
        min_data_in_leaf: int = 100
        feature_fraction: float = 0.9
        seed: int = 42
        verbose: int = -1
        num_threads: int = psutil.cpu_count(logical=False)

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
        stacking_alpha: float = 1.0
        mlp_hidden: tuple[int, ...] = (32, 16)
        mlp_alpha: float = 1e-3
        mlp_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
        lgbm_max_rounds: int = 2000
        lgbm_early_stopping_rounds: int = 100
        inner_valid_share: float = 0.2
        lgbm: StackerLgbParams = field(default_factory=StackerLgbParams)

    cfg = Config()
    EXP = "s12_black_magic"
    BASE = "s10_final"
    return BASE, EXP, cfg


@app.cell
def _(cfg, comp_dir, cv, detect, feature_set, mlit, pl):
    env = detect()
    data_dir = comp_dir(env, cfg.comp)
    train_raw = pl.read_csv(data_dir / cfg.train_file, infer_schema_length=0).with_columns(
        pl.col(cfg.id_col).cast(pl.Int64)
    )
    test_raw = pl.read_csv(data_dir / cfg.test_file, infer_schema_length=0).with_columns(
        pl.col(cfg.id_col).cast(pl.Int64)
    )
    sample = pl.read_csv(data_dir / cfg.sample_sub)
    assert test_raw[cfg.id_col].to_list() == sample[cfg.id_col].to_list()
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
    y_holdout = fs.y_log[va_idx]
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    in_q1 = train_folded[va_idx].select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy() == 2025.0
    return env, fs, holdout_ids, in_q1, sample, va_idx, y_holdout


@app.cell
def _(get_metric, np):
    rmsle = get_metric("rmsle")

    def holdout_rmsle(pred_log, y_log):
        return rmsle(np.expm1(y_log), np.expm1(pred_log))

    def bootstrap_diff(sq_a, sq_b, *, n=2000, seed=0):
        """RMSLE(a) - RMSLE(b) and its 95% range, resampling the same holdout rows for both."""
        idx = np.random.default_rng(seed).integers(0, len(sq_a), size=(n, len(sq_a)))
        diffs = np.sqrt(sq_a[idx].mean(axis=1)) - np.sqrt(sq_b[idx].mean(axis=1))
        low, high = np.percentile(diffs, [2.5, 97.5])
        return float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean())), float(low), float(high)

    return bootstrap_diff, holdout_rmsle


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. スタッキングの入力

    メンバーは S10 と同じ6つ（LightGBM の seed 平均、CatBoost、XGBoost、Ridge、RandomForest、MLP）で、保存してある予測をそのまま読み込みます。
    行の特徴は4つです。

    | 列 | 意味 |
    |---|---|
    | `contract` | 成約価格情報なら 1、取引価格情報なら 0 |
    | `log_area` | 面積の log |
    | `station_minutes` | 最寄駅からの分数（S3 で数値化したもの） |
    | `te_district` | 地区の価格水準（S5 の Target encoding） |

    holdout の行には holdout の設定で作った特徴（2024Q4 までのデータから計算）を、test の行には全期間で作った特徴を使います。各メンバーと同じ使い分けです。
    """)
    return


@app.cell
def _(env, fs, holdout_ids, np, pl, sample, va_idx):
    def saved(stage, run):
        """Holdout and test predictions of a saved run, on the log scale."""
        oof = pl.read_parquet(env.output_dir / stage / run / "oof.parquet")
        sub = pl.read_csv(env.output_dir / stage / run / "submission.csv")
        assert oof["id"].to_list() == holdout_ids["id"].to_list() and sub["id"].to_list() == sample["id"].to_list()
        return np.log1p(oof["pred"].to_numpy()), np.log1p(sub["price"].to_numpy())

    members = {
        "lightgbm_seed_avg": saved("s10_final", "lightgbm_seed_avg"),
        "catboost_long": saved("s10_final", "catboost_long"),
        "xgboost": saved("s6_gbdt", "xgboost"),
        "ridge": saved("s7_mlp_classic", "ridge"),
        "random_forest": saved("s7_mlp_classic", "random_forest"),
        "mlp": saved("s7_mlp_classic", "mlp"),
    }
    member_names = list(members)
    s10_stacking = saved("s10_final", "stacking")
    s10_weighted = saved("s10_final", "weighted_average")
    ROW_FEATURES = ["contract", "log_area", "station_minutes", "te_district"]

    def row_features(X):
        """The row features the stackers may condition on, in the order of ROW_FEATURES."""
        return np.column_stack(
            [
                (X["価格情報区分"] == "成約価格情報").to_numpy(dtype=float),
                np.log(X["area"].to_numpy()),
                X["station_minutes"].to_numpy(),
                X["te_district"].to_numpy(),
            ]
        )

    # Z: the members' predictions first, then the row features
    Z_holdout = np.hstack([np.column_stack([members[n][0] for n in member_names]), row_features(fs.X.iloc[va_idx])])
    Z_test = np.hstack([np.column_stack([members[n][1] for n in member_names]), row_features(fs.X_test)])
    {
        "rows (holdout, test)": (len(Z_holdout), len(Z_test)),
        "missing per row feature (holdout)": dict(
            zip(ROW_FEATURES, np.isnan(Z_holdout[:, len(member_names) :]).sum(axis=0).tolist(), strict=True)
        ),
        "contract share (holdout, test)": (
            round(float(Z_holdout[:, len(member_names)].mean()), 3),
            round(float(Z_test[:, len(member_names)].mean()), 3),
        ),
    }
    return Z_holdout, Z_test, member_names, s10_stacking, s10_weighted


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 4つのスタッキングと、比べるための1つ

    | run | 中身 |
    |---|---|
    | `ridge` | S10 の Ridge のスタッキングそのもの（基準）。メンバーの予測だけを見て、全行に同じ重みを使う |
    | `ridge_by_type` | 成約価格の行と取引価格の行で、メンバーの重みを別々にした Ridge |
    | `mlp` | MLP。入力は「各メンバーの予測と6つの平均とのずれ」「6つの平均」「行の特徴」。出力は6つの平均への補正。seed を5つ変えて平均する（記事と同じ） |
    | `lgbm` | 浅い LightGBM（葉15枚）。入力と出力は MLP と同じ。学習する四半期のうちランダムな20%で early stopping |
    | `lgbm_preds_only` | 比べるための run。`lgbm` から行の特徴を抜いたもの。スタッキングの平均には入れない |

    MLP と LightGBM に「6つの平均への補正」を予測させるのは、スタッキングの出発点を平均にしておくためです。
    何も学ばなければ平均になり、学べたところだけ平均から動きます。

    採点は S10 と同じです。2025Q1 で学習して Q2 を予測し、逆向きも行います。test 用には holdout 全体で学習し直します。
    """)
    return


@app.cell
def _(
    MLPRegressor,
    Ridge,
    SimpleImputer,
    StandardScaler,
    asdict,
    cfg,
    ensemble,
    lgb,
    make_pipeline,
    member_names,
    np,
):
    n_members = len(member_names)

    def fit_ridge(Z, y):
        return ensemble.fit_stacking(Z[:, :n_members], y, alpha=cfg.stacking_alpha)

    def predict_ridge(model, Z):
        return model.predict(Z[:, :n_members])

    def by_type(Z):
        """Member predictions split into a contract-price copy and a survey-price copy, plus a contract flag."""
        H, contract = Z[:, :n_members], Z[:, n_members : n_members + 1]
        return np.hstack([H * contract, H * (1 - contract), contract])

    def fit_ridge_by_type(Z, y):
        return Ridge(alpha=cfg.stacking_alpha).fit(by_type(Z), y)

    def predict_ridge_by_type(model, Z):
        return model.predict(by_type(Z))

    def correction_inputs(Z, *, with_rows=True):
        """Inputs for the nonlinear stackers (deviations from the members' mean, the mean, row features) and the mean."""
        H = Z[:, :n_members]
        mean = H.mean(axis=1)
        rows = [Z[:, n_members:]] if with_rows else []
        return np.hstack([H - mean[:, None], mean[:, None], *rows]), mean

    def fit_mlp(Z, y):
        X, mean = correction_inputs(Z)
        return [
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=cfg.mlp_hidden,
                    alpha=cfg.mlp_alpha,
                    early_stopping=True,
                    validation_fraction=cfg.inner_valid_share,
                    max_iter=500,
                    random_state=seed,
                ),
            ).fit(X, y - mean)
            for seed in cfg.mlp_seeds
        ]

    def predict_mlp(models, Z):
        X, mean = correction_inputs(Z)
        return mean + np.mean([m.predict(X) for m in models], axis=0)

    def fit_lgbm(Z, y, *, with_rows=True):
        X, mean = correction_inputs(Z, with_rows=with_rows)
        inner = np.random.default_rng(cfg.seed).random(len(y)) < cfg.inner_valid_share
        train_set = lgb.Dataset(X[~inner], label=(y - mean)[~inner])
        valid_set = lgb.Dataset(X[inner], label=(y - mean)[inner], reference=train_set)
        return lgb.train(
            asdict(cfg.lgbm),
            train_set,
            num_boost_round=cfg.lgbm_max_rounds,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(cfg.lgbm_early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
        )

    def predict_lgbm(booster, Z, *, with_rows=True):
        X, mean = correction_inputs(Z, with_rows=with_rows)
        return mean + booster.predict(X, num_iteration=booster.best_iteration)

    STACKERS = {
        "ridge": (fit_ridge, predict_ridge),
        "ridge_by_type": (fit_ridge_by_type, predict_ridge_by_type),
        "mlp": (fit_mlp, predict_mlp),
        "lgbm": (fit_lgbm, predict_lgbm),
    }
    # the LightGBM stacker without the row features: how much of its gain is nonlinearity on the predictions alone
    ABLATIONS = {
        "lgbm_preds_only": (
            lambda Z, y: fit_lgbm(Z, y, with_rows=False),
            lambda booster, Z: predict_lgbm(booster, Z, with_rows=False),
        ),
    }
    return ABLATIONS, STACKERS


@app.cell
def _(ABLATIONS, STACKERS, Z_holdout, Z_test, ensemble, in_q1, np, s10_stacking, s10_weighted, y_holdout):
    stacked = {}
    for _name, (_fit, _predict) in (STACKERS | ABLATIONS).items():
        stacked[_name] = (
            ensemble.cross_predict(Z_holdout, y_holdout, in_q1, _fit, _predict),
            _predict(_fit(Z_holdout, y_holdout), Z_test),
        )
    assert np.allclose(stacked["ridge"][0], s10_stacking[0], atol=1e-9)
    assert np.allclose(stacked["ridge"][1], s10_stacking[1], atol=1e-9)

    # Take Me Higher: the plain average of every stacker (not the ablation) and the S10 weighted average
    _averaged = [stacked[n] for n in STACKERS] + [s10_weighted]
    stacked["stacker_average"] = (
        np.mean([h for h, _ in _averaged], axis=0),
        np.mean([t for _, t in _averaged], axis=0),
    )
    return (stacked,)


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    bootstrap_diff,
    cfg,
    env,
    git_commit,
    holdout_ids,
    holdout_rmsle,
    lesson,
    member_names,
    np,
    pl,
    sample,
    stacked,
    validate_submission,
    write_submission,
    y_holdout,
):
    commit = git_commit(env.root)
    _base_sq = (stacked["ridge"][0] - y_holdout) ** 2
    _rows = []
    for _name, (_holdout_log, _test_log) in stacked.items():
        _score = holdout_rmsle(_holdout_log, y_holdout)
        _change, _low, _high = bootstrap_diff((_holdout_log - y_holdout) ** 2, _base_sq)
        _rows.append(
            {"run": _name, "holdout_rmsle": _score, "change_vs_s10": _change, "ci_low": _low, "ci_high": _high}
        )
        if _name == "ridge":
            continue  # the same predictions as output/s10_final/stacking
        _out = lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=_name,
            score=_score,
            metric=cfg.metric,
            config=asdict(cfg) | {"stacker": _name, "members": member_names},
            holdout=holdout_ids,
            pred=np.expm1(_holdout_log),
            notes=f"{_name} stacker on the S10 members (#35)",
            commit=commit,
        )
        _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(_test_log)})
        validate_submission(_submission, sample, cfg.id_col)
        write_submission(_out, _submission)
    comparison = pl.DataFrame(_rows)
    comparison
    return (comparison,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 物件の種類ごとに見る

    行の特徴を見せた狙いは、物件によって頼るメンバーを変えることでした。成約価格と取引価格に分けて、それぞれの holdout RMSLE を並べます。
    """)
    return


@app.cell
def _(Z_holdout, holdout_rmsle, member_names, pl, stacked, y_holdout):
    _contract = Z_holdout[:, len(member_names)] == 1.0
    by_type_scores = pl.DataFrame(
        [
            {
                "run": _name,
                "contract price": holdout_rmsle(_holdout_log[_contract], y_holdout[_contract]),
                "survey price": holdout_rmsle(_holdout_log[~_contract], y_holdout[~_contract]),
            }
            for _name, (_holdout_log, _) in stacked.items()
        ]
    )
    by_type_scores
    return (by_type_scores,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. test でも同じように補正しているか

    非線形のスタッキングには、holdout と test で条件が違うという不安があります。
    holdout でのメンバーの予測は 2024Q4 までで学習したモデルのもので、test でのメンバーの予測は全期間で学習し直したモデルのものです。
    スタッキングが覚えた「この形のずれならこう直す」が、test では合わないかもしれません。

    test の正解は分からないので、代わりに、Ridge のスタッキング（S10）からどれだけ予測を動かしたか（補正の大きさ）を holdout と test で比べます。
    補正の大きさが test で大きく変わっていれば、holdout で学んだことがそのまま移っていない兆候です。
    """)
    return


@app.cell
def _(np, pl, stacked):
    _ridge_holdout, _ridge_test = stacked["ridge"]
    corrections = pl.DataFrame(
        [
            {
                "run": _name,
                "holdout: sd": float(np.std(_h - _ridge_holdout)),
                "test: sd": float(np.std(_t - _ridge_test)),
                "holdout: 99th pct of |correction|": float(np.percentile(np.abs(_h - _ridge_holdout), 99)),
                "test: 99th pct of |correction|": float(np.percentile(np.abs(_t - _ridge_test), 99)),
            }
            for _name, (_h, _t) in stacked.items()
            if _name not in ("ridge", "ridge_by_type")
        ]
    )
    corrections
    return (corrections,)


@app.cell
def _(comparison, corrections, mo, pl):
    def row(run):
        return comparison.filter(pl.col("run") == run).row(0, named=True)

    def change(run):
        r = row(run)
        return f"{r['holdout_rmsle']:.4f}（S10 との差 {r['change_vs_s10']:+.4f}、95% の範囲 {r['ci_low']:+.4f}〜{r['ci_high']:+.4f}）"

    def sd(run, where):
        return corrections.filter(pl.col("run") == run)[f"{where}: sd"][0]

    mo.md(f"""
    ## 5. 結果を読む

    基準の S10 スタッキング（`ridge`）は {row("ridge")["holdout_rmsle"]:.4f} です。

    - 成約価格と取引価格で重みを分けた `ridge_by_type` は {change("ridge_by_type")} で、変わりませんでした。どちらの種類の物件でも、頼るべきメンバーはほぼ同じです。
    - 非線形のスタッキングは S10 より良くなりました。`mlp` は {change("mlp")}、`lgbm` は {change("lgbm")} です。
      `mlp` の範囲は 0 に近く、偶然の範囲すれすれです。
    - 行の特徴を抜いた `lgbm_preds_only` も {change("lgbm_preds_only")} でした。
      LightGBM のスタッキングの改善のうち、約半分は「メンバーの予測のずれ方に応じた補正」で、残りが行の特徴の分です。
    - いちばん良かったのは、スタッキングの平均 `stacker_average` の {change("stacker_average")} です。95% の範囲も最も狭くなりました。
      記事の Take Me Higher と同じく、作り方の違うスタッキングを平均すると、それぞれの癖が打ち消し合います。
    - test での補正の大きさ（標準偏差）は、`lgbm` が holdout の {sd("lgbm", "holdout"):.4f} から test の {sd("lgbm", "test"):.4f} に大きくなりました。
      `stacker_average` は {sd("stacker_average", "holdout"):.4f} と {sd("stacker_average", "test"):.4f} でほぼ同じです。平均は、test での振る舞いも安定しています。

    `stacker_average` を採用し、最終提出の候補に加えます（2026-10-16）。最終提出に何を選ぶかはユーザーが決めます。

    ただし、改善は 0.001 ほどです。これまでの提出では、LB は CV のおよそ半分しか動いていません。LB での改善は 0.0005 前後と見ておきます。
    記事では順位が大きく上がりましたが、MoA には206クラスの構造があり、予測の組み合わせ方を工夫する余地が大きかったと考えられます。
    この課題では、黒魔術は最後のひと押しです。効き目の大部分は、S3〜S6 で良い単独モデルを作ったところにあります。
    """)
    return


if __name__ == "__main__":
    app.run()
