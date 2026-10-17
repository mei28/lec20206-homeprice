import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from dataclasses import asdict, dataclass, field

    import marimo as mo
    import numpy as np
    import polars as pl
    import psutil
    from tabpfn import TabPFNRegressor
    from threadpoolctl import threadpool_limits

    from kgl import classic, cv, ensemble, feature_set, incontext, lesson, mlit, stacking
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, read_metrics, validate_submission, write_submission
    from kgl.metrics import get_metric

    return (
        TabPFNRegressor,
        asdict,
        classic,
        comp_dir,
        cv,
        dataclass,
        detect,
        ensemble,
        feature_set,
        field,
        get_metric,
        git_commit,
        incontext,
        lesson,
        mlit,
        mo,
        np,
        pl,
        psutil,
        read_metrics,
        stacking,
        threadpool_limits,
        validate_submission,
        write_submission,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S15. 2段目の学習に使う行を増やす

    スタッキングの2段目（各メンバーの予測を入力にして最終の予測を出すモデル）は、holdout の行でしか学習できません。
    2025Q1 と Q2 で交互に学習して採点するので、1回の学習に使えるのは1四半期分、約1万1千行です。
    S12 では LightGBM の2段目が Ridge より 0.001 良かったのに、S13 で TabPFN が入ると4つの2段目が 0.1916〜0.1918 に並びました。
    行が少ないと、非線形の2段目は「重みを付けて足す」以上のことを学びにくいのです。

    K-fold の交差検証なら、学習データの全行に「そのモデルが学習に使わなかった行への予測」（OOF）があるので、2段目も全行で学習できます。
    時系列の分け方ではそうはいきませんが、もっと前の時点でも同じ手順で予測を作れば、2段目の学習行を増やせます（#43）。

    この段階では、すべてのメンバーに 2024Q2 までで学習させて 2024Q3〜Q4 を予測させ、その 17,931 行を2段目の学習に足します。
    メンバーは S14 と同じです（探索した LightGBM・CatBoost・XGBoost・MLP、S7 の RandomForest と Ridge、S13 の TabPFN）。
    """)
    return


@app.cell
def _(dataclass, field, psutil):
    THREADS = psutil.cpu_count(logical=False)

    @dataclass
    class ForestParams:
        # S7's RandomForest
        n_estimators: int = 150
        min_samples_leaf: int = 10
        max_features: float = 0.33
        max_samples: float = 0.5
        random_state: int = 42
        n_jobs: int = THREADS

    @dataclass
    class TabPFNParams:
        # S13's TabPFN
        context_rows: int = 50000
        predict_chunk: int = 2000
        seed: int = 0

    @dataclass
    class StackerLgbParams:
        # S12-S14's LightGBM stacker
        objective: str = "regression"
        learning_rate: float = 0.03
        num_leaves: int = 15
        min_data_in_leaf: int = 100
        feature_fraction: float = 0.9
        seed: int = 42
        verbose: int = -1
        num_threads: int = THREADS

    @dataclass
    class Config:
        comp: str = "utokyo-mlapp-2026-ex1"
        target: str = "取引価格（総額）"
        id_col: str = "id"
        train_file: str = "train.csv"
        test_file: str = "test.csv"
        sample_sub: str = "sample_submission.csv"
        folds: str = "time_2024q3_2025q1"
        metric: str = "rmsle"
        trend_quarters: int = 8
        te_smoothing: float = 20.0
        te_folds: int = 5
        seed: int = 42
        tuned_stage: str = "s14_tuning"
        max_rounds: int = 10000
        catboost_max_rounds: int = 12000
        early_stopping_rounds: int = 100
        ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
        forest: ForestParams = field(default_factory=ForestParams)
        tabpfn: TabPFNParams = field(default_factory=TabPFNParams)
        stacking_alpha: float = 1.0
        stack_mlp_hidden: tuple[int, ...] = (32, 16)
        stack_mlp_alpha: float = 1e-3
        stack_mlp_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
        stack_lgbm_max_rounds: int = 2000
        stack_lgbm_early_stopping_rounds: int = 100
        inner_valid_share: float = 0.2
        stack_lgbm: StackerLgbParams = field(default_factory=StackerLgbParams)

    cfg = Config()
    EXP = "s15_more_stack_rows"
    BASE = "s14_tuning"
    return BASE, EXP, THREADS, cfg


@app.cell
def _(cfg, classic, comp_dir, cv, detect, feature_set, mlit, pl):
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
    _build = dict(
        te_smoothing=cfg.te_smoothing, te_folds=cfg.te_folds, seed=cfg.seed, trend_quarters=cfg.trend_quarters
    )

    # Holdout setting (S13, S14): train up to 2024Q4, validate on 2025Q1-Q2 (fold 1).
    tr_idx, va_idx = cv.split(train_folded["fold"].to_numpy(), 1, time=True)
    fs = feature_set.build(train_folded, test_raw, tr_idx, va_idx, **_build)
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    in_q1 = train_folded[va_idx].select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy() == 2025.0
    y_holdout = fs.y_log[va_idx]

    # The earlier origin (S14's search split): rows up to 2024Q4; train up to 2024Q2 (fold -1), predict 2024Q3-Q4 (fold 0).
    inner_raw = train_folded.filter(pl.col("fold") <= 0)
    tr_in, va_in = cv.split(inner_raw["fold"].to_numpy(), 0, time=True)
    fs_in = feature_set.build(inner_raw, test_raw, tr_in, va_in, **_build)
    y_in = fs_in.y_log - fs_in.offset
    y_block = fs_in.y_log[va_in]
    block_ids = inner_raw[va_in].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    extras_in = classic.numeric_extras(inner_raw)
    linear_in = classic.linear_frame(fs_in.X, extras_in)
    forest_in = classic.forest_frame(fs_in.X, extras_in)
    return (
        block_ids,
        env,
        forest_in,
        fs,
        fs_in,
        holdout_ids,
        in_q1,
        linear_in,
        sample,
        tr_in,
        va_idx,
        va_in,
        y_block,
        y_holdout,
        y_in,
    )


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
def _(cfg, env, holdout_ids, np, pl, read_metrics, sample):
    def saved(stage, run):
        """Holdout and test predictions of a saved run, on the log scale."""
        oof = pl.read_parquet(env.output_dir / stage / run / "oof.parquet")
        sub = pl.read_csv(env.output_dir / stage / run / "submission.csv")
        assert oof["id"].to_list() == holdout_ids["id"].to_list() and sub["id"].to_list() == sample["id"].to_list()
        return np.log1p(oof["pred"].to_numpy()), np.log1p(sub["price"].to_numpy())

    # S14's member set, in S14's order
    members = {
        "lightgbm": saved(cfg.tuned_stage, "lightgbm_tuned"),
        "catboost": saved(cfg.tuned_stage, "catboost_tuned"),
        "xgboost": saved(cfg.tuned_stage, "xgboost_tuned"),
        "ridge": saved("s7_mlp_classic", "ridge"),
        "random_forest": saved("s7_mlp_classic", "random_forest"),
        "mlp": saved(cfg.tuned_stage, "mlp_tuned"),
        "tabpfn": saved("s13_tabular_dl", "tabpfn"),
    }
    s14_average = saved(cfg.tuned_stage, "stacker_average_tuned")

    def tuned_params(name, base_key):
        """The full parameters S14 trained the tuned member with: its base settings and the chosen ones."""
        config = read_metrics(env.output_dir / cfg.tuned_stage / f"{name}_tuned" / "metrics.json").config
        return config[base_key] | config["chosen"], config

    return members, s14_average, tuned_params


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 2024Q3〜Q4 の予測を作る

    S14 の探索と同じ分け方（2024Q4 までの行で特徴量を作り、価格指数と Target encoding は 2024Q2 までの行から）で、
    各メンバーを 2024Q2 までの行で学習し、2024Q3〜Q4 を予測します。設定は holdout の予測を作ったときと同じです。

    - LightGBM・CatBoost・XGBoost・MLP: S14 で選んだ設定（学習率はメンバーと同じ）。LightGBM は seed を3つ平均
    - RandomForest・Ridge: S7 の設定
    - TabPFN: S13 と同じく、2024Q2 までの直近5万行を文脈にする

    GBDT は、holdout のときに holdout で early stopping したのと同じく、ここでは 2024Q3〜Q4 で early stopping します。
    """)
    return


@app.cell
def _(
    TabPFNRegressor,
    THREADS,
    asdict,
    cfg,
    classic,
    forest_in,
    fs_in,
    incontext,
    lesson,
    linear_in,
    np,
    threadpool_limits,
    tr_in,
    tuned_params,
    va_in,
    y_in,
):
    def gbdt_kwargs(max_rounds):
        return dict(max_rounds=max_rounds, early_stopping_rounds=cfg.early_stopping_rounds)

    def block_lightgbm():
        params, config = tuned_params("lightgbm", "lgbm")
        preds = [
            lesson.fit_lgbm_holdout(
                fs_in.X, y_in, tr_in, va_in, params | {"seed": seed}, **gbdt_kwargs(cfg.max_rounds)
            )[1]
            for seed in config["lgbm_seeds"]
        ]
        return np.mean(preds, axis=0)

    def block_catboost():
        params, _ = tuned_params("catboost", "cat")
        return lesson.fit_catboost_holdout(fs_in.X, y_in, tr_in, va_in, params, **gbdt_kwargs(cfg.catboost_max_rounds))[
            1
        ]

    def block_xgboost():
        params, _ = tuned_params("xgboost", "xgb")
        return lesson.fit_xgb_holdout(fs_in.X, y_in, tr_in, va_in, params, **gbdt_kwargs(cfg.max_rounds))[1]

    def block_mlp():
        params, _ = tuned_params("mlp", "mlp")
        with threadpool_limits(THREADS):
            model = classic.make_mlp(linear_in, params).fit(linear_in.iloc[tr_in], y_in[tr_in])
            return model.predict(linear_in.iloc[va_in])

    def block_ridge():
        model = classic.make_ridge(linear_in, alphas=cfg.ridge_alphas).fit(linear_in.iloc[tr_in], y_in[tr_in])
        return model.predict(linear_in.iloc[va_in])

    def block_forest():
        model = classic.make_forest(asdict(cfg.forest)).fit(forest_in.iloc[tr_in], y_in[tr_in])
        return model.predict(forest_in.iloc[va_in])

    def block_tabpfn():
        context = incontext.most_recent(tr_in, fs_in.X["trade_period"].to_numpy(), cfg.tabpfn.context_rows)
        model = TabPFNRegressor(device="cuda", random_state=cfg.tabpfn.seed)
        model.fit(fs_in.X.iloc[context], y_in[context])
        return incontext.predict_in_chunks(model, fs_in.X.iloc[va_in], cfg.tabpfn.predict_chunk)

    BLOCK = {
        "lightgbm": block_lightgbm,
        "catboost": block_catboost,
        "xgboost": block_xgboost,
        "ridge": block_ridge,
        "random_forest": block_forest,
        "mlp": block_mlp,
        "tabpfn": block_tabpfn,
    }
    return (BLOCK,)


@app.cell
def _(BLOCK, EXP, block_ids, env, fs_in, holdout_rmsle, members, pl, va_in, y_block, y_holdout):
    block_members = {_name: _predict() + fs_in.offset[va_in] for _name, _predict in BLOCK.items()}
    assert list(block_members) == list(members)
    (env.output_dir / EXP).mkdir(parents=True, exist_ok=True)
    block_ids.with_columns([pl.Series(_name, _pred) for _name, _pred in block_members.items()]).write_parquet(
        env.output_dir / EXP / "block_predictions.parquet"
    )
    pl.DataFrame(
        [
            {
                "member": _name,
                "2024Q3-Q4 (block)": holdout_rmsle(block_members[_name], y_block),
                "2025Q1-Q2 (holdout)": holdout_rmsle(members[_name][0], y_holdout),
            }
            for _name in members
        ]
    )
    return (block_members,)


@app.cell
def _(mo):
    mo.md(r"""
    表は各メンバーの RMSLE を、2つの期間で並べたものです。2024Q3〜Q4 の行を足すと、2段目はこの期間の「どのメンバーがどれだけ当たるか」も学びます。
    2つの期間でメンバーの順位が大きく違えば、足した行は2段目を迷わせます。

    ## 2. 行を足して2段目を学習し直す

    S14 と同じ4つの2段目（Ridge、種類別の Ridge、MLP、LightGBM）と重み付き平均を、3通りの学習行で比べます。

    - holdout だけ（S14 と同じ）: 2025Q1 の行は Q2 で学習した2段目で、Q2 の行は Q1 で学習した2段目で予測する
    - 2024Q3〜Q4 を足す: 上の学習行に、両方向とも 2024Q3〜Q4 の行を足す
    - 2024Q3〜Q4 だけ: 2024Q3〜Q4 だけで学習して 2025Q1〜Q2 を予測する。時間の順向きだけで、test の予測の仕方に一番近い

    提出用の2段目は、2024Q3〜2025Q2 の4四半期の行すべてで学習します。
    まず holdout だけの設定で S14 の `stacker_average_tuned` を再現できることを確かめます。
    """)
    return


@app.cell
def _(asdict, block_members, cfg, fs, fs_in, members, np, stacking, va_idx, va_in):
    names = list(members)
    found = stacking.stackers(
        len(names),
        alpha=cfg.stacking_alpha,
        mlp_hidden=cfg.stack_mlp_hidden,
        mlp_alpha=cfg.stack_mlp_alpha,
        mlp_seeds=cfg.stack_mlp_seeds,
        lgbm_params=asdict(cfg.stack_lgbm),
        lgbm_max_rounds=cfg.stack_lgbm_max_rounds,
        lgbm_early_stopping_rounds=cfg.stack_lgbm_early_stopping_rounds,
        valid_share=cfg.inner_valid_share,
        seed=cfg.seed,
    )
    H = np.column_stack([members[n][0] for n in names])
    T = np.column_stack([members[n][1] for n in names])
    H_block = np.column_stack([block_members[n] for n in names])
    rows_holdout = stacking.row_features(fs.X.iloc[va_idx])
    rows_test = stacking.row_features(fs.X_test)
    rows_block = stacking.row_features(fs_in.X.iloc[va_in])
    return H, H_block, T, found, rows_block, rows_holdout, rows_test


@app.cell
def _(
    H,
    H_block,
    T,
    ensemble,
    found,
    holdout_rmsle,
    in_q1,
    np,
    pl,
    rows_block,
    rows_holdout,
    rows_test,
    s14_average,
    stacking,
    y_block,
    y_holdout,
):
    holdout_only = stacking.stacker_average(H, T, rows_holdout, rows_test, y_holdout, in_q1, found)
    assert np.allclose(holdout_only[0], s14_average[0], atol=1e-9)
    assert np.allclose(holdout_only[1], s14_average[1], atol=1e-9)
    with_block = stacking.stacker_average_with_block(
        H, T, rows_holdout, rows_test, y_holdout, in_q1, H_block, rows_block, y_block, found
    )
    forward_only = stacking.stacker_average_forward(H, rows_holdout, H_block, rows_block, y_block, found)

    # Each stacker on its own, under the three choices of training rows
    _Z, _Z_block = np.hstack([H, rows_holdout]), np.hstack([H_block, rows_block])
    _per_stacker = {
        _name: (
            ensemble.cross_predict(_Z, y_holdout, in_q1, _fit, _predict),
            ensemble.cross_predict_with_block(_Z, y_holdout, in_q1, _Z_block, y_block, _fit, _predict),
            _predict(_fit(_Z_block, y_block), _Z),
        )
        for _name, (_fit, _predict) in found.items()
    }
    _weighted = ensemble.fit_weights
    _per_stacker["weighted_average"] = (
        ensemble.cross_predict(H, y_holdout, in_q1, _weighted, lambda w, Hp: Hp @ w),
        ensemble.cross_predict_with_block(H, y_holdout, in_q1, H_block, y_block, _weighted, lambda w, Hp: Hp @ w),
        H @ _weighted(H_block, y_block),
    )
    _per_stacker["average of the five"] = (holdout_only[0], with_block[0], forward_only)
    per_stacker = pl.DataFrame(
        [
            {
                "stacker": _name,
                "holdout rows only": holdout_rmsle(_a, y_holdout),
                "with 2024Q3-Q4": holdout_rmsle(_b, y_holdout),
                "2024Q3-Q4 only": holdout_rmsle(_c, y_holdout),
            }
            for _name, (_a, _b, _c) in _per_stacker.items()
        ]
    )
    per_stacker
    return forward_only, holdout_only, per_stacker, with_block


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    bootstrap_diff,
    cfg,
    env,
    forward_only,
    git_commit,
    holdout_ids,
    holdout_only,
    holdout_rmsle,
    lesson,
    members,
    np,
    pl,
    sample,
    validate_submission,
    with_block,
    write_submission,
    y_holdout,
):
    commit = git_commit(env.root)
    _base_sq = (holdout_only[0] - y_holdout) ** 2
    block_change = bootstrap_diff((with_block[0] - y_holdout) ** 2, _base_sq)
    forward_change = bootstrap_diff((forward_only - y_holdout) ** 2, _base_sq)
    block_score = holdout_rmsle(with_block[0], y_holdout)
    _out = lesson.record_run(
        env,
        exp=EXP,
        base=BASE,
        run="stacker_average_with_block",
        score=block_score,
        metric=cfg.metric,
        config=asdict(cfg) | {"members": list(members), "block": "2024Q3-Q4 (fold 0)"},
        holdout=holdout_ids,
        pred=np.expm1(with_block[0]),
        notes="S14 stacker average with the members' 2024Q3-Q4 predictions added to every stacker's training rows (#43)",
        commit=commit,
    )
    _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(with_block[1])})
    validate_submission(_submission, sample, cfg.id_col)
    write_submission(_out, _submission)
    pl.DataFrame(
        [
            {
                "run": "s14 stacker_average_tuned (holdout rows only)",
                "holdout_rmsle": holdout_rmsle(holdout_only[0], y_holdout),
            },
            {
                "run": "stacker_average_with_block",
                "holdout_rmsle": block_score,
                "change_vs_s14": block_change[0],
                "ci_low": block_change[1],
                "ci_high": block_change[2],
            },
            {
                "run": "2024Q3-Q4 only (forward in time)",
                "holdout_rmsle": holdout_rmsle(forward_only, y_holdout),
                "change_vs_s14": forward_change[0],
                "ci_low": forward_change[1],
                "ci_high": forward_change[2],
            },
        ],
        infer_schema_length=None,
    )
    return block_change, block_score, forward_change


@app.cell
def _(block_change, block_score, forward_change, mo):
    mo.md(f"""
    ## 3. 結果

    - 2024Q3〜Q4 を足したスタッキングの平均: {block_score:.4f}（S14 との差 {block_change[0]:+.4f}、95% の範囲 {block_change[1]:+.4f}〜{block_change[2]:+.4f}）
    - 2024Q3〜Q4 だけで学習（時間の順向きだけ）: S14 との差 {forward_change[0]:+.4f}（95% の範囲 {forward_change[1]:+.4f}〜{forward_change[2]:+.4f}）
    """)
    return


if __name__ == "__main__":
    app.run()
