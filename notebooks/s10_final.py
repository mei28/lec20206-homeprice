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

    from kgl import cv, ensemble, feature_set, lesson, mlit
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, validate_submission, write_submission
    from kgl.metrics import get_metric

    return (
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
    # S10. 仕上げ（seed 平均・CatBoost の本数・最終のアンサンブル）

    最後の段階では、新しい特徴量やモデルは足さず、これまでのモデルを仕上げます（#25）。

    1. seed 平均: 同じ設定のモデルを乱数の種（seed）だけ変えて何本か学習し、予測を平均する。木の作り方の偶然によるぶれが減る
    2. CatBoost の本数: S6 では上限 5,000 本の手前（4,796 本）まで改善が続いていたので、上限を上げて学習し直す
    3. 最終のアンサンブル: 1 と 2 の結果を入れて S9 と同じ方法で組み合わせ直し、最終提出の候補を選ぶ
    """)
    return


@app.cell
def _(dataclass, field, psutil):
    THREADS = psutil.cpu_count(logical=False)

    @dataclass
    class LgbParams:
        objective: str = "regression"
        learning_rate: float = 0.05
        num_leaves: int = 63
        min_data_in_leaf: int = 50
        feature_fraction: float = 0.8
        bagging_fraction: float = 0.8
        bagging_freq: int = 1
        seed: int = 42
        verbose: int = -1
        num_threads: int = THREADS

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
        train_file: str = "train.csv"
        test_file: str = "test.csv"
        sample_sub: str = "sample_submission.csv"
        folds: str = "holdout_2025q1"
        metric: str = "rmsle"
        max_rounds: int = 5000
        catboost_max_rounds: int = 12000
        early_stopping_rounds: int = 100
        trend_quarters: int = 8
        te_smoothing: float = 20.0
        te_folds: int = 5
        seed: int = 42
        lgbm_seeds: tuple[int, ...] = (42, 7, 2026)
        stacking_alpha: float = 1.0
        lgbm: LgbParams = field(default_factory=LgbParams)
        cat: CatParams = field(default_factory=CatParams)

    cfg = Config()
    EXP = "s10_final"
    BASE = "s9_ensemble"
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
    y_real, y_real_refit = fs.y_log - fs.offset, fs.y_log - fs.offset_refit
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    in_q1 = train_folded[va_idx].select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy() == 2025.0
    return env, fs, holdout_ids, in_q1, sample, test_raw, tr_idx, va_idx, y_real, y_real_refit


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
    ## 1. seed 平均（LightGBM）

    S5 の LightGBM と同じ設定で、seed だけを3通りに変えて学習します。LightGBM は行と列を毎回ランダムに一部だけ使う（bagging、feature fraction）ので、
    seed が変わると木の形が少しずつ変わります。1本ずつのスコアのぶれと、3本の予測を平均したスコアを比べます。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    cfg,
    env,
    fs,
    git_commit,
    holdout_ids,
    holdout_rmsle,
    lesson,
    lgb,
    np,
    sample,
    test_raw,
    tr_idx,
    va_idx,
    validate_submission,
    write_submission,
    pl,
    y_real,
    y_real_refit,
):
    commit = git_commit(env.root)
    seed_scores = {}
    seed_holdout, seed_test = [], []
    for _seed in cfg.lgbm_seeds:
        _params = asdict(cfg.lgbm) | {"seed": _seed}
        _booster, _pred_real = lesson.fit_lgbm_holdout(
            fs.X,
            y_real,
            tr_idx,
            va_idx,
            _params,
            max_rounds=cfg.max_rounds,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        _holdout_log = _pred_real + fs.offset[va_idx]
        seed_scores[_seed] = holdout_rmsle(_holdout_log, fs.y_log[va_idx])
        seed_holdout.append(_holdout_log)
        _rounds = lesson.refit_rounds(_booster.best_iteration, n_all=len(fs.y_log), n_train=len(tr_idx))
        _refit = lgb.train(_params, lgb.Dataset(fs.X_refit, label=y_real_refit), num_boost_round=_rounds)
        seed_test.append(_refit.predict(fs.X_test) + fs.offset_test)
    lgbm_avg_holdout = np.mean(seed_holdout, axis=0)
    lgbm_avg_test = np.mean(seed_test, axis=0)
    lgbm_avg_score = holdout_rmsle(lgbm_avg_holdout, fs.y_log[va_idx])
    _out = lesson.record_run(
        env,
        exp=EXP,
        base=BASE,
        run="lightgbm_seed_avg",
        score=lgbm_avg_score,
        metric=cfg.metric,
        config=asdict(cfg) | {"model": "lightgbm", "seed_scores": seed_scores},
        holdout=holdout_ids,
        pred=np.expm1(lgbm_avg_holdout),
        notes=f"LightGBM on the S5 feature set, averaged over seeds {list(cfg.lgbm_seeds)} (#25)",
        commit=commit,
    )
    _submission = pl.DataFrame({cfg.id_col: test_raw[cfg.id_col], "price": np.expm1(lgbm_avg_test)})
    validate_submission(_submission, sample, cfg.id_col)
    write_submission(_out, _submission)
    {"each seed": seed_scores, "average of the seeds": lgbm_avg_score}
    return commit, lgbm_avg_holdout, lgbm_avg_score, lgbm_avg_test, seed_holdout, seed_scores


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. CatBoost の本数の上限を上げる

    S6 と同じ設定のまま、本数の上限を 5,000 から 12,000 に上げます。early stopping が止めるところまで学習し、全期間での学習し直しも同じ規則で本数を決めます。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    cfg,
    commit,
    env,
    fs,
    holdout_ids,
    holdout_rmsle,
    lesson,
    np,
    sample,
    test_raw,
    tr_idx,
    va_idx,
    validate_submission,
    write_submission,
    pl,
    y_real,
    y_real_refit,
):
    cat_rounds, _cat_real = lesson.fit_catboost_holdout(
        fs.X,
        y_real,
        tr_idx,
        va_idx,
        asdict(cfg.cat),
        max_rounds=cfg.catboost_max_rounds,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    cat_holdout = _cat_real + fs.offset[va_idx]
    cat_score = holdout_rmsle(cat_holdout, fs.y_log[va_idx])
    cat_refit_rounds = lesson.refit_rounds(cat_rounds, n_all=len(fs.y_log), n_train=len(tr_idx))
    cat_test = (
        lesson.refit_catboost(fs.X_refit, y_real_refit, fs.X_test, asdict(cfg.cat), rounds=cat_refit_rounds)
        + fs.offset_test
    )
    _out = lesson.record_run(
        env,
        exp=EXP,
        base=BASE,
        run="catboost_long",
        score=cat_score,
        metric=cfg.metric,
        config=asdict(cfg) | {"model": "catboost", "best_rounds": cat_rounds, "refit_rounds": cat_refit_rounds},
        holdout=holdout_ids,
        pred=np.expm1(cat_holdout),
        notes=f"CatBoost with up to {cfg.catboost_max_rounds} rounds (#25)",
        commit=commit,
    )
    _submission = pl.DataFrame({cfg.id_col: test_raw[cfg.id_col], "price": np.expm1(cat_test)})
    validate_submission(_submission, sample, cfg.id_col)
    write_submission(_out, _submission)
    {"rounds": cat_rounds, "holdout_rmsle": cat_score, "refit_rounds": cat_refit_rounds}
    return cat_holdout, cat_rounds, cat_score, cat_test


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 最終のアンサンブル

    S9 と同じ方法（単純平均・重み付き平均・スタッキング、2分割の交差検証で採点）で組み合わせ直します。
    メンバーは、この段階の LightGBM（seed 平均）と CatBoost（本数の上限を上げたもの）、S6 の XGBoost、S7 の Ridge・RandomForest・MLP です。
    """)
    return


@app.cell
def _(cat_holdout, cat_test, env, fs, holdout_ids, lgbm_avg_holdout, lgbm_avg_test, np, pl, sample, va_idx):
    def saved(stage, run):
        oof = pl.read_parquet(env.output_dir / stage / run / "oof.parquet")
        sub = pl.read_csv(env.output_dir / stage / run / "submission.csv")
        assert oof["id"].to_list() == holdout_ids["id"].to_list() and sub["id"].to_list() == sample["id"].to_list()
        return np.log1p(oof["pred"].to_numpy()), np.log1p(sub["price"].to_numpy())

    final_members = {
        "lightgbm_seed_avg": (lgbm_avg_holdout, lgbm_avg_test),
        "catboost_long": (cat_holdout, cat_test),
        "xgboost": saved("s6_gbdt", "xgboost"),
        "ridge": saved("s7_mlp_classic", "ridge"),
        "random_forest": saved("s7_mlp_classic", "random_forest"),
        "mlp": saved("s7_mlp_classic", "mlp"),
    }
    member_names = list(final_members)
    H_final = np.column_stack([final_members[n][0] for n in member_names])
    T_final = np.column_stack([final_members[n][1] for n in member_names])
    y_holdout = fs.y_log[va_idx]
    return H_final, T_final, member_names, y_holdout


@app.cell
def _(H_final, bootstrap_diff, cfg, ensemble, holdout_rmsle, in_q1, member_names, np, pl, y_holdout):
    member_scores = {n: holdout_rmsle(H_final[:, i], y_holdout) for i, n in enumerate(member_names)}
    best_member = min(member_scores, key=member_scores.get)
    final_candidates = {
        f"best single ({best_member})": H_final[:, member_names.index(best_member)],
        "simple average": H_final.mean(axis=1),
        "weighted average": ensemble.cross_predict(
            H_final, y_holdout, in_q1, ensemble.fit_weights, lambda w, Hp: Hp @ w
        ),
        "stacking (ridge)": ensemble.cross_predict(
            H_final,
            y_holdout,
            in_q1,
            lambda Hf, yf: ensemble.fit_stacking(Hf, yf, alpha=cfg.stacking_alpha),
            lambda m, Hp: m.predict(Hp),
        ),
    }
    base_sq = (final_candidates[f"best single ({best_member})"] - y_holdout) ** 2
    final_comparison = pl.DataFrame(
        [
            {"method": name, "holdout_rmsle": holdout_rmsle(pred, y_holdout)}
            | dict(
                zip(
                    ("change_vs_best_single", "ci_low", "ci_high"),
                    bootstrap_diff((pred - y_holdout) ** 2, base_sq),
                    strict=True,
                )
            )
            for name, pred in final_candidates.items()
        ]
    )
    final_comparison
    return best_member, final_candidates, final_comparison, member_scores


@app.cell
def _(
    BASE,
    EXP,
    H_final,
    T_final,
    asdict,
    cfg,
    commit,
    ensemble,
    env,
    final_candidates,
    holdout_ids,
    holdout_rmsle,
    lesson,
    member_names,
    np,
    pl,
    sample,
    validate_submission,
    write_submission,
    y_holdout,
):
    weights = ensemble.fit_weights(H_final, y_holdout)
    stack = ensemble.fit_stacking(H_final, y_holdout, alpha=cfg.stacking_alpha)
    final_tests = {
        "simple_average": (final_candidates["simple average"], T_final.mean(axis=1)),
        "weighted_average": (final_candidates["weighted average"], T_final @ weights),
        "stacking": (final_candidates["stacking (ridge)"], stack.predict(T_final)),
    }
    for _method, (_holdout_log, _test_log) in final_tests.items():
        _out = lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=_method,
            score=holdout_rmsle(_holdout_log, y_holdout),
            metric=cfg.metric,
            config=asdict(cfg) | {"method": _method, "members": member_names},
            holdout=holdout_ids,
            pred=np.expm1(_holdout_log),
            notes=f"final {_method} (#25)",
            commit=commit,
        )
        _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(_test_log)})
        validate_submission(_submission, sample, cfg.id_col)
        write_submission(_out, _submission)
    final_weights = pl.DataFrame(
        {"member": member_names, "weight": np.round(weights, 3), "stacking_coefficient": np.round(stack.coef_, 3)}
    )
    final_weights
    return (final_weights,)


if __name__ == "__main__":
    app.run()
