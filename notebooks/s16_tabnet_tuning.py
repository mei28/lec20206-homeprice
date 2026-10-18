import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    # cuBLAS reads this when CUDA starts; with deterministic algorithms, a seed gives the same TabNet every time.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from dataclasses import asdict, dataclass, field

    import marimo as mo
    import numpy as np
    import optuna
    import polars as pl
    import psutil
    import torch

    from kgl import cv, ensemble, feature_set, lesson, mlit, stacking, tabnet
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, validate_submission, write_submission
    from kgl.metrics import get_metric

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
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
        mlit,
        mo,
        np,
        optuna,
        pl,
        psutil,
        stacking,
        tabnet,
        torch,
        validate_submission,
        write_submission,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S16. TabNet の設定を探す

    S13 で入れた TabNet は、最初に試した設定のまま holdout で 0.2201 でした。S14 で XGBoost の設定を探したら 0.2103 から 0.1996 まで良くなったので、
    同じことを TabNet でも試します（#44）。探し方は S14 と同じで、2024Q3〜Q4 で設定を選び、2025Q1〜Q2 の holdout は確かめに1回だけ使います。

    TabNet には、GBDT にない難しさがあります。同じ設定でも、学習のたびにスコアが変わることです。
    S13 では、同じ設定で 0.2172、0.2220、0.2201 が出ました。設定の違いによる差がこのぶれより小さいと、探索は「良い設定」ではなく「運の良かった1回」を選んでしまいます。
    このぶれには2つの原因があります。1つは GPU の計算の順番が毎回変わること、もう1つは乱数の種（seed）の違いです。
    1つ目は、PyTorch の「決定的なモード」で消せます。計算の順番を固定するので少し遅くなります（1回あたり約2〜3割）が、
    同じ seed なら毎回同じ結果になります（試したところ、同じ seed の2回が 0.21368 で完全に一致しました）。この段階は、このモードで動かします。
    2つ目は残るので、まず seed を変えたときのぶれの大きさを測ります。
    """)
    return


@app.cell
def _(dataclass, field, psutil):
    THREADS = psutil.cpu_count(logical=False)

    @dataclass
    class TabNetParams:
        # S13's TabNet
        n_d: int = 16
        n_a: int = 16
        n_steps: int = 4
        gamma: float = 1.3
        lambda_sparse: float = 1e-4
        cat_emb_dim: int = 4
        lr: float = 2e-2
        lr_step_size: int = 10
        lr_gamma: float = 0.9
        batch_size: int = 4096
        virtual_batch_size: int = 512
        max_epochs: int = 200
        patience: int = 20
        seed: int = 0

    @dataclass
    class StackerLgbParams:
        # S12-S15's LightGBM stacker
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
        noise_seeds: tuple[int, ...] = (0, 1, 2)
        trial_seeds: tuple[int, ...] = (0,)
        # One fit takes about 10-12 minutes in deterministic mode (2026-10-17), so 40 trials run overnight.
        trials: int = 40
        final_seeds: tuple[int, ...] = (0, 1, 2)
        tabnet: TabNetParams = field(default_factory=TabNetParams)
        stacking_alpha: float = 1.0
        stack_mlp_hidden: tuple[int, ...] = (32, 16)
        stack_mlp_alpha: float = 1e-3
        stack_mlp_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
        stack_lgbm_max_rounds: int = 2000
        stack_lgbm_early_stopping_rounds: int = 100
        inner_valid_share: float = 0.2
        stack_lgbm: StackerLgbParams = field(default_factory=StackerLgbParams)

    cfg = Config()
    EXP = "s16_tabnet_tuning"
    BASE = "s15_more_stack_rows"
    return BASE, EXP, cfg


@app.cell
def _(cfg, comp_dir, cv, detect, feature_set, mlit, pl, torch):
    assert torch.cuda.is_available()
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

    # Holdout setting (S13-S15): train up to 2024Q4, validate on 2025Q1-Q2 (fold 1).
    tr_idx, va_idx = cv.split(train_folded["fold"].to_numpy(), 1, time=True)
    fs = feature_set.build(train_folded, test_raw, tr_idx, va_idx, **_build)
    y_real, y_real_refit = fs.y_log - fs.offset, fs.y_log - fs.offset_refit
    y_holdout = fs.y_log[va_idx]
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    in_q1 = train_folded[va_idx].select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy() == 2025.0

    # Inner setting for the search (S14): rows up to 2024Q4; train up to 2024Q2, score 2024Q3-Q4.
    inner_raw = train_folded.filter(pl.col("fold") <= 0)
    tr_in, va_in = cv.split(inner_raw["fold"].to_numpy(), 0, time=True)
    fs_in = feature_set.build(inner_raw, test_raw, tr_in, va_in, **_build)
    y_in = fs_in.y_log - fs_in.offset
    return (
        env,
        fs,
        fs_in,
        holdout_ids,
        in_q1,
        sample,
        tr_idx,
        tr_in,
        va_idx,
        va_in,
        y_holdout,
        y_in,
        y_real,
        y_real_refit,
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
def _(fs_in, holdout_rmsle, np, tabnet, tr_in, va_in, y_in):
    def inner_fit(params, seeds):
        """Train on 2019Q1-2024Q2 once per seed and score the seeds' average prediction on 2024Q3-Q4."""
        runs = [tabnet.fit_holdout(fs_in.X, y_in, tr_in, va_in, params | {"seed": s}, device="cuda") for s in seeds]
        pred = np.mean([p for _, p in runs], axis=0) + fs_in.offset[va_in]
        each = [holdout_rmsle(p + fs_in.offset[va_in], fs_in.y_log[va_in]) for _, p in runs]
        return holdout_rmsle(pred, fs_in.y_log[va_in]), each, [e for e, _ in runs]

    return (inner_fit,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. ぶれの大きさ

    S13 の設定のまま、乱数の種（seed）だけを変えて、2024Q2 までで学習し 2024Q3〜Q4 で採点します。
    """)
    return


@app.cell
def _(asdict, cfg, inner_fit, np, pl):
    _avg, noise_each, _epochs = inner_fit(asdict(cfg.tabnet), cfg.noise_seeds)
    noise_spread = float(np.max(noise_each) - np.min(noise_each))
    pl.DataFrame({"seed": list(cfg.noise_seeds), "2024Q3-Q4 RMSLE": noise_each, "epochs": _epochs})
    return noise_each, noise_spread


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 探す

    探す範囲は、TabNet の大きさ（`n_d` = `n_a`、ステップ数 `n_steps`）、特徴の選び方の強さ（`gamma`、`lambda_sparse`）、学習率、バッチの大きさ、
    カテゴリの埋め込みの次元（`cat_emb_dim`）です。1回の学習に10分以上かかるので、時間がかかりすぎる設定（小さいバッチ、多いステップ）は範囲から外し、
    1回の試行は1回の学習（seed 0）にしました。試行の結果は S14 と同じく SQLite に保存し、止まっても続きから再開します。
    """)
    return


@app.cell
def _(EXP, asdict, cfg, env, inner_fit, optuna):
    def space(trial):
        width = trial.suggest_categorical("n_d", [8, 16, 24, 32, 48])
        return {
            "n_d": width,
            "n_a": width,
            "n_steps": trial.suggest_int("n_steps", 3, 7),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-2, log=True),
            "lr": trial.suggest_float("lr", 5e-3, 5e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [2048, 4096, 8192]),
            "virtual_batch_size": trial.suggest_categorical("virtual_batch_size", [128, 256, 512, 1024]),
            "cat_emb_dim": trial.suggest_int("cat_emb_dim", 1, 8),
        }

    def objective(trial):
        searched = space(trial)
        score, each, epochs = inner_fit(asdict(cfg.tabnet) | searched, cfg.trial_seeds)
        trial.set_user_attr("params", searched)
        trial.set_user_attr("each_seed", each)
        trial.set_user_attr("epochs", epochs)
        return score

    (env.output_dir / EXP).mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="tabnet",
        storage=f"sqlite:///{env.output_dir / EXP / 'optuna.db'}",
        direction="minimize",
        # Trials run one at a time; without the constant liar a trial left RUNNING by an interruption does not steer TPE.
        sampler=optuna.samplers.TPESampler(seed=cfg.seed, constant_liar=False),
        load_if_exists=True,
    )
    _done = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    study.optimize(objective, n_trials=max(0, cfg.trials - _done))
    return (study,)


@app.cell
def _(noise_each, np, optuna, pl, study):
    _values = [t.value for t in study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))]
    search_summary = pl.DataFrame(
        [
            {
                "trials": len(_values),
                "S13 settings (mean of the seeds)": float(np.mean(noise_each)),
                "best after 10": float(np.min(_values[:10])),
                "best": study.best_value,
                "best params": str(study.best_trial.user_attrs["params"]),
            }
        ]
    )
    search_summary
    return (search_summary,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. holdout で確かめる

    選んだ設定と S13 の設定を、どちらも同じ3つの seed で学習して予測を平均し、2025Q1〜Q2 で採点します。
    seed を平均するだけでも予測は安定して良くなるので、S13 の TabNet（1回だけの学習）と比べるだけでは、探索の効果と平均の効果が混ざります。
    「S13 の設定の平均」と比べた差が、探索の効果です。提出用には、同じ規則で全期間で学習し直します。
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
    np,
    pl,
    sample,
    study,
    tabnet,
    tr_idx,
    va_idx,
    validate_submission,
    write_submission,
    y_holdout,
    y_real,
    y_real_refit,
):
    commit = git_commit(env.root)

    def seed_average(params):
        holdout, test = [], []
        for seed in cfg.final_seeds:
            p = params | {"seed": seed}
            epochs, pred = tabnet.fit_holdout(fs.X, y_real, tr_idx, va_idx, p, device="cuda")
            holdout.append(pred)
            test.append(tabnet.refit(fs.X_refit, y_real_refit, fs.X_test, p, epochs=epochs, device="cuda"))
        return np.mean(holdout, axis=0) + fs.offset[va_idx], np.mean(test, axis=0) + fs.offset_test

    tabnet_runs = {
        "tabnet_s13_settings_seed_avg": (asdict(cfg.tabnet), "S13 TabNet settings, averaged over seeds"),
        "tabnet_tuned_seed_avg": (
            asdict(cfg.tabnet) | study.best_trial.user_attrs["params"],
            "TabNet with the settings Optuna chose on 2024Q3-Q4, averaged over seeds",
        ),
    }
    tabnet_members = {}
    for _run, (_params, _note) in tabnet_runs.items():
        tabnet_members[_run] = seed_average(_params)
        _out = lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=_run,
            score=holdout_rmsle(tabnet_members[_run][0], y_holdout),
            metric=cfg.metric,
            config=asdict(cfg) | {"model": "tabnet", "params": _params},
            holdout=holdout_ids,
            pred=np.expm1(tabnet_members[_run][0]),
            notes=f"{_note} {list(cfg.final_seeds)} (#44)",
            commit=commit,
        )
        _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(tabnet_members[_run][1])})
        validate_submission(_submission, sample, cfg.id_col)
        write_submission(_out, _submission)
    return commit, tabnet_members


@app.cell
def _(bootstrap_diff, env, holdout_ids, holdout_rmsle, np, pl, sample, tabnet_members, y_holdout):
    def saved(stage, run):
        """Holdout and test predictions of a saved run, on the log scale."""
        oof = pl.read_parquet(env.output_dir / stage / run / "oof.parquet")
        sub = pl.read_csv(env.output_dir / stage / run / "submission.csv")
        assert oof["id"].to_list() == holdout_ids["id"].to_list() and sub["id"].to_list() == sample["id"].to_list()
        return np.log1p(oof["pred"].to_numpy()), np.log1p(sub["price"].to_numpy())

    s13_tabnet = saved("s13_tabular_dl", "tabnet")
    _tuned_sq = (tabnet_members["tabnet_tuned_seed_avg"][0] - y_holdout) ** 2
    _rows = [{"run": "s13 tabnet (one fit)", "holdout_rmsle": holdout_rmsle(s13_tabnet[0], y_holdout)}]
    for _run, (_h, _) in tabnet_members.items():
        _rows.append({"run": _run, "holdout_rmsle": holdout_rmsle(_h, y_holdout)})
    tuning_change = bootstrap_diff(_tuned_sq, (tabnet_members["tabnet_s13_settings_seed_avg"][0] - y_holdout) ** 2)
    _rows[-1] |= {"change_vs_s13_settings": tuning_change[0], "ci_low": tuning_change[1], "ci_high": tuning_change[2]}
    tabnet_comparison = pl.DataFrame(_rows, infer_schema_length=None)
    tabnet_comparison
    return saved, tabnet_comparison, tuning_change


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. スタッキングに足す

    S14 のメンバー（探索した LightGBM・CatBoost・XGBoost・MLP、S7 の RandomForest と Ridge、S13 の TabPFN）に、
    探索した TabNet を足して、S14 と同じ手順でスタッキングし直します。まず S14 の `stacker_average_tuned` が再現できることを確かめます。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    bootstrap_diff,
    cfg,
    commit,
    ensemble,
    env,
    fs,
    holdout_ids,
    holdout_rmsle,
    in_q1,
    lesson,
    np,
    pl,
    sample,
    saved,
    stacking,
    tabnet_members,
    va_idx,
    validate_submission,
    write_submission,
    y_holdout,
):
    s14_members = {
        "lightgbm": saved(cfg.tuned_stage, "lightgbm_tuned"),
        "catboost": saved(cfg.tuned_stage, "catboost_tuned"),
        "xgboost": saved(cfg.tuned_stage, "xgboost_tuned"),
        "ridge": saved("s7_mlp_classic", "ridge"),
        "random_forest": saved("s7_mlp_classic", "random_forest"),
        "mlp": saved(cfg.tuned_stage, "mlp_tuned"),
        "tabpfn": saved("s13_tabular_dl", "tabpfn"),
    }
    s14_average = saved(cfg.tuned_stage, "stacker_average_tuned")
    _rows_h, _rows_t = stacking.row_features(fs.X.iloc[va_idx]), stacking.row_features(fs.X_test)

    def stacker_average(members):
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
        return stacking.stacker_average(H, T, _rows_h, _rows_t, y_holdout, in_q1, found)

    _reproduced = stacker_average(s14_members)
    assert np.allclose(_reproduced[0], s14_average[0], atol=1e-9)
    assert np.allclose(_reproduced[1], s14_average[1], atol=1e-9)

    members_with_tabnet = s14_members | {"tabnet": tabnet_members["tabnet_tuned_seed_avg"]}
    with_tabnet = stacker_average(members_with_tabnet)
    restack_change = bootstrap_diff((with_tabnet[0] - y_holdout) ** 2, (s14_average[0] - y_holdout) ** 2)
    restack_score = holdout_rmsle(with_tabnet[0], y_holdout)
    _out = lesson.record_run(
        env,
        exp=EXP,
        base=BASE,
        run="stacker_average_tuned_with_tabnet",
        score=restack_score,
        metric=cfg.metric,
        config=asdict(cfg) | {"members": list(members_with_tabnet)},
        holdout=holdout_ids,
        pred=np.expm1(with_tabnet[0]),
        notes="S14 stacker average with the tuned TabNet added (#44)",
        commit=commit,
    )
    _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(with_tabnet[1])})
    validate_submission(_submission, sample, cfg.id_col)
    write_submission(_out, _submission)
    _weights = ensemble.fit_weights(np.column_stack([m[0] for m in members_with_tabnet.values()]), y_holdout)
    tabnet_weight = float(_weights[list(members_with_tabnet).index("tabnet")])
    pl.DataFrame(
        [
            {"run": "s14 stacker_average_tuned", "holdout_rmsle": holdout_rmsle(s14_average[0], y_holdout)},
            {
                "run": "stacker_average_tuned_with_tabnet",
                "holdout_rmsle": restack_score,
                "change_vs_s14": restack_change[0],
                "ci_low": restack_change[1],
                "ci_high": restack_change[2],
                "tabnet weight": tabnet_weight,
            },
        ],
        infer_schema_length=None,
    )
    return restack_change, restack_score, tabnet_weight


@app.cell
def _(
    mo,
    noise_each,
    noise_spread,
    pl,
    restack_change,
    restack_score,
    search_summary,
    tabnet_comparison,
    tabnet_weight,
    tuning_change,
):
    def score(run):
        return tabnet_comparison.filter(pl.col("run") == run)["holdout_rmsle"][0]

    _s = search_summary.row(0, named=True)
    mo.md(f"""
    ## 5. 結果

    - ぶれ: S13 の設定のまま seed だけ変えると、2024Q3〜Q4 で {", ".join(f"{v:.4f}" for v in noise_each)}（幅 {noise_spread:.4f}）でした。
    - 探索（2024Q3〜Q4）: S13 の設定 {_s["S13 settings (mean of the seeds)"]:.4f} → 最良 {_s["best"]:.4f}（{_s["trials"]} 回）
    - holdout（2025Q1〜Q2）: S13 の TabNet（1回）{score("s13 tabnet (one fit)"):.4f}、S13 の設定の平均 {score("tabnet_s13_settings_seed_avg"):.4f}、
      探索した設定の平均 {score("tabnet_tuned_seed_avg"):.4f}（S13 の設定の平均との差 {tuning_change[0]:+.4f}、95% の範囲 {tuning_change[1]:+.4f}〜{tuning_change[2]:+.4f}）
    - スタッキングの平均: {restack_score:.4f}（S14 との差 {restack_change[0]:+.4f}、95% の範囲 {restack_change[1]:+.4f}〜{restack_change[2]:+.4f}）。
      TabNet の重み（非負の重み付き平均）は {tabnet_weight:.3f} でした。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. 結果を読む

    2026-10-18 の実行の結果です。

    - 先にぶれを測りました。S13 の設定のまま seed だけ変えると、2024Q3〜Q4 のスコアが 0.0024 の幅で動きます。これより小さい差は、設定の違いとは言えません。
    - 探索の最良（0.2069）は S13 の設定（0.2137）より、この幅を大きく超えて良くなりました。holdout でも seed 3つの平均どうしで -0.0023 で、
      95% の範囲は 0 をまたぎません。S14 の CatBoost や LightGBM と違い、改善が holdout まで残りました。選ばれたのは S13 より小さいネットワークです。
    - それでも、スタッキングの平均は S14 と変わりません（+0.00008）。TabNet が新しく当てられるようになった部分は、TabPFN・CatBoost・LightGBM がすでに当てていました。
    - S14 と同じ結論です。採用しません。
    """)
    return


if __name__ == "__main__":
    app.run()
