import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from dataclasses import asdict, dataclass, field

    import lightgbm as lgb
    import marimo as mo
    import numpy as np
    import optuna
    import polars as pl
    import psutil
    from threadpoolctl import threadpool_limits

    from kgl import classic, cv, ensemble, feature_set, lesson, mlit, stacking
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, validate_submission, write_submission
    from kgl.metrics import get_metric

    return (
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
        lesson,
        lgb,
        mlit,
        mo,
        np,
        optuna,
        pl,
        psutil,
        stacking,
        threadpool_limits,
        validate_submission,
        write_submission,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S14. ハイパーパラメータ探索（Optuna）

    ここまでのメンバーの設定（木の深さ、葉の数、正則化の強さなど）は、最初に手で決めたものをほとんど変えずに使ってきました。
    この段階では、CatBoost・LightGBM・MLP・XGBoost の設定を Optuna で探し直し、探した設定のメンバーで S13 と同じスタッキングをやり直します（#39〜#42）。

    探索では何百回も設定を試して、一番良かったものを選びます。このとき、選ぶのに使ったデータでのスコアは、実力より良く見えます。
    たまたまそのデータに合った設定も「良い設定」として選ばれるからです。
    そこで、2025Q1〜Q2 の holdout は探索に使わず、その手前の 2024Q3〜Q4 で探します（#38）。holdout は、選んだ設定を確かめるのに1回だけ使います。

    この段階でわかること:

    - 探索で、各モデルが単独でどれだけ良くなるか
    - 探索に使った期間（2024Q3〜Q4）での改善が、holdout（2025Q1〜Q2）でもそのまま出るか
    - メンバーが良くなったとき、スタッキングの平均（S13 で 0.1910）がどれだけ良くなるか
    """)
    return


@app.cell
def _(dataclass, field, psutil):
    THREADS = psutil.cpu_count(logical=False)

    @dataclass
    class LgbParams:
        # S5 and S10's LightGBM
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
        # S6 and S10's CatBoost
        learning_rate: float = 0.08
        depth: int = 8
        loss_function: str = "RMSE"
        random_seed: int = 42
        thread_count: int = THREADS

    @dataclass
    class XgbParams:
        # S6's XGBoost, on the GPU in this stage (the search and the final model alike)
        objective: str = "reg:squarederror"
        eta: float = 0.05
        max_depth: int = 8
        min_child_weight: float = 20.0
        subsample: float = 0.8
        colsample_bytree: float = 0.8
        tree_method: str = "hist"
        max_cat_to_onehot: int = 1
        seed: int = 42
        nthread: int = THREADS
        device: str = "cuda"

    @dataclass
    class MlpParams:
        # S7's MLP
        hidden_layer_sizes: tuple[int, ...] = (256, 128)
        alpha: float = 1e-4
        batch_size: int = 512
        learning_rate_init: float = 1e-3
        max_iter: int = 60
        early_stopping: bool = True
        n_iter_no_change: int = 5
        random_state: int = 42

    @dataclass
    class SearchRates:
        # A higher learning rate during the search makes each trial shorter; the chosen setting is trained at the members' rate.
        lightgbm: float = 0.1
        catboost: float = 0.15
        xgboost: float = 0.1

    @dataclass
    class Trials:
        # Set from one timed trial each (2026-10-17): LightGBM 15 s, XGBoost 7 s on the GPU, MLP 50 s, CatBoost 280 s.
        lightgbm: int = 300
        xgboost: int = 400
        mlp: int = 150
        catboost: int = 60

    @dataclass
    class StackerLgbParams:
        # S12 and S13's LightGBM stacker
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
        max_rounds: int = 10000
        catboost_max_rounds: int = 12000
        early_stopping_rounds: int = 100
        lgbm_seeds: tuple[int, ...] = (42, 7, 2026)
        lgbm: LgbParams = field(default_factory=LgbParams)
        cat: CatParams = field(default_factory=CatParams)
        xgb: XgbParams = field(default_factory=XgbParams)
        mlp: MlpParams = field(default_factory=MlpParams)
        search_rate: SearchRates = field(default_factory=SearchRates)
        trials: Trials = field(default_factory=Trials)
        stacking_alpha: float = 1.0
        stack_mlp_hidden: tuple[int, ...] = (32, 16)
        stack_mlp_alpha: float = 1e-3
        stack_mlp_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
        stack_lgbm_max_rounds: int = 2000
        stack_lgbm_early_stopping_rounds: int = 100
        inner_valid_share: float = 0.2
        stack_lgbm: StackerLgbParams = field(default_factory=StackerLgbParams)

    cfg = Config()
    EXP = "s14_tuning"
    BASE = "s13_tabular_dl"
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

    # Holdout setting, as in S13: train up to 2024Q4 (folds -1 and 0), validate on 2025Q1-Q2 (fold 1).
    tr_idx, va_idx = cv.split(train_folded["fold"].to_numpy(), 1, time=True)
    fs = feature_set.build(train_folded, test_raw, tr_idx, va_idx, **_build)
    y_real, y_real_refit = fs.y_log - fs.offset, fs.y_log - fs.offset_refit
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    in_q1 = train_folded[va_idx].select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy() == 2025.0

    # Inner setting for the search: only the rows up to 2024Q4; train up to 2024Q2 (fold -1), score 2024Q3-Q4 (fold 0).
    inner_raw = train_folded.filter(pl.col("fold") <= 0)
    tr_in, va_in = cv.split(inner_raw["fold"].to_numpy(), 0, time=True)
    fs_in = feature_set.build(inner_raw, test_raw, tr_in, va_in, **_build)
    y_in = fs_in.y_log - fs_in.offset

    # The MLP's inputs (S7)
    extras_train = classic.numeric_extras(train_folded)
    mlp_frame = classic.linear_frame(fs.X, extras_train)
    mlp_frame_refit = classic.linear_frame(fs.X_refit, extras_train)
    mlp_frame_test = classic.linear_frame(fs.X_test, classic.numeric_extras(test_raw))
    mlp_frame_in = classic.linear_frame(fs_in.X, classic.numeric_extras(inner_raw))
    return (
        env,
        fs,
        fs_in,
        holdout_ids,
        in_q1,
        mlp_frame,
        mlp_frame_in,
        mlp_frame_refit,
        mlp_frame_test,
        sample,
        test_raw,
        tr_idx,
        tr_in,
        train_folded,
        va_idx,
        va_in,
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
def _(mo):
    mo.md(r"""
    ## 1. 探索用の分け方

    fold ファイル `time_2024q3_2025q1` は、学習データを時期で3つに分けます。

    - fold -1（2019Q1〜2024Q2）: 探索で学習に使う
    - fold 0（2024Q3〜Q4）: 探索で各試行を採点する
    - fold 1（2025Q1〜Q2）: これまでと同じ holdout。選んだ設定を1回だけ確かめる

    探索の評価期間は、holdout と同じく「学習期間の直後の2四半期」です。価格指数を1〜2四半期先へ外挿する部分も、holdout と同じように試されます。
    価格指数と Target encoding は fold -1 だけから作ります。fold 1 の行は探索では一切使いません。
    """)
    return


@app.cell
def _(pl, train_folded):
    train_folded.group_by("fold").agg(
        pl.len().alias("rows"),
        pl.col("取引時期").min().alias("from"),
        pl.col("取引時期").max().alias("to"),
        (pl.col("価格情報区分") == "成約価格情報").mean().round(3).alias("contract share"),
    ).sort("fold")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 何を探すか

    Optuna は、前の試行の結果を見て次に試す設定を選びます（標準の TPE という方法）。良かった試行の設定が集まっている辺りを重点的に試すので、
    やみくもに試すより少ない回数で良い設定にたどり着きます。乱数の種を固定しているので、同じ順に試します。

    探す範囲は、各モデルの「木の形」と「正則化」が中心です。

    - LightGBM: 葉の数（`num_leaves`）、葉に入る最小の行数、列と行の抜き取りの割合、L1/L2 正則化、カテゴリの扱い（`cat_smooth`、`cat_l2`）
    - CatBoost: 木の深さ（`depth`）、L2 正則化、分割に混ぜる乱数の強さ、行の抜き取りの割合、数値の区切りの細かさ、one-hot にするカテゴリの大きさ
    - XGBoost: 木の深さ、葉の重みの下限、列と行の抜き取りの割合、L1/L2 正則化、分割の最小の改善幅、カテゴリの扱い
    - MLP: 層の数（1〜3）と各層のユニット数、L2 正則化（`alpha`）、学習率、バッチの大きさ

    学習率は探さず、探索中は高め（LightGBM と XGBoost は 0.1、CatBoost は 0.15）に固定します。学習率を上げると、少ない本数で学習が終わり、1回の試行が短くなります。
    選んだ設定は、これまでのメンバーと同じ学習率（0.05、0.08、0.05）で学習し直します。木の形や正則化の良し悪しは、学習率を変えてもおおむね保たれます。

    比べる基準として、いまの設定のままの各モデルも、同じ条件（探索用の分け方と学習率）で1回ずつ採点します。
    """)
    return


@app.cell
def _(
    THREADS,
    asdict,
    cfg,
    classic,
    fs_in,
    holdout_rmsle,
    lesson,
    mlp_frame_in,
    threadpool_limits,
    tr_in,
    va_in,
    y_in,
):
    def inner_score(pred_real):
        return holdout_rmsle(pred_real + fs_in.offset[va_in], fs_in.y_log[va_in])

    def lgbm_space(trial):
        return {
            "num_leaves": trial.suggest_int("num_leaves", 15, 511, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 500, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
            "cat_l2": trial.suggest_float("cat_l2", 1.0, 100.0, log=True),
        }

    def cat_space(trial):
        return {
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 16),
        }

    def xgb_space(trial):
        return {
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 200.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 30.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 1, 32, log=True),
            "max_cat_threshold": trial.suggest_int("max_cat_threshold", 8, 256, log=True),
        }

    def mlp_space(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        return {
            "hidden_layer_sizes": [trial.suggest_int(f"units_{i}", 32, 512, log=True) for i in range(n_layers)],
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
        }

    def fit_lgbm_inner(params):
        booster, pred = lesson.fit_lgbm_holdout(
            fs_in.X,
            y_in,
            tr_in,
            va_in,
            params,
            max_rounds=cfg.max_rounds,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        return inner_score(pred), booster.best_iteration

    def fit_cat_inner(params):
        rounds, pred = lesson.fit_catboost_holdout(
            fs_in.X,
            y_in,
            tr_in,
            va_in,
            params,
            max_rounds=cfg.catboost_max_rounds,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        return inner_score(pred), rounds

    def fit_xgb_inner(params):
        rounds, pred = lesson.fit_xgb_holdout(
            fs_in.X,
            y_in,
            tr_in,
            va_in,
            params,
            max_rounds=cfg.max_rounds,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        return inner_score(pred), rounds

    def fit_mlp_inner(params):
        with threadpool_limits(THREADS):
            model = classic.make_mlp(mlp_frame_in, params).fit(mlp_frame_in.iloc[tr_in], y_in[tr_in])
            pred = model.predict(mlp_frame_in.iloc[va_in])
        return inner_score(pred), model.regressor_[-1].n_iter_

    # name: (search space, the current settings at the search learning rate, fit on fold -1 and score fold 0)
    SEARCH = {
        "lightgbm": (lgbm_space, asdict(cfg.lgbm) | {"learning_rate": cfg.search_rate.lightgbm}, fit_lgbm_inner),
        "xgboost": (xgb_space, asdict(cfg.xgb) | {"eta": cfg.search_rate.xgboost}, fit_xgb_inner),
        "mlp": (mlp_space, asdict(cfg.mlp), fit_mlp_inner),
        "catboost": (cat_space, asdict(cfg.cat) | {"learning_rate": cfg.search_rate.catboost}, fit_cat_inner),
    }
    return (SEARCH,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 探索する

    試行の結果は `output/s14_tuning/optuna.db`（SQLite）に1回ずつ保存します。途中で止まっても、もう一度実行すれば続きから再開し、決めた回数に達したモデルは探索を飛ばします。
    回数は、各モデルの1回の時間を測ってから決めました（LightGBM 約15秒、XGBoost 約7秒、MLP 約50秒、CatBoost 約280秒）。
    XGBoost だけは GPU で学習します（CPU では約40秒）。GPU と CPU では計算の細部が違い、同じ設定でもスコアが少し変わるので、探索も最後の学習も GPU にそろえます。
    """)
    return


@app.cell
def _(EXP, SEARCH, cfg, env, optuna):
    (env.output_dir / EXP).mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{env.output_dir / EXP / 'optuna.db'}"

    def objective(name):
        space, base, fit = SEARCH[name]

        def run(trial):
            searched = space(trial)
            score, rounds = fit(base | searched)
            trial.set_user_attr("params", searched)
            trial.set_user_attr("rounds", int(rounds))
            return score

        return run

    studies = {}
    for _name in SEARCH:
        _study = optuna.create_study(
            study_name=_name,
            storage=storage,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=cfg.seed),
            load_if_exists=True,
        )
        _done = len(_study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
        _study.optimize(objective(_name), n_trials=max(0, getattr(cfg.trials, _name) - _done))
        studies[_name] = _study
    return (studies,)


@app.cell
def _(SEARCH, np, optuna, pl, studies):
    baseline_inner = {_name: fit(base)[0] for _name, (_, base, fit) in SEARCH.items()}

    def best_after(study, n):
        values = [t.value for t in study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))]
        return float(np.min(values[:n]))

    search_summary = pl.DataFrame(
        [
            {
                "model": _name,
                "trials": len(_study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))),
                "current settings": baseline_inner[_name],
                "best after 10": best_after(_study, 10),
                "best after 50": best_after(_study, 50),
                "best": _study.best_value,
                "change": _study.best_value - baseline_inner[_name],
            }
            for _name, _study in studies.items()
        ]
    )
    search_summary
    return baseline_inner, search_summary


@app.cell
def _(mo):
    mo.md(r"""
    表は探索用の評価期間（2024Q3〜Q4）での RMSLE です。`current settings` がいまの設定、`best after 10` と `best after 50` は最初の10回・50回までの最良、`best` が全体の最良です。
    回数を増やしたときに、最良がどれだけ下がり続けるかを見ます。

    次の表は、それぞれのモデルで選ばれた設定と、スコアへの影響が大きかった設定の上位3つです（Optuna の fANOVA による重要度）。
    """)
    return


@app.cell
def _(optuna, pl, studies):
    best_settings = pl.DataFrame(
        [
            {
                "model": _name,
                "best params": str(_study.best_trial.user_attrs["params"]),
                "rounds": _study.best_trial.user_attrs["rounds"],
                "most important": ", ".join(
                    f"{k} {v:.2f}"
                    for k, v in list(
                        optuna.importance.get_param_importances(
                            _study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0)
                        ).items()
                    )[:3]
                ),
            }
            for _name, _study in studies.items()
        ]
    )
    best_settings
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. holdout で確かめる

    選んだ設定を、これまでのメンバーと同じ学習率に戻して、S13 までと同じ条件（2024Q4 まで学習、2025Q1〜Q2 で採点）で1回だけ学習します。
    LightGBM は S10 と同じく seed を3つ変えて平均します。提出用には、同じ規則で本数を決めて全期間で学習し直します。

    比べる相手は、S13 のスタッキングに入っているメンバー（S10 の LightGBM と CatBoost、S6 の XGBoost、S7 の MLP）です。
    """)
    return


@app.cell
def _(env, holdout_ids, np, pl, sample):
    def saved(stage, run):
        """Holdout and test predictions of a saved run, on the log scale."""
        oof = pl.read_parquet(env.output_dir / stage / run / "oof.parquet")
        sub = pl.read_csv(env.output_dir / stage / run / "submission.csv")
        assert oof["id"].to_list() == holdout_ids["id"].to_list() and sub["id"].to_list() == sample["id"].to_list()
        return np.log1p(oof["pred"].to_numpy()), np.log1p(sub["price"].to_numpy())

    s13_members = {
        "lightgbm": saved("s10_final", "lightgbm_seed_avg"),
        "catboost": saved("s10_final", "catboost_long"),
        "xgboost": saved("s6_gbdt", "xgboost"),
        "ridge": saved("s7_mlp_classic", "ridge"),
        "random_forest": saved("s7_mlp_classic", "random_forest"),
        "mlp": saved("s7_mlp_classic", "mlp"),
        "tabpfn": saved("s13_tabular_dl", "tabpfn"),
    }
    s13_average = saved("s13_tabular_dl", "stacker_average_with_tabpfn")
    return s13_average, s13_members


@app.cell
def _(
    THREADS,
    asdict,
    cfg,
    classic,
    fs,
    lesson,
    lgb,
    mlp_frame,
    mlp_frame_refit,
    mlp_frame_test,
    np,
    studies,
    threadpool_limits,
    tr_idx,
    va_idx,
    y_real,
    y_real_refit,
):
    def chosen(name):
        return studies[name].best_trial.user_attrs["params"]

    def scaled(rounds):
        return lesson.refit_rounds(rounds, n_all=len(fs.y_log), n_train=len(tr_idx))

    def final_lightgbm():
        holdout, test, rounds = [], [], []
        for seed in cfg.lgbm_seeds:
            params = asdict(cfg.lgbm) | chosen("lightgbm") | {"seed": seed}
            booster, pred = lesson.fit_lgbm_holdout(
                fs.X,
                y_real,
                tr_idx,
                va_idx,
                params,
                max_rounds=cfg.max_rounds,
                early_stopping_rounds=cfg.early_stopping_rounds,
            )
            refit = lgb.train(
                params, lgb.Dataset(fs.X_refit, label=y_real_refit), num_boost_round=scaled(booster.best_iteration)
            )
            holdout.append(pred)
            test.append(refit.predict(fs.X_test))
            rounds.append(booster.best_iteration)
        return np.mean(holdout, axis=0), np.mean(test, axis=0), rounds

    def final_catboost():
        params = asdict(cfg.cat) | chosen("catboost")
        rounds, pred = lesson.fit_catboost_holdout(
            fs.X,
            y_real,
            tr_idx,
            va_idx,
            params,
            max_rounds=cfg.catboost_max_rounds,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        return pred, lesson.refit_catboost(fs.X_refit, y_real_refit, fs.X_test, params, rounds=scaled(rounds)), [rounds]

    def final_xgboost():
        params = asdict(cfg.xgb) | chosen("xgboost")
        rounds, pred = lesson.fit_xgb_holdout(
            fs.X,
            y_real,
            tr_idx,
            va_idx,
            params,
            max_rounds=cfg.max_rounds,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        return pred, lesson.refit_xgb(fs.X_refit, y_real_refit, fs.X_test, params, rounds=scaled(rounds)), [rounds]

    def final_mlp():
        params = asdict(cfg.mlp) | chosen("mlp")
        with threadpool_limits(THREADS):
            model = classic.make_mlp(mlp_frame, params).fit(mlp_frame.iloc[tr_idx], y_real[tr_idx])
            refit = classic.make_mlp(mlp_frame_refit, params).fit(mlp_frame_refit, y_real_refit)
            return model.predict(mlp_frame.iloc[va_idx]), refit.predict(mlp_frame_test), [model.regressor_[-1].n_iter_]

    FINAL = {"lightgbm": final_lightgbm, "xgboost": final_xgboost, "mlp": final_mlp, "catboost": final_catboost}
    return (FINAL,)


@app.cell
def _(
    BASE,
    EXP,
    FINAL,
    asdict,
    baseline_inner,
    bootstrap_diff,
    cfg,
    env,
    fs,
    git_commit,
    holdout_ids,
    holdout_rmsle,
    lesson,
    np,
    pl,
    s13_members,
    sample,
    studies,
    validate_submission,
    va_idx,
    write_submission,
):
    commit = git_commit(env.root)
    y_holdout = fs.y_log[va_idx]
    _issues = {"lightgbm": "#40", "xgboost": "#42", "mlp": "#41", "catboost": "#39"}
    tuned_members = {}
    _rows = []
    for _name, _final in FINAL.items():
        _pred_real, _test_real, _rounds = _final()
        _holdout_log, _test_log = _pred_real + fs.offset[va_idx], _test_real + fs.offset_test
        tuned_members[_name] = (_holdout_log, _test_log)
        _score = holdout_rmsle(_holdout_log, y_holdout)
        _old = s13_members[_name][0]
        _change, _low, _high = bootstrap_diff((_holdout_log - y_holdout) ** 2, (_old - y_holdout) ** 2)
        _rows.append(
            {
                "model": _name,
                "s13 member": holdout_rmsle(_old, y_holdout),
                "tuned": _score,
                "change": _change,
                "ci_low": _low,
                "ci_high": _high,
                "change on 2024Q3-Q4": studies[_name].best_value - baseline_inner[_name],
            }
        )
        _out = lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=f"{_name}_tuned",
            score=_score,
            metric=cfg.metric,
            config=asdict(cfg)
            | {
                "model": _name,
                "chosen": studies[_name].best_trial.user_attrs["params"],
                "inner_best": studies[_name].best_value,
                "inner_current": baseline_inner[_name],
                "trials": len(studies[_name].trials),
                "rounds": _rounds,
            },
            holdout=holdout_ids,
            pred=np.expm1(_holdout_log),
            notes=f"{_name} with the settings Optuna chose on 2024Q3-Q4 ({_issues[_name]})",
            commit=commit,
        )
        _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(_test_log)})
        validate_submission(_submission, sample, cfg.id_col)
        write_submission(_out, _submission)
    tuned_comparison = pl.DataFrame(_rows)
    tuned_comparison
    return commit, tuned_comparison, tuned_members, y_holdout


@app.cell
def _(mo):
    mo.md(r"""
    `change` は S13 のメンバーとの差（holdout、マイナスが改善）と、その 95% の範囲（holdout の行を選び直す bootstrap）です。
    `change on 2024Q3-Q4` は、探索用の評価期間での、いまの設定との差です。holdout での改善がこれよりずっと小さければ、探索が 2024Q3〜Q4 に合わせ込んだということです。

    ## 5. スタッキングし直す

    S13 の `stacker_average_with_tabpfn` と同じ手順（4つのスタッキングと重み付き平均の平均、2025Q1 と Q2 で交互に学習して採点）で、
    メンバーを探索したものに入れ替えて組み直します。RandomForest、Ridge、TabPFN は S13 と同じものを使います。
    まず S13 のメンバーで組み直して、S13 の結果が再現できることを確かめます。
    """)
    return


@app.cell
def _(asdict, cfg, ensemble, fs, in_q1, np, stacking, va_idx, y_holdout):
    rows_holdout = stacking.row_features(fs.X.iloc[va_idx])
    rows_test = stacking.row_features(fs.X_test)

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
        return stacking.stacker_average(
            np.column_stack([members[n][0] for n in names]),
            np.column_stack([members[n][1] for n in names]),
            rows_holdout,
            rows_test,
            y_holdout,
            in_q1,
            found,
        )

    def weights(members):
        """Non-negative weights per member, fit on every holdout row (how much each member is used)."""
        return ensemble.fit_weights(np.column_stack([m[0] for m in members.values()]), y_holdout)

    return stacker_average, weights


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    bootstrap_diff,
    cfg,
    commit,
    env,
    holdout_ids,
    holdout_rmsle,
    lesson,
    np,
    pl,
    s13_average,
    s13_members,
    sample,
    stacker_average,
    tuned_members,
    validate_submission,
    weights,
    write_submission,
    y_holdout,
):
    _reproduced = stacker_average(s13_members)
    assert np.allclose(_reproduced[0], s13_average[0], atol=1e-9)
    assert np.allclose(_reproduced[1], s13_average[1], atol=1e-9)

    members_tuned = s13_members | tuned_members
    restacked_tuned = stacker_average(members_tuned)
    _holdout_log, _test_log = restacked_tuned
    restack_score = holdout_rmsle(_holdout_log, y_holdout)
    restack_change = bootstrap_diff((_holdout_log - y_holdout) ** 2, (s13_average[0] - y_holdout) ** 2)
    _out = lesson.record_run(
        env,
        exp=EXP,
        base=BASE,
        run="stacker_average_tuned",
        score=restack_score,
        metric=cfg.metric,
        config=asdict(cfg) | {"members": list(members_tuned), "tuned": list(tuned_members)},
        holdout=holdout_ids,
        pred=np.expm1(_holdout_log),
        notes="S13 stacker average with the Optuna-tuned LightGBM, CatBoost, XGBoost and MLP (#38-#42)",
        commit=commit,
    )
    _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(_test_log)})
    validate_submission(_submission, sample, cfg.id_col)
    write_submission(_out, _submission)
    member_weights = pl.DataFrame(
        {
            "member": list(s13_members),
            "weight in s13": np.round(weights(s13_members), 3),
            "weight with tuned members": np.round(weights(members_tuned), 3),
        }
    )
    pl.DataFrame(
        [
            {"run": "s13 stacker_average_with_tabpfn", "holdout_rmsle": holdout_rmsle(s13_average[0], y_holdout)},
            {
                "run": "stacker_average_tuned",
                "holdout_rmsle": restack_score,
                "change_vs_s13": restack_change[0],
                "ci_low": restack_change[1],
                "ci_high": restack_change[2],
            },
        ],
        infer_schema_length=None,
    )
    return member_weights, restack_change, restack_score, restacked_tuned


@app.cell
def _(member_weights):
    member_weights
    return


@app.cell
def _(mo, pl, restack_change, restack_score, search_summary, tuned_comparison):
    def tuned(name):
        r = tuned_comparison.filter(pl.col("model") == name).row(0, named=True)
        return f"{r['s13 member']:.4f} → {r['tuned']:.4f}（{r['change']:+.4f}、95% の範囲 {r['ci_low']:+.4f}〜{r['ci_high']:+.4f}）"

    def inner(name):
        r = search_summary.filter(pl.col("model") == name).row(0, named=True)
        return f"{r['current settings']:.4f} → {r['best']:.4f}（{r['trials']} 回）"

    mo.md(f"""
    ## 6. 結果

    探索用の評価期間（2024Q3〜Q4）での、いまの設定と探索の最良:

    - LightGBM: {inner("lightgbm")}
    - XGBoost: {inner("xgboost")}
    - MLP: {inner("mlp")}
    - CatBoost: {inner("catboost")}

    holdout（2025Q1〜Q2）での、S13 のメンバーと探索した設定:

    - LightGBM: {tuned("lightgbm")}
    - XGBoost: {tuned("xgboost")}
    - MLP: {tuned("mlp")}
    - CatBoost: {tuned("catboost")}

    スタッキングの平均: {restack_score:.4f}（S13 の 0.1910 との差 {restack_change[0]:+.4f}、95% の範囲 {restack_change[1]:+.4f}〜{restack_change[2]:+.4f}）
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. 結果を読む

    2026-10-17 の実行の結果です。

    - 探索期間（2024Q3〜Q4）では、4つとも良くなりました。holdout でも良くなったのは XGBoost（-0.0106）と MLP（-0.0051）だけです。
      CatBoost（+0.0004）と LightGBM（-0.0000）の改善は消えました。探索した期間にだけ合う設定を選んでいたことになります。
      holdout を探索に使わず、選んだ設定の確かめに1回だけ使ったから、これが見えました。
    - XGBoost は、値が31種類以下のカテゴリを one-hot で扱う設定に変わり、単独で CatBoost と同じくらいになりました。
      その代わり、誤差の出方が CatBoost に似ました（誤差の相関 0.876 → 0.952）。
    - スタッキングの平均は S13 と変わりません。重みもほとんど動かず（TabPFN 0.558、CatBoost 0.216、LightGBM 0.205、XGBoost 0.022、ほかは 0）、
      良くなった XGBoost と MLP が持ち込んだのは、スタッキングがすでに持っている情報でした。
    - 単独のメンバーが強くなっても、ほかのメンバーと同じ外し方をするなら、アンサンブルは良くなりません。S14 は採用しません。
    """)
    return


if __name__ == "__main__":
    app.run()
