import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from dataclasses import asdict, dataclass, field

    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import psutil
    import torch
    from pytorch_tabnet.tab_model import TabNetRegressor
    from tabpfn import TabPFNRegressor

    from kgl import cv, feature_set, incontext, lesson, mlit, stacking
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, validate_submission, write_submission
    from kgl.metrics import get_metric

    return (
        TabNetRegressor,
        TabPFNRegressor,
        asdict,
        comp_dir,
        cv,
        dataclass,
        detect,
        feature_set,
        field,
        get_metric,
        git_commit,
        incontext,
        lesson,
        mlit,
        mo,
        np,
        pd,
        pl,
        psutil,
        stacking,
        torch,
        validate_submission,
        write_submission,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S13. 表データ向けの深層学習（TabNet と TabPFN）

    S12 までのメンバーは、GBDT 3つ（LightGBM、CatBoost、XGBoost）と、S7 の MLP・RandomForest・Ridge でした。
    この段階では、表データ向けに作られたニューラルネットを新しいメンバーとして足します（#36）。

    TabNet は、行ごとに「どの特徴を見るか」を注意機構（attention）で選びながら、何段階かに分けて予測を組み立てるニューラルネットです。
    木を使わないので、GBDT とは誤差の出方が違うことが期待できます。S12 で参考にした黒魔術の記事のコンペ（MoA）でも、ベースモデルの1つでした。

    TabPFN は、人工的に作った大量の表データで事前学習済みのモデルです（#37）。このデータに合わせて重みを学習し直すことはせず、
    学習データの行を「文脈」として読み込み、その場で予測します（文脈内学習、in-context learning）。
    使うのは TabPFN-3.5 で、重みは非商用のライセンスに同意して取得します。学習済みモデルの利用はコンペのルールで認められています。

    単独で GBDT に勝つことはあまり期待していません。見たいのは、アンサンブルに入れたときに、ほかのメンバーが持っていない情報を足せるかどうかです。
    """)
    return


@app.cell
def _(dataclass, field, psutil):
    @dataclass
    class TabNetParams:
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
    class TabPFNParams:
        # 50,000 rows is what fits this 8 GB GPU (a probe on 2026-10-16: 6.2 GiB peak, 0.2038 on 2,000 holdout rows)
        context_rows: int = 50000
        predict_chunk: int = 2000
        seed: int = 0

    @dataclass
    class MemberLgbParams:
        # S5 and S10's LightGBM, used in section 4 to compare training windows
        objective: str = "regression"
        learning_rate: float = 0.05
        num_leaves: int = 63
        min_data_in_leaf: int = 50
        feature_fraction: float = 0.8
        bagging_fraction: float = 0.8
        bagging_freq: int = 1
        seed: int = 42
        verbose: int = -1
        num_threads: int = psutil.cpu_count(logical=False)

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
        member_max_rounds: int = 5000
        member_early_stopping_rounds: int = 100
        tabnet: TabNetParams = field(default_factory=TabNetParams)
        tabpfn: TabPFNParams = field(default_factory=TabPFNParams)
        member_lgbm: MemberLgbParams = field(default_factory=MemberLgbParams)
        lgbm: StackerLgbParams = field(default_factory=StackerLgbParams)

    cfg = Config()
    EXP = "s13_tabular_dl"
    BASE = "s12_black_magic"
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
    y_real, y_real_refit = fs.y_log - fs.offset, fs.y_log - fs.offset_refit
    y_holdout = fs.y_log[va_idx]
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    in_q1 = train_folded[va_idx].select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy() == 2025.0
    holdout_build_year = train_folded[va_idx].select(mlit.build_year("建築年"))["建築年"].to_numpy()
    return (
        env,
        fs,
        holdout_build_year,
        holdout_ids,
        in_q1,
        sample,
        tr_idx,
        va_idx,
        y_holdout,
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
    ## 1. TabNet

    特徴量は S5 と同じものを使います。ニューラルネットに入れるために、次の下ごしらえをします。統計量は学習に使う行だけから計算します。

    - カテゴリ（地区名、駅名、間取りなど12列）は番号にして、埋め込み（embedding）で4次元のベクトルにする。欠損は番号 0
    - 数値の列は、欠損を中央値で埋めてから、平均 0・標準偏差 1 にそろえる
    - 目的変数は S4 からと同じ「log 価格 − その四半期の価格指数」で、さらに平均を引いてから学習する（出力の初期値 0 から学び始められるように）

    GBDT と同じく 2024Q4 までで学習して holdout で early stopping し、提出用には全期間で、同じエポック数だけ学習し直します。
    """)
    return


@app.cell
def _(TabNetRegressor, asdict, cfg, np, pd, torch):
    def tabnet_inputs(X_fit, X_apply):
        """Category codes (0 = missing) and standardized numeric columns, with statistics from X_fit only."""
        cat_cols = [c for c in X_fit.columns if isinstance(X_fit[c].dtype, pd.CategoricalDtype)]
        num_cols = [c for c in X_fit.columns if c not in cat_cols]
        median = X_fit[num_cols].median()
        filled = X_fit[num_cols].fillna(median)
        mean, std = filled.mean(), filled.std()
        assert (std > 0).all(), std[std == 0]

        def matrix(X):
            num = ((X[num_cols].fillna(median) - mean) / std).to_numpy(dtype=np.float32)
            cat = np.column_stack([X[c].cat.codes.to_numpy() + 1 for c in cat_cols]).astype(np.float32)
            return np.hstack([cat, num])

        cat_dims = [len(X_fit[c].cat.categories) + 1 for c in cat_cols]
        return [matrix(X) for X in X_apply], cat_dims

    def make_tabnet(cat_dims):
        p = asdict(cfg.tabnet)
        return TabNetRegressor(
            n_d=p["n_d"],
            n_a=p["n_a"],
            n_steps=p["n_steps"],
            gamma=p["gamma"],
            lambda_sparse=p["lambda_sparse"],
            cat_idxs=list(range(len(cat_dims))),
            cat_dims=cat_dims,
            cat_emb_dim=p["cat_emb_dim"],
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": p["lr"]},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": p["lr_step_size"], "gamma": p["lr_gamma"]},
            seed=p["seed"],
            device_name="cuda",
            verbose=0,
        )

    return make_tabnet, tabnet_inputs


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
    make_tabnet,
    np,
    pl,
    sample,
    tabnet_inputs,
    torch,
    tr_idx,
    va_idx,
    validate_submission,
    write_submission,
    y_holdout,
    y_real,
    y_real_refit,
):
    assert torch.cuda.is_available()
    commit = git_commit(env.root)
    _fit_kwargs = {
        "batch_size": cfg.tabnet.batch_size,
        "virtual_batch_size": cfg.tabnet.virtual_batch_size,
    }

    (_X_tr, _X_va), _cat_dims = tabnet_inputs(fs.X.iloc[tr_idx], [fs.X.iloc[tr_idx], fs.X.iloc[va_idx]])
    _center = float(y_real[tr_idx].mean())
    _model = make_tabnet(_cat_dims)
    _model.fit(
        _X_tr,
        (y_real[tr_idx] - _center).reshape(-1, 1),
        eval_set=[(_X_va, (y_real[va_idx] - _center).reshape(-1, 1))],
        eval_metric=["rmse"],
        max_epochs=cfg.tabnet.max_epochs,
        patience=cfg.tabnet.patience,
        **_fit_kwargs,
    )
    tabnet_epochs = int(_model.best_epoch) + 1
    tabnet_holdout = _model.predict(_X_va).ravel().astype(np.float64) + _center + fs.offset[va_idx]

    (_X_all, _X_test), _cat_dims = tabnet_inputs(fs.X_refit, [fs.X_refit, fs.X_test])
    _center_all = float(y_real_refit.mean())
    _refit = make_tabnet(_cat_dims)
    _refit.fit(_X_all, (y_real_refit - _center_all).reshape(-1, 1), max_epochs=tabnet_epochs, **_fit_kwargs)
    tabnet_test = _refit.predict(_X_test).ravel().astype(np.float64) + _center_all + fs.offset_test

    tabnet_score = holdout_rmsle(tabnet_holdout, y_holdout)
    _out = lesson.record_run(
        env,
        exp=EXP,
        base=BASE,
        run="tabnet",
        score=tabnet_score,
        metric=cfg.metric,
        config=asdict(cfg) | {"model": "tabnet", "epochs": tabnet_epochs},
        holdout=holdout_ids,
        pred=np.expm1(tabnet_holdout),
        notes="TabNet on the S5 feature set (#36)",
        commit=commit,
    )
    _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(tabnet_test)})
    validate_submission(_submission, sample, cfg.id_col)
    write_submission(_out, _submission)
    {"holdout_rmsle": tabnet_score, "epochs (best + 1)": tabnet_epochs}
    return commit, tabnet_epochs, tabnet_holdout, tabnet_score, tabnet_test


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. TabPFN

    TabPFN は学習データの行を全部「文脈」として読み込むので、行が多いほど GPU のメモリと時間が要ります。
    この GPU（8GB）で試したところ、5万行が上限でした（2,000行の予測に約3.5分）。
    そこで、文脈には取引時期が新しい順に5万行を使います。holdout 用には 2024Q4 までの直近5万行、提出用には全期間の直近5万行です。
    test は学習期間より後の取引なので、新しい行のほうが test に近いと考えました。

    特徴量は S5 と同じものをそのまま渡します（カテゴリの扱いは TabPFN に任せます）。目的変数も同じく「log 価格 − その四半期の価格指数」です。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    TabPFNRegressor,
    asdict,
    cfg,
    commit,
    env,
    fs,
    holdout_ids,
    holdout_rmsle,
    incontext,
    lesson,
    np,
    pl,
    sample,
    tr_idx,
    va_idx,
    validate_submission,
    write_submission,
    y_holdout,
    y_real,
    y_real_refit,
):
    most_recent = incontext.most_recent

    def tabpfn_predict(X_context, y_context, X_apply):
        model = TabPFNRegressor(device="cuda", random_state=cfg.tabpfn.seed)
        model.fit(X_context, y_context)
        return incontext.predict_in_chunks(model, X_apply, cfg.tabpfn.predict_chunk)

    _context = most_recent(tr_idx, fs.X["trade_period"].to_numpy(), cfg.tabpfn.context_rows)
    tabpfn_holdout = tabpfn_predict(fs.X.iloc[_context], y_real[_context], fs.X.iloc[va_idx]) + fs.offset[va_idx]
    _context_all = most_recent(
        np.arange(len(fs.X_refit)), fs.X_refit["trade_period"].to_numpy(), cfg.tabpfn.context_rows
    )
    tabpfn_test = tabpfn_predict(fs.X_refit.iloc[_context_all], y_real_refit[_context_all], fs.X_test) + fs.offset_test

    tabpfn_score = holdout_rmsle(tabpfn_holdout, y_holdout)
    _out = lesson.record_run(
        env,
        exp=EXP,
        base=BASE,
        run="tabpfn",
        score=tabpfn_score,
        metric=cfg.metric,
        config=asdict(cfg) | {"model": "tabpfn-3.5"},
        holdout=holdout_ids,
        pred=np.expm1(tabpfn_holdout),
        notes=f"TabPFN-3.5, context = the {cfg.tabpfn.context_rows} most recent rows (#37)",
        commit=commit,
    )
    _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(tabpfn_test)})
    validate_submission(_submission, sample, cfg.id_col)
    write_submission(_out, _submission)
    {
        "holdout_rmsle": tabpfn_score,
        "context periods (holdout)": (
            float(fs.X["trade_period"].iloc[_context].min()),
            float(fs.X["trade_period"].iloc[_context].max()),
        ),
        "context periods (test)": (
            float(fs.X_refit["trade_period"].iloc[_context_all].min()),
            float(fs.X_refit["trade_period"].iloc[_context_all].max()),
        ),
    }
    return most_recent, tabpfn_holdout, tabpfn_score, tabpfn_test


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. ほかのメンバーとどれだけ違うか

    アンサンブルで効くのは、単独で強いメンバーだけではありません。ほかのメンバーと違う間違え方をするメンバーも効きます。
    そこで、各メンバーの holdout での誤差（予測 − 正解、log）が、CatBoost の誤差とどれだけ相関するかを並べます。相関が低いほど、CatBoost と違う間違え方をしています。
    """)
    return


@app.cell
def _(
    env,
    holdout_ids,
    holdout_rmsle,
    np,
    pl,
    sample,
    tabnet_holdout,
    tabnet_test,
    tabpfn_holdout,
    tabpfn_test,
    y_holdout,
):
    def saved(stage, run):
        """Holdout and test predictions of a saved run, on the log scale."""
        oof = pl.read_parquet(env.output_dir / stage / run / "oof.parquet")
        sub = pl.read_csv(env.output_dir / stage / run / "submission.csv")
        assert oof["id"].to_list() == holdout_ids["id"].to_list() and sub["id"].to_list() == sample["id"].to_list()
        return np.log1p(oof["pred"].to_numpy()), np.log1p(sub["price"].to_numpy())

    s12_members = {
        "lightgbm_seed_avg": saved("s10_final", "lightgbm_seed_avg"),
        "catboost_long": saved("s10_final", "catboost_long"),
        "xgboost": saved("s6_gbdt", "xgboost"),
        "ridge": saved("s7_mlp_classic", "ridge"),
        "random_forest": saved("s7_mlp_classic", "random_forest"),
        "mlp": saved("s7_mlp_classic", "mlp"),
    }
    s12_average = saved("s12_black_magic", "stacker_average")
    new_members = {"tabnet": (tabnet_holdout, tabnet_test), "tabpfn": (tabpfn_holdout, tabpfn_test)}
    _all = s12_members | new_members
    _cat_error = _all["catboost_long"][0] - y_holdout
    diversity = pl.DataFrame(
        [
            {
                "member": _name,
                "holdout_rmsle": holdout_rmsle(_h, y_holdout),
                "error correlation with catboost": float(np.corrcoef(_h - y_holdout, _cat_error)[0, 1]),
            }
            for _name, (_h, _) in _all.items()
        ]
    )
    diversity
    return diversity, new_members, s12_average, s12_members


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. TabPFN の強さはどこから来るか

    TabPFN は、GBDT と違って直近5万行しか見ていません。良いスコアの理由が「モデル」ではなく「新しい行だけを使ったこと」なら、GBDT でも直近の行だけで学習すれば同じように良くなるはずです。
    そこで、S5・S10 と同じ設定の LightGBM を、2024Q4 までの全行と、TabPFN と同じ直近5万行の2通りで学習して比べます。
    比べるときは、違いを1つだけにそろえます（ここでは学習に使う行）。

    続けて、holdout の行をいくつかのグループに分け、CatBoost（S10 の `catboost_long`）と TabPFN の RMSLE をグループごとに並べます。
    改善が一部の行に偏っているのか、全体に広がっているのかを見ます。
    """)
    return


@app.cell
def _(
    asdict,
    bootstrap_diff,
    cfg,
    fs,
    holdout_rmsle,
    lesson,
    most_recent,
    pl,
    s12_members,
    tabpfn_holdout,
    tr_idx,
    va_idx,
    y_holdout,
    y_real,
):
    _windows = {
        "all rows up to 2024Q4": tr_idx,
        "the most recent rows (TabPFN's context)": most_recent(
            tr_idx, fs.X["trade_period"].to_numpy(), cfg.tabpfn.context_rows
        ),
    }
    _rows = []
    for _name, _idx in _windows.items():
        _, _pred_real = lesson.fit_lgbm_holdout(
            fs.X,
            y_real,
            _idx,
            va_idx,
            asdict(cfg.member_lgbm),
            max_rounds=cfg.member_max_rounds,
            early_stopping_rounds=cfg.member_early_stopping_rounds,
        )
        _rows.append(
            {
                "model": "lightgbm",
                "training rows": _name,
                "n": len(_idx),
                "holdout_rmsle": holdout_rmsle(_pred_real + fs.offset[va_idx], y_holdout),
            }
        )
    _rows.append(
        {
            "model": "tabpfn",
            "training rows": "the most recent rows (TabPFN's context)",
            "n": cfg.tabpfn.context_rows,
            "holdout_rmsle": holdout_rmsle(tabpfn_holdout, y_holdout),
        }
    )
    window_comparison = pl.DataFrame(_rows)
    tabpfn_vs_catboost = bootstrap_diff(
        (tabpfn_holdout - y_holdout) ** 2, (s12_members["catboost_long"][0] - y_holdout) ** 2
    )
    window_comparison
    return tabpfn_vs_catboost, window_comparison


@app.cell
def _(fs, holdout_build_year, holdout_rmsle, pl, s12_members, tabpfn_holdout, va_idx, y_holdout):
    _X = fs.X.iloc[va_idx]
    _contract = (_X["価格情報区分"] == "成約価格情報").to_numpy()
    _area = _X["area"].to_numpy()
    _new_build = holdout_build_year >= 2025
    _groups = {
        "contract price (成約価格)": _contract,
        "transaction price (取引価格)": ~_contract,
        "area < 30 m2": _area < 30,
        "area 30-70 m2": (_area >= 30) & (_area < 70),
        "area >= 70 m2": _area >= 70,
        "built in 2025 or later": _new_build,
        "built before 2025 (or unknown)": ~_new_build,
    }
    _cat = s12_members["catboost_long"][0]
    by_group = pl.DataFrame(
        [
            {
                "group": _name,
                "n": int(_mask.sum()),
                "catboost_long": holdout_rmsle(_cat[_mask], y_holdout[_mask]),
                "tabpfn": holdout_rmsle(tabpfn_holdout[_mask], y_holdout[_mask]),
            }
            for _name, _mask in _groups.items()
        ]
    ).with_columns((pl.col("tabpfn") - pl.col("catboost_long")).alias("change"))
    by_group
    return (by_group,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. S12 のスタッキングの平均に足す

    S12 と同じ4つのスタッキング（Ridge、種類別の Ridge、MLP、LightGBM）と重み付き平均を、新しいメンバーを足した顔ぶれで作り直し、その平均を取ります。
    採点も S12 と同じです（2025Q1 で学習して Q2 を予測、逆向きも）。
    まず S12 の6メンバーで作り直して、S12 の `stacker_average` が再現できることを確かめます。
    """)
    return


@app.cell
def _(asdict, cfg, fs, in_q1, np, stacking, va_idx, y_holdout):
    rows_holdout = stacking.row_features(fs.X.iloc[va_idx])
    rows_test = stacking.row_features(fs.X_test)

    def stacker_average(members):
        """S12's recipe on a member set: the four stackers and the weighted average, then their plain average."""
        names = list(members)
        found = stacking.stackers(
            len(names),
            alpha=cfg.stacking_alpha,
            mlp_hidden=cfg.mlp_hidden,
            mlp_alpha=cfg.mlp_alpha,
            mlp_seeds=cfg.mlp_seeds,
            lgbm_params=asdict(cfg.lgbm),
            lgbm_max_rounds=cfg.lgbm_max_rounds,
            lgbm_early_stopping_rounds=cfg.lgbm_early_stopping_rounds,
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

    return (stacker_average,)


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
    new_members,
    np,
    pl,
    s12_average,
    s12_members,
    sample,
    stacker_average,
    validate_submission,
    write_submission,
    y_holdout,
):
    _reproduced = stacker_average(s12_members)
    assert np.allclose(_reproduced[0], s12_average[0], atol=1e-9)
    assert np.allclose(_reproduced[1], s12_average[1], atol=1e-9)

    _s12_sq = (s12_average[0] - y_holdout) ** 2
    _rows = [{"run": "s12 stacker_average", "holdout_rmsle": holdout_rmsle(s12_average[0], y_holdout)}]
    restacked = {}
    for _variant, _names in {"tabnet": ["tabnet"], "tabpfn": ["tabpfn"], "both": ["tabnet", "tabpfn"]}.items():
        _run = f"stacker_average_with_{_variant}"
        restacked[_run] = stacker_average(s12_members | {n: new_members[n] for n in _names})
        _holdout_log, _test_log = restacked[_run]
        _score = holdout_rmsle(_holdout_log, y_holdout)
        _change, _low, _high = bootstrap_diff((_holdout_log - y_holdout) ** 2, _s12_sq)
        _rows.append({"run": _run, "holdout_rmsle": _score, "change_vs_s12": _change, "ci_low": _low, "ci_high": _high})
        _out = lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=_run,
            score=_score,
            metric=cfg.metric,
            config=asdict(cfg) | {"members": list(s12_members) + _names},
            holdout=holdout_ids,
            pred=np.expm1(_holdout_log),
            notes=f"S12 stacker average with {' and '.join(_names)} added (#36 #37)",
            commit=commit,
        )
        _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(_test_log)})
        validate_submission(_submission, sample, cfg.id_col)
        write_submission(_out, _submission)
    restack_comparison = pl.DataFrame(_rows, infer_schema_length=None)
    restack_comparison
    return restack_comparison, restacked


@app.cell
def _(
    bootstrap_diff,
    by_group,
    diversity,
    fs,
    mo,
    new_members,
    np,
    pl,
    restack_comparison,
    restacked,
    s12_average,
    s12_members,
    tabpfn_vs_catboost,
    window_comparison,
    y_holdout,
):
    def member(name):
        return diversity.filter(pl.col("member") == name).row(0, named=True)

    def change(run):
        r = restack_comparison.filter(pl.col("run") == run).row(0, named=True)
        return (
            f"{r['holdout_rmsle']:.4f}（S12 との差 {r['change_vs_s12']:+.4f}、"
            f"95% の範囲 {r['ci_low']:+.4f}〜{r['ci_high']:+.4f}）"
        )

    def group(name):
        return by_group.filter(pl.col("group") == name).row(0, named=True)

    def lightgbm(rows):
        return window_comparison.filter(
            (pl.col("model") == "lightgbm") & pl.col("training rows").str.starts_with(rows)
        )["holdout_rmsle"][0]

    _with_tabpfn = restacked["stacker_average_with_tabpfn"]
    _both_vs_tabpfn = bootstrap_diff(
        (restacked["stacker_average_with_both"][0] - y_holdout) ** 2, (_with_tabpfn[0] - y_holdout) ** 2
    )
    _shift_holdout = float(np.mean(_with_tabpfn[0] - s12_average[0]))
    _shift_test = float(np.mean(_with_tabpfn[1] - s12_average[1]))
    _new, _old = group("built in 2025 or later"), group("built before 2025 (or unknown)")
    _test_period = fs.X_test["trade_period"].to_numpy()
    _test_gap = new_members["tabpfn"][1] - s12_members["catboost_long"][1]
    _gap_by_quarter = "、".join(
        f"{int(_q)}Q{int(round((_q % 1) * 4)) + 1} {_test_gap[_test_period == _q].mean():+.4f}"
        for _q in np.unique(_test_period)
    )
    mo.md(f"""
    ## 6. 結果を読む

    - TabNet の holdout RMSLE は {member("tabnet")["holdout_rmsle"]:.4f} で、MLP（S7）の {member("mlp")["holdout_rmsle"]:.4f} と同じくらいです。
      CatBoost（{member("catboost_long")["holdout_rmsle"]:.4f}）には届きません。
      CatBoost との誤差の相関は {member("tabnet")["error correlation with catboost"]:.3f} で、XGBoost（{member("xgboost")["error correlation with catboost"]:.3f}）や MLP（{member("mlp")["error correlation with catboost"]:.3f}）と変わりません。
      なお、GPU で学習するニューラルネットは、seed を固定しても実行のたびに結果が少し変わります。同じ設定のこれまでの実行では、0.2172 と 0.2220 でした。
      スタッキングの平均に足したときは {change("stacker_average_with_tabnet")} で、改善とは言えません。TabNet は採用しません。
    - TabPFN の holdout RMSLE は {member("tabpfn")["holdout_rmsle"]:.4f} で、これまでのどの単独モデルよりも良くなりました。
      CatBoost との差は {tabpfn_vs_catboost[0]:+.4f}（95% の範囲 {tabpfn_vs_catboost[1]:+.4f}〜{tabpfn_vs_catboost[2]:+.4f}）です。
      このデータに合わせた学習を一切せず、学習データの3割弱しか見ていないのに、です。冒頭の「単独で GBDT に勝つことはあまり期待していない」という予想は外れました。
    - 4 の比較では、LightGBM を直近5万行だけで学習すると {lightgbm("the most recent"):.4f} で、全行の {lightgbm("all rows"):.4f} より悪くなりました。
      新しい行だけを使うのは、LightGBM にとっては損です。TabPFN の強さは、行の選び方ではなく、モデルそのものから来ています。
    - グループ別では、どのグループでも TabPFN が CatBoost より良くなっています。
      差が特に大きいのは、建築年が2025年以降の新築です（{_new["n"]} 行、{_new["catboost_long"]:.4f} → {_new["tabpfn"]:.4f}）。学習期間（2024Q4 まで）にはほとんどない建築年で、CatBoost は大きく外しています。
      ただし行数は少なく、新築を除いても {_old["catboost_long"]:.4f} → {_old["tabpfn"]:.4f} です。改善は一部の行に偏らず、全体に広がっています。
    - スタッキングの平均に TabPFN を足すと {change("stacker_average_with_tabpfn")} です。S12 の黒魔術による改善（-0.0012）の2倍以上あります。
      TabNet も一緒に足した `with_both` は {change("stacker_average_with_both")} で、TabPFN だけの場合との差は {_both_vs_tabpfn[0]:+.4f}（95% の範囲 {_both_vs_tabpfn[1]:+.4f}〜{_both_vs_tabpfn[2]:+.4f}）です。TabNet を足す意味はありません。
    - TabPFN の誤差と CatBoost の誤差の相関は {member("tabpfn")["error correlation with catboost"]:.3f} で、ほかのメンバーよりむしろ高めです。それでも、アンサンブルを最も大きく良くしました。
      違う間違え方をすることも大事ですが、単独で強いことのほうが効く場合もあります。

    `stacker_average_with_tabpfn` を採用し、最終提出の候補に加えます（2026-10-16）。最終提出に何を選ぶかはユーザーが決めます。

    気をつける点が4つあります。

    - TabPFN を足すと、予測の平均（log）が holdout では {_shift_holdout:+.4f}、test では {_shift_test:+.4f} 動きます。
      test では全体を少し安く見積もる方向です。価格指数の外挿はほかのメンバーと共通なので、これは TabPFN 自身の見立ての差です。合っているかは LB で確かめます。
    - holdout が測っているのは、学習期間の1〜2四半期先までです。test は3四半期先（2026Q1）まであり、TabPFN は文脈より先の取引時期をそのまま数値として受け取ります。
      test の四半期ごとに、TabPFN と CatBoost の予測の平均の差（log）を見ると {_gap_by_quarter} で、いちばん先の四半期だけずれが大きくなってはいません。
    - TabPFN の予測には、この GPU で約1時間半かかります（holdout と test を合わせて約5万行）。重みは非商用のライセンスで、使う前に Prior Labs のサイトでの同意が要ります。
    - これまでの5回の提出では、LB は CV の変化のおよそ6割しか動いていません。LB での改善は 0.0015〜0.002 ほどと見ておきます。
    """)
    return


if __name__ == "__main__":
    app.run()
