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
    import torch
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
        torch,
        validate_submission,
        write_submission,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S19. TabPFN の切り方を増やし、2段目の学習行も増やす

    S17 では、市区町村ごとに分けた TabPFN が、単独では良くないのにスタッキングで重み 0.188 を得ました。
    外し方が違うメンバーは、単独で多少弱くても、スタッキングを良くします。この段階では、ほかの切り方を試します（#47）。

    | 名前 | グループ | 文脈 |
    |---|---|---|
    | `by_price_type` | 価格情報区分（成約価格と取引価格） | グループごとの直近2万行 |
    | `by_area_band` | 面積の帯（30㎡未満、30〜60㎡、60㎡以上） | グループごとの直近2万行 |
    | `by_region` | 23区とそれ以外 | グループごとの直近2万行 |

    成約価格と取引価格は集め方が違い、30㎡未満のワンルームは投資用の別の市場で取引されます。
    同じ種類の取引だけを見せると、TabPFN の外し方が変わるはずです。駅や地区まで細かく分けると、文脈のない行が出るので使いません。

    文脈を2万行にしたのは、GPU のメモリのためです。TabPFN は、1回に渡された行が多いときだけ行を分けて処理し、
    メモリが足りなければ分ける幅を半分にしてやり直します。グループの文脈が4〜5万行だと「分けるほど多くはない」と判断されて
    そのまま処理し、8GB の GPU では足りなくなりました。2万行なら、どのグループでも分けずに収まります。

    もう1つ、S15 の「2段目の学習に 2024Q3〜Q4 の行も使う」を、S17 のメンバーでやり直します（#48）。
    S15 では -0.00017 で、採用のルールにわずかに届きませんでした。
    そのとき 2024Q3〜Q4 の予測を作ったのは S14 で探索したメンバーで、holdout と test で使うメンバー（S13 の組）とは違っていました。
    ここでは S13 の組のメンバーを、記録された設定のまま 2024Q2 までで学習し直して、2024Q3〜Q4 を予測させます。
    この予測は、TabPFN の切り方を選ぶ内側の比較にも使います。内側と holdout で、スタッキングのメンバーがそろいます。
    """)
    return


@app.cell
def _(dataclass, field, psutil):
    THREADS = psutil.cpu_count(logical=False)

    @dataclass
    class TabPFNParams:
        seed: int = 0

    @dataclass
    class StackerLgbParams:
        # S12-S18's LightGBM stacker
        objective: str = "regression"
        learning_rate: float = 0.03
        num_leaves: int = 15
        min_data_in_leaf: int = 100
        feature_fraction: float = 0.9
        seed: int = 42
        verbose: int = -1
        num_threads: int = THREADS

    def _variants():
        # Each variant fits one TabPFN per group, with that group's most recent rows as context. Every group of
        # the holdout and the test has rows in its context; stations and districts would leave 14-232 rows without.
        # TabPFN-3.5 splits a call into row chunks, and halves them when the GPU runs out of memory, only when the
        # call holds more rows than one chunk (about 50,000 here; more when it drops empty columns, as it does for
        # contract prices). A group context of 38,000-50,000 rows plus the predicted rows stays below that and
        # ran out of the 8 GB unchunked (probes on 2026-10-18). 20,000 + 2,000 rows peak at 4.1 GiB in every group.
        return {
            "by_price_type": {"group": "price_type", "context_rows": 20000, "predict_chunk": 2000},
            "by_area_band": {"group": "area_band", "context_rows": 20000, "predict_chunk": 2000},
            "by_region": {"group": "region", "context_rows": 20000, "predict_chunk": 2000},
        }

    def _member_runs():
        # S13's members and where their settings are recorded, in S13's order
        return {
            "lightgbm": ["s10_final", "lightgbm_seed_avg"],
            "catboost": ["s10_final", "catboost_long"],
            "xgboost": ["s6_gbdt", "xgboost"],
            "ridge": ["s7_mlp_classic", "ridge"],
            "random_forest": ["s7_mlp_classic", "random_forest"],
            "mlp": ["s7_mlp_classic", "mlp"],
        }

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
        member_runs: dict = field(default_factory=_member_runs)
        area_bands: tuple[float, ...] = (30.0, 60.0)
        tabpfn: TabPFNParams = field(default_factory=TabPFNParams)
        variants: dict = field(default_factory=_variants)
        s17_stage: str = "s17_tabpfn_context"
        stacking_alpha: float = 1.0
        stack_mlp_hidden: tuple[int, ...] = (32, 16)
        stack_mlp_alpha: float = 1e-3
        stack_mlp_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
        stack_lgbm_max_rounds: int = 2000
        stack_lgbm_early_stopping_rounds: int = 100
        inner_valid_share: float = 0.2
        stack_lgbm: StackerLgbParams = field(default_factory=StackerLgbParams)

    cfg = Config()
    EXP = "s19_tabpfn_groups"
    BASE = "s18_comparable_sales"
    return BASE, EXP, THREADS, cfg


@app.cell
def _(cfg, classic, comp_dir, cv, detect, feature_set, mlit, pl, torch):
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

    # Holdout setting (S13-S18): train up to 2024Q4, validate on 2025Q1-Q2 (fold 1).
    tr_idx, va_idx = cv.split(train_folded["fold"].to_numpy(), 1, time=True)
    fs = feature_set.build(train_folded, test_raw, tr_idx, va_idx, **_build)
    y_real, y_real_refit = fs.y_log - fs.offset, fs.y_log - fs.offset_refit
    y_holdout = fs.y_log[va_idx]
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    in_q1 = train_folded[va_idx].select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy() == 2025.0

    # Inner setting (S14): rows up to 2024Q4; train up to 2024Q2, score 2024Q3-Q4.
    inner_raw = train_folded.filter(pl.col("fold") <= 0)
    tr_in, va_in = cv.split(inner_raw["fold"].to_numpy(), 0, time=True)
    fs_in = feature_set.build(inner_raw, test_raw, tr_in, va_in, **_build)
    y_in = fs_in.y_log - fs_in.offset
    y_block = fs_in.y_log[va_in]
    block_ids = inner_raw[va_in].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    in_q3 = inner_raw[va_in].select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy() == 2024.5
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
        in_q3,
        linear_in,
        sample,
        tr_idx,
        tr_in,
        va_idx,
        va_in,
        y_block,
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
def _(TabPFNRegressor, cfg, incontext, np):
    def groups_of(X, group):
        """The group of every row of X, for a variant's group name."""
        if group == "price_type":
            return X["価格情報区分"].astype(str).to_numpy()
        if group == "area_band":
            area = X["area"].to_numpy()
            assert not np.isnan(area).any()
            return np.digitize(area, cfg.area_bands)
        if group == "region":
            return np.where(X["市区町村名"].astype(str).str.endswith("区"), "23 wards", "Tama and islands")
        raise ValueError(f"unknown group: {group}")

    def tabpfn_predict(X_context, y_context, X_apply, variant):
        """TabPFN's predictions (deflated log scale) for X_apply, one TabPFN per group of the variant."""
        periods = X_context["trade_period"].to_numpy()
        group_context, group_apply = groups_of(X_context, variant["group"]), groups_of(X_apply, variant["group"])
        everything = np.arange(len(X_context))
        pred = np.empty(len(X_apply))
        for g in np.unique(group_apply):
            at = np.flatnonzero(group_apply == g)
            pool = everything[group_context == g]
            assert len(pool) > 0, g
            context = incontext.most_recent(pool, periods, variant["context_rows"])
            model = TabPFNRegressor(device="cuda", random_state=cfg.tabpfn.seed)
            model.fit(X_context.iloc[context], y_context[context])
            pred[at] = incontext.predict_in_chunks(model, X_apply.iloc[at], variant["predict_chunk"])
        return pred

    return (tabpfn_predict,)


@app.cell
def _(asdict, cfg, stacking):
    def found_stackers(n_members):
        return stacking.stackers(
            n_members,
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

    return (found_stackers,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. S13 のメンバーで 2024Q3〜Q4 を予測する

    S13 の組のメンバーを、holdout の予測を作ったときと同じ設定で 2024Q2 までの行で学習し、2024Q3〜Q4 を予測します。

    - LightGBM（S10、seed 3つの平均）、CatBoost（S10）、XGBoost（S6）: GBDT は、holdout のときに holdout で early stopping したのと同じく、2024Q3〜Q4 で early stopping します
    - MLP・RandomForest・Ridge: S7 の設定
    - 2本の TabPFN（S17）: S17 の内側の比較で作った予測をそのまま使います

    1回の実行で約1時間かかるので、予測はファイルに保存し、次からは読み直します。
    """)
    return


@app.cell
def _(
    THREADS,
    cfg,
    classic,
    env,
    forest_in,
    fs_in,
    lesson,
    linear_in,
    np,
    read_metrics,
    threadpool_limits,
    tr_in,
    va_in,
    y_in,
):
    def member_config(name):
        stage, run = cfg.member_runs[name]
        return read_metrics(env.output_dir / stage / run / "metrics.json").config

    def gbdt_kwargs(config, max_rounds_key):
        return dict(max_rounds=config[max_rounds_key], early_stopping_rounds=config["early_stopping_rounds"])

    def block_lightgbm():
        config = member_config("lightgbm")
        preds = [
            lesson.fit_lgbm_holdout(
                fs_in.X, y_in, tr_in, va_in, config["lgbm"] | {"seed": seed}, **gbdt_kwargs(config, "max_rounds")
            )[1]
            for seed in config["lgbm_seeds"]
        ]
        return np.mean(preds, axis=0)

    def block_catboost():
        config = member_config("catboost")
        return lesson.fit_catboost_holdout(
            fs_in.X, y_in, tr_in, va_in, config["cat"], **gbdt_kwargs(config, "catboost_max_rounds")
        )[1]

    def block_xgboost():
        config = member_config("xgboost")
        return lesson.fit_xgb_holdout(fs_in.X, y_in, tr_in, va_in, config["xgb"], **gbdt_kwargs(config, "max_rounds"))[
            1
        ]

    def block_ridge():
        config = member_config("ridge")
        model = classic.make_ridge(linear_in, alphas=tuple(config["ridge_alphas"]))
        return model.fit(linear_in.iloc[tr_in], y_in[tr_in]).predict(linear_in.iloc[va_in])

    def block_forest():
        config = member_config("random_forest")
        model = classic.make_forest(config["forest"])
        return model.fit(forest_in.iloc[tr_in], y_in[tr_in]).predict(forest_in.iloc[va_in])

    def block_mlp():
        config = member_config("mlp")
        with threadpool_limits(THREADS):
            model = classic.make_mlp(linear_in, config["mlp"]).fit(linear_in.iloc[tr_in], y_in[tr_in])
            return model.predict(linear_in.iloc[va_in])

    BLOCK = {
        "lightgbm": block_lightgbm,
        "catboost": block_catboost,
        "xgboost": block_xgboost,
        "ridge": block_ridge,
        "random_forest": block_forest,
        "mlp": block_mlp,
    }
    assert list(BLOCK) == list(cfg.member_runs)
    return (BLOCK,)


@app.cell
def _(BLOCK, EXP, block_ids, cfg, env, fs_in, holdout_ids, holdout_rmsle, np, pl, sample, va_in, y_block, y_holdout):
    def saved(stage, run):
        """Holdout and test predictions of a saved run, on the log scale."""
        oof = pl.read_parquet(env.output_dir / stage / run / "oof.parquet")
        sub = pl.read_csv(env.output_dir / stage / run / "submission.csv")
        assert oof["id"].to_list() == holdout_ids["id"].to_list() and sub["id"].to_list() == sample["id"].to_list()
        return np.log1p(oof["pred"].to_numpy()), np.log1p(sub["price"].to_numpy())

    def s17_inner(variant):
        """S17's 2024Q3-Q4 prediction of a TabPFN variant (with the offset)."""
        _saved = pl.read_parquet(env.output_dir / cfg.s17_stage / f"inner_{variant}.parquet")
        assert _saved[cfg.id_col].to_list() == block_ids[cfg.id_col].to_list()
        return _saved["pred"].to_numpy()

    # S17's members on the holdout and the test, in S17's order
    s17_members = {_name: saved(*_run) for _name, _run in cfg.member_runs.items()}
    s17_members["tabpfn"] = saved(cfg.s17_stage, "tabpfn_context_large")
    s17_members["tabpfn_by_municipality"] = saved(cfg.s17_stage, "tabpfn_by_municipality")
    s17_average = saved(cfg.s17_stage, "stacker_average_s17")

    (env.output_dir / EXP).mkdir(parents=True, exist_ok=True)
    _path = env.output_dir / EXP / "block_predictions.parquet"
    if not _path.exists():
        block_ids.with_columns(
            [pl.Series(_name, _predict() + fs_in.offset[va_in]) for _name, _predict in BLOCK.items()]
        ).write_parquet(_path)
    _block = pl.read_parquet(_path)
    assert _block[cfg.id_col].to_list() == block_ids[cfg.id_col].to_list()
    block_members = {_name: _block[_name].to_numpy() for _name in BLOCK}
    block_members["tabpfn"] = s17_inner("context_large")
    block_members["tabpfn_by_municipality"] = s17_inner("by_municipality")
    assert list(block_members) == list(s17_members)

    block_table = pl.DataFrame(
        [
            {
                "member": _name,
                "2024Q3-Q4 (block)": holdout_rmsle(block_members[_name], y_block),
                "2025Q1-Q2 (holdout)": holdout_rmsle(s17_members[_name][0], y_holdout),
            }
            for _name in s17_members
        ]
    )
    block_table
    return block_members, block_table, s17_average, s17_members, saved


@app.cell
def _(mo):
    mo.md(r"""
    表は、各メンバーの RMSLE を2つの期間で並べたものです。どのメンバーも、2024Q3〜Q4 のほうが当てやすい期間です（S15 と同じ）。

    ## 2. 内側の分割で TabPFN の切り方を選ぶ

    3つの切り方で、2024Q2 までの行を文脈にして 2024Q3〜Q4 を予測します。予測は1通りずつファイルに保存します。
    そのうえで、S17 のメンバーの 2024Q3〜Q4 の予測に1本ずつ足してスタッキングを組みます。
    一番良くなるものを足し、残りでまた比べることを、良くならなくなるまで繰り返します。holdout は見ません。
    """)
    return


@app.cell
def _(EXP, block_ids, cfg, env, fs_in, pl, tabpfn_predict, tr_in, va_in, y_in):
    inner_preds = {}
    for _name, _variant in cfg.variants.items():
        _path = env.output_dir / EXP / f"inner_{_name}.parquet"
        if not _path.exists():
            _pred = tabpfn_predict(fs_in.X.iloc[tr_in], y_in[tr_in], fs_in.X.iloc[va_in], _variant)
            pl.DataFrame({cfg.id_col: block_ids[cfg.id_col], "pred": _pred + fs_in.offset[va_in]}).write_parquet(_path)
        _saved = pl.read_parquet(_path)
        assert _saved[cfg.id_col].to_list() == block_ids[cfg.id_col].to_list()
        inner_preds[_name] = _saved["pred"].to_numpy()
    return (inner_preds,)


@app.cell
def _(block_members, found_stackers, fs_in, holdout_rmsle, in_q3, inner_preds, np, pl, stacking, va_in, y_block):
    _rows = stacking.row_features(fs_in.X.iloc[va_in])

    def block_stack(members):
        """The stacker average on 2024Q3-Q4, fit on Q3 and scored on Q4 and the other way round."""
        H = np.column_stack(list(members.values()))
        return holdout_rmsle(
            stacking.stacker_average(H, H, _rows, _rows, y_block, in_q3, found_stackers(len(members)))[0], y_block
        )

    _chosen = dict(block_members)
    _score = block_stack(_chosen)
    _steps = [{"step": 0, "variant": "(S17 members)", "alone": None, "stack": _score}]
    added = []
    _left = list(inner_preds)
    while _left:
        _tries = {_name: block_stack(_chosen | {f"tabpfn_{_name}": inner_preds[_name]}) for _name in _left}
        _best = min(_tries, key=lambda _n: _tries[_n])
        for _name in _left:
            _steps.append(
                {
                    "step": len(added) + 1,
                    "variant": _name,
                    "alone": holdout_rmsle(inner_preds[_name], y_block),
                    "stack": _tries[_name],
                }
            )
        if _tries[_best] >= _score:
            break
        added.append(_best)
        _chosen[f"tabpfn_{_best}"] = inner_preds[_best]
        _score = _tries[_best]
        _left.remove(_best)
    selection_table = pl.DataFrame(_steps, infer_schema_length=None)
    selection_table
    return added, selection_table


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. holdout と test を予測する

    選んだ切り方で、2024Q4 までを文脈にして holdout を、全期間を文脈にして test を1回ずつ予測します。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    added,
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
    tabpfn_predict,
    tr_idx,
    va_idx,
    validate_submission,
    write_submission,
    y_holdout,
    y_real,
    y_real_refit,
):
    commit = git_commit(env.root)
    new_tabpfn = {}
    for _name in added:
        _variant = cfg.variants[_name]
        _path_holdout = env.output_dir / EXP / f"holdout_{_name}.parquet"
        if not _path_holdout.exists():
            _pred = tabpfn_predict(fs.X.iloc[tr_idx], y_real[tr_idx], fs.X.iloc[va_idx], _variant)
            pl.DataFrame({cfg.id_col: holdout_ids[cfg.id_col], "pred": _pred + fs.offset[va_idx]}).write_parquet(
                _path_holdout
            )
        _path_test = env.output_dir / EXP / f"test_{_name}.parquet"
        if not _path_test.exists():
            _pred = tabpfn_predict(fs.X_refit, y_real_refit, fs.X_test, _variant)
            pl.DataFrame({cfg.id_col: sample[cfg.id_col], "pred": _pred + fs.offset_test}).write_parquet(_path_test)
        _holdout, _test = pl.read_parquet(_path_holdout), pl.read_parquet(_path_test)
        assert _holdout[cfg.id_col].to_list() == holdout_ids[cfg.id_col].to_list()
        assert _test[cfg.id_col].to_list() == sample[cfg.id_col].to_list()
        new_tabpfn[_name] = (_holdout["pred"].to_numpy(), _test["pred"].to_numpy())
        _out = lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=f"tabpfn_{_name}",
            score=holdout_rmsle(new_tabpfn[_name][0], y_holdout),
            metric=cfg.metric,
            config=asdict(cfg) | {"model": "tabpfn-3.5", "variant": _name},
            holdout=holdout_ids,
            pred=np.expm1(new_tabpfn[_name][0]),
            notes=f"TabPFN-3.5, one per group, variant {_name} (#47)",
            commit=commit,
        )
        _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(new_tabpfn[_name][1])})
        validate_submission(_submission, sample, cfg.id_col)
        write_submission(_out, _submission)
    return commit, new_tabpfn


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. スタッキングし直す

    まず、S17 のメンバーで S17 の `stacker_average_s17` を再現できることを確かめます。
    そのうえで、選んだ TabPFN を足して2通りに組み直します。

    - `stacker_average_s19`: 2段目は S17 と同じく holdout の行だけで学習（#47 の効果）
    - `stacker_average_s19_with_block`: 2段目の学習行に 2024Q3〜Q4 を足す（#48 の効果）
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    added,
    asdict,
    block_members,
    bootstrap_diff,
    cfg,
    commit,
    ensemble,
    env,
    found_stackers,
    fs,
    fs_in,
    holdout_ids,
    holdout_rmsle,
    in_q1,
    inner_preds,
    lesson,
    new_tabpfn,
    np,
    pl,
    s17_average,
    s17_members,
    sample,
    stacking,
    va_idx,
    va_in,
    validate_submission,
    write_submission,
    y_block,
    y_holdout,
):
    _rows_h, _rows_t = stacking.row_features(fs.X.iloc[va_idx]), stacking.row_features(fs.X_test)
    _rows_b = stacking.row_features(fs_in.X.iloc[va_in])

    def matrices(members):
        return np.column_stack([m[0] for m in members.values()]), np.column_stack([m[1] for m in members.values()])

    def stacker_average(members):
        H, T = matrices(members)
        return stacking.stacker_average(H, T, _rows_h, _rows_t, y_holdout, in_q1, found_stackers(len(members)))

    def stacker_average_with_block(members, block):
        H, T = matrices(members)
        H_block = np.column_stack(list(block.values()))
        return stacking.stacker_average_with_block(
            H, T, _rows_h, _rows_t, y_holdout, in_q1, H_block, _rows_b, y_block, found_stackers(len(members))
        )

    _reproduced = stacker_average(s17_members)
    assert np.allclose(_reproduced[0], s17_average[0], atol=1e-9)
    assert np.allclose(_reproduced[1], s17_average[1], atol=1e-9)

    s19_members = dict(s17_members)
    s19_block = dict(block_members)
    for _name in added:
        s19_members[f"tabpfn_{_name}"] = new_tabpfn[_name]
        s19_block[f"tabpfn_{_name}"] = inner_preds[_name]
    assert list(s19_members) == list(s19_block)

    _results = {
        "stacker_average_s19": (stacker_average(s19_members), "S17 members plus TabPFN by group (#47)"),
        "stacker_average_s19_with_block": (
            stacker_average_with_block(s19_members, s19_block),
            "stacker_average_s19 with 2024Q3-Q4 added to every stacker's training rows (#47, #48)",
        ),
    }
    _base_sq = (s17_average[0] - y_holdout) ** 2
    _rows = [{"run": "s17 stacker_average_s17", "holdout_rmsle": holdout_rmsle(s17_average[0], y_holdout)}]
    for _run, ((_h, _t), _notes) in _results.items():
        _change = bootstrap_diff((_h - y_holdout) ** 2, _base_sq)
        _rows.append(
            {
                "run": _run,
                "holdout_rmsle": holdout_rmsle(_h, y_holdout),
                "change_vs_s17": _change[0],
                "ci_low": _change[1],
                "ci_high": _change[2],
            }
        )
        _out = lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=_run,
            score=holdout_rmsle(_h, y_holdout),
            metric=cfg.metric,
            config=asdict(cfg) | {"members": list(s19_members), "added": added},
            holdout=holdout_ids,
            pred=np.expm1(_h),
            notes=_notes,
            commit=commit,
        )
        _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(_t)})
        validate_submission(_submission, sample, cfg.id_col)
        write_submission(_out, _submission)

    # The block rows on their own: same members, stacked on the holdout rows only versus with 2024Q3-Q4
    block_change = bootstrap_diff(
        (_results["stacker_average_s19_with_block"][0][0] - y_holdout) ** 2,
        (_results["stacker_average_s19"][0][0] - y_holdout) ** 2,
    )
    _weights = ensemble.fit_weights(matrices(s19_members)[0], y_holdout)
    s19_weights = dict(zip(s19_members, np.round(_weights, 3).tolist(), strict=True))
    holdout_table = pl.DataFrame(_rows, infer_schema_length=None)
    holdout_table
    return block_change, holdout_table, s19_weights


@app.cell
def _(added, block_change, block_table, holdout_table, mo, pl, s19_weights, selection_table):
    def _row(run):
        return holdout_table.filter(pl.col("run") == run).row(0, named=True)

    def _change(run):
        r = _row(run)
        return f"{r['holdout_rmsle']:.5f}（S17 との差 {r['change_vs_s17']:+.5f}、95% の範囲 {r['ci_low']:+.5f}〜{r['ci_high']:+.5f}）"

    _block = "\n".join(
        f"    | {r['member']} | {r['2024Q3-Q4 (block)']:.4f} | {r['2025Q1-Q2 (holdout)']:.4f} |"
        for r in block_table.iter_rows(named=True)
    )

    def _alone(value):
        return "" if value is None else f"{value:.4f}"

    _steps = "\n".join(
        f"    | {r['step']} | `{r['variant']}` | {_alone(r['alone'])} | {r['stack']:.5f} |"
        for r in selection_table.iter_rows(named=True)
    )
    mo.md(f"""
    ## 5. 結果

    S13 の組のメンバーの、2つの期間での RMSLE:

    | メンバー | 2024Q3〜Q4 | 2025Q1〜Q2 |
    |---|---|---|
{_block}

    内側の分割での選択（各段で、残りの切り方を1本ずつ足したスタッキングの平均）:

    | 段 | variant | 単独 | スタッキング |
    |---|---|---|---|
{_steps}

    - 足したもの: {added}
    - holdout: S17 {_row("s17 stacker_average_s17")["holdout_rmsle"]:.5f}
    - `stacker_average_s19`: {_change("stacker_average_s19")}
    - `stacker_average_s19_with_block`: {_change("stacker_average_s19_with_block")}
    - 2段目に 2024Q3〜Q4 を足した効果だけ: {block_change[0]:+.5f}（95% の範囲 {block_change[1]:+.5f}〜{block_change[2]:+.5f}）
    - 重み（非負の重み付き平均）: {s19_weights}
    """)
    return


if __name__ == "__main__":
    app.run()
