import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import itertools
    from dataclasses import asdict, dataclass, field

    import marimo as mo
    import numpy as np
    import polars as pl
    import psutil
    from sklearn.neighbors import NearestNeighbors

    from kgl import classic, cv, ensemble, feature_set, lesson, mlit, stacking
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, read_metrics, validate_submission, write_submission
    from kgl.metrics import get_metric

    return (
        NearestNeighbors,
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
        itertools,
        lesson,
        mlit,
        mo,
        np,
        pl,
        psutil,
        read_metrics,
        stacking,
        validate_submission,
        write_submission,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S18. 似た物件の取引から当てる

    S13 のスタッキングでは、holdout の 62% の行で7メンバー全員が同じ向きに外していて、その行にスタッキングの平均の二乗誤差の 95% が集まっていました。
    全員が同じ向きに外した行は、重みをどう変えても打ち消せません。S14 と S16 でメンバーを良くしても動かなかったのはこのためです。
    スタッキングを良くするには、今のメンバーと違う外し方をするメンバーが要ります（#46）。

    この段階では、不動産鑑定の「取引事例比較法」をまねたメンバーを作ります。考え方は単純です。

    1. 価格を比べられる形にする。S4 の価格指数で実質化した log ㎡単価にすれば、違う四半期・違う広さの取引どうしを比べられます。
    2. 同じ地区の過去の取引のうち、建築年・広さ・駅からの分・取引時期・価格情報区分（成約か取引か）が近いものを k 件選ぶ。
       地区に k 件ない行は、同じ駅、同じ市区町村、東京全体の順に広げます。
    3. その k 件の実質 log ㎡単価の平均に、その行の log 面積と価格指数を足して予測にする。

    木のモデルや TabPFN は、地区の情報を target encoding や分岐でまとめて学びます。この方法は、その行のすぐ近くの実際の取引をそのまま使います。
    単独では弱くても、外し方が違えばスタッキングで重みがつくはずです。

    k と、距離の測り方（どの列をどれだけ重く見るか）は、2024Q3〜Q4 の内側の分割で選びます。holdout は確かめに1回だけ使います。
    """)
    return


@app.cell
def _(dataclass, field, psutil):
    THREADS = psutil.cpu_count(logical=False)

    @dataclass
    class StackerLgbParams:
        # S12-S17's LightGBM stacker
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
        # From the narrowest group to the widest: a row uses the first one with at least k past sales.
        levels: tuple[str, ...] = ("district", "station", "municipality", "tokyo")
        # The distance columns, in the order of the weights below.
        distance_columns: tuple[str, ...] = (
            "build_year",
            "log_area",
            "station_minutes",
            "trade_period",
            "contract",
        )
        # A first grid on the inner split (2026-10-18, k 10-40, build-year weight 1-2) scored 0.25-0.31 alone;
        # fewer neighbors and a heavier build year (the same building, in practice) did better, hence this grid.
        grid_k: tuple[int, ...] = (5, 10)
        grid_weights: tuple[tuple[float, ...], ...] = (
            (2.0, 1.0, 0.5, 0.5, 1.0),
            (4.0, 1.0, 1.0, 0.5, 0.5),
            (4.0, 1.0, 0.5, 0.25, 0.5),
            (8.0, 1.0, 0.5, 0.5, 0.5),
        )
        # The residual check of section 4: the past sales whose errors are compared.
        residual_members: tuple[str, ...] = ("tabpfn", "catboost", "lightgbm")
        residual_k: tuple[int, ...] = (1, 3, 10)
        block_stage: str = "s15_more_stack_rows"
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
    EXP = "s18_comparable_sales"
    BASE = "s17_tabpfn_context"
    return BASE, EXP, cfg


@app.cell
def _(classic, np, pl):
    def comparable_inputs(df, X):
        """The distance columns (build year, log area, minutes, period, contract) and the group keys of each row."""
        extras = classic.numeric_extras(df)
        numbers = np.column_stack(
            [
                extras["build_year_num"].to_numpy(),
                extras["log_area"].to_numpy(),
                X["station_minutes"].to_numpy(dtype=float),
                X["trade_period"].to_numpy(dtype=float),
                (X["価格情報区分"] == "成約価格情報").to_numpy(dtype=float),
            ]
        )
        keys = {
            "district": (df["市区町村名"] + "|" + df["地区名"]).to_numpy(),
            "station": df["最寄駅：名称"].fill_null("").to_numpy(),
            "municipality": df["市区町村名"].to_numpy(),
            "tokyo": np.full(df.height, "tokyo"),
        }
        return numbers, keys, extras["log_area"].to_numpy()

    def groups_of(keys):
        """Row positions per key."""
        frame = pl.DataFrame({"key": keys, "row": np.arange(len(keys))}).group_by("key").agg("row")
        return dict(zip(frame["key"].to_list(), frame["row"].to_list(), strict=True))

    return comparable_inputs, groups_of


@app.cell
def _(NearestNeighbors, cfg, groups_of, np):
    def comparable_sales(fit, apply, target, k, weights):
        """For each apply row, the mean target of its k nearest fit rows in the narrowest group that has k of them.

        `fit` and `apply` are (numbers, keys) pairs from `comparable_inputs`. Missing numbers take the fit rows'
        median; every column is divided by its standard deviation over the fit rows, then multiplied by its weight.
        """
        (fit_numbers, fit_keys), (apply_numbers, apply_keys) = fit, apply
        median = np.nanmedian(fit_numbers, axis=0)
        scale = np.nanstd(fit_numbers, axis=0)
        assert (scale > 0).all()
        Z_fit = np.where(np.isnan(fit_numbers), median, fit_numbers) / scale * np.asarray(weights)
        Z_apply = np.where(np.isnan(apply_numbers), median, apply_numbers) / scale * np.asarray(weights)
        pred = np.full(len(apply_numbers), np.nan)
        for level in cfg.levels:
            candidates = groups_of(fit_keys[level])
            open_rows = np.flatnonzero(np.isnan(pred))
            for key, rows in groups_of(apply_keys[level][open_rows]).items():
                pool = np.asarray(candidates.get(key, []))
                if key == "" or len(pool) < k:
                    continue
                at = open_rows[np.asarray(rows)]
                _, nearest = NearestNeighbors(n_neighbors=k).fit(Z_fit[pool]).kneighbors(Z_apply[at])
                pred[at] = target[pool[nearest]].mean(axis=1)
        assert not np.isnan(pred).any()
        return pred

    return (comparable_sales,)


@app.cell
def _(cfg, comp_dir, comparable_inputs, cv, detect, feature_set, mlit, pl):
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

    # Holdout setting (S13-S17): sales up to 2024Q4, validate on 2025Q1-Q2 (fold 1).
    tr_idx, va_idx = cv.split(train_folded["fold"].to_numpy(), 1, time=True)
    fs = feature_set.build(train_folded, test_raw, tr_idx, va_idx, **_build)
    y_holdout = fs.y_log[va_idx]
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    in_q1 = train_folded[va_idx].select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy() == 2025.0
    numbers, keys, log_area = comparable_inputs(train_folded, fs.X)
    numbers_test, keys_test, log_area_test = comparable_inputs(test_raw, fs.X_test)

    # Inner setting (S14): rows up to 2024Q4; sales up to 2024Q2, score 2024Q3-Q4.
    inner_raw = train_folded.filter(pl.col("fold") <= 0)
    tr_in, va_in = cv.split(inner_raw["fold"].to_numpy(), 0, time=True)
    fs_in = feature_set.build(inner_raw, test_raw, tr_in, va_in, **_build)
    numbers_in, keys_in, log_area_in = comparable_inputs(inner_raw, fs_in.X)
    y_block = fs_in.y_log[va_in]
    block_ids = inner_raw[va_in][cfg.id_col].to_list()
    in_q3 = inner_raw[va_in].select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy() == 2024.5
    return (
        block_ids,
        env,
        fs,
        fs_in,
        holdout_ids,
        in_q1,
        in_q3,
        keys,
        keys_in,
        keys_test,
        log_area,
        log_area_in,
        log_area_test,
        numbers,
        numbers_in,
        numbers_test,
        sample,
        tr_idx,
        tr_in,
        train_folded,
        va_idx,
        va_in,
        y_block,
        y_holdout,
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
    ## 1. 内側の分割で k と距離を選ぶ

    2024Q2 までの取引から 2024Q3〜Q4 を当てます。距離の重みは、建築年・log 面積・駅からの分・取引時期・価格情報区分の順です。
    単独のスコアだけでなく、S17 で選んだメンバーの 2024Q3〜Q4 の予測に足してスタッキングを組み、その良さで選びます。
    メンバーの予測は S15 で作ったもので、TabPFN だけ S17 で選んだ文脈のものに替えます。
    欲しいのは「単独で強いメンバー」ではなく「スタッキングに新しい情報を持ち込むメンバー」だからです。
    """)
    return


@app.cell
def _(
    block_ids,
    cfg,
    comparable_sales,
    env,
    fs_in,
    found_stackers,
    holdout_rmsle,
    in_q3,
    itertools,
    keys_in,
    log_area_in,
    np,
    numbers_in,
    pl,
    read_metrics,
    stacking,
    tr_in,
    va_in,
    y_block,
):
    _block = pl.read_parquet(env.output_dir / cfg.block_stage / "block_predictions.parquet")
    assert _block[cfg.id_col].to_list() == block_ids
    block_members = {c: _block[c].to_numpy() for c in _block.columns if c not in (cfg.id_col, cfg.target, "fold")}

    def _s17_inner(variant):
        _saved = pl.read_parquet(env.output_dir / cfg.s17_stage / f"inner_{variant}.parquet")
        assert _saved[cfg.id_col].to_list() == block_ids
        return _saved["pred"].to_numpy()

    # S17's members on 2024Q3-Q4: S15's predictions with TabPFN swapped for the context S17 chose.
    _s17 = read_metrics(env.output_dir / cfg.s17_stage / "stacker_average_s17" / "metrics.json").config
    if _s17["replacement"] != "s13":
        block_members["tabpfn"] = _s17_inner(_s17["replacement"])
    if _s17["addition"] is not None:
        block_members[f"tabpfn_{_s17['addition']}"] = _s17_inner(_s17["addition"])
    assert list(block_members) == _s17["members"]
    _rows = stacking.row_features(fs_in.X.iloc[va_in])

    def block_stack(members):
        """The stacker average on 2024Q3-Q4, fit on Q3 and scored on Q4 and the other way round."""
        H = np.column_stack(list(members.values()))
        return stacking.stacker_average(H, H, _rows, _rows, y_block, in_q3, found_stackers(len(members)))[0]

    _target = fs_in.y_log - fs_in.offset - log_area_in
    _fit = (numbers_in[tr_in], {k: v[tr_in] for k, v in keys_in.items()})
    _apply = (numbers_in[va_in], {k: v[va_in] for k, v in keys_in.items()})
    _table = [
        {
            "k": None,
            "weights": "(no comparable sales)",
            "alone": None,
            "stack": holdout_rmsle(block_stack(block_members), y_block),
        }
    ]
    for _k, _w in itertools.product(cfg.grid_k, cfg.grid_weights):
        _pred = comparable_sales(_fit, _apply, _target[tr_in], _k, _w) + log_area_in[va_in] + fs_in.offset[va_in]
        _table.append(
            {
                "k": _k,
                "weights": str(_w),
                "alone": holdout_rmsle(_pred, y_block),
                "stack": holdout_rmsle(block_stack(block_members | {"comparable": _pred}), y_block),
            }
        )
    grid_table = pl.DataFrame(_table, infer_schema_length=None)
    _best = grid_table.filter(pl.col("k").is_not_null()).sort("stack").row(0, named=True)
    chosen_k, chosen_weights = _best["k"], cfg.grid_weights[[str(w) for w in cfg.grid_weights].index(_best["weights"])]
    grid_table
    return chosen_k, chosen_weights, grid_table


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. holdout と test を予測する

    選んだ k と重みで、2024Q4 までの取引から holdout を、全期間の取引から test を当てます。
    test の価格指数は、S4 と同じく全期間で作った指数を外挿したものです。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    cfg,
    chosen_k,
    chosen_weights,
    comparable_sales,
    env,
    fs,
    git_commit,
    holdout_ids,
    holdout_rmsle,
    keys,
    keys_test,
    lesson,
    log_area,
    log_area_test,
    np,
    numbers,
    numbers_test,
    pl,
    sample,
    tr_idx,
    va_idx,
    validate_submission,
    write_submission,
    y_holdout,
):
    commit = git_commit(env.root)
    _target = fs.y_log - fs.offset - log_area
    _fit = (numbers[tr_idx], {k: v[tr_idx] for k, v in keys.items()})
    _apply = (numbers[va_idx], {k: v[va_idx] for k, v in keys.items()})
    comparable_holdout = (
        comparable_sales(_fit, _apply, _target[tr_idx], chosen_k, chosen_weights) + log_area[va_idx] + fs.offset[va_idx]
    )
    _target_refit = fs.y_log - fs.offset_refit - log_area
    comparable_test = (
        comparable_sales((numbers, keys), (numbers_test, keys_test), _target_refit, chosen_k, chosen_weights)
        + log_area_test
        + fs.offset_test
    )
    comparable_score = holdout_rmsle(comparable_holdout, y_holdout)
    _out = lesson.record_run(
        env,
        exp=EXP,
        base=BASE,
        run="comparable_sales",
        score=comparable_score,
        metric=cfg.metric,
        config=asdict(cfg) | {"model": "comparable_sales", "k": chosen_k, "weights": list(chosen_weights)},
        holdout=holdout_ids,
        pred=np.expm1(comparable_holdout),
        notes=f"Comparable sales, k={chosen_k}, weights {list(chosen_weights)} (#46)",
        commit=commit,
    )
    _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(comparable_test)})
    validate_submission(_submission, sample, cfg.id_col)
    write_submission(_out, _submission)
    comparable_score
    return commit, comparable_holdout, comparable_score, comparable_test


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. スタッキングに足す

    S17 で選んだメンバーに足して、同じ手順でスタッキングを組み直し、S17 の `stacker_average_s17` と比べます。
    まず S17 の結果が再現できることを確かめます。
    """)
    return


@app.cell
def _(cfg, env, holdout_ids, np, pl, read_metrics, sample):
    def saved(stage, run):
        """Holdout and test predictions of a saved run, on the log scale."""
        oof = pl.read_parquet(env.output_dir / stage / run / "oof.parquet")
        sub = pl.read_csv(env.output_dir / stage / run / "submission.csv")
        assert oof["id"].to_list() == holdout_ids["id"].to_list() and sub["id"].to_list() == sample["id"].to_list()
        return np.log1p(oof["pred"].to_numpy()), np.log1p(sub["price"].to_numpy())

    s17_members = {
        "lightgbm": saved("s10_final", "lightgbm_seed_avg"),
        "catboost": saved("s10_final", "catboost_long"),
        "xgboost": saved("s6_gbdt", "xgboost"),
        "ridge": saved("s7_mlp_classic", "ridge"),
        "random_forest": saved("s7_mlp_classic", "random_forest"),
        "mlp": saved("s7_mlp_classic", "mlp"),
        "tabpfn": saved("s13_tabular_dl", "tabpfn"),
    }
    _s17 = read_metrics(env.output_dir / cfg.s17_stage / "stacker_average_s17" / "metrics.json").config
    if _s17["replacement"] != "s13":
        s17_members["tabpfn"] = saved(cfg.s17_stage, f"tabpfn_{_s17['replacement']}")
    if _s17["addition"] is not None:
        s17_members[f"tabpfn_{_s17['addition']}"] = saved(cfg.s17_stage, f"tabpfn_{_s17['addition']}")
    assert list(s17_members) == _s17["members"]
    s17_average = saved(cfg.s17_stage, "stacker_average_s17")
    return s17_average, s17_members


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    bootstrap_diff,
    cfg,
    commit,
    comparable_holdout,
    comparable_test,
    ensemble,
    env,
    found_stackers,
    fs,
    holdout_ids,
    holdout_rmsle,
    in_q1,
    lesson,
    np,
    pl,
    s17_average,
    s17_members,
    sample,
    stacking,
    va_idx,
    validate_submission,
    write_submission,
    y_holdout,
):
    _rows_h, _rows_t = stacking.row_features(fs.X.iloc[va_idx]), stacking.row_features(fs.X_test)

    def stacker_average(members):
        H = np.column_stack([m[0] for m in members.values()])
        T = np.column_stack([m[1] for m in members.values()])
        return stacking.stacker_average(H, T, _rows_h, _rows_t, y_holdout, in_q1, found_stackers(len(members)))

    _reproduced = stacker_average(s17_members)
    assert np.allclose(_reproduced[0], s17_average[0], atol=1e-9)
    assert np.allclose(_reproduced[1], s17_average[1], atol=1e-9)

    members_with_comparable = s17_members | {"comparable_sales": (comparable_holdout, comparable_test)}
    restacked = stacker_average(members_with_comparable)
    restack_change = bootstrap_diff((restacked[0] - y_holdout) ** 2, (s17_average[0] - y_holdout) ** 2)
    restack_score = holdout_rmsle(restacked[0], y_holdout)
    _out = lesson.record_run(
        env,
        exp=EXP,
        base=BASE,
        run="stacker_average_with_comparable",
        score=restack_score,
        metric=cfg.metric,
        config=asdict(cfg) | {"members": list(members_with_comparable)},
        holdout=holdout_ids,
        pred=np.expm1(restacked[0]),
        notes="S17 stacker average with comparable sales added (#46)",
        commit=commit,
    )
    _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(restacked[1])})
    validate_submission(_submission, sample, cfg.id_col)
    write_submission(_out, _submission)

    _H = np.column_stack([m[0] for m in members_with_comparable.values()])
    comparable_weight = float(
        ensemble.fit_weights(_H, y_holdout)[list(members_with_comparable).index("comparable_sales")]
    )
    _residuals = {n: m[0] - y_holdout for n, m in members_with_comparable.items()}
    comparable_correlation = {
        n: float(np.corrcoef(_residuals["comparable_sales"], r)[0, 1])
        for n, r in _residuals.items()
        if n != "comparable_sales"
    }
    pl.DataFrame(
        [
            {"run": "s17 stacker_average_s17", "holdout_rmsle": holdout_rmsle(s17_average[0], y_holdout)},
            {
                "run": "stacker_average_with_comparable",
                "holdout_rmsle": restack_score,
                "change_vs_s17": restack_change[0],
                "ci_low": restack_change[1],
                "ci_high": restack_change[2],
                "comparable weight": comparable_weight,
            },
        ],
        infer_schema_length=None,
    )
    return comparable_correlation, comparable_weight, restack_change, restack_score


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 近い取引で、メンバーは同じように外しているか

    似た物件の価格を足しても効かないなら、メンバーはその情報をもう持っていることになります。これを、外れ方（予測 − 正解）で直接確かめます。
    S15 で、2024Q2 までで学習したメンバーが 2024Q3〜Q4 を予測しています。holdout の各行について、2024Q3〜Q4 の近い取引（1節と同じ距離）を
    k 件選び、その外れ方の平均と、S17 のスタッキングの外れ方の相関を測ります。
    同じ建物の物件をメンバーがいつも高めに（低めに）当てているなら、相関は正になり、2段目でその分を直せるはずです。
    """)
    return


@app.cell
def _(
    cfg,
    chosen_weights,
    comparable_sales,
    env,
    keys,
    np,
    numbers,
    pl,
    s17_average,
    train_folded,
    va_idx,
    y_holdout,
    fs,
):
    _block = pl.read_parquet(env.output_dir / cfg.block_stage / "block_predictions.parquet")
    _block_rows = np.flatnonzero(train_folded["fold"].to_numpy() == 0)
    assert _block[cfg.id_col].to_list() == train_folded[_block_rows][cfg.id_col].to_list()
    _block_residual = np.mean([_block[m].to_numpy() for m in cfg.residual_members], axis=0) - fs.y_log[_block_rows]
    _stack_residual = s17_average[0] - y_holdout
    _fit = (numbers[_block_rows], {k: v[_block_rows] for k, v in keys.items()})
    _apply = (numbers[va_idx], {k: v[va_idx] for k, v in keys.items()})
    residual_table = pl.DataFrame(
        [
            {
                "k": _k,
                "corr(neighbors' residual, holdout residual)": float(
                    np.corrcoef(comparable_sales(_fit, _apply, _block_residual, _k, chosen_weights), _stack_residual)[
                        0, 1
                    ]
                ),
            }
            for _k in cfg.residual_k
        ]
    )
    residual_table
    return (residual_table,)


@app.cell
def _(
    chosen_k,
    chosen_weights,
    comparable_correlation,
    comparable_score,
    comparable_weight,
    grid_table,
    mo,
    pl,
    residual_table,
    restack_change,
    restack_score,
):
    _base = grid_table.filter(pl.col("k").is_null())["stack"][0]
    _best = grid_table.filter(pl.col("k").is_not_null()).sort("stack").row(0, named=True)
    _corr = ", ".join(f"k={r['k']} {r[residual_table.columns[1]]:+.3f}" for r in residual_table.iter_rows(named=True))
    mo.md(f"""
    ## 5. 結果

    - 内側の分割（2024Q3〜Q4）: スタッキングの平均は、足す前 {_base:.4f}、足したあと最良 {_best["stack"]:.4f}（k={chosen_k}、重み {list(chosen_weights)}、単独 {_best["alone"]:.4f}）
    - holdout（2025Q1〜Q2）: 単独 {comparable_score:.4f}
    - 誤差の相関: {", ".join(f"{n} {c:.3f}" for n, c in comparable_correlation.items())}
    - スタッキングの平均: {restack_score:.4f}（S17 との差 {restack_change[0]:+.4f}、95% の範囲 {restack_change[1]:+.4f}〜{restack_change[2]:+.4f}）。重みは {comparable_weight:.3f}
    - 近い取引の外れ方と holdout の外れ方の相関: {_corr}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. 結果を読む

    2026-10-18 の実行の結果です。

    - 似た物件の取引の単独のスコアは、holdout で 0.2456 です。S5 の LightGBM（0.2035）にも届きません。
      同じ地区の近い取引の平均を使うだけでは、取引ごとの違いを拾いきれません。
    - ほかのメンバーとの誤差の相関は 0.75〜0.83 で、2本の TabPFN どうし（0.94）より低く、外し方は違います。
      それでも、スタッキングでの重みは 0 でした。外し方が違っても、自分の誤差が大きすぎると、スタッキングは使えません。
    - 内側の分割では、グリッドのどの設定でも、足さないほうが良い結果でした。holdout でも S17 との差は -0.00002 で、偶然の範囲です。
    - 4 で、近い取引の外れ方とスタッキングの外れ方の相関を見ると、ほぼ 0（-0.000〜+0.007）でした。
      近くの取引が持つ価格の情報は、すでにメンバーに入っています。スタッキングが外している部分は取引ごとの事情で、近くの取引からは当てられません。

    S18 は採用しません。メンバーを足して良くなるのは、違う外し方をして、しかも単独でもある程度強いメンバーです（S13 の TabPFN、S17 の市区町村ごとの TabPFN）。
    """)
    return


if __name__ == "__main__":
    app.run()
