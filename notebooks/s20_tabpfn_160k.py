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

    from kgl import cv, ensemble, feature_set, incontext, lesson, mlit, stacking
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, validate_submission, write_submission
    from kgl.metrics import get_metric

    return (
        TabPFNRegressor,
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
        incontext,
        lesson,
        mlit,
        mo,
        np,
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
    # S20. TabPFN の文脈を16万行にする

    S17 では、TabPFN の文脈を直近5万行から12万行に増やすと、単独で 0.1948 → 0.1886 になりました。
    S17 の前の試行では、8万行 → 12万行でもまだ良くなっていて、頭打ちになっていません（#49）。

    はじめは学習データの全行（test では 202,323 行）を文脈にする予定でした。しかし全行は、1回に 1,000 行しか予測しなくても
    8GB の GPU のメモリが足りませんでした。16万行なら、1回 4,000 行の予測でピーク 7.27GiB とぎりぎり収まります。
    そこで、2024Q3〜Q4 の評価、holdout、test のどれでも、直近16万行を文脈にします。2024Q3〜Q4 の評価では、学習側の 162,701 行のほぼ全部です。

    TabPFN は呼び出しのたびに文脈の全行を読み直すので、1回 4,000 行の予測に約24分かかります。
    2024Q3〜Q4 で約2時間、holdout と test を合わせて約5時間半です。

    進め方は S17 と同じです。まず 2024Q3〜Q4 で、S17 のメンバーのスタッキングの中の TabPFN を入れ替えた場合と、2本目として足した場合を比べます。
    どちらも良くならなければ、holdout と test は予測しません。
    内側のスタッキングには、S19 で作った S13 のメンバー自身の 2024Q3〜Q4 の予測を使います。
    """)
    return


@app.cell
def _(dataclass, field, psutil):
    THREADS = psutil.cpu_count(logical=False)

    @dataclass
    class StackerLgbParams:
        # S12-S19's LightGBM stacker
        objective: str = "regression"
        learning_rate: float = 0.03
        num_leaves: int = 15
        min_data_in_leaf: int = 100
        feature_fraction: float = 0.9
        seed: int = 42
        verbose: int = -1
        num_threads: int = THREADS

    # More than 50,000 context rows need the memory options and lifting the official limit, as in S17.
    # Probes on 2026-10-18, test setting: every row (202,323) ran out of the 8 GB GPU even at 1,000 rows per call;
    # 160,000 rows fit at 4,000 rows per call (peak 7.27 GiB, 1,458 s per call).
    LARGE = {"fit_mode": "low_memory", "memory_saving_mode": True, "ignore_pretraining_limits": True}

    def _member_runs():
        # S17's members and where their holdout and test predictions are, in S17's order
        return {
            "lightgbm": ["s10_final", "lightgbm_seed_avg"],
            "catboost": ["s10_final", "catboost_long"],
            "xgboost": ["s6_gbdt", "xgboost"],
            "ridge": ["s7_mlp_classic", "ridge"],
            "random_forest": ["s7_mlp_classic", "random_forest"],
            "mlp": ["s7_mlp_classic", "mlp"],
            "tabpfn": ["s17_tabpfn_context", "tabpfn_context_large"],
            "tabpfn_by_municipality": ["s17_tabpfn_context", "tabpfn_by_municipality"],
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
        tabpfn_seed: int = 0
        tabpfn_options: dict = field(default_factory=lambda: dict(LARGE))
        context_rows: int = 160000
        predict_chunk: int = 4000
        member_runs: dict = field(default_factory=_member_runs)
        s17_stage: str = "s17_tabpfn_context"
        s19_stage: str = "s19_tabpfn_groups"
        stacking_alpha: float = 1.0
        stack_mlp_hidden: tuple[int, ...] = (32, 16)
        stack_mlp_alpha: float = 1e-3
        stack_mlp_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
        stack_lgbm_max_rounds: int = 2000
        stack_lgbm_early_stopping_rounds: int = 100
        inner_valid_share: float = 0.2
        stack_lgbm: StackerLgbParams = field(default_factory=StackerLgbParams)

    cfg = Config()
    EXP = "s20_tabpfn_160k"
    BASE = "s19_tabpfn_groups"
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

    # Holdout setting (S13-S19): train up to 2024Q4, validate on 2025Q1-Q2 (fold 1).
    tr_idx, va_idx = cv.split(train_folded["fold"].to_numpy(), 1, time=True)
    fs = feature_set.build(train_folded, test_raw, tr_idx, va_idx, **_build)
    y_real, y_real_refit = fs.y_log - fs.offset, fs.y_log - fs.offset_refit
    y_holdout = fs.y_log[va_idx]
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    in_q1 = train_folded[va_idx].select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy() == 2025.0

    # Inner setting (S14): rows up to 2024Q4; context up to 2024Q2, score 2024Q3-Q4.
    inner_raw = train_folded.filter(pl.col("fold") <= 0)
    tr_in, va_in = cv.split(inner_raw["fold"].to_numpy(), 0, time=True)
    fs_in = feature_set.build(inner_raw, test_raw, tr_in, va_in, **_build)
    y_in = fs_in.y_log - fs_in.offset
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
    def tabpfn_160k(X_context, y_context, X_apply):
        """TabPFN's predictions (deflated log scale) for X_apply, with the most recent rows of X_context as context."""
        context = incontext.most_recent(
            np.arange(len(X_context)), X_context["trade_period"].to_numpy(), cfg.context_rows
        )
        model = TabPFNRegressor(device="cuda", random_state=cfg.tabpfn_seed, **cfg.tabpfn_options)
        model.fit(X_context.iloc[context], y_context[context])
        return incontext.predict_in_chunks(model, X_apply, cfg.predict_chunk)

    return (tabpfn_160k,)


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
    ## 1. 内側の分割で比べる

    2024Q2 までの直近16万行を文脈にして 2024Q3〜Q4 を予測します。予測はファイルに保存します。
    S17 のメンバーの 2024Q3〜Q4 の予測（S19 で作った S13 のメンバー自身の予測と、S17 の2本の TabPFN）でスタッキングを組み、
    12万行の TabPFN を16万行の TabPFN に入れ替えた場合と、2本目の全体の TabPFN として足した場合を比べます。
    """)
    return


@app.cell
def _(EXP, block_ids, cfg, env, fs_in, pl, tabpfn_160k, tr_in, va_in, y_in):
    (env.output_dir / EXP).mkdir(parents=True, exist_ok=True)
    _path = env.output_dir / EXP / "inner_context_160k.parquet"
    if not _path.exists():
        _pred = tabpfn_160k(fs_in.X.iloc[tr_in], y_in[tr_in], fs_in.X.iloc[va_in])
        pl.DataFrame({cfg.id_col: block_ids, "pred": _pred + fs_in.offset[va_in]}).write_parquet(_path)
    _saved = pl.read_parquet(_path)
    assert _saved[cfg.id_col].to_list() == block_ids
    inner_160k = _saved["pred"].to_numpy()
    return (inner_160k,)


@app.cell
def _(block_ids, cfg, env, pl):
    def _s17_inner(variant):
        _saved = pl.read_parquet(env.output_dir / cfg.s17_stage / f"inner_{variant}.parquet")
        assert _saved[cfg.id_col].to_list() == block_ids
        return _saved["pred"].to_numpy()

    _block = pl.read_parquet(env.output_dir / cfg.s19_stage / "block_predictions.parquet")
    assert _block[cfg.id_col].to_list() == block_ids
    block_members = {_name: _block[_name].to_numpy() for _name in list(cfg.member_runs)[:6]}
    block_members["tabpfn"] = _s17_inner("context_large")
    block_members["tabpfn_by_municipality"] = _s17_inner("by_municipality")
    assert list(block_members) == list(cfg.member_runs)
    return (block_members,)


@app.cell
def _(
    block_members,
    found_stackers,
    fs_in,
    holdout_rmsle,
    in_q3,
    inner_160k,
    np,
    pl,
    stacking,
    va_in,
    y_block,
):
    _rows = stacking.row_features(fs_in.X.iloc[va_in])

    def block_stack(members):
        """The stacker average on 2024Q3-Q4, fit on Q3 and scored on Q4 and the other way round."""
        H = np.column_stack(list(members.values()))
        return holdout_rmsle(
            stacking.stacker_average(H, H, _rows, _rows, y_block, in_q3, found_stackers(len(members)))[0], y_block
        )

    _candidates = {
        "S17 members": block_members,
        "TabPFN replaced by 160,000 rows": block_members | {"tabpfn": inner_160k},
        "160,000-row TabPFN added": block_members | {"tabpfn_160k": inner_160k},
    }
    inner_table = pl.DataFrame(
        [{"stack": _name, "inner stacker average": block_stack(_members)} for _name, _members in _candidates.items()]
    )
    _scores = dict(zip(inner_table["stack"], inner_table["inner stacker average"], strict=True))
    _best = min(_scores, key=lambda _n: _scores[_n])
    use = {"S17 members": None, "TabPFN replaced by 160,000 rows": "replace", "160,000-row TabPFN added": "add"}[_best]
    inner_alone = {
        "120,000 rows (S17)": holdout_rmsle(block_members["tabpfn"], y_block),
        "160,000 rows": holdout_rmsle(inner_160k, y_block),
    }
    inner_table
    return inner_alone, inner_table, use


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. holdout と test で確かめる

    1 で入れ替えか追加が選ばれたときだけ、2024Q4 までの直近16万行を文脈にして holdout を、全期間の直近16万行を文脈にして test を予測します。
    そのうえで S17 と同じ手順でスタッキングを組み直し、S17 の `stacker_average_s17` と比べます。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    bootstrap_diff,
    cfg,
    ensemble,
    env,
    found_stackers,
    fs,
    git_commit,
    holdout_ids,
    holdout_rmsle,
    in_q1,
    lesson,
    np,
    pl,
    sample,
    stacking,
    tabpfn_160k,
    tr_idx,
    use,
    va_idx,
    validate_submission,
    write_submission,
    y_holdout,
    y_real,
    y_real_refit,
):
    def saved(stage, run):
        """Holdout and test predictions of a saved run, on the log scale."""
        oof = pl.read_parquet(env.output_dir / stage / run / "oof.parquet")
        sub = pl.read_csv(env.output_dir / stage / run / "submission.csv")
        assert oof["id"].to_list() == holdout_ids["id"].to_list() and sub["id"].to_list() == sample["id"].to_list()
        return np.log1p(oof["pred"].to_numpy()), np.log1p(sub["price"].to_numpy())

    _rows_h, _rows_t = stacking.row_features(fs.X.iloc[va_idx]), stacking.row_features(fs.X_test)

    def stacker_average(members):
        H = np.column_stack([m[0] for m in members.values()])
        T = np.column_stack([m[1] for m in members.values()])
        return stacking.stacker_average(H, T, _rows_h, _rows_t, y_holdout, in_q1, found_stackers(len(members)))

    s17_members = {_name: saved(*_run) for _name, _run in cfg.member_runs.items()}
    s17_average = saved(cfg.s17_stage, "stacker_average_s17")
    _reproduced = stacker_average(s17_members)
    assert np.allclose(_reproduced[0], s17_average[0], atol=1e-9)
    assert np.allclose(_reproduced[1], s17_average[1], atol=1e-9)

    holdout_table = None
    s20_weights = None
    if use is not None:
        commit = git_commit(env.root)
        _path_holdout = env.output_dir / EXP / "holdout_context_160k.parquet"
        if not _path_holdout.exists():
            _pred = tabpfn_160k(fs.X.iloc[tr_idx], y_real[tr_idx], fs.X.iloc[va_idx])
            pl.DataFrame({cfg.id_col: holdout_ids[cfg.id_col], "pred": _pred + fs.offset[va_idx]}).write_parquet(
                _path_holdout
            )
        _path_test = env.output_dir / EXP / "test_context_160k.parquet"
        if not _path_test.exists():
            _pred = tabpfn_160k(fs.X_refit, y_real_refit, fs.X_test)
            pl.DataFrame({cfg.id_col: sample[cfg.id_col], "pred": _pred + fs.offset_test}).write_parquet(_path_test)
        _holdout, _test = pl.read_parquet(_path_holdout), pl.read_parquet(_path_test)
        assert _holdout[cfg.id_col].to_list() == holdout_ids[cfg.id_col].to_list()
        assert _test[cfg.id_col].to_list() == sample[cfg.id_col].to_list()
        context_160k = (_holdout["pred"].to_numpy(), _test["pred"].to_numpy())
        _out = lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run="tabpfn_160k",
            score=holdout_rmsle(context_160k[0], y_holdout),
            metric=cfg.metric,
            config=asdict(cfg) | {"model": "tabpfn-3.5", "context": "most recent 160,000 rows"},
            holdout=holdout_ids,
            pred=np.expm1(context_160k[0]),
            notes="TabPFN-3.5 with the most recent 160,000 rows as context (#49)",
            commit=commit,
        )
        _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(context_160k[1])})
        validate_submission(_submission, sample, cfg.id_col)
        write_submission(_out, _submission)

        s20_members = dict(s17_members)
        if use == "replace":
            s20_members["tabpfn"] = context_160k
        else:
            s20_members["tabpfn_160k"] = context_160k
        restacked = stacker_average(s20_members)
        _restack_change = bootstrap_diff((restacked[0] - y_holdout) ** 2, (s17_average[0] - y_holdout) ** 2)
        _out = lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run="stacker_average_s20",
            score=holdout_rmsle(restacked[0], y_holdout),
            metric=cfg.metric,
            config=asdict(cfg) | {"members": list(s20_members), "use": use},
            holdout=holdout_ids,
            pred=np.expm1(restacked[0]),
            notes=f"S17 stacker average with the 160,000-row TabPFN ({use}) (#49)",
            commit=commit,
        )
        _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(restacked[1])})
        validate_submission(_submission, sample, cfg.id_col)
        write_submission(_out, _submission)

        _alone_change = bootstrap_diff((context_160k[0] - y_holdout) ** 2, (s17_members["tabpfn"][0] - y_holdout) ** 2)
        _weights = ensemble.fit_weights(np.column_stack([m[0] for m in s20_members.values()]), y_holdout)
        s20_weights = dict(zip(s20_members, np.round(_weights, 3).tolist(), strict=True))
        holdout_table = pl.DataFrame(
            [
                {
                    "run": "s17 tabpfn_context_large",
                    "holdout_rmsle": holdout_rmsle(s17_members["tabpfn"][0], y_holdout),
                },
                {
                    "run": "tabpfn_160k",
                    "holdout_rmsle": holdout_rmsle(context_160k[0], y_holdout),
                    "change": _alone_change[0],
                    "ci_low": _alone_change[1],
                    "ci_high": _alone_change[2],
                },
                {"run": "s17 stacker_average_s17", "holdout_rmsle": holdout_rmsle(s17_average[0], y_holdout)},
                {
                    "run": "stacker_average_s20",
                    "holdout_rmsle": holdout_rmsle(restacked[0], y_holdout),
                    "change": _restack_change[0],
                    "ci_low": _restack_change[1],
                    "ci_high": _restack_change[2],
                },
            ],
            infer_schema_length=None,
        )
    holdout_table
    return holdout_table, s20_weights


@app.cell
def _(holdout_table, inner_alone, inner_table, mo, s20_weights, use):
    _inner = "\n".join(
        f"    | {r['stack']} | {r['inner stacker average']:.5f} |" for r in inner_table.iter_rows(named=True)
    )
    if holdout_table is None:
        _holdout = "    内側で良くならなかったので、holdout と test は予測していません。"
    else:
        _holdout = "\n".join(
            f"    - `{r['run']}`: {r['holdout_rmsle']:.5f}"
            + (
                ""
                if r["change"] is None
                else f"（差 {r['change']:+.5f}、95% の範囲 {r['ci_low']:+.5f}〜{r['ci_high']:+.5f}）"
            )
            for r in holdout_table.iter_rows(named=True)
        )
        _holdout += f"\n    - 重み（非負の重み付き平均）: {s20_weights}"
    mo.md(f"""
    ## 3. 結果

    2024Q3〜Q4 の TabPFN 単独: 12万行 {inner_alone["120,000 rows (S17)"]:.4f}、16万行 {inner_alone["160,000 rows"]:.4f}

    | スタッキング（2024Q3〜Q4） | スタッキングの平均 |
    |---|---|
{_inner}

    選んだもの: {use}

    holdout（2025Q1〜Q2）:

{_holdout}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 結果を読む

    2026-10-18〜19 の実行の結果です。予測にかかった時間は、2024Q3〜Q4 が約2時間、holdout が約2時間半、test が約3時間でした。

    - 2024Q3〜Q4 では、TabPFN 単独が 12万行 0.1830 から 16万行 0.1816 になりました。
      スタッキングの平均は、S17 のメンバー 0.18246 に対して、入れ替えが 0.18128、追加が 0.18117 で、追加を選びました。
    - holdout でも同じ向きです。TabPFN 単独は 0.18864 → 0.18674（差 -0.0019、95% の範囲 -0.0024〜-0.0014）です。
      スタッキングの平均は 0.18697 → 0.18584（差 -0.00113、範囲 -0.00147〜-0.00079）で、範囲が 0 をまたぎません。
      2024Q3〜Q4 での差（-0.0013）が、ほぼそのまま holdout にも出ました。
    - 改善は一部の行に偏っていません。2025Q1（0.18634 → 0.18509）と Q2（0.18760 → 0.18659）、
      不動産取引価格情報（0.21042 → 0.20911）と成約価格情報（0.17100 → 0.16999）の、どれでも良くなりました。
    - 12万行と16万行の TabPFN の誤差の相関は 0.991 です。外し方はほぼ同じで、16万行のほうが少しずつ正確です。
      非負の重み付き平均の重みも、16万行 0.823、12万行 0 で、12万行は16万行に置き換わりました。
      「追加」を選びましたが、中身は「入れ替え」とほとんど同じです（2024Q3〜Q4 での差 0.00011）。
    - 文脈の行数と TabPFN 単独の holdout は、5万行 0.1948、12万行 0.1886、16万行 0.1867 です。
      1万行あたりの改善は小さくなってきましたが、まだ止まっていません。8GB の GPU では、これ以上は増やせませんでした。
    - test の予測は、S17 より平均で 0.12% 低くなりました（log の差の平均 -0.0012）。holdout での差の平均は +0.0002 です。
      test の文脈には 2025Q1〜Q2 の行が入るので、holdout とは文脈の期間が違います。この差は CV では確かめられず、LB で確かめます。

    S20 を採用します。S17 のスタッキングの平均に、16万行の文脈の TabPFN を足したものが、次の提出の候補です。
    """)
    return


if __name__ == "__main__":
    app.run()
