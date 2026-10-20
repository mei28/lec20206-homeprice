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
    # S21. TabPFN に㎡単価を当てさせる・価格情報区分ごとに分ける

    S17 と S20 では、スタッキングで重みの一番大きい TabPFN を良くすると、その改善がスタッキングにも LB にも出ました
    （S17 は CV -0.0041、LB -0.0038）。S20 の TabPFN（直近16万行の文脈）は、重み 0.82 のメンバーです。
    この段階では、TabPFN の使い方を2通り変えてみます。

    - 案 A（#50）: TabPFN に価格ではなく ㎡単価（価格 ÷ 面積）を当てさせ、面積を掛けて価格に戻します。
      価格の差の大きな部分は面積で決まります。それを外すと、TabPFN は場所や築年数、時期による違いに集中できます。
    - 案 B（#51）: 価格情報区分（取引当事者へのアンケートか、レインズの成約情報か）ごとに TabPFN を分け、
      それぞれの区分の全行を文脈にします。S19 では文脈が2万行しかなく、単独で弱いメンバーでした。
      文脈が約6万行を超えると TabPFN は行を分けて処理するので、各区分の全行（6.8万〜11万行）でも 8GB の GPU に収まります。

    進め方は S20 と同じです。まず 2024Q3〜Q4 で、S20 のメンバーのスタッキングに足す場合と、16万行の TabPFN と入れ替える場合を比べます。
    どれも良くならなければ、holdout と test は予測しません。
    """)
    return


@app.cell
def _(dataclass, field, psutil):
    THREADS = psutil.cpu_count(logical=False)

    @dataclass
    class StackerLgbParams:
        # S12-S20's LightGBM stacker
        objective: str = "regression"
        learning_rate: float = 0.03
        num_leaves: int = 15
        min_data_in_leaf: int = 100
        feature_fraction: float = 0.9
        seed: int = 42
        verbose: int = -1
        num_threads: int = THREADS

    # More than 50,000 context rows need the memory options and lifting the official limit, as in S17 and S20.
    LARGE = {"fit_mode": "low_memory", "memory_saving_mode": True, "ignore_pretraining_limits": True}

    def _variants():
        # target: "price" is the deflated log price (S13-S20); "ppm2" subtracts log area and adds it back after predicting.
        # group: None is one TabPFN for every row; "price_type" is one TabPFN per 価格情報区分, each with its own context.
        # Probe on 2026-10-20, test setting: every row of a price type fits at 4,000 rows per call
        # (不動産取引価格情報 111,526 rows, 677 s per call; 成約価格情報 90,797 rows, 436 s; peak 6.24 GiB).
        return {
            "ppm2": {"target": "ppm2", "group": None},
            "by_price_type": {"target": "price", "group": "price_type"},
        }

    def _member_runs():
        # S20's members and where their holdout and test predictions are, in S20's order
        return {
            "lightgbm": ["s10_final", "lightgbm_seed_avg"],
            "catboost": ["s10_final", "catboost_long"],
            "xgboost": ["s6_gbdt", "xgboost"],
            "ridge": ["s7_mlp_classic", "ridge"],
            "random_forest": ["s7_mlp_classic", "random_forest"],
            "mlp": ["s7_mlp_classic", "mlp"],
            "tabpfn": ["s17_tabpfn_context", "tabpfn_context_large"],
            "tabpfn_by_municipality": ["s17_tabpfn_context", "tabpfn_by_municipality"],
            "tabpfn_160k": ["s20_tabpfn_160k", "tabpfn_160k"],
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
        variants: dict = field(default_factory=_variants)
        member_runs: dict = field(default_factory=_member_runs)
        s17_stage: str = "s17_tabpfn_context"
        s19_stage: str = "s19_tabpfn_groups"
        s20_stage: str = "s20_tabpfn_160k"
        stacking_alpha: float = 1.0
        stack_mlp_hidden: tuple[int, ...] = (32, 16)
        stack_mlp_alpha: float = 1e-3
        stack_mlp_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
        stack_lgbm_max_rounds: int = 2000
        stack_lgbm_early_stopping_rounds: int = 100
        inner_valid_share: float = 0.2
        stack_lgbm: StackerLgbParams = field(default_factory=StackerLgbParams)

    cfg = Config()
    EXP = "s21_tabpfn_ppm2_price_type"
    BASE = "s20_tabpfn_160k"
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

    # Holdout setting (S13-S20): train up to 2024Q4, validate on 2025Q1-Q2 (fold 1).
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
    def _target_shift(X, target):
        """What a variant's target subtracts from the deflated log price: log area for ppm2, nothing for price."""
        if target == "ppm2":
            area = X["area"].to_numpy()
            assert (area > 0).all()
            return np.log(area)
        assert target == "price", target
        return np.zeros(len(X))

    def _groups(X, group):
        """The group of every row of X: one group for every row, or the row's 価格情報区分."""
        if group is None:
            return np.zeros(len(X), dtype=int)
        assert group == "price_type", group
        return X["価格情報区分"].astype(str).to_numpy()

    def tabpfn_variant(X_context, y_context, X_apply, variant):
        """TabPFN's predictions (deflated log price) for X_apply, with the most recent rows of each group as context."""
        periods = X_context["trade_period"].to_numpy()
        shift_context, shift_apply = (
            _target_shift(X_context, variant["target"]),
            _target_shift(X_apply, variant["target"]),
        )
        group_context, group_apply = _groups(X_context, variant["group"]), _groups(X_apply, variant["group"])
        everything = np.arange(len(X_context))
        pred = np.empty(len(X_apply))
        for g in np.unique(group_apply):
            at = np.flatnonzero(group_apply == g)
            pool = everything[group_context == g]
            assert len(pool) > 0, g
            context = incontext.most_recent(pool, periods, cfg.context_rows)
            model = TabPFNRegressor(device="cuda", random_state=cfg.tabpfn_seed, **cfg.tabpfn_options)
            model.fit(X_context.iloc[context], y_context[context] - shift_context[context])
            pred[at] = incontext.predict_in_chunks(model, X_apply.iloc[at], cfg.predict_chunk) + shift_apply[at]
        return pred

    return (tabpfn_variant,)


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

    def candidates(base, extra):
        """The stacks to compare: S20's members, each variant added or replacing the 160,000-row TabPFN, and all added."""
        out = {"S20 members": base}
        for _name, _pred in extra.items():
            out[f"{_name} added"] = base | {_name: _pred}
            out[f"{_name} replacing tabpfn_160k"] = {
                (_name if _k == "tabpfn_160k" else _k): (_pred if _k == "tabpfn_160k" else _v)
                for _k, _v in base.items()
            }
        out["all variants added"] = base | extra
        return out

    return candidates, found_stackers


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 内側の分割で比べる

    2024Q2 までの行を文脈にして、2つの案で 2024Q3〜Q4 を予測します。予測はファイルに保存します。
    S20 のメンバーの 2024Q3〜Q4 の予測（S19 で作った S13 のメンバー自身の予測、S17 の2本の TabPFN、S20 の16万行の TabPFN）で
    スタッキングを組み、それぞれの案を足した場合と、16万行の TabPFN と入れ替えた場合、両方を足した場合を比べます。
    """)
    return


@app.cell
def _(EXP, block_ids, cfg, env, fs_in, pl, tabpfn_variant, tr_in, va_in, y_in):
    (env.output_dir / EXP).mkdir(parents=True, exist_ok=True)
    inner_preds = {}
    for _name, _variant in cfg.variants.items():
        _path = env.output_dir / EXP / f"inner_{_name}.parquet"
        if not _path.exists():
            _pred = tabpfn_variant(fs_in.X.iloc[tr_in], y_in[tr_in], fs_in.X.iloc[va_in], _variant)
            pl.DataFrame({cfg.id_col: block_ids, "pred": _pred + fs_in.offset[va_in]}).write_parquet(_path)
        _saved = pl.read_parquet(_path)
        assert _saved[cfg.id_col].to_list() == block_ids
        inner_preds[_name] = _saved["pred"].to_numpy()
    return (inner_preds,)


@app.cell
def _(block_ids, cfg, env, pl):
    def _inner(stage, name):
        _saved = pl.read_parquet(env.output_dir / stage / name)
        assert _saved[cfg.id_col].to_list() == block_ids
        return _saved["pred"].to_numpy()

    _block = pl.read_parquet(env.output_dir / cfg.s19_stage / "block_predictions.parquet")
    assert _block[cfg.id_col].to_list() == block_ids
    block_members = {_name: _block[_name].to_numpy() for _name in list(cfg.member_runs)[:6]}
    block_members["tabpfn"] = _inner(cfg.s17_stage, "inner_context_large.parquet")
    block_members["tabpfn_by_municipality"] = _inner(cfg.s17_stage, "inner_by_municipality.parquet")
    block_members["tabpfn_160k"] = _inner(cfg.s20_stage, "inner_context_160k.parquet")
    assert list(block_members) == list(cfg.member_runs)
    return (block_members,)


@app.cell
def _(
    block_members,
    candidates,
    found_stackers,
    fs_in,
    holdout_rmsle,
    in_q3,
    inner_preds,
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

    _inner_candidates = candidates(block_members, inner_preds)
    inner_table = pl.DataFrame(
        [
            {"stack": _name, "inner stacker average": block_stack(_members)}
            for _name, _members in _inner_candidates.items()
        ]
    )
    _scores = dict(zip(inner_table["stack"], inner_table["inner stacker average"], strict=True))
    chosen = min(_scores, key=lambda _n: _scores[_n])
    _err_160k = block_members["tabpfn_160k"] - y_block
    inner_alone = pl.DataFrame(
        [{"tabpfn": "tabpfn_160k (S20)", "inner rmsle": holdout_rmsle(block_members["tabpfn_160k"], y_block)}]
        + [
            {
                "tabpfn": _name,
                "inner rmsle": holdout_rmsle(_pred, y_block),
                "error corr with tabpfn_160k": float(np.corrcoef(_pred - y_block, _err_160k)[0, 1]),
            }
            for _name, _pred in inner_preds.items()
        ],
        infer_schema_length=None,
    )
    needed = [_name for _name in inner_preds if _name in _inner_candidates[chosen]]
    inner_table
    return chosen, inner_alone, inner_table, needed


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. holdout と test で確かめる

    1 で S20 のメンバーより良いスタックが選ばれたときだけ、選ばれた案で holdout と test を予測します。
    holdout の文脈は 2024Q4 まで、test の文脈は全期間で、どちらも各グループの直近16万行（区分ごとなら、その区分の全行）です。
    そのうえで S20 と同じ手順でスタッキングを組み直し、S20 の `stacker_average_s20` と比べます。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    bootstrap_diff,
    candidates,
    cfg,
    chosen,
    ensemble,
    env,
    found_stackers,
    fs,
    git_commit,
    holdout_ids,
    holdout_rmsle,
    in_q1,
    lesson,
    needed,
    np,
    pl,
    sample,
    stacking,
    tabpfn_variant,
    tr_idx,
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

    s20_members = {_name: saved(*_run) for _name, _run in cfg.member_runs.items()}
    s20_average = saved(cfg.s20_stage, "stacker_average_s20")
    _reproduced = stacker_average(s20_members)
    assert np.allclose(_reproduced[0], s20_average[0], atol=1e-9)
    assert np.allclose(_reproduced[1], s20_average[1], atol=1e-9)

    holdout_table = None
    s21_weights = None
    if chosen != "S20 members":
        commit = git_commit(env.root)
        variant_preds = {}
        for _name in needed:
            _variant = cfg.variants[_name]
            _path_holdout = env.output_dir / EXP / f"holdout_{_name}.parquet"
            if not _path_holdout.exists():
                _pred = tabpfn_variant(fs.X.iloc[tr_idx], y_real[tr_idx], fs.X.iloc[va_idx], _variant)
                pl.DataFrame({cfg.id_col: holdout_ids[cfg.id_col], "pred": _pred + fs.offset[va_idx]}).write_parquet(
                    _path_holdout
                )
            _path_test = env.output_dir / EXP / f"test_{_name}.parquet"
            if not _path_test.exists():
                _pred = tabpfn_variant(fs.X_refit, y_real_refit, fs.X_test, _variant)
                pl.DataFrame({cfg.id_col: sample[cfg.id_col], "pred": _pred + fs.offset_test}).write_parquet(_path_test)
            _holdout, _test = pl.read_parquet(_path_holdout), pl.read_parquet(_path_test)
            assert _holdout[cfg.id_col].to_list() == holdout_ids[cfg.id_col].to_list()
            assert _test[cfg.id_col].to_list() == sample[cfg.id_col].to_list()
            variant_preds[_name] = (_holdout["pred"].to_numpy(), _test["pred"].to_numpy())
            _out = lesson.record_run(
                env,
                exp=EXP,
                base=BASE,
                run=f"tabpfn_{_name}",
                score=holdout_rmsle(variant_preds[_name][0], y_holdout),
                metric=cfg.metric,
                config=asdict(cfg) | {"model": "tabpfn-3.5", "variant": _name},
                holdout=holdout_ids,
                pred=np.expm1(variant_preds[_name][0]),
                notes=f"TabPFN-3.5, variant {_name} (#50 #51)",
                commit=commit,
            )
            _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(variant_preds[_name][1])})
            validate_submission(_submission, sample, cfg.id_col)
            write_submission(_out, _submission)

        s21_members = candidates(s20_members, variant_preds)[chosen]
        restacked = stacker_average(s21_members)
        _restack_change = bootstrap_diff((restacked[0] - y_holdout) ** 2, (s20_average[0] - y_holdout) ** 2)
        _out = lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run="stacker_average_s21",
            score=holdout_rmsle(restacked[0], y_holdout),
            metric=cfg.metric,
            config=asdict(cfg) | {"members": list(s21_members), "chosen": chosen},
            holdout=holdout_ids,
            pred=np.expm1(restacked[0]),
            notes=f"S20 stacker average, {chosen} (#50 #51)",
            commit=commit,
        )
        _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(restacked[1])})
        validate_submission(_submission, sample, cfg.id_col)
        write_submission(_out, _submission)

        # Shown only: fit on the whole holdout, not cross-fit, and not used for scoring.
        _weights = ensemble.fit_weights(np.column_stack([m[0] for m in s21_members.values()]), y_holdout)
        s21_weights = dict(zip(s21_members, np.round(_weights, 3).tolist(), strict=True))
        _rows = [{"run": "s20 tabpfn_160k", "holdout_rmsle": holdout_rmsle(s20_members["tabpfn_160k"][0], y_holdout)}]
        for _name, _pair in variant_preds.items():
            _change = bootstrap_diff((_pair[0] - y_holdout) ** 2, (s20_members["tabpfn_160k"][0] - y_holdout) ** 2)
            _rows.append(
                {
                    "run": f"tabpfn_{_name}",
                    "holdout_rmsle": holdout_rmsle(_pair[0], y_holdout),
                    "change": _change[0],
                    "ci_low": _change[1],
                    "ci_high": _change[2],
                }
            )
        _rows.append({"run": "s20 stacker_average_s20", "holdout_rmsle": holdout_rmsle(s20_average[0], y_holdout)})
        _rows.append(
            {
                "run": "stacker_average_s21",
                "holdout_rmsle": holdout_rmsle(restacked[0], y_holdout),
                "change": _restack_change[0],
                "ci_low": _restack_change[1],
                "ci_high": _restack_change[2],
            }
        )
        holdout_table = pl.DataFrame(_rows, infer_schema_length=None)
    holdout_table
    return holdout_table, s21_weights


@app.cell
def _(chosen, holdout_table, inner_alone, inner_table, mo, s21_weights):
    _alone = "\n".join(
        f"    | {r['tabpfn']} | {r['inner rmsle']:.5f} | "
        + ("" if r["error corr with tabpfn_160k"] is None else f"{r['error corr with tabpfn_160k']:.3f}")
        + " |"
        for r in inner_alone.iter_rows(named=True)
    )
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
        _holdout += f"\n    - 重み（holdout の全行で合わせた非負の重み。表示だけ）: {s21_weights}"
    mo.md(f"""
    ## 3. 結果

    2024Q3〜Q4 の TabPFN 単独:

    | TabPFN | RMSLE | 16万行の TabPFN との誤差の相関 |
    |---|---|---|
{_alone}

    | スタッキング（2024Q3〜Q4） | スタッキングの平均 |
    |---|---|
{_inner}

    選んだもの: {chosen}

    holdout（2025Q1〜Q2）:

{_holdout}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 結果を読む

    2026-10-20 の実行の結果です。予測にかかった時間は、2024Q3〜Q4 が㎡単価で約1時間50分、区分ごとで約40分、
    ㎡単価の holdout が約2時間20分、test が約3時間でした。

    - 2024Q3〜Q4 では、㎡単価の TabPFN が単独で 0.18161 → 0.18132 と少し良くなりました。区分ごとの TabPFN は 0.18486 で、0.0033 悪くなりました。
      スタッキングの平均は、S20 のメンバー 0.18117 に対して、㎡単価の追加 0.18093、入れ替え 0.18109、
      区分ごとの追加 0.18124、入れ替え 0.18202、両方の追加 0.18102 で、㎡単価の追加を選びました。
    - 区分ごとの TabPFN は、どちらの区分でも S20 の TabPFN より悪くなりました
      （2024Q3〜Q4 で、不動産取引価格情報 0.20285 → 0.20594、成約価格情報 0.16104 → 0.16451）。
      区分で分けると、1本の TabPFN が見る文脈は16万行から区分の行数（2024Q2 までで 9.5万行と 6.8万行）に減り、もう一方の区分の取引も見えなくなります。
      S20 で見た「文脈が多いほど良い」の効き方のほうが、区分をそろえる効き方より大きかったと読めます。
    - holdout では、㎡単価の TabPFN は単独で 0.18674 → 0.18685（差 +0.0001、95% の範囲 -0.0006〜+0.0009）で、
      2024Q3〜Q4 で見えた改善は消えました。区分別では、成約価格情報が良くなり（0.17072 → 0.17024）、
      不動産取引価格情報が悪くなりました（0.21028 → 0.21115）。
    - スタッキングの平均は 0.18584 → 0.18560（差 -0.00024、範囲 -0.00057〜+0.00016）で、範囲が 0 をまたぎます。
      2024Q3〜Q4 での差（-0.00024）と同じ大きさですが、偶然のぶれと区別できない大きさです。
      2025Q1（0.18509 → 0.18479）と Q2（0.18659 → 0.18641）、成約価格情報（0.16999 → 0.16944）は良くなり、
      不動産取引価格情報（0.20911 → 0.20930）は悪くなりました。
    - ㎡単価と16万行の TabPFN の誤差の相関は 0.990 です。holdout の全行で合わせた非負の重み（表示するだけで、採点には使いません）は、
      ㎡単価 0.474、16万行 0.401 で、2本をほぼ半々に混ぜています。単独の精度は同じくらいで、外し方が少しだけ違うので、
      平均すると少しだけ良くなります。目的変数を㎡単価にしても単独では変わらないのは、S4 の LightGBM（0.2048 と 0.2050）と同じです。
    - test の予測は S20 より平均で 0.05% 高く（log の差の平均 +0.0005）、holdout での差の平均（+0.0003）とほぼ同じです。
      S20 のときのような、test でだけ水準がずれる動きはほぼありません。

    S21 は採用しません。スタッキングの平均の改善が、偶然の範囲と区別できないからです。
    次の提出の候補は、引き続き S20 の `stacker_average_s20` です。㎡単価を足した `stacker_average_s21`（CV 0.1856）は記録として残します。
    """)
    return


if __name__ == "__main__":
    app.run()
