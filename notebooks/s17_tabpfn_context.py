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
    # S17. TabPFN の文脈を変える

    S13 から、スタッキングの重みの半分以上（0.56）は TabPFN が持っています。S14 と S16 では、重みの小さいメンバーを良くしても
    スタッキングは動きませんでした。動かすなら、重みの大きい TabPFN を良くするのが近道です（#45）。

    TabPFN は、学習データの行を「文脈」として読み込み、その場で予測するモデルです。S13 では、東京全体の直近5万行を文脈にしました。
    5万行は TabPFN-3.5 が公式に対応する上限で、この 8GB の GPU にも既定の設定でちょうど収まる大きさでした。
    この段階では、文脈の選び方と予測の回数を変えた4通りを比べます。

    | 名前 | 文脈 | 変えたところ |
    |---|---|---|
    | `s13` | 東京全体の直近5万行 | S13 のまま（比べる基準） |
    | `estimators_16` | 東京全体の直近5万行 | 予測の回数（特徴量の並べ方を変えた予測の平均）を 8 → 16 |
    | `context_large` | 東京全体の直近12万行 | メモリを節約する設定にして、上限を超える行数を読ませる |
    | `by_municipality` | 同じ市区町村の行だけ（全期間） | 市区町村ごとに TabPFN を分ける |

    最後の `by_municipality` は、近くの取引だけを見せる案です。大きなデータで TabPFN を使うとき、予測する行に近い行だけを文脈にすると
    良くなることが知られています（LoCalPFN、2024）。ここでは「近い」を「同じ市区町村」で近似します。
    文脈の行は少なくなりますが、その市区町村の昔の取引まで見られます。今の TabPFN と外し方が変わるので、置き換えでなく「足す」使い方も確かめます。

    選び方は S14 と同じです。2024Q3〜Q4 の内側の分割で比べ、holdout（2025Q1〜Q2）は選んだものの確かめに1回だけ使います。
    内側の分割では、S15 で作った 2024Q3〜Q4 の全メンバーの予測を使って、スタッキングまで組んで比べます。
    """)
    return


@app.cell
def _(dataclass, field, psutil):
    THREADS = psutil.cpu_count(logical=False)

    @dataclass
    class TabPFNParams:
        # S13's TabPFN: 50,000 rows is TabPFN-3.5's official limit and what fits the 8 GB GPU with its defaults
        context_rows: int = 50000
        predict_chunk: int = 2000
        seed: int = 0

    @dataclass
    class StackerLgbParams:
        # S12-S16's LightGBM stacker
        objective: str = "regression"
        learning_rate: float = 0.03
        num_leaves: int = 15
        min_data_in_leaf: int = 100
        feature_fraction: float = 0.9
        seed: int = 42
        verbose: int = -1
        num_threads: int = THREADS

    # The memory options let TabPFN read more rows than its official limit on 8 GB. Probes on 2026-10-18, rows of
    # 2024Q3-Q4: on 1,000 rows, 50,000 context rows scored 0.1791 and 80,000 0.1769; on 6,000 rows, 80,000 scored
    # 0.1788 and 120,000 0.1761 (peak 7.2 GiB, 860 s per call). A call's time is mostly the context, not the rows
    # it predicts (80,000 rows: 388 s for 1,000 rows, 411 s for 6,000), so the variants predict in large chunks.
    # Every row up to the cutoff (162,701 rows on the inner split) ran out of memory at 15,000 rows per call: the
    # predicted distributions of the rows in a call take about 2 GB. Smaller chunks mean more calls, each reading
    # the whole context again (an estimated 10 hours for the holdout and the test), so 120,000 is the largest here.
    LARGE = {"fit_mode": "low_memory", "memory_saving_mode": True, "ignore_pretraining_limits": True}

    def _variants():
        return {
            "s13": {"context_rows": 50000, "by_municipality": False, "options": {}, "predict_chunk": 2000},
            "estimators_16": {
                "context_rows": 50000,
                "by_municipality": False,
                "options": {"n_estimators": 16},
                "predict_chunk": 6000,
            },
            "context_large": {
                "context_rows": 120000,
                "by_municipality": False,
                "options": LARGE,
                "predict_chunk": 12000,
            },
            "by_municipality": {"context_rows": 50000, "by_municipality": True, "options": {}, "predict_chunk": 6000},
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
        tabpfn: TabPFNParams = field(default_factory=TabPFNParams)
        variants: dict = field(default_factory=_variants)
        block_stage: str = "s15_more_stack_rows"
        stacking_alpha: float = 1.0
        stack_mlp_hidden: tuple[int, ...] = (32, 16)
        stack_mlp_alpha: float = 1e-3
        stack_mlp_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
        stack_lgbm_max_rounds: int = 2000
        stack_lgbm_early_stopping_rounds: int = 100
        inner_valid_share: float = 0.2
        stack_lgbm: StackerLgbParams = field(default_factory=StackerLgbParams)

    cfg = Config()
    EXP = "s17_tabpfn_context"
    BASE = "s16_tabnet_tuning"
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

    # Holdout setting (S13-S16): train up to 2024Q4, validate on 2025Q1-Q2 (fold 1).
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
    def tabpfn_predict(X_context, y_context, X_apply, variant):
        """TabPFN's predictions (on the deflated log scale) for X_apply, with the context the variant picks."""

        def fit_predict(context_rows, apply_rows):
            model = TabPFNRegressor(device="cuda", random_state=cfg.tabpfn.seed, **variant["options"])
            model.fit(X_context.iloc[context_rows], y_context[context_rows])
            return incontext.predict_in_chunks(model, X_apply.iloc[apply_rows], variant["predict_chunk"])

        periods = X_context["trade_period"].to_numpy()
        everything = np.arange(len(X_context))
        if not variant["by_municipality"]:
            context = incontext.most_recent(everything, periods, variant["context_rows"])
            return fit_predict(context, np.arange(len(X_apply)))
        town_context, town_apply = X_context["市区町村名"].to_numpy(), X_apply["市区町村名"].to_numpy()
        pred = np.empty(len(X_apply))
        for town in np.unique(town_apply):
            at = np.flatnonzero(town_apply == town)
            pool = everything[town_context == town]
            assert len(pool) > 0, town
            pred[at] = fit_predict(incontext.most_recent(pool, periods, variant["context_rows"]), at)
        return pred

    return (tabpfn_predict,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 内側の分割で比べる

    2024Q2 までの行を文脈にして 2024Q3〜Q4 を予測します。1通りに15分から1時間近くかかるので、予測は1通りずつファイルに保存し、
    止まっても続きから再開します。`s13` は S15 で同じ手順で作った予測があるので、それを使います。

    スタッキングは、S15 の 2024Q3〜Q4 の予測（S14 で探索した LightGBM・CatBoost・XGBoost・MLP、S7 の RandomForest と Ridge、S13 の TabPFN）で組みます。
    2024Q3 と Q4 で交互に学習して採点する、S13 と同じ手順です。TabPFN を入れ替えた場合と、足した場合の両方を見ます。
    """)
    return


@app.cell
def _(EXP, block_ids, cfg, env, fs_in, np, pl, tabpfn_predict, tr_in, va_in, y_in):
    block = pl.read_parquet(env.output_dir / cfg.block_stage / "block_predictions.parquet")
    assert block[cfg.id_col].to_list() == block_ids
    inner_preds = {"s13": block["tabpfn"].to_numpy()}
    (env.output_dir / EXP).mkdir(parents=True, exist_ok=True)
    for _name, _variant in cfg.variants.items():
        if _name == "s13":
            continue
        _path = env.output_dir / EXP / f"inner_{_name}.parquet"
        if not _path.exists():
            _pred = tabpfn_predict(fs_in.X.iloc[tr_in], y_in[tr_in], fs_in.X.iloc[va_in], _variant)
            pl.DataFrame({cfg.id_col: block_ids, "pred": _pred + fs_in.offset[va_in]}).write_parquet(_path)
        _saved = pl.read_parquet(_path)
        assert _saved[cfg.id_col].to_list() == block_ids
        inner_preds[_name] = _saved["pred"].to_numpy()
    block_members = {_c: block[_c].to_numpy() for _c in block.columns if _c not in (cfg.id_col, cfg.target, "fold")}
    assert np.allclose(block_members["tabpfn"], inner_preds["s13"])
    return block_members, inner_preds


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
def _(block_members, fs_in, found_stackers, holdout_rmsle, in_q3, inner_preds, np, pl, stacking, va_in, y_block):
    _rows = stacking.row_features(fs_in.X.iloc[va_in])

    def block_stack(members):
        """The stacker average on 2024Q3-Q4, fit on Q3 and scored on Q4 and the other way round."""
        H = np.column_stack(list(members.values()))
        return stacking.stacker_average(H, H, _rows, _rows, y_block, in_q3, found_stackers(len(members)))[0]

    _base = holdout_rmsle(block_stack(block_members), y_block)
    _table = []
    for _name, _pred in inner_preds.items():
        _replaced = block_stack(block_members | {"tabpfn": _pred})
        _added = block_stack(block_members | {f"tabpfn_{_name}": _pred}) if _name != "s13" else _replaced
        _table.append(
            {
                "variant": _name,
                "TabPFN alone": holdout_rmsle(_pred, y_block),
                "stack, TabPFN replaced": holdout_rmsle(_replaced, y_block),
                "stack, TabPFN added": holdout_rmsle(_added, y_block),
            }
        )
    inner_table = pl.DataFrame(_table)
    assert np.isclose(inner_table["stack, TabPFN replaced"][0], _base)
    inner_table
    return (inner_table,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. holdout で確かめるものを選ぶ

    規則は次の2つです。どちらも 2024Q3〜Q4 のスタッキングの平均で決め、holdout は見ません。

    - 置き換え: TabPFN を入れ替えたスタッキングが一番良い variant。`s13` より良くなければ置き換えない。
    - 追加: 置き換えたあとに、さらにもう1本の TabPFN を足したほうが良ければ、それも足す。
    """)
    return


@app.cell
def _(inner_table, pl):
    _others = inner_table.filter(pl.col("variant") != "s13")
    _s13 = inner_table.filter(pl.col("variant") == "s13")["stack, TabPFN replaced"][0]
    _best = _others.sort("stack, TabPFN replaced").row(0, named=True)
    replacement = _best["variant"] if _best["stack, TabPFN replaced"] < _s13 else "s13"
    _added = _others.filter(pl.col("variant") != replacement).sort("stack, TabPFN added").row(0, named=True)
    _reference = _s13 if replacement == "s13" else _best["stack, TabPFN replaced"]
    addition = _added["variant"] if _added["stack, TabPFN added"] < _reference else None
    {"replacement": replacement, "addition": addition}
    return addition, replacement


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. holdout で確かめる

    選んだ variant で、2024Q4 までを文脈にして holdout を、全期間を文脈にして test を、1回ずつ予測します。
    そのうえで、S13 のメンバー（S10 の LightGBM と CatBoost、S6 の XGBoost、S7 の MLP・RandomForest・Ridge、S13 の TabPFN）で
    S13 と同じ手順のスタッキングを組み直し、S13 の `stacker_average_with_tabpfn` と比べます。まず S13 が再現できることを確かめます。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    addition,
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
    replacement,
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
    for _name in [n for n in (replacement, addition) if n not in (None, "s13")]:
        _variant = cfg.variants[_name]
        # A large context takes hours on the test, so each prediction is kept and a rerun reads it back.
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
        _holdout, _test = new_tabpfn[_name]
        _out = lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=f"tabpfn_{_name}",
            score=holdout_rmsle(_holdout, y_holdout),
            metric=cfg.metric,
            config=asdict(cfg) | {"model": "tabpfn-3.5", "variant": _name},
            holdout=holdout_ids,
            pred=np.expm1(_holdout),
            notes=f"TabPFN-3.5, variant {_name} (#45)",
            commit=commit,
        )
        _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(_test)})
        validate_submission(_submission, sample, cfg.id_col)
        write_submission(_out, _submission)
    return commit, new_tabpfn


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
    BASE,
    EXP,
    addition,
    asdict,
    bootstrap_diff,
    cfg,
    commit,
    ensemble,
    env,
    found_stackers,
    fs,
    holdout_ids,
    holdout_rmsle,
    in_q1,
    lesson,
    new_tabpfn,
    np,
    pl,
    replacement,
    s13_average,
    s13_members,
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

    _reproduced = stacker_average(s13_members)
    assert np.allclose(_reproduced[0], s13_average[0], atol=1e-9)
    assert np.allclose(_reproduced[1], s13_average[1], atol=1e-9)

    s17_members = dict(s13_members)
    if replacement != "s13":
        s17_members["tabpfn"] = new_tabpfn[replacement]
    if addition is not None:
        s17_members[f"tabpfn_{addition}"] = new_tabpfn[addition]
    restacked = stacker_average(s17_members)
    restack_change = bootstrap_diff((restacked[0] - y_holdout) ** 2, (s13_average[0] - y_holdout) ** 2)
    restack_score = holdout_rmsle(restacked[0], y_holdout)
    _out = lesson.record_run(
        env,
        exp=EXP,
        base=BASE,
        run="stacker_average_s17",
        score=restack_score,
        metric=cfg.metric,
        config=asdict(cfg) | {"members": list(s17_members), "replacement": replacement, "addition": addition},
        holdout=holdout_ids,
        pred=np.expm1(restacked[0]),
        notes=f"S13 stacker average, TabPFN replaced by {replacement}, added {addition} (#45)",
        commit=commit,
    )
    _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(restacked[1])})
    validate_submission(_submission, sample, cfg.id_col)
    write_submission(_out, _submission)

    _weights = ensemble.fit_weights(np.column_stack([m[0] for m in s17_members.values()]), y_holdout)
    s17_weights = dict(zip(s17_members, np.round(_weights, 3).tolist(), strict=True))
    _rows = [{"run": "s13 tabpfn", "holdout_rmsle": holdout_rmsle(s13_members["tabpfn"][0], y_holdout)}]
    for _name, (_h, _) in new_tabpfn.items():
        _change = bootstrap_diff((_h - y_holdout) ** 2, (s13_members["tabpfn"][0] - y_holdout) ** 2)
        _rows.append(
            {
                "run": f"tabpfn_{_name}",
                "holdout_rmsle": holdout_rmsle(_h, y_holdout),
                "change": _change[0],
                "ci_low": _change[1],
                "ci_high": _change[2],
            }
        )
    _rows += [
        {"run": "s13 stacker_average_with_tabpfn", "holdout_rmsle": holdout_rmsle(s13_average[0], y_holdout)},
        {
            "run": "stacker_average_s17",
            "holdout_rmsle": restack_score,
            "change": restack_change[0],
            "ci_low": restack_change[1],
            "ci_high": restack_change[2],
        },
    ]
    holdout_table = pl.DataFrame(_rows, infer_schema_length=None)
    holdout_table
    return holdout_table, restack_change, restack_score, s17_weights


@app.cell
def _(addition, holdout_table, inner_table, mo, pl, replacement, restack_change, restack_score, s17_weights):
    def _score(run):
        return holdout_table.filter(pl.col("run") == run)["holdout_rmsle"][0]

    _inner = "\n".join(
        f"    | `{r['variant']}` | {r['TabPFN alone']:.4f} | {r['stack, TabPFN replaced']:.4f} | {r['stack, TabPFN added']:.4f} |"
        for r in inner_table.iter_rows(named=True)
    )
    mo.md(f"""
    ## 4. 結果

    内側の分割（2024Q3〜Q4）:

    | variant | TabPFN 単独 | スタッキング（入れ替え） | スタッキング（追加） |
    |---|---|---|---|
{_inner}

    - 選んだもの: 置き換え `{replacement}`、追加 `{addition}`
    - holdout（2025Q1〜Q2）: S13 の TabPFN {_score("s13 tabpfn"):.4f}
    - スタッキングの平均: {restack_score:.4f}（S13 との差 {restack_change[0]:+.4f}、95% の範囲 {restack_change[1]:+.4f}〜{restack_change[2]:+.4f}）
    - 重み（非負の重み付き平均）: {s17_weights}
    """)
    return


if __name__ == "__main__":
    app.run()
