import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from dataclasses import asdict, dataclass, field

    import lightgbm as lgb
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import psutil

    from kgl import cv, encoding, feature_set, homeprice, lesson, mlit, price_index
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, validate_submission, write_submission
    from kgl.metrics import get_metric

    return (
        asdict,
        comp_dir,
        cv,
        dataclass,
        detect,
        encoding,
        feature_set,
        field,
        get_metric,
        git_commit,
        homeprice,
        lesson,
        lgb,
        mlit,
        mo,
        np,
        pd,
        pl,
        price_index,
        psutil,
        validate_submission,
        write_submission,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S5. Target encoding・集約特徴・件数

    S4 までで、時間による価格水準の動きはモデルの外に出せました（holdout RMSLE 0.2050）。この段階では、場所の情報の渡し方を工夫します。

    場所（市区町村、地区、駅）は、これまでカテゴリとして LightGBM に渡していました。地区名は 1,210 種類あり、行の少ない地区も多くあります。
    そこで、場所ごとの性質を数値にまとめて渡す方法を3つ試します。

    | run | 足すもの | Issue |
    |---|---|---|
    | `te` | 地区・駅ごとの平均の実質㎡単価（Target encoding） | #18 |
    | `aggregates` | 地区・駅ごとの物件の性質（面積・建築年の中央値、容積率の平均、成約価格情報の割合） | #19 |
    | `counts` | 間取り・地区・駅の件数（どれだけよくある値か） | #33 |
    | `all_three` | 上の3つ全部 | |

    比べる相手は S4 で採用した `hedonic_linear` です。目的変数と価格指数の扱いは S4 と同じにします。
    """)
    return


@app.cell
def _(dataclass, field, psutil):
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
        # One thread per physical core (see S1: all logical cores made LightGBM 150x slower here).
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
        max_rounds: int = 5000
        early_stopping_rounds: int = 100
        trend_quarters: int = 8
        # Target encoding: weight of the prior (in rows) and the folds used to encode the fitting rows.
        te_smoothing: float = 20.0
        te_folds: int = 5
        seed: int = 42
        params: LgbParams = field(default_factory=LgbParams)

    cfg = Config()
    EXP = "s5_te"
    BASE = "s4_trend"
    ADOPTED = {"type", "period", "notation"}
    return ADOPTED, BASE, EXP, cfg


@app.cell
def _(ADOPTED, cfg, comp_dir, cv, detect, homeprice, mlit, pl):
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

    y_log = train_folded[cfg.target].cast(pl.Float64).log1p().to_numpy()
    t_train = train_folded.select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy()
    t_test = test_raw.select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy()
    log_area = train_folded.select(mlit.area_m2("面積（㎡）").log().alias("a"))["a"].to_numpy()
    log_ppm2 = y_log - log_area

    vocab = homeprice.make_vocab([train_folded, test_raw])
    X_base = homeprice.features(train_folded, ADOPTED, vocab)
    X_test_base = homeprice.features(test_raw, ADOPTED, vocab)
    return (
        X_base,
        X_test_base,
        env,
        log_area,
        log_ppm2,
        sample,
        t_test,
        t_train,
        test_raw,
        tr_idx,
        train_folded,
        va_idx,
        y_log,
    )


@app.cell
def _(cfg, homeprice, log_ppm2, np, price_index, t_test, t_train, tr_idx, train_folded):
    # S4's adopted trend: the hedonic index, looked up for covered quarters and extended linearly beyond them.
    periods_block, index_block = price_index.hedonic_index(
        t_train[tr_idx], log_ppm2[tr_idx], homeprice.hedonic_controls(train_folded[tr_idx])
    )
    offset_holdout = price_index.offset(periods_block, index_block, t_train, k=cfg.trend_quarters)
    all_rows = np.arange(len(t_train))
    periods_all, index_all = price_index.hedonic_index(
        t_train, log_ppm2, homeprice.hedonic_controls(train_folded[all_rows])
    )
    offset_all = price_index.offset(periods_all, index_all, t_train, k=cfg.trend_quarters)
    offset_test = price_index.offset(periods_all, index_all, t_test, k=cfg.trend_quarters)
    return offset_all, offset_holdout, offset_test


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Target encoding とリーク

    Target encoding（TE）は、カテゴリの値ごとに目的変数の平均を計算して、特徴量にする方法です。ここでは地区・駅ごとに、
    実質㎡単価（log 価格 − log 面積 − 価格指数）の平均を使います。時期と広さの影響を除いてあるので、平均は「その場所の値段の高さ」を表します。

    2つの工夫をします。

    - 平滑化: 行の少ない地区の平均はぶれるので、全体の平均に寄せます。行数 n、合計 s の地区は (s + 20 × 全体平均) / (n + 20) にします。
    - out-of-fold: 学習に使う行の TE を、その行自身の値を含めて計算すると、モデルは「TE を見れば答えが分かる」と学んでしまいます。
      学習期間の行をランダムに5つに分け、各行の TE は残りの4つから計算します。holdout と test の行の TE は、学習期間の全行から計算します。

    out-of-fold にしなかった場合にどうなるかも、比較のために2つ走らせます。平滑化ありで自分の行を含めた `te_in_sample` と、
    平滑化もなしで自分の行を含めた `te_raw_leak` です。
    """)
    return


@app.cell
def _(cfg, encoding, np, pd):
    TE_KEYS = {"te_district": "地区名", "te_station": "最寄駅：名称"}

    def target_encodings(fit_df, fit_target, apply_df, *, out_of_fold, m):
        """TE columns for the fitting rows (out-of-fold or in-sample) and for the rows to predict."""
        fold_of_row = np.random.default_rng(cfg.seed).integers(0, cfg.te_folds, fit_df.height)
        prior = float(fit_target.mean())
        fit_columns, apply_columns = {}, {}
        for name, key in TE_KEYS.items():
            if out_of_fold:
                fit_columns[name] = encoding.oof_smoothed_mean(fit_df[key], fit_target, fold_of_row, m=m)
            else:
                fit_columns[name] = encoding.smoothed_mean(fit_df[key], fit_target, fit_df[key], prior=prior, m=m)
            apply_columns[name] = encoding.smoothed_mean(fit_df[key], fit_target, apply_df[key], prior=prior, m=m)
        return pd.DataFrame(fit_columns), pd.DataFrame(apply_columns)

    return (target_encodings,)


@app.cell
def _(cfg, log_area, offset_holdout, pd, target_encodings, tr_idx, train_folded, va_idx, y_log):
    real_ppm2_block = (y_log - log_area - offset_holdout)[tr_idx]

    def holdout_te(out_of_fold, m):
        """TE for every training row in the holdout setting: fitting rows from the block, holdout rows applied."""
        fit_cols, apply_cols = target_encodings(
            train_folded[tr_idx], real_ppm2_block, train_folded[va_idx], out_of_fold=out_of_fold, m=m
        )
        full = pd.DataFrame(index=range(train_folded.height), columns=fit_cols.columns, dtype=float)
        full.iloc[tr_idx] = fit_cols.to_numpy()
        full.iloc[va_idx] = apply_cols.to_numpy()
        return full

    te_holdout = holdout_te(out_of_fold=True, m=cfg.te_smoothing)
    te_holdout_in_sample = holdout_te(out_of_fold=False, m=cfg.te_smoothing)
    # The leak at its worst: a row's own price inside its encoding, with no pull toward the overall mean.
    te_holdout_raw_leak = holdout_te(out_of_fold=False, m=0.0)
    return te_holdout, te_holdout_in_sample, te_holdout_raw_leak


@app.cell
def _(homeprice, pl, test_raw, train_folded):
    # Target-free features are computed over every training row (2019Q1-2025Q2) for both the holdout and the test.
    def target_free(apply_df):
        aggregates = homeprice.location_aggregates(train_folded, apply_df, "地区名", prefix="district").hstack(
            homeprice.location_aggregates(train_folded, apply_df, "最寄駅：名称", prefix="station")
        )
        counts = homeprice.count_features(train_folded, apply_df, ["間取り", "地区名", "最寄駅：名称"])
        return aggregates.to_pandas(), counts.to_pandas()

    aggregates_train, counts_train = target_free(train_folded)
    aggregates_test, counts_test = target_free(test_raw)
    return aggregates_test, aggregates_train, counts_test, counts_train


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 足して比べる

    各 run は S4 と同じく、実質価格（log 価格 − 価格指数）で学習し、holdout では外挿した指数を足し戻して採点します。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    X_base,
    aggregates_train,
    asdict,
    cfg,
    counts_train,
    env,
    get_metric,
    git_commit,
    lesson,
    np,
    offset_holdout,
    pd,
    pl,
    te_holdout,
    te_holdout_in_sample,
    te_holdout_raw_leak,
    tr_idx,
    train_folded,
    va_idx,
    y_log,
):
    rmsle = get_metric("rmsle")
    commit = git_commit(env.root)
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    EXTRA = {
        "te": [te_holdout],
        "te_in_sample": [te_holdout_in_sample],
        "te_raw_leak": [te_holdout_raw_leak],
        "aggregates": [aggregates_train],
        "counts": [counts_train],
        "all_three": [te_holdout, aggregates_train, counts_train],
    }
    ISSUES = {
        "te": "#18",
        "te_in_sample": "#18",
        "te_raw_leak": "#18",
        "aggregates": "#19",
        "counts": "#33",
        "all_three": "#18 #19 #33",
    }
    results = {}
    boosters = {}
    holdout_logs = {}
    for _run, _blocks in EXTRA.items():
        _X = pd.concat([X_base, *_blocks], axis=1)
        _booster, _pred_real = lesson.fit_lgbm_holdout(
            _X,
            y_log - offset_holdout,
            tr_idx,
            va_idx,
            asdict(cfg.params),
            max_rounds=cfg.max_rounds,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        _pred_log = _pred_real + offset_holdout[va_idx]
        _score = rmsle(np.expm1(y_log[va_idx]), np.expm1(_pred_log))
        results[_run] = {
            "run": _run,
            "issue": ISSUES[_run],
            "holdout_rmsle": _score,
            "bias": float((_pred_log - y_log[va_idx]).mean()),
            "best_iteration": _booster.best_iteration,
        }
        boosters[_run] = _booster
        holdout_logs[_run] = _pred_log
        lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=_run,
            score=_score,
            metric=cfg.metric,
            config=asdict(cfg)
            | {
                "base_run": "s4_trend/hedonic_linear",
                "extra_columns": [c for b in _blocks for c in b.columns],
                "best_iteration": _booster.best_iteration,
                "refit_rounds": lesson.refit_rounds(_booster.best_iteration, n_all=len(y_log), n_train=len(tr_idx)),
            },
            holdout=holdout_ids,
            pred=np.expm1(_pred_log),
            notes=f"S4 hedonic_linear + {_run} ({ISSUES[_run]})",
            commit=commit,
        )
    return EXTRA, boosters, holdout_ids, holdout_logs, results


@app.cell
def _(env, holdout_ids, holdout_logs, np, pl, results, va_idx, y_log):
    s4_oof = pl.read_parquet(env.output_dir / "s4_trend" / "hedonic_linear" / "oof.parquet")
    assert s4_oof["id"].to_list() == holdout_ids["id"].to_list()
    s4_log = np.log1p(s4_oof["pred"].to_numpy())
    squared_s4 = (s4_log - y_log[va_idx]) ** 2
    boot_idx = np.random.default_rng(0).integers(0, len(va_idx), size=(2000, len(va_idx)))

    def paired_diff(sq_a, sq_b):
        """RMSLE(a) - RMSLE(b) on the holdout and its 95% bootstrap range, resampling the same rows for both."""
        diffs = np.sqrt(sq_a[boot_idx].mean(axis=1)) - np.sqrt(sq_b[boot_idx].mean(axis=1))
        low, high = np.percentile(diffs, [2.5, 97.5])
        return float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean())), float(low), float(high)

    summary = pl.DataFrame(
        [
            {
                "run": "s4 hedonic_linear",
                "issue": "#16",
                "holdout_rmsle": float(np.sqrt(squared_s4.mean())),
                "bias": float((s4_log - y_log[va_idx]).mean()),
            }
        ]
        + [
            {k: r[k] for k in ("run", "issue", "holdout_rmsle", "bias")}
            | dict(
                zip(
                    ("change", "ci_low", "ci_high"),
                    paired_diff((holdout_logs[r["run"]] - y_log[va_idx]) ** 2, squared_s4),
                    strict=True,
                )
            )
            for r in results.values()
        ],
        infer_schema_length=None,
    )
    summary
    return paired_diff, summary


@app.cell
def _(mo, pl, summary):
    def ci(run):
        r = summary.filter(pl.col("run") == run).row(0, named=True)
        return f"{r['holdout_rmsle']:.4f}（S4 との差 {r['change']:+.4f}、95% の範囲 {r['ci_low']:+.4f}〜{r['ci_high']:+.4f}）"

    mo.md(f"""
    ## 3. 結果を読む

    - Target encoding（`te`、#18）: {ci("te")}
    - out-of-fold にしない TE（`te_in_sample`）: {ci("te_in_sample")}
    - out-of-fold にも平滑化にもしない TE（`te_raw_leak`）: {ci("te_raw_leak")}
    - 集約特徴（`aggregates`、#19）: {ci("aggregates")}
    - 件数（`counts`、#33）: {ci("counts")}
    - 3つ全部（`all_three`）: {ci("all_three")}
    """)
    return


@app.cell
def _(pl, tr_idx, train_folded):
    group_sizes = pl.DataFrame(
        [
            {
                "key": key,
                "groups": g.height,
                "median_rows_per_group": float(g["len"].median()),
                "share_of_rows_in_groups_under_10": float(g.filter(pl.col("len") < 10)["len"].sum() / len(tr_idx)),
            }
            for key in ("地区名", "最寄駅：名称")
            for g in [train_folded[tr_idx].group_by(key).len()]
        ]
    )
    group_sizes
    return (group_sizes,)


@app.cell
def _(group_sizes, mo, pl, summary):
    def r(run):
        return summary.filter(pl.col("run") == run).row(0, named=True)

    district = group_sizes.row(0, named=True)
    mo.md(f"""
    - Target encoding は、S4 から {r("te")["change"]:+.4f} と、偶然の範囲を超えて良くなりました。地区名のカテゴリだけでは、行の少ない地区の値段の高さをうまく学べていなかったと考えられます。
    - リークする TE（`te_in_sample` と `te_raw_leak`）は、予想に反して正しい TE とほとんど変わりませんでした。
      上の表のとおり、学習期間の地区は1つあたり中央値 {district["median_rows_per_group"]:.0f} 行あり、10行未満の小さな地区の行は全体の {district["share_of_rows_in_groups_under_10"]:.2%} だけです。
      1行の価格が平均に占める重みは1%前後しかないので、自分の行を含めても答えはほとんど漏れません。
      リークの害はグループが小さいほど大きくなります。建物ごとのような小さなグループで同じことをすれば、CV は大きく崩れます。
      このデータでは目立たなかったというだけで、out-of-fold で作るのが基本です。
    - 集約特徴（{r("aggregates")["change"]:+.4f}）と件数（{r("counts")["change"]:+.4f}）は、単独ではどちらも偶然の範囲でした。
    - 3つ全部（`all_three`）は {r("all_three")["change"]:+.4f} でした。TE だけとの差（-0.0003）は偶然の範囲ですが、
      S6 以降で別の GBDT やアンサンブルを試すときに特徴量が多いほうが有利な可能性をとって、3つ全部を採用しました（2026-10-10、ユーザーが決定）。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 残っているずれ

    採用した `all_three` の残差を、S4 と同じ切り口で見ます。
    """)
    return


@app.cell
def _(holdout_logs, mlit, pl, train_folded, va_idx, y_log):
    FINAL = "all_three"
    residual_frame = train_folded[va_idx].with_columns(
        residual=pl.Series(holdout_logs[FINAL] - y_log[va_idx]),
        age=mlit.trade_period("取引時期").floor() - mlit.build_year("建築年"),
        area=mlit.area_m2("面積（㎡）"),
    )

    def residual_by(frame, key):
        return (
            frame.group_by(key)
            .agg(
                pl.len().alias("rows"),
                pl.col("residual").mean().round(3).alias("bias"),
                (pl.col("residual") ** 2).mean().sqrt().round(3).alias("rmsle"),
            )
            .sort(key)
        )

    residual_tables = {
        "quarter": residual_by(residual_frame, "取引時期"),
        "type": residual_by(residual_frame, "価格情報区分"),
        "age": residual_by(
            residual_frame.drop_nulls("age")
            .filter(pl.col("age").is_between(0, 59))
            .with_columns(age_bin=pl.col("age") // 10 * 10),
            "age_bin",
        ),
    }
    mo_tables = (residual_tables["quarter"], residual_tables["type"], residual_tables["age"])
    return FINAL, mo_tables, residual_tables


@app.cell
def _(mo, mo_tables):
    mo.hstack(list(mo_tables), justify="start", gap=2)
    return


@app.cell
def _(mo, residual_tables):
    age_bias = residual_tables["age"]["bias"].to_list()
    mo.md(f"""
    築年数による違いは、築10年未満で {age_bias[0]:+.3f}、築50年代で {age_bias[-1]:+.3f} です（S4 では -0.025 と +0.013）。
    場所の情報を足しても、新しい物件ほど低めという傾向はあまり変わりませんでした。S6 以降で、モデルを変えたときにこの傾向が変わるかを見ます。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. 提出ファイルを作る

    S5 までの特徴量一式は `kgl.feature_set`（テスト付き）にまとめ、S6 以降もこれを使います。
    まず、`feature_set` が作る holdout 用の特徴量が、上の `all_three` で使ったものと完全に一致することを確かめます。
    そのうえで、全期間の行で価格指数と Target encoding を作り直し、全期間で学習し直したモデルで test を予測します。
    """)
    return


@app.cell
def _(
    EXP,
    FINAL,
    X_base,
    aggregates_train,
    asdict,
    boosters,
    cfg,
    counts_train,
    env,
    feature_set,
    lesson,
    lgb,
    np,
    offset_holdout,
    pd,
    pl,
    sample,
    te_holdout,
    test_raw,
    tr_idx,
    train_folded,
    va_idx,
    validate_submission,
    write_submission,
    y_log,
):
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
    pd.testing.assert_frame_equal(fs.X, pd.concat([X_base, te_holdout, aggregates_train, counts_train], axis=1))
    np.testing.assert_array_equal(fs.offset, offset_holdout)
    refit_rounds = lesson.refit_rounds(boosters[FINAL].best_iteration, n_all=len(y_log), n_train=len(tr_idx))
    refit_booster = lgb.train(
        asdict(cfg.params), lgb.Dataset(fs.X_refit, label=y_log - fs.offset_refit), num_boost_round=refit_rounds
    )
    submission = pl.DataFrame(
        {cfg.id_col: test_raw[cfg.id_col], "price": np.expm1(refit_booster.predict(fs.X_test) + fs.offset_test)}
    )
    validate_submission(submission, sample, cfg.id_col)
    submission_path = write_submission(env.output_dir / EXP / FINAL, submission)
    (refit_rounds, str(submission_path))
    return


if __name__ == "__main__":
    app.run()
