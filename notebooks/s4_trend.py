import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from dataclasses import asdict, dataclass, field

    import lightgbm as lgb
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import psutil

    from kgl import cv, homeprice, lesson, mlit, price_index
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, validate_submission, write_submission
    from kgl.metrics import get_metric

    return (
        asdict,
        comp_dir,
        cv,
        dataclass,
        detect,
        field,
        get_metric,
        git_commit,
        homeprice,
        lesson,
        lgb,
        mlit,
        mo,
        np,
        pl,
        plt,
        price_index,
        psutil,
        validate_submission,
        write_submission,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S4. 価格指数と時間トレンド

    S3 で取引時期を数値にし、holdout RMSLE は 0.2090 になりました。それでも予測は平均で約 4% 低く、
    四半期が進むほど低めになっていました（2025年第1四半期 -0.031、第2四半期 -0.050）。

    原因は木のモデルの性質です。木は、学習で見た値の範囲の中で予測を組み立てます。
    学習期間（2024年まで）より高い価格が続く未来に対しては、学習期間の最後の水準より上を出せません。

    この段階では、価格の水準の動きをモデルの外に出します。

    1. 学習データから、四半期ごとの価格の水準（価格指数）を作る
    2. 目的変数を指数で割った「実質価格」にして学習する（log では「log 価格 − log 指数」）
    3. 予測するときは、未来の四半期の指数を外挿して掛け戻す

    外部の物価や地価の系列は使いません。2025年第3四半期以降の不動産の価格系列はテスト期間の取引から作られているので、ルール上使えないからです（#16）。
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
        # Linear extrapolation fits the last k quarters of the index (2 years).
        trend_quarters: int = 8
        params: LgbParams = field(default_factory=LgbParams)

    cfg = Config()
    EXP = "s4_trend"
    BASE = "s3_features"
    # The features S3 adopted (#12, #9): S1's columns with the trade period and the notation parsed.
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
    log_area_test = test_raw.select(mlit.area_m2("面積（㎡）").log().alias("a"))["a"].to_numpy()
    # Price per m2 on the log scale is what the indices track.
    log_ppm2 = y_log - log_area

    vocab = homeprice.make_vocab([train_folded, test_raw])
    X = homeprice.features(train_folded, ADOPTED, vocab)
    X_test = homeprice.features(test_raw, ADOPTED, vocab)
    return (
        X,
        X_test,
        env,
        log_area,
        log_area_test,
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
def _(mo):
    mo.md(r"""
    ## 1. 価格指数を2通りで作る

    どちらも log ㎡単価の水準を四半期ごとに測り、最初の四半期（2019年第1四半期）を 0 とした値です。

    - 中央値の指数: 四半期ごとの log ㎡単価の中央値。単純ですが、その四半期に取引された物件の顔ぶれが変わると、値も動いてしまいます。
    - ヘドニック指数: log ㎡単価を「四半期」と「物件の性質（市区町村、価格情報区分、面積、築年数、駅距離）」で回帰し、四半期の効果だけを取り出したもの。
      同じ性質の物件の価格が、時期によってどれだけ違うかを測るので、顔ぶれの変化に引っぱられません。

    holdout の評価に使う指数は学習期間（2024年まで）の行だけで、提出に使う指数は train 全体（2025年第2四半期まで）で作ります。
    """)
    return


@app.cell
def _(homeprice, log_ppm2, np, price_index, t_train, tr_idx, train_folded):
    all_rows = np.arange(len(t_train))
    indices = {}
    for _scope, _rows in (("train_block", tr_idx), ("all_train", all_rows)):
        indices[(_scope, "median")] = price_index.median_index(t_train[_rows], log_ppm2[_rows])
        indices[(_scope, "hedonic")] = price_index.hedonic_index(
            t_train[_rows], log_ppm2[_rows], homeprice.hedonic_controls(train_folded[_rows])
        )
    return all_rows, indices


@app.cell
def _(cfg, indices, np, plt, price_index, t_test, t_train, va_idx):
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    fig_index, axes_index = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    panels = (
        (axes_index[0], "train_block", np.unique(t_train[va_idx]), "fit on 2019-2024, extrapolated to the holdout"),
        (axes_index[1], "all_train", np.unique(t_test), "fit on 2019-2025Q2, extrapolated to the test"),
    )
    for _ax, _scope, _ahead, _title in panels:
        for _kind, _color in (("median", BLUE), ("hedonic", ORANGE)):
            _periods, _index = indices[(_scope, _kind)]
            _extrapolated = price_index.extrapolate(_periods, _index, _ahead, how="linear", k=cfg.trend_quarters)
            _ax.plot(_periods, _index, color=_color, linewidth=2, marker="o", markersize=3)
            _ax.plot(
                np.r_[_periods[-1], _ahead], np.r_[_index[-1], _extrapolated], color=_color, linewidth=2, linestyle="--"
            )
            _ax.text(_ahead[-1] + 0.05, _extrapolated[-1], _kind, fontsize=8, color="#333333")
        _ax.set_title(_title, fontsize=9)
        _ax.set_xlabel("trade quarter")
        _ax.set_xlim(2018.8, 2026.6)
        _ax.spines[["top", "right"]].set_visible(False)
    axes_index[0].set_ylabel("log price-per-m2 level vs 2019Q1")
    fig_index.tight_layout()
    fig_index
    return


@app.cell
def _(cfg, indices, mo, np):
    def yearly_slope(scope, kind):
        periods, index = indices[(scope, kind)]
        return float(np.polyfit(periods[-cfg.trend_quarters :], index[-cfg.trend_quarters :], 1)[0])

    slopes = {key: yearly_slope(*key) for key in indices}
    mo.md(f"""
    実線が指数、破線が直近8四半期の直線を延ばした外挿です。

    - 学習期間だけで作ると（左）、2つの指数の伸びは近く、直近2年の傾きは中央値が年 {slopes[("train_block", "median")]:+.3f}、ヘドニックが年 {slopes[("train_block", "hedonic")]:+.3f}（log）です。
    - train 全体で作ると（右）、傾きは中央値が年 {slopes[("all_train", "median")]:+.3f}、ヘドニックが年 {slopes[("all_train", "hedonic")]:+.3f} と大きく離れます。

    右で中央値の指数が伸び悩むのは、2025年に成約価格情報（㎡単価が低めの区分）の件数が急に増え、取引された物件の顔ぶれが変わったからです（S0）。
    同じ物件の値段が下がったわけではありません。ヘドニック指数は価格情報区分をそろえて比べるので、この影響を受けません。

    同じことは2021年第1四半期にも起きています。中央値の指数はこの四半期に急に下がりますが、これは成約価格情報がデータに加わった時期です。
    物件の値段が下がったのではなく、単価が低めの区分の行が増えただけです。ヘドニック指数は、この時期もなめらかに上がっています。
    """)
    return (slopes,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 実質価格で学習し、指数を外挿して戻す

    各 run は S3 で採用した特徴量（`plus_notation`）をそのまま使い、目的変数と指数の作り方だけを変えます。

    | run | 指数 | 外挿 | ほかの変更 | Issue |
    |---|---|---|---|---|
    | `median_flat` | 中央値 | 最後の水準のまま | | #16 |
    | `median_linear` | 中央値 | 直近8四半期の直線 | | #16 |
    | `hedonic_linear` | ヘドニック | 直近8四半期の直線 | | #16 |
    | `hedonic_by_age` | 築年数帯ごとのヘドニック | 直近8四半期の直線 | | #32 |
    | `hedonic_by_area` | 面積帯ごとのヘドニック | 直近8四半期の直線 | | #32 |
    | `hedonic_ppm2` | ヘドニック | 直近8四半期の直線 | 目的変数を㎡単価にする | #15 |
    | `hedonic_recent` | ヘドニック | 直近8四半期の直線 | 新しい四半期ほど重く学習する（半減期2年） | #17 |

    holdout の行の指数は、学習期間の指数を外挿した値です。holdout の正解はどこにも使いません。early stopping も、外挿した指数で割った holdout の値で判定します。
    """)
    return


@app.cell
def _(cfg, homeprice, log_ppm2, mlit, np, pl, price_index, t_train, tr_idx, train_folded, va_idx):
    age_years = train_folded.select((mlit.trade_period("取引時期").floor() - mlit.build_year("建築年")).alias("age"))[
        "age"
    ].to_numpy()
    area_m2 = train_folded.select(mlit.area_m2("面積（㎡）").alias("area"))["area"].to_numpy()
    SEGMENTS = {
        "age": np.select([np.isnan(age_years), age_years < 10, age_years < 30], [3, 0, 1], 2),
        "area": np.select([area_m2 < 30, area_m2 < 70], [0, 1], 2),
    }

    def index_for_rows(kind, how, segment=None):
        """Index value per training row: in-sample for the training block, extrapolated for the holdout."""
        out = np.full(len(t_train), np.nan)
        groups = (
            [np.ones(len(t_train), bool)]
            if segment is None
            else [SEGMENTS[segment] == g for g in np.unique(SEGMENTS[segment])]
        )
        for in_group in groups:
            fit_rows, ahead_rows = tr_idx[in_group[tr_idx]], va_idx[in_group[va_idx]]
            if kind == "median":
                periods, index = price_index.median_index(t_train[fit_rows], log_ppm2[fit_rows])
            else:
                periods, index = price_index.hedonic_index(
                    t_train[fit_rows], log_ppm2[fit_rows], homeprice.hedonic_controls(train_folded[fit_rows])
                )
            out[fit_rows] = price_index.lookup(periods, index, t_train[fit_rows])
            out[ahead_rows] = price_index.extrapolate(
                periods, index, t_train[ahead_rows], how=how, k=cfg.trend_quarters
            )
        assert not np.isnan(out).any()
        return out

    return SEGMENTS, index_for_rows


@app.cell
def _(
    BASE,
    EXP,
    X,
    asdict,
    cfg,
    env,
    get_metric,
    git_commit,
    index_for_rows,
    lesson,
    log_area,
    np,
    pl,
    t_train,
    tr_idx,
    train_folded,
    va_idx,
    y_log,
):
    rmsle = get_metric("rmsle")
    commit = git_commit(env.root)
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    recent_weight = 0.5 ** ((t_train[tr_idx].max() - t_train) / 2.0)

    RUNS = [
        # (run, index kind, extrapolation, segment, per-m2 target, recent weights, compared with, Issue)
        ("median_flat", "median", "flat", None, False, False, "s3 plus_notation", 16),
        ("median_linear", "median", "linear", None, False, False, "s3 plus_notation", 16),
        ("hedonic_linear", "hedonic", "linear", None, False, False, "s3 plus_notation", 16),
        ("hedonic_by_age", "hedonic", "linear", "age", False, False, "hedonic_linear", 32),
        ("hedonic_by_area", "hedonic", "linear", "area", False, False, "hedonic_linear", 32),
        ("hedonic_ppm2", "hedonic", "linear", None, True, False, "hedonic_linear", 15),
        ("hedonic_recent", "hedonic", "linear", None, False, True, "hedonic_linear", 17),
    ]
    results = {}
    boosters = {}
    holdout_logs = {}
    for _run, _kind, _how, _segment, _ppm2, _recent, _compared, _issue in RUNS:
        # Everything removed from the target here is added back to the prediction below.
        _offset = index_for_rows(_kind, _how, _segment) + (log_area if _ppm2 else 0.0)
        _booster, _pred_real = lesson.fit_lgbm_holdout(
            X,
            y_log - _offset,
            tr_idx,
            va_idx,
            asdict(cfg.params),
            max_rounds=cfg.max_rounds,
            early_stopping_rounds=cfg.early_stopping_rounds,
            weight=recent_weight if _recent else None,
        )
        _pred_log = _pred_real + _offset[va_idx]
        _err = _pred_log - y_log[va_idx]
        _score = rmsle(np.expm1(y_log[va_idx]), np.expm1(_pred_log))
        results[_run] = {
            "run": _run,
            "issue": f"#{_issue}",
            "compared_with": _compared,
            "holdout_rmsle": _score,
            "bias": float(_err.mean()),
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
                "steps": ["notation", "period", "type"],
                "index": _kind,
                "extrapolation": _how,
                "segment": _segment,
                "target": "log price per m2" if _ppm2 else "log price",
                "recent_weight_half_life_years": 2.0 if _recent else None,
                "best_iteration": _booster.best_iteration,
                "refit_rounds": lesson.refit_rounds(_booster.best_iteration, n_all=len(y_log), n_train=len(tr_idx)),
            },
            holdout=holdout_ids,
            pred=np.expm1(_pred_log),
            notes=f"{_run} (#{_issue})",
            commit=commit,
        )
    return boosters, holdout_ids, holdout_logs, results


@app.cell
def _(env, holdout_ids, holdout_logs, np, pl, results, va_idx, y_log):
    s3_oof = pl.read_parquet(env.output_dir / "s3_features" / "plus_notation" / "oof.parquet")
    assert s3_oof["id"].to_list() == holdout_ids["id"].to_list()
    squared = {"s3 plus_notation": (np.log1p(s3_oof["pred"].to_numpy()) - y_log[va_idx]) ** 2} | {
        name: (pred - y_log[va_idx]) ** 2 for name, pred in holdout_logs.items()
    }
    boot_idx = np.random.default_rng(0).integers(0, len(va_idx), size=(2000, len(va_idx)))

    def paired_diff(sq_a, sq_b):
        """RMSLE(a) - RMSLE(b) on the holdout and its 95% bootstrap range, resampling the same rows for both."""
        diffs = np.sqrt(sq_a[boot_idx].mean(axis=1)) - np.sqrt(sq_b[boot_idx].mean(axis=1))
        low, high = np.percentile(diffs, [2.5, 97.5])
        return float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean())), float(low), float(high)

    s3_score = float(np.sqrt(squared["s3 plus_notation"].mean()))
    summary = pl.DataFrame(
        [
            {
                "run": "s3 plus_notation",
                "issue": "#12 #9",
                "compared_with": "",
                "holdout_rmsle": s3_score,
                "bias": float(np.mean(np.log1p(s3_oof["pred"].to_numpy()) - y_log[va_idx])),
            }
        ]
        + [
            {k: r[k] for k in ("run", "issue", "compared_with", "holdout_rmsle", "bias")}
            | dict(
                zip(
                    ("change", "ci_low", "ci_high"),
                    paired_diff(squared[r["run"]], squared[r["compared_with"]]),
                    strict=True,
                )
            )
            for r in results.values()
        ],
        infer_schema_length=None,
    )
    summary
    return paired_diff, s3_score, summary


@app.cell
def _(mo, pl, summary):
    def row(run):
        return summary.filter(pl.col("run") == run).row(0, named=True)

    def ci(run):
        r = row(run)
        return f"{r['change']:+.4f}（95% の範囲 {r['ci_low']:+.4f}〜{r['ci_high']:+.4f}）"

    mo.md(f"""
    ## 3. 結果を読む

    - 中央値の指数で、最後の水準のまま（`median_flat`）: {ci("median_flat")}。外挿しなければ、S3 とほとんど変わりません。実質価格にするだけでは足りないということです。
    - 直近8四半期の直線で外挿（`median_linear`）: {ci("median_linear")}。平均のずれが {row("s3 plus_notation")["bias"]:+.3f} から {row("median_linear")["bias"]:+.3f} になりました。
    - ヘドニック指数で外挿（`hedonic_linear`）: {ci("hedonic_linear")}。holdout では中央値の指数とほぼ同じです。
    - 築年数帯・面積帯ごとの指数（#32）: {ci("hedonic_by_age")}、{ci("hedonic_by_area")}。区分ごとに分けると1つの指数あたりの行が減り、伸びの推定がぶれます。全体で1本の指数より良くはなりませんでした。
    - 目的変数を㎡単価にする（#15）: {ci("hedonic_ppm2")}。差は偶然の範囲です。
    - 新しい四半期を重く学習する（#17）: {ci("hedonic_recent")}。指数で水準をそろえたあとでは、重みを付けても良くなりませんでした。
    """)
    return (row,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 中央値の指数とヘドニック指数のどちらを使うか

    holdout では2つの指数の差はほとんどありません。しかし第1節の図のとおり、提出に使う train 全体の指数では、伸びの傾きが大きく違います。
    test は 2025年第3四半期から2026年第1四半期で、学習の最後から1〜3四半期先です。傾きの差は先に行くほど広がります。

    holdout でこの違いを確かめることはできません。holdout の指数を作る期間（2024年まで）には、2025年の顔ぶれの変化が入っていないからです。
    そこで補助の確認として、2025年第1四半期までで学習して第2四半期を予測し、顔ぶれの変化を含む期間で外挿した場合を比べます。
    """)
    return


@app.cell
def _(X, asdict, cfg, homeprice, lesson, log_area, log_ppm2, np, price_index, t_train, train_folded, y_log):
    fit_q2 = np.flatnonzero(t_train <= 2025.0)
    ahead_q2 = np.flatnonzero(t_train == 2025.25)

    def backtest(kind):
        if kind == "median":
            periods, index = price_index.median_index(t_train[fit_q2], log_ppm2[fit_q2])
        else:
            periods, index = price_index.hedonic_index(
                t_train[fit_q2], log_ppm2[fit_q2], homeprice.hedonic_controls(train_folded[fit_q2])
            )
        offset = np.empty(len(t_train))
        offset[fit_q2] = price_index.lookup(periods, index, t_train[fit_q2])
        offset[ahead_q2] = price_index.extrapolate(
            periods, index, t_train[ahead_q2], how="linear", k=cfg.trend_quarters
        )
        offset = offset + log_area
        _, pred_real = lesson.fit_lgbm_holdout(
            X,
            y_log - offset,
            fit_q2,
            ahead_q2,
            asdict(cfg.params),
            max_rounds=cfg.max_rounds,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        err = pred_real + offset[ahead_q2] - y_log[ahead_q2]
        return float(np.sqrt((err**2).mean())), float(err.mean())

    backtest_q2 = {kind: backtest(kind) for kind in ("median", "hedonic")}
    backtest_q2
    return (backtest_q2,)


@app.cell
def _(backtest_q2, mo, slopes):
    mo.md(f"""
    2025年第2四半期の補助の確認では、RMSLE は中央値の指数で {backtest_q2["median"][0]:.4f}（平均のずれ {backtest_q2["median"][1]:+.3f}）、
    ヘドニック指数で {backtest_q2["hedonic"][0]:.4f}（{backtest_q2["hedonic"][1]:+.3f}）でした。1四半期先では、傾きの差がまだ小さいので、結果もほぼ同じです。

    データで決着がつかないときは、test の条件に照らして理屈で選びます。test では成約価格情報の割合がさらに高く（67%）、
    中央値の指数の伸び（年 {slopes[("all_train", "median")]:+.3f}）は顔ぶれの変化で小さく出ていると考えられます。
    ヘドニック指数（年 {slopes[("all_train", "hedonic")]:+.3f}）は、同じ性質の物件の値動きを測っています。
    そこで提出にはヘドニック指数を使います（`hedonic_linear`）。目的変数を㎡単価にする案（#15）は差が偶然の範囲だったので、単純なほうを選びます。
    ㎡単価を目的変数にしたモデルは、S9 のアンサンブルで別のメンバーとして使う候補に残します。採用は 2026-10-10 にユーザーと決めました。

    この選択は、holdout では確かめられない判断を含みます。最初の LB で、CV との差を確かめる必要があります。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. 残っているずれ

    採用する run（`hedonic_linear`）の残差を、S3 と同じ切り口で見ます。
    """)
    return


@app.cell
def _(holdout_logs, mlit, pl, train_folded, va_idx, y_log):
    FINAL = "hedonic_linear"
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
        "area": residual_by(
            residual_frame.with_columns(area_bin=pl.col("area").cut([20, 30, 50, 70, 100])), "area_bin"
        ),
    }
    return FINAL, residual_tables


@app.cell
def _(mo, residual_tables):
    mo.vstack(
        [
            mo.hstack([residual_tables["quarter"], residual_tables["type"]], justify="start", gap=2),
            mo.hstack([residual_tables["age"], residual_tables["area"]], justify="start", gap=2),
        ]
    )
    return


@app.cell
def _(mo, residual_tables):
    by_quarter = residual_tables["quarter"]["bias"].to_list()
    by_age = residual_tables["age"]
    mo.md(f"""
    四半期ごとの平均のずれは {by_quarter[0]:+.3f} と {by_quarter[1]:+.3f} になり、S3（-0.031 と -0.050）からほぼ消えました。
    一方、築年数による違いは残っています。築10年未満は {by_age["bias"][0]:+.3f}、築50年代は {by_age["bias"][-1]:+.3f} で、
    新しい物件ほど低め、古い物件ほど高めに予測しています。区分ごとの指数（#32）ではこの差を縮められなかったので、
    S5 の Target encoding や S6・S7 の別のモデルで変わるかを見ていきます。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. 提出ファイルを作る

    提出用には、train 全体（2025年第2四半期まで）でヘドニック指数を作り直し、test の四半期（2025年第3四半期〜2026年第1四半期）へ直線で外挿します。
    モデルも全期間で学習し直し、本数は S1 と同じ規則（holdout での本数 × 行数の比）で決めます。提出するかどうかは別に決めます。
    """)
    return


@app.cell
def _(
    EXP,
    FINAL,
    X,
    X_test,
    all_rows,
    asdict,
    boosters,
    cfg,
    env,
    indices,
    lesson,
    lgb,
    np,
    pl,
    price_index,
    sample,
    t_test,
    t_train,
    test_raw,
    tr_idx,
    validate_submission,
    write_submission,
    y_log,
):
    periods_all, index_all = indices[("all_train", "hedonic")]
    offset_train = price_index.lookup(periods_all, index_all, t_train[all_rows])
    offset_test = price_index.extrapolate(periods_all, index_all, t_test, how="linear", k=cfg.trend_quarters)
    refit_rounds = lesson.refit_rounds(boosters[FINAL].best_iteration, n_all=len(y_log), n_train=len(tr_idx))
    refit_booster = lgb.train(
        asdict(cfg.params), lgb.Dataset(X, label=y_log - offset_train), num_boost_round=refit_rounds
    )
    test_log = refit_booster.predict(X_test) + offset_test
    submission = pl.DataFrame({cfg.id_col: test_raw[cfg.id_col], "price": np.expm1(test_log)})
    validate_submission(submission, sample, cfg.id_col)
    submission_path = write_submission(env.output_dir / EXP / FINAL, submission)
    test_index_by_quarter = {
        float(q): float(
            price_index.extrapolate(periods_all, index_all, np.array([q]), how="linear", k=cfg.trend_quarters)[0]
            - index_all[-1]
        )
        for q in np.unique(t_test)
    }
    (refit_rounds, str(submission_path), test_index_by_quarter)
    return


if __name__ == "__main__":
    app.run()
