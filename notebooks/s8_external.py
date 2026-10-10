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

    from kgl import cv, external, feature_set, lesson
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, validate_submission, write_submission
    from kgl.metrics import get_metric

    return (
        asdict,
        comp_dir,
        cv,
        dataclass,
        detect,
        external,
        feature_set,
        field,
        get_metric,
        git_commit,
        lesson,
        lgb,
        mo,
        np,
        pd,
        pl,
        psutil,
        validate_submission,
        write_submission,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S8. 外部データ（駅と地区の位置）

    これまでモデルが知っていた場所の情報は、市区町村名・地区名・駅名という「名前」と、S5 で作った地区・駅ごとの平均価格だけでした。
    名前からは、2つの地区が隣り合っているのか遠く離れているのかは分かりません。この段階では、公開されている地理のデータを足します。

    | データ | 出典 | 使う値 | Issue |
    |---|---|---|---|
    | 駅別乗降客数（2024年） | 国土数値情報 S12（国土交通省、CC BY 4.0） | 駅の緯度・経度、1日の乗降客数 | #21 |
    | 町丁目の代表点 | Geolonia japanese-addresses（国土交通省 位置参照情報をもとに作成、CC BY 4.0） | 地区の緯度・経度、東京駅までの距離 | #22 |

    どちらも公開・無料で、参加者全員が同じように入手できます。不動産の取引データではないので、コンペのルール（テスト期間の取引データの禁止）にも触れません。
    ダウンロードは `just external` で再現できます。

    名前の表記は少し違います。こちらのデータの「大森(東京)」は S12 では「大森」、「幡ケ谷」は「幡ヶ谷」です。括弧の補足を外し、ヶ をケにそろえて結合します。
    地区名は町丁目の「〜丁目」を外した町の単位なので、丁目ごとの代表点を平均します。
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
        te_smoothing: float = 20.0
        te_folds: int = 5
        seed: int = 42
        stations_file: str = "external/s12_2024/S12-25_GML/UTF-8/S12-25_NumberOfPassengers.geojson"
        towns_file: str = "external/geolonia/latest.csv"
        params: LgbParams = field(default_factory=LgbParams)

    cfg = Config()
    EXP = "s8_external"
    BASE = "s5_te"
    return BASE, EXP, cfg


@app.cell
def _(cfg, comp_dir, cv, detect, external, feature_set, np, pl):
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
    stations = external.load_stations(env.input_dir / cfg.stations_file)
    towns = external.load_towns(env.input_dir / cfg.towns_file)

    def geo(df):
        station = (
            external.station_features(df, stations)
            .with_columns(pl.col("station_riders").cast(pl.Float64).log1p().alias("log_station_riders"))
            .drop("station_riders")
        )
        return station.to_pandas(), external.district_features(df, towns).to_pandas()

    (station_train, district_train), (station_test, district_test) = geo(train_folded), geo(test_raw)
    match_rates = {
        "station rows matched": float(np.isfinite(station_train["station_lat"]).mean()),
        "district rows matched": float(np.isfinite(district_train["district_lat"]).mean()),
    }
    match_rates
    return (
        district_test,
        district_train,
        env,
        fs,
        sample,
        station_test,
        station_train,
        test_raw,
        tr_idx,
        train_folded,
        va_idx,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 足して比べる

    S5 で採用した LightGBM（`all_three`）に、駅の情報（緯度・経度・乗降客数の log）、地区の位置（緯度・経度・東京駅までの距離）、両方、をそれぞれ足します。
    外部データには目的変数が入らないので、holdout の設定でも提出の設定でも同じ列を使えます。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    cfg,
    district_train,
    env,
    fs,
    get_metric,
    git_commit,
    lesson,
    np,
    pd,
    pl,
    station_train,
    tr_idx,
    train_folded,
    va_idx,
):
    rmsle = get_metric("rmsle")
    commit = git_commit(env.root)
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    y_real = fs.y_log - fs.offset
    EXTRA = {
        "stations": ([station_train], "#21"),
        "districts": ([district_train], "#22"),
        "both": ([station_train, district_train], "#21 #22"),
    }
    results = {}
    boosters = {}
    holdout_logs = {}
    for _run, (_blocks, _issue) in EXTRA.items():
        _X = pd.concat([fs.X, *_blocks], axis=1)
        _booster, _pred_real = lesson.fit_lgbm_holdout(
            _X,
            y_real,
            tr_idx,
            va_idx,
            asdict(cfg.params),
            max_rounds=cfg.max_rounds,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        _pred_log = _pred_real + fs.offset[va_idx]
        _score = rmsle(np.expm1(fs.y_log[va_idx]), np.expm1(_pred_log))
        results[_run] = {
            "run": _run,
            "issue": _issue,
            "holdout_rmsle": _score,
            "bias": float((_pred_log - fs.y_log[va_idx]).mean()),
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
                "features": "kgl.feature_set (S5 all_three) + external",
                "extra_columns": [c for b in _blocks for c in b.columns],
                "best_iteration": _booster.best_iteration,
                "refit_rounds": lesson.refit_rounds(_booster.best_iteration, n_all=len(fs.y_log), n_train=len(tr_idx)),
            },
            holdout=holdout_ids,
            pred=np.expm1(_pred_log),
            notes=f"S5 all_three + {_run} ({_issue})",
            commit=commit,
        )
    return EXTRA, boosters, holdout_ids, holdout_logs, results


@app.cell
def _(env, fs, holdout_ids, holdout_logs, np, pl, results, va_idx):
    s5_oof = pl.read_parquet(env.output_dir / "s5_te" / "all_three" / "oof.parquet")
    assert s5_oof["id"].to_list() == holdout_ids["id"].to_list()
    s5_squared = (np.log1p(s5_oof["pred"].to_numpy()) - fs.y_log[va_idx]) ** 2
    boot_idx = np.random.default_rng(0).integers(0, len(va_idx), size=(2000, len(va_idx)))

    def paired_diff(sq_a, sq_b):
        """RMSLE(a) - RMSLE(b) on the holdout and its 95% bootstrap range, resampling the same rows for both."""
        diffs = np.sqrt(sq_a[boot_idx].mean(axis=1)) - np.sqrt(sq_b[boot_idx].mean(axis=1))
        low, high = np.percentile(diffs, [2.5, 97.5])
        return float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean())), float(low), float(high)

    summary = pl.DataFrame(
        [{"run": "s5 all_three", "holdout_rmsle": float(np.sqrt(s5_squared.mean()))}]
        + [
            {k: r[k] for k in ("run", "issue", "holdout_rmsle", "bias", "best_iteration")}
            | dict(
                zip(
                    ("change", "ci_low", "ci_high"),
                    paired_diff((holdout_logs[r["run"]] - fs.y_log[va_idx]) ** 2, s5_squared),
                    strict=True,
                )
            )
            for r in results.values()
        ],
        infer_schema_length=None,
    )
    summary
    return (summary,)


@app.cell
def _(mo, pl, summary):
    def ci(run):
        r = summary.filter(pl.col("run") == run).row(0, named=True)
        return f"{r['holdout_rmsle']:.4f}（S5 との差 {r['change']:+.4f}、95% の範囲 {r['ci_low']:+.4f}〜{r['ci_high']:+.4f}）"

    mo.md(f"""
    ## 2. 結果を読む

    - 駅の位置と乗降客数（#21）: {ci("stations")}
    - 地区の位置と東京駅までの距離（#22）: {ci("districts")}
    - 両方: {ci("both")}

    どれも偶然の範囲で、外部データを足しても良くなりませんでした。

    LightGBM は、地区名・駅名をカテゴリとして持ち、S5 で地区・駅ごとの平均価格（Target encoding）も持っています。
    学習データにある地区や駅については、その場所の値段の高さをすでに直接知っているので、緯度・経度という間接的な手がかりを足しても上積みがありません。
    座標が効くのは、学習データにない場所が test に多いとき（このデータでは test にだけある地区は4行、駅は1行）や、
    場所のカテゴリを扱えない線形モデルのときです。

    外部データは、足せば良くなるとは限りません。モデルがすでに何を知っているかを考えてから選び、効いたかどうかは CV で確かめます。
    この段階の変更は採用せず、S9 のアンサンブルは S5〜S7 のモデルで組みます（2026-10-10）。
    """)
    return


if __name__ == "__main__":
    app.run()
