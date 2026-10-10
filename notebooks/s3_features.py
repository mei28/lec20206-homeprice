import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from dataclasses import asdict, dataclass, field

    import lightgbm as lgb
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import psutil

    from kgl import cv, lesson, mlit
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
        json,
        lesson,
        lgb,
        mlit,
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
    # S3. 前処理と特徴量

    S1 では、数字として読めない値を欠損にし、文字列はそのままカテゴリにしていました。S2 の EDA で、そのせいで捨てていた情報が見えました。
    この段階では、それを1つずつ拾い直し、そのたびに CV がどう変わるかを確かめます。

    | 順番 | 変えること | Issue |
    |---|---|---|
    | 1 | 取引時期を数値にする（年 + (四半期 − 1)/4） | #12 |
    | 2 | 駅距離のレンジと面積の表記を数値にする | #9 |
    | 3 | 建築年を数値にし、築年数（取引年 − 建築年）を足す | #10 |
    | 4 | 間取りから部屋数と L/D/K/S/R の有無を足す | #11 |

    1〜4 は前の変更に積み上げます（累積）。最後に、4 までを入れた状態から1つだけ外す・変える比較を2つします。

    - 価格情報区分を外す（#14）
    - 場所の列（市区町村、地区、駅）をカテゴリではなく整数のコードで渡す（#13）

    モデルと CV は S1 と同じです（LightGBM、log1p、2025年第1〜第2四半期の時系列 holdout）。変えるのは特徴量だけなので、CV の差はそのまま特徴量の効果と読めます。
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
        params: LgbParams = field(default_factory=LgbParams)

    cfg = Config()
    EXP = "s3_features"
    BASE = "s1_baseline"
    return BASE, EXP, cfg


@app.cell
def _(cfg, comp_dir, cv, detect, pl):
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
    y_log = train_folded[cfg.target].cast(pl.Float64).log1p().to_numpy()
    tr_idx, va_idx = cv.split(train_folded["fold"].to_numpy(), 0, time=True)
    return env, sample, test_raw, tr_idx, train_folded, va_idx, y_log


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 特徴量を作る関数

    `steps` に入っている変更だけを S1 の特徴量に加えます。表記の読み取りは `kgl.mlit`（テスト付き、S2 と共通）を使います。
    カテゴリの語彙は S1 と同じく train と test の値から作ります（目的変数は使いません）。
    """)
    return


@app.cell
def _(mlit, pd, pl, test_raw, train_folded):
    LOCATION = ["市区町村名", "地区名", "最寄駅：名称"]
    OTHER_CATEGORICAL = ["間取り", "建物の構造", "用途", "今後の利用目的", "都市計画", "改装", "取引の事情等"]

    def build_frame(df, steps):
        """Return the feature frame (polars) for the given set of steps."""
        exprs = [
            pl.col("建ぺい率（％）").cast(pl.Float64),
            pl.col("容積率（％）").cast(pl.Float64),
        ]
        categorical = LOCATION + OTHER_CATEGORICAL
        if "type" in steps:
            categorical = categorical + ["価格情報区分"]
        if "period" in steps:
            exprs.append(mlit.trade_period("取引時期").alias("trade_period"))
        else:
            categorical = categorical + ["取引時期"]
        if "notation" in steps:
            exprs += [
                mlit.station_minutes("最寄駅：距離（分）").alias("station_minutes"),
                mlit.area_m2("面積（㎡）").alias("area"),
            ]
        else:
            # S1's conversion: anything that is not a plain number becomes missing.
            exprs += [
                pl.col("最寄駅：距離（分）").cast(pl.Float64, strict=False).alias("station_minutes"),
                pl.col("面積（㎡）").cast(pl.Float64, strict=False).alias("area"),
            ]
        if "age" in steps:
            year = mlit.build_year("建築年")
            exprs += [
                year.alias("build_year"),
                (mlit.trade_period("取引時期").floor() - year).alias("age"),
            ]
        else:
            categorical = categorical + ["建築年"]
        if "layout" in steps:
            exprs += [mlit.layout_rooms("間取り").alias("rooms"), *mlit.layout_flags("間取り")]
        return df.select(exprs + [pl.col(c) for c in categorical]), categorical

    def to_pandas(frame, categorical, vocab, location_codes):
        """Categories with a fixed vocabulary; with `location_codes`, the location columns become integer codes."""
        out = frame.to_pandas()
        for c in categorical:
            out[c] = pd.Categorical(out[c], categories=vocab[c])
            if location_codes and c in LOCATION:
                out[c] = out[c].cat.codes.astype(float).where(lambda codes: codes >= 0)
        return out

    ALL_CATEGORICAL = LOCATION + OTHER_CATEGORICAL + ["価格情報区分", "取引時期", "建築年"]
    vocab = {
        c: sorted(set(train_folded[c].drop_nulls().to_list()) | set(test_raw[c].drop_nulls().to_list()))
        for c in ALL_CATEGORICAL
    }

    def features(df, steps, location_codes=False):
        frame, categorical = build_frame(df, steps)
        return to_pandas(frame, categorical, vocab, location_codes)

    return (features,)


@app.cell
def _(features, train_folded):
    features(train_folded, {"type", "period", "notation", "age", "layout"}).head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 1つずつ足して CV を測る

    各 run は holdout で early stopping し、holdout の RMSLE、平均のずれ（予測の log − 正解の log の平均）、価格情報区分ごとの RMSLE を記録します。
    結果は `output/s3_features/<run>/` に保存し、`just exp-table` で一覧にできます。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    cfg,
    env,
    features,
    get_metric,
    git_commit,
    lesson,
    np,
    pl,
    tr_idx,
    train_folded,
    va_idx,
    y_log,
):
    rmsle = get_metric("rmsle")
    commit = git_commit(env.root)
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    y_price_va = np.expm1(y_log[va_idx])
    is_contract = (train_folded[va_idx]["価格情報区分"] == "成約価格情報").to_numpy()

    RUNS = [
        # (run name, steps, location as codes, previous run it is compared with, Issue, note)
        ("plus_period", {"type", "period"}, False, "s1 lgbm_raw", 12, "trade period as a number"),
        ("plus_notation", {"type", "period", "notation"}, False, "plus_period", 9, "station minutes and area parsed"),
        ("plus_age", {"type", "period", "notation", "age"}, False, "plus_notation", 10, "build year as a number, age"),
        ("plus_layout", {"type", "period", "notation", "age", "layout"}, False, "plus_age", 11, "rooms and L/D/K/S/R"),
        ("minus_type", {"period", "notation", "age", "layout"}, False, "plus_layout", 14, "without 価格情報区分"),
        (
            "location_codes",
            {"type", "period", "notation", "age", "layout"},
            True,
            "plus_layout",
            13,
            "locations as codes",
        ),
    ]
    results = {}
    boosters = {}
    for _run, _steps, _codes, _compared, _issue, _note in RUNS:
        _X = features(train_folded, _steps, location_codes=_codes)
        _booster, _pred_log = lesson.fit_lgbm_holdout(
            _X,
            y_log,
            tr_idx,
            va_idx,
            asdict(cfg.params),
            max_rounds=cfg.max_rounds,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        _pred = np.expm1(_pred_log)
        _err = _pred_log - y_log[va_idx]
        results[_run] = {
            "run": _run,
            "issue": f"#{_issue}",
            "compared_with": _compared,
            "holdout_rmsle": rmsle(y_price_va, _pred),
            "bias": float(_err.mean()),
            "rmsle_survey": float(np.sqrt((_err[~is_contract] ** 2).mean())),
            "rmsle_contract": float(np.sqrt((_err[is_contract] ** 2).mean())),
            "best_iteration": _booster.best_iteration,
        }
        boosters[_run] = _booster
        lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=_run,
            score=results[_run]["holdout_rmsle"],
            metric=cfg.metric,
            config=asdict(cfg)
            | {
                "steps": sorted(_steps),
                "location_codes": _codes,
                "best_iteration": _booster.best_iteration,
                "refit_rounds": lesson.refit_rounds(_booster.best_iteration, n_all=len(y_log), n_train=len(tr_idx)),
            },
            holdout=holdout_ids,
            pred=_pred,
            notes=f"{_note} (#{_issue})",
            commit=commit,
        )
    return boosters, results


@app.cell
def _(env, json, pl, results):
    s1 = json.loads((env.output_dir / "s1_baseline" / "lgbm_raw" / "metrics.json").read_text())
    table = pl.DataFrame(
        [{"run": "s1 lgbm_raw", "issue": "#6", "compared_with": "", "holdout_rmsle": s1["cv"]}]
        + [{k: r[k] for k in ("run", "issue", "compared_with", "holdout_rmsle")} for r in results.values()]
    )
    scores = dict(zip(table["run"].to_list(), table["holdout_rmsle"].to_list(), strict=True))
    summary = table.with_columns(
        pl.col("compared_with").replace_strict(scores, default=None).alias("previous_rmsle")
    ).with_columns((pl.col("holdout_rmsle") - pl.col("previous_rmsle")).alias("change"))
    details = pl.DataFrame(list(results.values())).select(
        "run", "bias", "rmsle_survey", "rmsle_contract", "best_iteration"
    )
    summary
    return details, scores, summary


@app.cell
def _(details):
    details
    return


@app.cell
def _(mo, scores):
    mo.md(f"""
    ## 3. 結果を読む

    - 取引時期を数値にする（#12）だけで、holdout RMSLE は {scores["s1 lgbm_raw"]:.4f} から {scores["plus_period"]:.4f} に下がりました。
      S1 の振り返りで見たとおり、カテゴリのままでは検証期間の四半期が「見たことのない値」になり、2021年ごろの価格水準で予測されていました。
      数値にすると、検証期間は「学習期間の最後より後ろ」として扱われ、直近の価格水準で予測されます。
    - 駅距離と面積の表記（#9）: {scores["plus_period"]:.4f} → {scores["plus_notation"]:.4f}。数値に直る行はレンジ表記の駅距離と面積で全体の 0.4% だけなので、差は小さいと予想していました。
    - 築年数（#10）: {scores["plus_notation"]:.4f} → {scores["plus_age"]:.4f}。
    - 間取りの分解（#11）: {scores["plus_age"]:.4f} → {scores["plus_layout"]:.4f}。
    - 価格情報区分を外す（#14）: {scores["plus_layout"]:.4f} → {scores["minus_type"]:.4f}。
    - 場所を整数コードにする（#13）: {scores["plus_layout"]:.4f} → {scores["location_codes"]:.4f}。

    holdout は約2万行なので、0.001 程度の差は偶然の範囲に入りえます。小さな差を「効いた」と読むときは注意が要ります。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 残っているずれ

    4 までを入れた run（`plus_layout`）の残差を、S2 と同じ切り口で見直します。S1 で大きかった全体の低めのずれが、どこまで減ったかを確かめます。
    """)
    return


@app.cell
def _(boosters, features, mlit, np, pl, train_folded, va_idx, y_log):
    final_X = features(train_folded, {"type", "period", "notation", "age", "layout"})
    final_pred_log = boosters["plus_layout"].predict(
        final_X.iloc[va_idx], num_iteration=boosters["plus_layout"].best_iteration
    )
    residual_frame = train_folded[va_idx].with_columns(
        residual=pl.Series(np.asarray(final_pred_log) - y_log[va_idx]),
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
    return (residual_tables,)


@app.cell
def _(mo, residual_tables):
    mo.hstack([residual_tables["quarter"], residual_tables["age"], residual_tables["area"]], justify="start", gap=2)
    return


@app.cell
def _(mo, residual_tables):
    quarter_bias = residual_tables["quarter"]["bias"].to_list()
    mo.md(f"""
    全体の低めのずれは小さくなりましたが、消えてはいません。四半期ごとの平均のずれは {quarter_bias[0]:+.3f} と {quarter_bias[1]:+.3f} で、
    学習期間の終わりから離れるほど低めになります。木のモデルは、学習期間の最後の価格水準より上を予測できないためです。
    test は 2025年第3四半期から2026年第1四半期で、holdout よりさらに先なので、このずれはもっと大きくなる見込みです。
    これを価格指数で補うのが S4 です。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. 提出ファイルを作る

    4 までを入れた `plus_layout` を、全期間で学習し直して test を予測します。本数は S1 と同じ規則（holdout での本数 × 行数の比）で決めます。
    ここでは提出ファイルを作って検査するところまでで、提出するかどうかは別に決めます。
    """)
    return


@app.cell
def _(
    EXP,
    asdict,
    boosters,
    cfg,
    env,
    features,
    lesson,
    lgb,
    np,
    pl,
    sample,
    test_raw,
    tr_idx,
    train_folded,
    validate_submission,
    write_submission,
    y_log,
):
    final_steps = {"type", "period", "notation", "age", "layout"}
    refit_rounds = lesson.refit_rounds(boosters["plus_layout"].best_iteration, n_all=len(y_log), n_train=len(tr_idx))
    refit_booster = lgb.train(
        asdict(cfg.params), lgb.Dataset(features(train_folded, final_steps), label=y_log), num_boost_round=refit_rounds
    )
    submission = pl.DataFrame(
        {cfg.id_col: test_raw[cfg.id_col], "price": np.expm1(refit_booster.predict(features(test_raw, final_steps)))}
    )
    validate_submission(submission, sample, cfg.id_col)
    submission_path = write_submission(env.output_dir / EXP / "plus_layout", submission)
    (refit_rounds, str(submission_path))
    return


if __name__ == "__main__":
    app.run()
