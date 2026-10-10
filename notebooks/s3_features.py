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
    import polars as pl
    import psutil

    from kgl import cv, homeprice, lesson, mlit
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
        json,
        lesson,
        lgb,
        mlit,
        mo,
        np,
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

    1〜4 は前の変更に積み上げます（累積）。次に、4 までを入れた状態から1つだけ外す・変える比較を2つします。最後に、S2 で見つけた極端な価格の行を学習から外す案（#31）を試します。

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

    `steps` に入っている変更だけを S1 の特徴量に加えます。関数は `kgl.homeprice`（テスト付き、S4 以降と共通）にあり、表記の読み取りには `kgl.mlit`（S2 と共通）を使います。
    カテゴリの語彙は S1 と同じく train と test の値から作ります（目的変数は使いません）。
    """)
    return


@app.cell
def _(homeprice, test_raw, train_folded):
    vocab = homeprice.make_vocab([train_folded, test_raw])

    def features(df, steps, location_codes=False):
        return homeprice.features(df, steps, vocab, location_codes=location_codes)

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
    holdout_logs = {}
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
        holdout_logs[_run] = _pred_log
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
    return boosters, commit, holdout_ids, holdout_logs, results, rmsle


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
def _(mo):
    mo.md(r"""
    ### 差は偶然の範囲か

    holdout は約2万行です。行の選ばれ方が少し違うだけでも、RMSLE は 0.001 程度動きえます。そこで、holdout の行を重複を許して選び直す
    「ブートストラップ」を 2,000 回くり返し、2つの run の RMSLE の差がどの範囲に収まるかを見ます。
    同じ行を両方の run で使う（ペアにする）ので、行の難しさの違いは打ち消し合います。95% の範囲が 0 をまたいでいれば、差は偶然の範囲と読みます。
    """)
    return


@app.cell
def _(env, holdout_ids, holdout_logs, np, pl, results, va_idx, y_log):
    s1_oof = pl.read_parquet(env.output_dir / "s1_baseline" / "lgbm_raw" / "oof.parquet")
    # S1 wrote its holdout rows in the same fold order; compare row by row only if the ids line up.
    assert s1_oof["id"].to_list() == holdout_ids["id"].to_list()
    s1_log = np.log1p(s1_oof["pred"].to_numpy())
    squared = {"s1 lgbm_raw": (s1_log - y_log[va_idx]) ** 2} | {
        name: (pred - y_log[va_idx]) ** 2 for name, pred in holdout_logs.items()
    }
    boot_idx = np.random.default_rng(0).integers(0, len(va_idx), size=(2000, len(va_idx)))

    def paired_diff(sq_a, sq_b):
        """RMSLE(a) - RMSLE(b) on the holdout and its 95% bootstrap range, resampling the same rows for both."""
        diffs = np.sqrt(sq_a[boot_idx].mean(axis=1)) - np.sqrt(sq_b[boot_idx].mean(axis=1))
        low, high = np.percentile(diffs, [2.5, 97.5])
        return float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean())), float(low), float(high)

    bootstrap = pl.DataFrame(
        [
            dict(
                zip(
                    ("run", "compared_with", "change", "ci_low", "ci_high"),
                    (r["run"], r["compared_with"], *paired_diff(squared[r["run"]], squared[r["compared_with"]])),
                    strict=True,
                )
            )
            for r in results.values()
        ]
    ).with_columns(within_noise=(pl.col("ci_low") < 0) & (pl.col("ci_high") > 0))
    bootstrap
    return bootstrap, paired_diff, squared


@app.cell
def _(bootstrap, mo, pl, scores):
    def ci(run):
        row = bootstrap.filter(pl.col("run") == run).row(0, named=True)
        return f"{row['change']:+.4f}（95% の範囲 {row['ci_low']:+.4f}〜{row['ci_high']:+.4f}）"

    mo.md(f"""
    ## 3. 結果を読み、採用する変更を決める

    - 取引時期を数値にする（#12）: {ci("plus_period")}。holdout RMSLE は {scores["s1 lgbm_raw"]:.4f} から {scores["plus_period"]:.4f} に下がりました。
      カテゴリのままでは検証期間の四半期が「見たことのない値」になり、2021年ごろの価格水準で予測されていました。
      数値にすると、検証期間は「学習期間の最後より後ろ」として扱われ、直近の価格水準で予測されます。採用します。
    - 駅距離と面積の表記（#9）: {ci("plus_notation")}。数値に直る行は全体の 0.4% だけで、差は偶然の範囲です。
      害はなく、表記の扱いとして正しいので採用します。
    - 築年数（#10）: {ci("plus_age")}。EDA では築年数と単価のきれいな関係が見えましたが、CV は良くなりませんでした。
      取引時期が数値になったことで、モデルは「建築年」と「取引時期」の組み合わせから築年数に当たる情報をすでに読み取れます。採用しません。
    - 間取りの分解（#11）: {ci("plus_layout")}。間取りの49種類のカテゴリがすでに部屋数の情報を持っていました。採用しません。
    - 価格情報区分を外す（#14）: {ci("minus_type")}。差は偶然の範囲です。区分による誤差の違いを読めるように、S1 と同じく残します。
    - 場所を整数コードにする（#13）: {ci("location_codes")}。コードにすると「番号が近い場所は似ている」という意味のない順序が入り、はっきり悪くなりました。
      場所はカテゴリのままにします。

    EDA で関係が見えても、モデルがすでに同じ情報を別の列から得ていれば、CV は良くなりません。特徴量は、足すたびに CV で確かめます。
    採用の判断は 2026-10-10 にユーザーと決めました（#12 と #9 を採用）。以降の段階は、この構成（`plus_notation`）を土台にします。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 極端な価格の行を学習から外す（#31）

    S2 で見つけた、㎡単価が10万円未満または500万円超の行を、学習（2024年まで）からだけ外します。holdout は全行のまま採点するので、スコアはそのまま比べられます。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    cfg,
    commit,
    env,
    features,
    holdout_ids,
    lesson,
    mlit,
    np,
    paired_diff,
    pl,
    rmsle,
    squared,
    tr_idx,
    train_folded,
    va_idx,
    y_log,
):
    ADOPTED = {"type", "period", "notation"}
    ppm2 = train_folded.select((pl.col(cfg.target).cast(pl.Float64) / mlit.area_m2("面積（㎡）")).alias("ppm2"))[
        "ppm2"
    ].to_numpy()
    is_extreme = (ppm2 < 1e5) | (ppm2 > 5e6)
    tr_kept = tr_idx[~is_extreme[tr_idx]]
    extreme_booster, extreme_log = lesson.fit_lgbm_holdout(
        features(train_folded, ADOPTED),
        y_log,
        tr_kept,
        va_idx,
        asdict(cfg.params),
        max_rounds=cfg.max_rounds,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    extreme_score = rmsle(np.expm1(y_log[va_idx]), np.expm1(extreme_log))
    lesson.record_run(
        env,
        exp=EXP,
        base=BASE,
        run="drop_extremes",
        score=extreme_score,
        metric=cfg.metric,
        config=asdict(cfg)
        | {
            "steps": sorted(ADOPTED),
            "drop_ppm2_below": 1e5,
            "drop_ppm2_above": 5e6,
            "best_iteration": extreme_booster.best_iteration,
            "refit_rounds": lesson.refit_rounds(extreme_booster.best_iteration, n_all=len(y_log), n_train=len(tr_kept)),
        },
        holdout=holdout_ids,
        pred=np.expm1(extreme_log),
        notes="adopted S3 features, extreme price-per-m2 rows dropped from training (#31)",
        commit=commit,
    )
    extreme_change = paired_diff((extreme_log - y_log[va_idx]) ** 2, squared["plus_notation"])
    dropped_rows = int(is_extreme[tr_idx].sum())
    (dropped_rows, extreme_score, extreme_change)
    return ADOPTED, dropped_rows, extreme_change, extreme_score


@app.cell
def _(dropped_rows, extreme_change, extreme_score, mo):
    mo.md(f"""
    学習期間から {dropped_rows} 行を外すと、holdout RMSLE は {extreme_score:.4f} になりました。
    採用した構成（`plus_notation`）との差は {extreme_change[0]:+.4f}（95% の範囲 {extreme_change[1]:+.4f}〜{extreme_change[2]:+.4f}）で、悪くなりました。

    holdout にも同じような極端な価格の行があります。学習からそれを外すと、モデルはその価格帯をまったく予測しなくなり、
    holdout の極端な行での誤差が大きくなります。RMSLE は誤差を二乗するので、その分が全体を押し上げました。
    学習データの「外れ値」でも、test に同じ種類の行がある限り、捨てると損をすることがあります。この案は採用しません。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 残っているずれ

    採用した構成（`plus_notation`）の残差を、S2 と同じ切り口で見直します。S1 で大きかった全体の低めのずれが、どこまで減ったかを確かめます。
    """)
    return


@app.cell
def _(holdout_logs, mlit, pl, train_folded, va_idx, y_log):
    residual_frame = train_folded[va_idx].with_columns(
        residual=pl.Series(holdout_logs["plus_notation"] - y_log[va_idx]),
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

    採用した構成（`plus_notation`）を、全期間で学習し直して test を予測します。本数は S1 と同じ規則（holdout での本数 × 行数の比）で決めます。
    ここでは提出ファイルを作って検査するところまでで、提出するかどうかは別に決めます。
    """)
    return


@app.cell
def _(
    ADOPTED,
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
    refit_rounds = lesson.refit_rounds(boosters["plus_notation"].best_iteration, n_all=len(y_log), n_train=len(tr_idx))
    refit_booster = lgb.train(
        asdict(cfg.params), lgb.Dataset(features(train_folded, ADOPTED), label=y_log), num_boost_round=refit_rounds
    )
    submission = pl.DataFrame(
        {cfg.id_col: test_raw[cfg.id_col], "price": np.expm1(refit_booster.predict(features(test_raw, ADOPTED)))}
    )
    validate_submission(submission, sample, cfg.id_col)
    submission_path = write_submission(env.output_dir / EXP / "plus_notation", submission)
    (refit_rounds, str(submission_path))
    return


if __name__ == "__main__":
    app.run()
