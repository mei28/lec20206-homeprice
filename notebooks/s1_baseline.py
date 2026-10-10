import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime as dt
    from dataclasses import asdict, dataclass, field

    import lightgbm as lgb
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import psutil

    from kgl import cv
    from kgl.env import comp_dir, detect, run_dir
    from kgl.io import (
        Metrics,
        git_commit,
        validate_submission,
        write_config,
        write_metrics,
        write_oof,
        write_submission,
    )
    from kgl.metrics import get_metric

    return (
        Metrics,
        asdict,
        comp_dir,
        cv,
        dataclass,
        detect,
        dt,
        field,
        get_metric,
        git_commit,
        lgb,
        mo,
        np,
        pd,
        pl,
        plt,
        psutil,
        run_dir,
        validate_submission,
        write_config,
        write_metrics,
        write_oof,
        write_submission,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S1. ベースライン

    最初の一歩として、データをほとんど加工せずに LightGBM で予測し、提出までを一通り動かします。
    ここで得た CV（手元のスコア）が、以降のすべての段階の比較の基準になります。

    この段階でやることは4つです。

    1. 列を最小限だけ変換する。数字として読める列は数値に、それ以外の文字列はそのままカテゴリにする
    2. 目的変数を $\log(1+y)$ にして LightGBM で学習する（S0 で見たとおり、これで RMSLE をそのまま最小化できる）
    3. 時系列 holdout（2024年までで学習、2025年第1〜第2四半期で検証）でスコアを測る
    4. 全期間で学習し直したモデルで test を予測し、提出ファイルを作る

    比較のために、全件に学習期間の中央値を出すだけの「何もしない予測」のスコアも測ります（#7）。
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
        # One thread per physical core. With every logical core LightGBM ran 150x slower here (163 s vs 1.1 s
        # per 100 rounds on 6 cores / 12 threads): the OpenMP threads fight over the shared cores.
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
    EXP = "s1_baseline"
    BASE = "none"
    return BASE, EXP, cfg


@app.cell
def _(cfg, comp_dir, cv, detect, pl):
    env = detect()
    data_dir = comp_dir(env, cfg.comp)
    # Read every column as text: the raw notation decides what "minimal conversion" can parse.
    train_raw = pl.read_csv(data_dir / cfg.train_file, infer_schema_length=0).with_columns(
        pl.col(cfg.id_col).cast(pl.Int64)
    )
    test_raw = pl.read_csv(data_dir / cfg.test_file, infer_schema_length=0).with_columns(
        pl.col(cfg.id_col).cast(pl.Int64)
    )
    sample = pl.read_csv(data_dir / cfg.sample_sub)

    fold_path = env.root / "folds" / f"{cfg.folds}.parquet"
    train_folded = cv.attach_folds(train_raw, cv.load_folds(fold_path), cfg.id_col)
    y_log = train_folded[cfg.target].cast(pl.Float64).log1p().to_numpy()
    folds = train_folded["fold"].to_numpy()
    tr_idx, va_idx = cv.split(folds, 0, time=True)
    return env, sample, test_raw, tr_idx, train_folded, va_idx, y_log


@app.cell
def _(mo, tr_idx, va_idx):
    mo.md(f"""
    ## 1. 学習用と検証用に分ける

    `folds/holdout_2025q1.parquet` に、行ごとの役割が書いてあります。fold が -1 の行（2019〜2024年）で学習し、
    fold が 0 の行（2025年第1〜第2四半期）で検証します。学習に {len(tr_idx):,} 行、検証に {len(va_idx):,} 行を使います。

    検証用の行は、学習には一切使いません。こうすると「まだ見ていない未来の取引をどれだけ当てられるか」を手元で測れます。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 何もしない予測（下限）

    最初に、学習期間の価格の中央値を全件にそのまま出した場合のスコアを測ります。
    モデルのスコアは、この値からどれだけ下がったかで読みます。
    """)
    return


@app.cell
def _(get_metric, np, tr_idx, va_idx, y_log):
    rmsle = get_metric("rmsle")
    y_price = np.expm1(y_log)
    floor_pred = np.full(len(va_idx), np.median(y_price[tr_idx]))
    floor_score = rmsle(y_price[va_idx], floor_pred)
    floor_score
    return floor_pred, floor_score, rmsle, y_price


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 最小限の変換

    S1 ではデータをほとんど加工しません。

    - 数値の列（`最寄駅：距離（分）`、`面積（㎡）`、`建ぺい率（％）`、`容積率（％）`）は数値に変換します。
      `30分～60分` のように数字として読めない値は、ここでは欠損になります（情報を捨てています。S3 で直します）。
    - それ以外の文字列の列はカテゴリとして渡します。`建築年` の `2013年` や `取引時期` の `2025年第1四半期` も、ここではただの文字列です。
    - `市区町村コード` は `市区町村名` と同じ情報なので使いません。`id` も使いません。

    LightGBM はカテゴリの列と欠損をそのまま扱えるので、これだけで学習できます。
    カテゴリの種類（語彙）は train と test の値を合わせて決めます。目的変数を使わないので、リークにはなりません。
    """)
    return


@app.cell
def _(pd, pl, test_raw, train_folded):
    NUMERIC = ["最寄駅：距離（分）", "面積（㎡）", "建ぺい率（％）", "容積率（％）"]
    CATEGORICAL = [
        "価格情報区分",
        "市区町村名",
        "地区名",
        "最寄駅：名称",
        "間取り",
        "建築年",
        "建物の構造",
        "用途",
        "今後の利用目的",
        "都市計画",
        "取引時期",
        "改装",
        "取引の事情等",
    ]
    vocab = {
        c: sorted(set(train_folded[c].drop_nulls().to_list()) | set(test_raw[c].drop_nulls().to_list()))
        for c in CATEGORICAL
    }

    def to_features(df):
        out = df.select(
            [pl.col(c).cast(pl.Float64, strict=False) for c in NUMERIC] + [pl.col(c) for c in CATEGORICAL]
        ).to_pandas()
        for c in CATEGORICAL:
            out[c] = pd.Categorical(out[c], categories=vocab[c])
        return out

    X = to_features(train_folded)
    X_test = to_features(test_raw)
    X.head()
    return X, X_test


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. LightGBM で学習する

    目的変数は $\log(1+y)$、損失は二乗誤差（`regression`）です。木の本数は、検証用の行でスコアが 100 回続けて改善しなくなったところで止めます（early stopping）。

    early stopping は検証用の行を見て本数を選ぶので、CV がわずかに楽観的になります。
    2025年第1四半期で本数を選んで第2四半期で測る、という入れ子の確認では最良の本数が変わらず、楽観の幅は 0.0005 未満でした（`docs/reviews/20261010-s1_baseline-leak.md`）。
    """)
    return


@app.cell
def _(X, asdict, cfg, lgb, np, rmsle, tr_idx, va_idx, y_log, y_price):
    train_set = lgb.Dataset(X.iloc[tr_idx], label=y_log[tr_idx])
    valid_set = lgb.Dataset(X.iloc[va_idx], label=y_log[va_idx], reference=train_set)
    booster = lgb.train(
        asdict(cfg.params),
        train_set,
        num_boost_round=cfg.max_rounds,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )
    oof_log = booster.predict(X.iloc[va_idx], num_iteration=booster.best_iteration)
    oof_price = np.expm1(oof_log)
    cv_score = rmsle(y_price[va_idx], oof_price)
    (booster.best_iteration, cv_score)
    return booster, cv_score, oof_log, oof_price


@app.cell
def _(cv_score, floor_score, mo):
    mo.md(f"""
    | 予測 | holdout RMSLE |
    |---|---|
    | 何もしない（学習期間の中央値） | {floor_score:.4f} |
    | LightGBM（最小限の変換） | {cv_score:.4f} |
    | 参考: AI単独ベースライン（Public LB、別のデータで測った値） | 0.19726 |

    中央値の予測から大きく下がりました。ただし AI単独ベースラインの値は Public LB（test の一部）で測ったもので、
    holdout とは採点に使う行が違います。並べて比べられるのは、提出して LB が出てからです。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. どこで外しているかを見る

    スコアを1つ見ただけで次の手を決めず、まず誤差の中身を見ます。区分別・四半期別の誤差と、外れの大きい20件を確認します。
    """)
    return


@app.cell
def _(np, oof_log, pl, rmsle, train_folded, va_idx, y_log, y_price, oof_price):
    holdout_rows = train_folded[va_idx].with_columns(
        price=pl.Series(y_price[va_idx]),
        pred=pl.Series(oof_price),
        log_error=pl.Series(oof_log - y_log[va_idx]),
    )
    by_group = pl.concat(
        [
            holdout_rows.group_by(pl.col(key).alias("group"))
            .agg(
                pl.len().alias("rows"),
                pl.col("log_error").mean().alias("mean_log_error"),
                (pl.col("log_error") ** 2).mean().sqrt().alias("rmsle"),
            )
            .with_columns(pl.lit(key).alias("by"))
            for key in ["価格情報区分", "取引時期"]
        ]
    ).select("by", "group", "rows", "mean_log_error", "rmsle")
    assert np.isclose(
        rmsle(holdout_rows["price"].to_numpy(), holdout_rows["pred"].to_numpy()),
        np.sqrt((holdout_rows["log_error"] ** 2).mean()),
        atol=1e-6,
    )
    by_group
    return by_group, holdout_rows


@app.cell
def _(by_group, holdout_rows, mo, np, pl):
    survey = by_group.filter(pl.col("group") == "不動産取引価格情報").row(0, named=True)
    contract = by_group.filter(pl.col("group") == "成約価格情報").row(0, named=True)
    bias = holdout_rows["log_error"].mean()
    # RMSE^2 = bias^2 + variance: removing a constant offset leaves only the spread of the errors.
    debiased = float(np.sqrt((holdout_rows["log_error"] ** 2).mean() - bias**2))
    mo.md(f"""
    `mean_log_error` は「予測の log − 正解の log」の平均で、負なら低めに予測しています。

    - 取引価格情報の行は RMSLE {survey["rmsle"]:.3f}、成約価格情報の行は {contract["rmsle"]:.3f} です。
      test は成約価格情報が 67% を占めるので、成約側の精度が LB に強く効きます。
    - 平均の誤差は、取引価格情報で {survey["mean_log_error"]:+.3f}、成約価格情報で {contract["mean_log_error"]:+.3f}、全体で {bias:+.3f} です。
      価格に直すと、予測は全体で約 {-np.expm1(bias):.0%} 低めです。どちらの区分でも、どちらの四半期でも同じ向きにずれています。

    仮に平均のずれだけを取り除けたとすると、RMSLE は {debiased:.3f} まで下がる計算です
    （二乗誤差の平均 = ずれの二乗 + ばらつき、という分解）。このずれがどこから来るのかを、次のセルで確かめます。
    """)
    return


@app.cell
def _(X, booster, mo, np, pd, va_idx, y_log):
    def holdout_with_quarter(quarter):
        """Predict the holdout as if every row had been traded in `quarter` (None = missing)."""
        X_va = X.iloc[va_idx].copy()
        X_va["取引時期"] = pd.Categorical([quarter] * len(X_va), categories=X["取引時期"].cat.categories)
        err = booster.predict(X_va, num_iteration=booster.best_iteration) - y_log[va_idx]
        return float(np.sqrt((err**2).mean())), float(err.mean())

    as_missing = holdout_with_quarter(None)
    as_last_train_quarter = holdout_with_quarter("2024年第4四半期")
    mo.md(f"""
    ### ずれの正体: 学習データにない「取引時期」

    `取引時期` はカテゴリとして渡しています。検証期間の値（`2025年第1四半期`、`2025年第2四半期`）は、学習データに1行もありません。
    LightGBM は、学習で見たことのないカテゴリを **欠損と同じ扱い** にします。下の表のとおり、検証用の行の取引時期を欠損に置き換えても、
    予測はまったく変わりません。欠損の扱いは学習の初期（2021年ごろ）の価格水準に寄っており、そのため予測が低めに出ていました。

    | 検証用の行の `取引時期` | holdout RMSLE | 平均のずれ |
    |---|---|---|
    | そのまま（学習にない値） | {as_missing[0]:.4f} | {as_missing[1]:+.3f} |
    | 学習期間の最後（2024年第4四半期）に置き換え | {as_last_train_quarter[0]:.4f} | {as_last_train_quarter[1]:+.3f} |

    最後の学習期間の値を渡すだけで、ずれの大半が消えます。残るずれ（約 {as_last_train_quarter[1]:+.2f}）が、
    木のモデルが学習期間の価格水準より上を出せないことによる分で、S4 の課題です。
    S3 では、取引時期を順序のある数値に直してこの問題を解きます（#12）。

    この置き換えは原因を確かめるための実験で、S1 の提出には使いません。S1 は「最小限の変換」の基準として、このまま残します。
    """)
    return


@app.cell
def _(holdout_rows, pl):
    worst = (
        holdout_rows.with_columns(pl.col("log_error").abs().alias("abs_log_error"))
        .sort("abs_log_error", descending=True)
        .head(20)
        .select(
            "id",
            "価格情報区分",
            "市区町村名",
            "地区名",
            "面積（㎡）",
            "間取り",
            "建築年",
            "取引時期",
            "取引の事情等",
            "price",
            "pred",
            "log_error",
        )
    )
    worst
    return


@app.cell
def _(mo):
    mo.md(r"""
    外れの大きい行は、両方向にあります。数十万〜数百万円の極端に安い取引を数千万円と予測した行と、
    1億円を超える高い取引を数千万円と予測した行です（最大は 160㎡ で 160億円）。`取引の事情等` はどれも空欄で、
    手元の列からは普通の取引と見分けがつきません。誤記や特殊な取引の可能性がありますが、確かめる手段はありません。
    RMSLE では、こうした1件の誤差（log で 2〜3）が二乗されて効きます。どこまで手当てするかは S3 以降で考えます。

    ## 6. どの列を使っているか

    LightGBM の重要度（gain）は、その列での分割によって誤差がどれだけ減ったかの合計です。
    """)
    return


@app.cell
def _(X, booster, pl, plt):
    importance = pl.DataFrame(
        {"feature": X.columns.tolist(), "gain": booster.feature_importance(importance_type="gain")}
    ).sort("gain")
    fig_importance, ax_importance = plt.subplots(figsize=(7, 4.5))
    ax_importance.barh(
        range(importance.height), importance["gain"].to_numpy() / importance["gain"].sum(), color="#2a78d6"
    )
    ax_importance.set_yticks(range(importance.height))
    # Japanese column names need a CJK font; label by position and show the names in the table below.
    ax_importance.set_yticklabels([f"f{i}" for i in range(importance.height)])
    ax_importance.set_xlabel("share of total gain")
    ax_importance.spines[["top", "right"]].set_visible(False)
    fig_importance.tight_layout()
    importance_table = importance.reverse().with_columns(
        (pl.col("gain") / pl.col("gain").sum()).round(3).alias("share")
    )
    fig_importance
    return (importance_table,)


@app.cell
def _(importance_table):
    importance_table
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. 全期間で学習し直して test を予測する

    holdout 用のモデルは 2025年第1〜第2四半期を見ていません。この2四半期は test に最も近い期間なので、
    提出用のモデルは全期間（2019年〜2025年第2四半期）で学習し直します（#8）。

    木の本数は、holdout で選んだ本数に行数の比（全期間 ÷ 学習期間、約 1.12）を掛けて決めます。
    データが増えた分だけ、同じ学習率でも少し多くの木が要るという考え方です。この判断は CV では確かめられないので、LB で確かめます。
    """)
    return


@app.cell
def _(X, X_test, asdict, booster, cfg, lgb, np, test_raw, tr_idx, y_log, pl):
    refit_rounds = int(round(booster.best_iteration * len(y_log) / len(tr_idx)))
    refit_booster = lgb.train(asdict(cfg.params), lgb.Dataset(X, label=y_log), num_boost_round=refit_rounds)
    test_pred = np.expm1(refit_booster.predict(X_test))
    submission = pl.DataFrame({cfg.id_col: test_raw[cfg.id_col], "price": test_pred})
    submission.describe()
    return refit_booster, refit_rounds, submission


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. 記録する

    結果は `output/s1_baseline/<run>/` に保存します。`metrics.json`（スコアと設定）、`oof.parquet`（検証用の行への予測）、
    `submission.csv`（提出ファイル）です。提出前に、行数・列名・id の順番・欠損を `sample_submission.csv` と照らし合わせます。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    Metrics,
    asdict,
    booster,
    cfg,
    cv_score,
    dt,
    env,
    floor_pred,
    floor_score,
    git_commit,
    oof_price,
    pl,
    refit_rounds,
    run_dir,
    sample,
    submission,
    train_folded,
    va_idx,
    validate_submission,
    write_config,
    write_metrics,
    write_oof,
    write_submission,
):
    finished = dt.datetime.now().isoformat(timespec="seconds")
    commit = git_commit(env.root)
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")

    def record(run: str, score: float, pred, config: dict, notes: str):
        out = run_dir(env, EXP, run)
        write_config(out, config, [])
        write_oof(out, holdout_ids.with_columns(pl.Series("pred", pred)))
        write_metrics(
            out,
            Metrics(
                exp=EXP,
                base=BASE,
                run=run,
                cv=score,
                cv_folds=[score],
                metric=cfg.metric,
                config=config,
                argv=[],
                git_commit=commit,
                started_at=finished,
                finished_at=finished,
                notes=notes,
            ),
        )
        return out

    floor_dir = record(
        "floor_median", floor_score, floor_pred, asdict(cfg) | {"model": "median"}, "constant training median (#7)"
    )
    lgbm_config = asdict(cfg) | {
        "model": "lightgbm",
        "best_iteration": booster.best_iteration,
        "refit_rounds": refit_rounds,
    }
    lgbm_dir = record("lgbm_raw", cv_score, oof_price, lgbm_config, "minimal conversion, log1p target (#6, #8)")
    validate_submission(submission, sample, cfg.id_col)
    write_submission(lgbm_dir, submission)
    (str(floor_dir), str(lgbm_dir))
    return


@app.cell
def _(cv_score, floor_score, mo):
    mo.md(f"""
    ## まとめ

    | run | holdout RMSLE | 内容 |
    |---|---|---|
    | `floor_median` | {floor_score:.4f} | 学習期間の中央値を全件に出す |
    | `lgbm_raw` | {cv_score:.4f} | 最小限の変換 + log1p + LightGBM |

    次の S2 で特徴量のための EDA をし、S3 でこの段階で捨てた情報を拾います。駅からの距離のレンジ、建築年から築年数、間取りの中身、取引時期の順序です。
    なかでも取引時期は、上の実験で学習期間の最後の値を渡しただけで 0.2988 が 0.2108 になったので、最も大きく効く見込みです。
    """)
    return


if __name__ == "__main__":
    app.run()
