import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from dataclasses import asdict, dataclass

    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy.optimize import nnls
    from sklearn.linear_model import Ridge

    from kgl import cv, lesson, mlit
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, validate_submission, write_submission
    from kgl.metrics import get_metric

    return (
        Ridge,
        asdict,
        comp_dir,
        cv,
        dataclass,
        detect,
        get_metric,
        git_commit,
        lesson,
        mlit,
        mo,
        nnls,
        np,
        pl,
        validate_submission,
        write_submission,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S9. アンサンブル（平均・重み付き平均・スタッキング）

    S5〜S8 で、仕組みの違うモデルをいくつも作りました。この段階では、それらの予測を組み合わせます。
    どのモデルも外す行はありますが、外し方が違えば、組み合わせたときに誤差が打ち消し合います。

    | 方法 | 組み合わせ方 | Issue |
    |---|---|---|
    | 単純平均 | 全モデルの予測（log）をそのまま平均する | #23 |
    | 重み付き平均 | 重みを 0 以上にして、holdout で誤差が最小になる重みを求める | #23 |
    | スタッキング | 各モデルの予測を入力にした Ridge 回帰（メタモデル）に、組み合わせ方を学ばせる。重みは負にもなれる | #30 |

    重みやメタモデルを holdout で学ぶと、同じ holdout で採点したときに良く見えすぎます。そこで holdout を四半期で2つに分け、
    2025年第1四半期で学んで第2四半期で採点し、逆向きも行って、2つを合わせたスコアで比べます（2分割の交差検証）。
    提出用には、holdout 全体で学び直した重み・メタモデルを、各モデルの test 予測に当てはめます。
    """)
    return


@app.cell
def _(dataclass):
    @dataclass
    class Config:
        comp: str = "utokyo-mlapp-2026-ex1"
        target: str = "取引価格（総額）"
        id_col: str = "id"
        train_file: str = "train.csv"
        sample_sub: str = "sample_submission.csv"
        folds: str = "holdout_2025q1"
        metric: str = "rmsle"
        stacking_alpha: float = 1.0
        seed: int = 0

    cfg = Config()
    EXP = "s9_ensemble"
    BASE = "s7_mlp_classic"
    # Every member has holdout predictions (oof.parquet) and a refit test prediction (submission.csv).
    MEMBERS = {
        "lightgbm": ("s5_te", "all_three"),
        "xgboost": ("s6_gbdt", "xgboost"),
        "catboost": ("s6_gbdt", "catboost"),
        "ridge": ("s7_mlp_classic", "ridge"),
        "random_forest": ("s7_mlp_classic", "random_forest"),
        "mlp": ("s7_mlp_classic", "mlp"),
    }
    return BASE, EXP, MEMBERS, cfg


@app.cell
def _(MEMBERS, cfg, comp_dir, cv, detect, mlit, np, pl):
    env = detect()
    data_dir = comp_dir(env, cfg.comp)
    train_raw = pl.read_csv(data_dir / cfg.train_file, infer_schema_length=0).with_columns(
        pl.col(cfg.id_col).cast(pl.Int64)
    )
    sample = pl.read_csv(data_dir / cfg.sample_sub)
    train_folded = cv.attach_folds(train_raw, cv.load_folds(env.root / "folds" / f"{cfg.folds}.parquet"), cfg.id_col)
    _, va_idx = cv.split(train_folded["fold"].to_numpy(), 0, time=True)
    holdout = train_folded[va_idx]
    y_log = holdout[cfg.target].cast(pl.Float64).log1p().to_numpy()
    quarter = holdout.select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy()
    in_q1 = quarter == 2025.0

    def load_member(stage, run):
        oof = pl.read_parquet(env.output_dir / stage / run / "oof.parquet")
        sub = pl.read_csv(env.output_dir / stage / run / "submission.csv")
        assert oof["id"].to_list() == holdout[cfg.id_col].to_list(), f"{stage}/{run}: holdout order"
        assert sub[cfg.id_col].to_list() == sample[cfg.id_col].to_list(), f"{stage}/{run}: test order"
        return np.log1p(oof["pred"].to_numpy()), np.log1p(sub["price"].to_numpy())

    loaded = {name: load_member(*where) for name, where in MEMBERS.items()}
    names = list(MEMBERS)
    H = np.column_stack([loaded[n][0] for n in names])  # holdout predictions, log scale
    T = np.column_stack([loaded[n][1] for n in names])  # test predictions, log scale
    return H, T, env, holdout, in_q1, names, sample, y_log


@app.cell
def _(H, names, np, pl, y_log):
    def rmse(pred):
        return float(np.sqrt(np.mean((pred - y_log) ** 2)))

    errors = H - y_log[:, None]
    members = pl.DataFrame({"member": names, "holdout_rmsle": [rmse(H[:, i]) for i in range(len(names))]}).sort(
        "holdout_rmsle"
    )
    correlation = pl.DataFrame(
        {"member": names}
        | {
            b: [round(float(np.corrcoef(errors[:, i], errors[:, j])[0, 1]), 3) for i in range(len(names))]
            for j, b in enumerate(names)
        }
    )
    members
    return correlation, members, rmse


@app.cell
def _(correlation, mo):
    mo.vstack([mo.md("各モデルの holdout 誤差の相関（1 に近いほど同じ行で同じように外す）"), correlation])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 3つの組み合わせ方を2分割の交差検証で比べる

    - 単純平均は何も学ばないので、holdout 全体の予測をそのまま採点します。
    - 重み付き平均（`nnls`、重みは 0 以上）とスタッキング（Ridge、切片あり）は、片方の四半期で学び、もう片方を予測します。
    - 比べる相手は、単独で最も良いモデルです。
    """)
    return


@app.cell
def _(H, Ridge, cfg, in_q1, nnls, np):
    def fit_weights(Hf, yf):
        weights, _ = nnls(Hf, yf)
        return weights

    def fit_stack(Hf, yf):
        return Ridge(alpha=cfg.stacking_alpha).fit(Hf, yf)

    def cross_predict(fit, predict, y):
        """Predict Q2 from a combiner fit on Q1 and Q1 from one fit on Q2."""
        out = np.empty(len(y))
        for train_rows in (in_q1, ~in_q1):
            combiner = fit(H[train_rows], y[train_rows])
            out[~train_rows] = predict(combiner, H[~train_rows])
        return out

    return cross_predict, fit_stack, fit_weights


@app.cell
def _(H, cross_predict, fit_stack, fit_weights, members, names, np, pl, rmse, y_log):
    best_single = members["member"][0]
    candidates = {
        f"best single ({best_single})": H[:, names.index(best_single)],
        "simple average": H.mean(axis=1),
        "weighted average": cross_predict(fit_weights, lambda w, Hp: Hp @ w, y_log),
        "stacking (ridge)": cross_predict(fit_stack, lambda m, Hp: m.predict(Hp), y_log),
    }
    boot_idx = np.random.default_rng(0).integers(0, len(y_log), size=(2000, len(y_log)))
    base_sq = (candidates[f"best single ({best_single})"] - y_log) ** 2

    def paired_diff(sq_a, sq_b):
        diffs = np.sqrt(sq_a[boot_idx].mean(axis=1)) - np.sqrt(sq_b[boot_idx].mean(axis=1))
        low, high = np.percentile(diffs, [2.5, 97.5])
        return float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean())), float(low), float(high)

    comparison = pl.DataFrame(
        [
            {"method": name, "holdout_rmsle": rmse(pred)}
            | dict(
                zip(
                    ("change_vs_best_single", "ci_low", "ci_high"),
                    paired_diff((pred - y_log) ** 2, base_sq),
                    strict=True,
                )
            )
            for name, pred in candidates.items()
        ]
    )
    comparison
    return best_single, candidates, comparison


@app.cell
def _(H, fit_stack, fit_weights, names, np, pl, y_log):
    final_weights = fit_weights(H, y_log)
    final_stack = fit_stack(H, y_log)
    combiners = pl.DataFrame(
        {
            "member": names,
            "weighted_average_weight": np.round(final_weights, 3),
            "stacking_coefficient": np.round(final_stack.coef_, 3),
        }
    )
    combiners
    return combiners, final_stack, final_weights


@app.cell
def _(best_single, combiners, comparison, final_stack, final_weights, mo, pl):
    def row(method):
        return comparison.filter(pl.col("method") == method).row(0, named=True)

    def ci(method):
        r = row(method)
        return f"{r['holdout_rmsle']:.4f}（単独最良との差 {r['change_vs_best_single']:+.4f}、95% の範囲 {r['ci_low']:+.4f}〜{r['ci_high']:+.4f}）"

    top = combiners.sort("weighted_average_weight", descending=True).head(3)
    mo.md(f"""
    ## 2. 結果を読む

    単独で最も良いモデルは `{best_single}`（holdout RMSLE {row(f"best single ({best_single})")["holdout_rmsle"]:.4f}）です。

    - 単純平均: {ci("simple average")}
    - 重み付き平均: {ci("weighted average")}
    - スタッキング: {ci("stacking (ridge)")}

    holdout 全体で求め直した重み付き平均の重みは、大きい順に
    {", ".join(f"`{m}` {w:.2f}" for m, w in zip(top["member"], top["weighted_average_weight"], strict=True))} で、重みの合計は {final_weights.sum():.3f} です。
    スタッキングのメタモデル（Ridge）の切片は {final_stack.intercept_:.3f} です。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 提出ファイルを作る

    3つの組み合わせ方それぞれについて、holdout 全体で求め直した重み・メタモデルを各モデルの test 予測に当てはめ、提出ファイルを作ります。
    どれを提出するかは、CV と、のちに見る LB で決めます。holdout の予測（2分割の交差検証で作ったもの）も記録します。

    test 予測は、各モデルを全期間で学習し直したものです。重みは holdout の予測（学習期間だけで学習したモデルの予測）で求めたので、
    test では同じ重みが少しずれた条件で使われます。この差は、提出して CV と LB を比べるときの注意点です。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    T,
    asdict,
    candidates,
    cfg,
    env,
    final_stack,
    final_weights,
    get_metric,
    git_commit,
    holdout,
    lesson,
    np,
    pl,
    sample,
    validate_submission,
    write_submission,
    y_log,
):
    rmsle = get_metric("rmsle")
    commit = git_commit(env.root)
    holdout_ids = holdout.select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    test_logs = {
        "simple_average": T.mean(axis=1),
        "weighted_average": T @ final_weights,
        "stacking": final_stack.predict(T),
    }
    holdout_preds = {
        "simple_average": candidates["simple average"],
        "weighted_average": candidates["weighted average"],
        "stacking": candidates["stacking (ridge)"],
    }
    paths = {}
    for _method, _test_log in test_logs.items():
        _out = lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=_method,
            score=rmsle(np.expm1(y_log), np.expm1(holdout_preds[_method])),
            metric=cfg.metric,
            config=asdict(cfg) | {"method": _method, "holdout_scoring": "fit on one quarter, score the other"},
            holdout=holdout_ids,
            pred=np.expm1(holdout_preds[_method]),
            notes=f"{_method} of the S5-S7 members (#23, #30)",
            commit=commit,
        )
        _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(_test_log)})
        validate_submission(_submission, sample, cfg.id_col)
        paths[_method] = str(write_submission(_out, _submission))
    paths
    return


if __name__ == "__main__":
    app.run()
