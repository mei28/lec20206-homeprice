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

    from kgl import cv, ensemble, feature_set, lesson, mlit
    from kgl.env import comp_dir, detect
    from kgl.io import git_commit, read_metrics, validate_submission, write_submission
    from kgl.metrics import get_metric

    return (
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
        lesson,
        lgb,
        mlit,
        mo,
        np,
        pd,
        pl,
        psutil,
        read_metrics,
        validate_submission,
        write_submission,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S11. pseudo label（test への予測を仮の正解にする）

    pseudo label は、モデルが test に出した予測を「仮の正解」として、test の行を学習データに足す方法です（#34）。
    この段階では、S10 のアンサンブル（教師）の予測を pseudo label にして、LightGBM（生徒）を学習し直します。

    pseudo label から生徒が学べるのは、教師がすでに知っていることだけです。それでも効くことがあるのは、次の2つのためです。

    - 蒸留: 教師（S10 スタッキング、holdout 0.1950）は生徒（S5 の LightGBM、0.2035）より強い。test の物件について、教師の答えを生徒に見せられる
    - 構成: test の期間は、成約価格の割合（67%、学習データは 43%）などの構成が学習データと違う。pseudo の行を足すと、その構成の行が学習に増える

    逆に、特徴が同じでも価格がばらつく分（RMSE 0.11〜0.14）は減らせません。価格指数の外挿も、教師のものをそのまま受け継ぎます。

    ## holdout で手順を再現する

    pseudo label の効果は提出しないと分からないように見えますが、holdout で同じ手順を再現できます。

    | | test への提出 | holdout での再現 |
    |---|---|---|
    | 教師が学習する期間 | 2019Q1〜2025Q2 | 2019Q1〜2024Q4 |
    | pseudo label を付ける行 | test（2025Q3〜2026Q1） | holdout（2025Q1〜Q2） |
    | 生徒の採点 | LB | holdout の本当のラベル |

    生徒は holdout の正解を一度も見ないので、この採点は test での使い方と同じ条件です。

    気をつける点が3つあります。

    1. 尺度: S4 からの目的変数は「log 価格 − その四半期の価格指数」です。pseudo label も同じ尺度に直してから足します。
    2. 本数: 生徒の学習では holdout の正解を一切使いません。early stopping はせず、S5 で決まった本数を行数（重みの合計）に比例して増やします。S1 から使っている、全期間で学習し直すときの規則と同じです。
    3. Target encoding: 本当のラベルだけで計算します（S5 のまま）。pseudo label は混ぜません。
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
        trend_quarters: int = 8
        te_smoothing: float = 20.0
        te_folds: int = 5
        seed: int = 42
        base_run: str = "s5_te/all_three"
        stacking_alpha: float = 1.0
        # teacher: whose predictions become pseudo labels; weight: of each pseudo row;
        # share: the fraction of pseudo rows kept, those where the six members agree most
        students: dict = field(
            default_factory=lambda: {
                "stacking_w1": {"teacher": "stacking", "weight": 1.0, "share": 1.0},
                "stacking_w05": {"teacher": "stacking", "weight": 0.5, "share": 1.0},
                "stacking_confident": {"teacher": "stacking", "weight": 1.0, "share": 0.8},
                "catboost_w1": {"teacher": "catboost", "weight": 1.0, "share": 1.0},
            }
        )
        lgbm: LgbParams = field(default_factory=LgbParams)

    cfg = Config()
    EXP = "s11_pseudo"
    BASE = "s10_final"
    return BASE, EXP, cfg


@app.cell
def _(cfg, comp_dir, cv, detect, feature_set, mlit, pl, read_metrics):
    env = detect()
    data_dir = comp_dir(env, cfg.comp)
    train_raw = pl.read_csv(data_dir / cfg.train_file, infer_schema_length=0).with_columns(
        pl.col(cfg.id_col).cast(pl.Int64)
    )
    test_raw = pl.read_csv(data_dir / cfg.test_file, infer_schema_length=0).with_columns(
        pl.col(cfg.id_col).cast(pl.Int64)
    )
    sample = pl.read_csv(data_dir / cfg.sample_sub)
    assert test_raw[cfg.id_col].to_list() == sample[cfg.id_col].to_list()
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
    y_real, y_real_refit = fs.y_log - fs.offset, fs.y_log - fs.offset_refit
    y_holdout = fs.y_log[va_idx]
    holdout_ids = train_folded[va_idx].select(cfg.id_col, pl.col(cfg.target).cast(pl.Float64), "fold")
    in_q1 = train_folded[va_idx].select(mlit.trade_period("取引時期").alias("t"))["t"].to_numpy() == 2025.0
    base_rounds = read_metrics(env.output_dir / cfg.base_run / "metrics.json").config["best_iteration"]
    return (
        base_rounds,
        env,
        fs,
        holdout_ids,
        in_q1,
        sample,
        tr_idx,
        va_idx,
        y_holdout,
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
def _(mo):
    mo.md(r"""
    ## 1. 教師の予測を読み込む

    教師の予測は、S10 までに保存した holdout への予測（`oof.parquet`）と test への予測（`submission.csv`）をそのまま使います。学習し直す必要はありません。

    - 教師 `stacking`: S10 のスタッキング。holdout への予測は、holdout の2つの四半期で「片方でメタモデルを学習し、もう片方を予測する」形で作ったものです。
      そのため、もう一方の四半期のラベルの情報が、メタモデルの係数（7個）を通してわずかに入っています。
    - 教師 `catboost`: CatBoost 単独（S6、S10 と同じもの）。holdout のラベルを early stopping にしか使っていない、汚れの少ない教師です。2つの教師で結果がそろうかを確かめます。

    「自信のある行だけ使う」run のために、6つのメンバーの予測（log）の標準偏差を、行ごとの「意見の割れ方」として計算します。回帰には分類のような確率がないので、その代わりです。
    """)
    return


@app.cell
def _(env, holdout_ids, np, pl, sample):
    def saved(stage, run):
        """Holdout and test predictions of a saved run, on the log scale."""
        oof = pl.read_parquet(env.output_dir / stage / run / "oof.parquet")
        sub = pl.read_csv(env.output_dir / stage / run / "submission.csv")
        assert oof["id"].to_list() == holdout_ids["id"].to_list() and sub["id"].to_list() == sample["id"].to_list()
        return np.log1p(oof["pred"].to_numpy()), np.log1p(sub["price"].to_numpy())

    members = {
        "lightgbm_seed_avg": saved("s10_final", "lightgbm_seed_avg"),
        "catboost_long": saved("s10_final", "catboost_long"),
        "xgboost": saved("s6_gbdt", "xgboost"),
        "ridge": saved("s7_mlp_classic", "ridge"),
        "random_forest": saved("s7_mlp_classic", "random_forest"),
        "mlp": saved("s7_mlp_classic", "mlp"),
    }
    member_names = list(members)
    teachers = {"stacking": saved("s10_final", "stacking"), "catboost": members["catboost_long"]}
    base_holdout = saved("s5_te", "all_three")[0]
    disagreement_holdout = np.std([members[n][0] for n in member_names], axis=0)
    disagreement_test = np.std([members[n][1] for n in member_names], axis=0)
    {
        "disagreement, holdout (median, 90th pct)": np.round(np.percentile(disagreement_holdout, [50, 90]), 4),
        "disagreement, test (median, 90th pct)": np.round(np.percentile(disagreement_test, [50, 90]), 4),
    }
    return (
        base_holdout,
        disagreement_holdout,
        disagreement_test,
        member_names,
        members,
        teachers,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 生徒を学習する

    生徒は S5 と同じ設定の LightGBM（seed 42）です。まず pseudo label を足さずに S5 の本数で学習し、S5 の予測と一致することを確かめます。
    一致すれば、この後の差は pseudo label だけによるものです。

    run ごとに、holdout での再現（採点用）と、全期間 + test の pseudo label での学習（提出用）を同じ設定で行います。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    base_holdout,
    base_rounds,
    bootstrap_diff,
    cfg,
    disagreement_holdout,
    disagreement_test,
    env,
    fs,
    git_commit,
    holdout_ids,
    holdout_rmsle,
    lesson,
    lgb,
    np,
    pd,
    pl,
    sample,
    teachers,
    tr_idx,
    va_idx,
    validate_submission,
    write_submission,
    y_holdout,
    y_real,
    y_real_refit,
):
    def fit_student(X_true, y_true, X_pseudo, y_pseudo, pseudo_weight, rounds):
        """LightGBM on true-label rows (weight 1) plus pseudo-label rows (weight `pseudo_weight`), fixed rounds."""
        X = pd.concat([X_true, X_pseudo], ignore_index=True)
        assert (X.dtypes == X_true.dtypes).all()
        y = np.concatenate([y_true, y_pseudo])
        w = np.concatenate([np.ones(len(y_true)), np.full(len(y_pseudo), pseudo_weight)])
        return lgb.train(asdict(cfg.lgbm), lgb.Dataset(X, label=y, weight=w), num_boost_round=rounds)

    _check = lgb.train(
        asdict(cfg.lgbm), lgb.Dataset(fs.X.iloc[tr_idx], label=y_real[tr_idx]), num_boost_round=base_rounds
    )
    no_pseudo_gap = float(np.abs(_check.predict(fs.X.iloc[va_idx]) + fs.offset[va_idx] - base_holdout).max())
    assert no_pseudo_gap < 1e-6, no_pseudo_gap

    commit = git_commit(env.root)
    base_sq = (base_holdout - y_holdout) ** 2
    student_preds = {}
    student_rows = []
    for _run, _spec in cfg.students.items():
        _teacher_holdout, _teacher_test = teachers[_spec["teacher"]]
        _keep_holdout = disagreement_holdout <= np.quantile(disagreement_holdout, _spec["share"])
        _keep_test = disagreement_test <= np.quantile(disagreement_test, _spec["share"])

        _rounds = lesson.refit_rounds(
            base_rounds, n_all=len(tr_idx) + _spec["weight"] * _keep_holdout.sum(), n_train=len(tr_idx)
        )
        _student = fit_student(
            fs.X.iloc[tr_idx],
            y_real[tr_idx],
            fs.X.iloc[va_idx[_keep_holdout]],
            (_teacher_holdout - fs.offset[va_idx])[_keep_holdout],
            _spec["weight"],
            _rounds,
        )
        _holdout_log = _student.predict(fs.X.iloc[va_idx]) + fs.offset[va_idx]

        _rounds_refit = lesson.refit_rounds(
            base_rounds, n_all=len(fs.y_log) + _spec["weight"] * _keep_test.sum(), n_train=len(tr_idx)
        )
        _refit = fit_student(
            fs.X_refit,
            y_real_refit,
            fs.X_test.iloc[np.flatnonzero(_keep_test)],
            (_teacher_test - fs.offset_test)[_keep_test],
            _spec["weight"],
            _rounds_refit,
        )
        _test_log = _refit.predict(fs.X_test) + fs.offset_test
        student_preds[_run] = (_holdout_log, _test_log)

        _score = holdout_rmsle(_holdout_log, y_holdout)
        _change, _low, _high = bootstrap_diff((_holdout_log - y_holdout) ** 2, base_sq)
        student_rows.append(
            {"run": _run, "holdout_rmsle": _score, "change_vs_s5": _change, "ci_low": _low, "ci_high": _high}
            | {"pseudo_rows": int(_keep_holdout.sum()), "rounds": _rounds}
        )
        _out = lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=_run,
            score=_score,
            metric=cfg.metric,
            config=asdict(cfg) | {"student": _spec, "rounds": _rounds, "refit_rounds": _rounds_refit},
            holdout=holdout_ids,
            pred=np.expm1(_holdout_log),
            notes=f"LightGBM student, pseudo labels from {_spec['teacher']} (#34)",
            commit=commit,
        )
        _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(_test_log)})
        validate_submission(_submission, sample, cfg.id_col)
        write_submission(_out, _submission)

    students = pl.DataFrame(
        [{"run": "s5 all_three (no pseudo labels)", "holdout_rmsle": float(np.sqrt(base_sq.mean()))}] + student_rows,
        infer_schema_length=None,
    )
    students
    return commit, student_preds, students


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 生徒をアンサンブルに戻す

    生徒単独が良くなっても、アンサンブルに入れたときに効くとは限りません。教師の答えを写した生徒は、教師の中のほかのメンバーと似た予測をするからです。
    S10 のスタッキングで LightGBM（seed 平均）を生徒に差し替え、S10 と同じ方法（2つの四半期で交差させて採点）で比べます。
    差し替える前の構成で S10 のスタッキングの予測が再現できることも確かめます。
    """)
    return


@app.cell
def _(
    BASE,
    EXP,
    asdict,
    bootstrap_diff,
    cfg,
    commit,
    ensemble,
    env,
    holdout_ids,
    holdout_rmsle,
    in_q1,
    lesson,
    member_names,
    members,
    np,
    pl,
    sample,
    student_preds,
    teachers,
    validate_submission,
    write_submission,
    y_holdout,
):
    def stack(H, T):
        """S10's stacking: holdout predictions cross-fit on the two quarters, test predictions from a fit on both."""

        def fit(Hf, yf):
            return ensemble.fit_stacking(Hf, yf, alpha=cfg.stacking_alpha)

        holdout_log = ensemble.cross_predict(H, y_holdout, in_q1, fit, lambda m, Hp: m.predict(Hp))
        return holdout_log, fit(H, y_holdout).predict(T)

    H_s10 = np.column_stack([members[n][0] for n in member_names])
    T_s10 = np.column_stack([members[n][1] for n in member_names])
    s10_holdout, s10_test = stack(H_s10, T_s10)
    assert np.allclose(s10_holdout, teachers["stacking"][0], atol=1e-9)
    assert np.allclose(s10_test, teachers["stacking"][1], atol=1e-9)

    s10_sq = (s10_holdout - y_holdout) ** 2
    stack_rows = [{"run": "s10 stacking", "holdout_rmsle": holdout_rmsle(s10_holdout, y_holdout)}]
    _lgbm_col = member_names.index("lightgbm_seed_avg")
    for _run, (_student_holdout, _student_test) in student_preds.items():
        _H, _T = H_s10.copy(), T_s10.copy()
        _H[:, _lgbm_col], _T[:, _lgbm_col] = _student_holdout, _student_test
        _holdout_log, _test_log = stack(_H, _T)
        _score = holdout_rmsle(_holdout_log, y_holdout)
        _change, _low, _high = bootstrap_diff((_holdout_log - y_holdout) ** 2, s10_sq)
        stack_rows.append(
            {
                "run": f"stack_{_run}",
                "holdout_rmsle": _score,
                "change_vs_s10": _change,
                "ci_low": _low,
                "ci_high": _high,
            }
        )
        _out = lesson.record_run(
            env,
            exp=EXP,
            base=BASE,
            run=f"stack_{_run}",
            score=_score,
            metric=cfg.metric,
            config=asdict(cfg) | {"method": "stacking", "members": member_names, "lightgbm_replaced_by": _run},
            holdout=holdout_ids,
            pred=np.expm1(_holdout_log),
            notes=f"S10 stacking with LightGBM replaced by the {_run} student (#34)",
            commit=commit,
        )
        _submission = pl.DataFrame({cfg.id_col: sample[cfg.id_col], "price": np.expm1(_test_log)})
        validate_submission(_submission, sample, cfg.id_col)
        write_submission(_out, _submission)

    stacks = pl.DataFrame(stack_rows, infer_schema_length=None)
    stacks
    return (stacks,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 生徒は何を覚えたのか

    生徒は、pseudo label を付けた行、つまり採点する行そのものを学習しています。その行では教師の答えを覚えるので、予測は教師の側に寄るはずです。
    そこで、生徒の予測が「S5 の予測 × a + 教師の予測 × (1 − a)」でどこまで説明できるかを調べます。
    a は生徒の予測に最も近くなる値を最小二乗で求めます。正解のラベルは使いません。
    同じ a で S5 と教師を単純に平均したときのスコアも並べます。
    """)
    return


@app.cell
def _(base_holdout, cfg, holdout_rmsle, np, pl, student_preds, teachers, y_holdout):
    def own_share(student, own, teacher):
        """The a for which a * own + (1 - a) * teacher is closest to the student's predictions (least squares)."""
        d = own - teacher
        return float(((student - teacher) * d).sum() / (d * d).sum())

    _rows = []
    for _run, _spec in cfg.students.items():
        _student = student_preds[_run][0]
        _teacher = teachers[_spec["teacher"]][0]
        _a = own_share(_student, base_holdout, _teacher)
        _blend = _a * base_holdout + (1 - _a) * _teacher
        _rows.append(
            {
                "run": _run,
                "teacher_rmsle": holdout_rmsle(_teacher, y_holdout),
                "student_rmsle": holdout_rmsle(_student, y_holdout),
                "own_share_a": _a,
                "blend_rmsle": holdout_rmsle(_blend, y_holdout),
                "corr_student_blend": float(np.corrcoef(_student, _blend)[0, 1]),
            }
        )
    memorization = pl.DataFrame(_rows)
    memorization
    return (memorization,)


@app.cell
def _(cfg, memorization, mo, pl, stacks, students):
    def student(run):
        return students.filter(pl.col("run") == run).row(0, named=True)

    def memo(run):
        return memorization.filter(pl.col("run") == run).row(0, named=True)

    def change(r, key):
        return f"{r[key]:+.4f}（95% の範囲 {r['ci_low']:+.4f}〜{r['ci_high']:+.4f}）"

    _table = "\n".join(
        [
            "| run | 教師 | 重み | pseudo の行 | holdout RMSLE | S5 との差（95% の範囲） | S5 の割合 a | 同じ a の単純な平均 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        + [
            f"| `{run}` | {spec['teacher']} | {spec['weight']} | {student(run)['pseudo_rows']:,} | {student(run)['holdout_rmsle']:.4f}"
            f" | {change(student(run), 'change_vs_s5')} | {memo(run)['own_share_a']:.2f} | {memo(run)['blend_rmsle']:.4f} |"
            for run, spec in cfg.students.items()
        ]
    )
    _stack_changes = stacks.drop_nulls("change_vs_s10")
    _intro = mo.md(f"""
    ## 5. 結果を読む

    S5（pseudo label なし）の holdout RMSLE は {students["holdout_rmsle"][0]:.4f} です。
    """)
    _reading = mo.md(f"""
    - 生徒単独では、どの run も S5 より良くなり、95% の範囲も 0 をまたぎませんでした。
    - ところが、生徒の予測は「S5 と教師の重み付き平均」とほとんど同じです（相関 {memorization["corr_student_blend"].min():.4f} 以上）。
      重みを 0.5 にしたり、意見のそろった行だけにしたりすると、覚える量が減って a が大きく（S5 寄りに）なり、スコアも S5 の側に戻りました。
    - 同じ割合の単純な平均と比べると、生徒は同じか、それより悪い結果です。改善のほとんどは、教師の答えを覚えて混ぜた分でした。
    - 教師がスタッキングのとき、生徒（{memo("stacking_w1")["student_rmsle"]:.4f}）は教師（{memo("stacking_w1")["teacher_rmsle"]:.4f}）より悪くなりました。
      教師が CatBoost のとき生徒（{memo("catboost_w1")["student_rmsle"]:.4f}）が教師（{memo("catboost_w1")["teacher_rmsle"]:.4f}）より良くなったのは、
      LightGBM と CatBoost を混ぜた効果です。S9 で見たアンサンブルの効果と同じものです。
    - 生徒を S10 のスタッキングに戻すと、S10 との差は {_stack_changes["change_vs_s10"].min():+.5f}〜{_stack_changes["change_vs_s10"].max():+.5f} で、どれも偶然の範囲でした。

    test でも同じことが起きます。test の行に pseudo label を付けて学習し、同じ test の行を予測するので、生徒の予測は自分と教師の平均に近づきます。

    この段階の変更は採用しません（2026-10-16）。このデータでの pseudo label は、手間のかかるアンサンブルと同じことしかしていませんでした。

    pseudo label を試すときは、生徒を元のモデル（ここでは S5）とだけ比べると、効いたように見えてしまいます。比べる相手は次の2つです。

    1. 教師そのもの: 生徒が教師より悪ければ、教師の予測を出せば済みます。
    2. 元のモデルと教師の単純な平均: 生徒がこれと同じなら、効いているのは覚えた分だけです。
    """)
    mo.vstack([_intro, mo.md(_table), _reading])
    return


if __name__ == "__main__":
    app.run()
