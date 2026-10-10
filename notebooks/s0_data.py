import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from kgl.env import comp_dir, detect
    from kgl.metrics import get_metric

    return comp_dir, detect, get_metric, mo, np, pl, plt


@app.cell
def _(mo):
    mo.md(r"""
    # S0. データを知る

    モデルを作る前に、データと評価のしくみを確かめる段階です。ここで見つけたことが、S1 以降でやることの理由になります。

    この段階で答える問いは3つです。

    1. 何を予測し、どう採点されるのか（目的変数と RMSLE）
    2. test はどう作られたのか（時系列の分割）
    3. train と test で何が違うのか（価格情報区分、価格の推移、表記ゆれ）

    この段階ではモデルを作らず、提出もしません。結論は `docs/eda.md` にまとめます。
    """)
    return


@app.cell
def _(comp_dir, detect, pl):
    COMP = "utokyo-mlapp-2026-ex1"
    TARGET = "取引価格（総額）"
    BLUE, ORANGE = "#2a78d6", "#eb6834"

    data_dir = comp_dir(detect(), COMP)
    # Read every column as text first: the values keep MLIT's raw notation ("30分～60分", "戦前", full-width letters).
    train_raw = pl.read_csv(data_dir / "train.csv", infer_schema_length=0)
    test_raw = pl.read_csv(data_dir / "test.csv", infer_schema_length=0)
    return BLUE, ORANGE, TARGET, test_raw, train_raw


@app.cell
def _(mo, test_raw, train_raw):
    mo.vstack(
        [
            mo.md(
                f"""
                ## 1. データの大きさと列

                train は {train_raw.height:,} 行 × {train_raw.width} 列、test は {test_raw.height:,} 行 × {test_raw.width} 列です。
                test にない列は目的変数 `取引価格（総額）` だけです。すべての列を文字列として読んでいるので、元データの表記がそのまま見えます。
                """
            ),
            train_raw.head(5),
        ]
    )
    return


@app.cell
def _(TARGET, pl, test_raw, train_raw):
    QUARTER = (
        pl.col("取引時期").str.extract(r"^(\d{4})年第(\d)四半期$", 1).cast(pl.Int32)
        + (pl.col("取引時期").str.extract(r"^(\d{4})年第(\d)四半期$", 2).cast(pl.Int32) - 1) / 4
    )

    train = train_raw.with_columns(
        price=pl.col(TARGET).cast(pl.Float64),
        # One row says "2,000㎡以上"; it becomes null here and is handled in S2.
        area=pl.col("面積（㎡）").cast(pl.Float64, strict=False),
        t=QUARTER,
    ).with_columns(log1p_price=pl.col("price").log1p(), price_per_m2=pl.col("price") / pl.col("area"))
    test = test_raw.with_columns(t=QUARTER)
    return test, train


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 目的変数と評価指標

    予測するのは取引価格の総額（円）です。評価指標は RMSLE です。

    $$\mathrm{RMSLE} = \sqrt{\frac{1}{n}\sum_i \left(\log(1+\hat{y}_i) - \log(1+y_i)\right)^2}$$

    式を見ると、RMSLE は「予測と正解をそれぞれ $\log(1+x)$ に変換してから RMSE を取ったもの」と同じです。
    したがって、目的変数を最初から $\log(1+y)$ に変換して、二乗誤差を小さくするように学習すれば、RMSLE をそのまま最小化できます。
    S1 以降のモデルはすべてこの形で学習します。

    下の図の左が円のままの分布、右が $\log(1+y)$ に変換した分布です。円のままだと一部の高額物件に引っぱられて右に長く裾を引きますが、
    log をとると左右対称に近い山になります。
    """)
    return


@app.cell
def _(BLUE, plt, train):
    fig_target, (ax_raw, ax_log) = plt.subplots(1, 2, figsize=(10, 3.2))
    ax_raw.hist(train["price"].to_numpy() / 1e6, bins=100, range=(0, 300), color=BLUE)
    ax_raw.set_xlabel("price (million yen, clipped at 300)")
    ax_raw.set_ylabel("rows")
    ax_log.hist(train["log1p_price"].to_numpy(), bins=100, color=BLUE)
    ax_log.set_xlabel("log1p(price)")
    for _ax in (ax_raw, ax_log):
        _ax.spines[["top", "right"]].set_visible(False)
    fig_target.tight_layout()
    fig_target
    return


@app.cell
def _(get_metric, mo, np):
    rmsle = get_metric("rmsle")
    truth = np.array([30_000_000.0])
    examples = {
        "10% 高く予測": rmsle(truth, truth * 1.1),
        "10% 低く予測": rmsle(truth, truth * 0.9),
        "2倍で予測": rmsle(truth, truth * 2),
        "半分で予測": rmsle(truth, truth * 0.5),
    }
    table = "\n".join(["| 予測 | RMSLE |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in examples.items()])
    mo.md(
        "RMSLE は「何円ずれたか」ではなく「何倍ずれたか」を測ります。3,000万円の物件で試すと次のとおりです。\n\n"
        + table
        + "\n\n2倍と半分は同じ誤差になります。10% のずれはおよそ 0.1 です。"
        "順位表の AI単独ベースライン 0.197 は、典型的に 2 割前後ずれる予測だと読めます。"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. test はどう作られたか

    データの説明によると、train は 2019年第1四半期〜2025年第2四半期、test は 2025年第3四半期〜2026年第1四半期です。
    四半期ごとの件数を並べると、test は train の直後の3四半期だと分かります。つまり **過去のデータから未来の取引価格を当てる課題** です。

    手元で精度を測るとき（CV）も、この作り方をまねる必要があります。本講座では、2024年第4四半期までで学習し、
    2025年第1〜第2四半期で検証する「時系列 holdout」を使います（`docs/validation.md`）。
    """)
    return


@app.cell
def _(BLUE, ORANGE, pl, plt, test, train):
    per_quarter = pl.concat(
        [
            train.group_by("t").len().with_columns(split=pl.lit("train")),
            test.group_by("t").len().with_columns(split=pl.lit("test")),
        ]
    ).sort("t")
    fig_split, ax_split = plt.subplots(figsize=(10, 3))
    for _split, _color in (("train", BLUE), ("test", ORANGE)):
        _rows = per_quarter.filter(pl.col("split") == _split)
        ax_split.bar(_rows["t"].to_numpy(), _rows["len"].to_numpy(), width=0.2, color=_color, label=_split)
    ax_split.axvline(2025.0 - 0.125, color="#888888", linestyle="--", linewidth=1)
    ax_split.set_ylim(0, per_quarter["len"].max() * 1.18)
    ax_split.text(2025.0 - 0.1, per_quarter["len"].max() * 1.08, "holdout starts", fontsize=8, color="#555555")
    ax_split.set_xlabel("trade quarter")
    ax_split.set_ylabel("rows")
    ax_split.legend(frameon=False, loc="upper left")
    ax_split.spines[["top", "right"]].set_visible(False)
    fig_split.tight_layout()
    fig_split
    return


@app.cell
def _(mo):
    mo.md(r"""
    2020年から2021年にかけて件数が倍に増えています。次の節で見るとおり、これは2021年から「成約価格情報」が加わったためです。

    ## 4. 2つの価格情報区分

    `価格情報区分` には2種類の値があります。

    - 不動産取引価格情報: 国交省が取引の当事者にアンケートして集めた価格
    - 成約価格情報: 不動産会社のネットワーク（レインズ）に登録された成約価格を、国交省が加工したもの。2021年から

    出どころが違うので、件数の推移も、埋まっている列も違います。
    """)
    return


@app.cell
def _(BLUE, ORANGE, pl, plt, test, train):
    by_type = (
        pl.concat([train.select("t", "価格情報区分"), test.select("t", "価格情報区分")])
        .group_by("t", "価格情報区分")
        .len()
        .sort("t")
    )
    fig_type, ax_type = plt.subplots(figsize=(10, 3))
    for _kind, _color, _label in (
        ("不動産取引価格情報", BLUE, "survey price"),
        ("成約価格情報", ORANGE, "contract price"),
    ):
        _rows = by_type.filter(pl.col("価格情報区分") == _kind)
        ax_type.plot(
            _rows["t"].to_numpy(), _rows["len"].to_numpy(), color=_color, linewidth=2, marker="o", markersize=4
        )
        ax_type.text(_rows["t"].to_numpy()[-1] + 0.05, _rows["len"].to_numpy()[-1], _label, color="#333333", fontsize=8)
    ax_type.axvline(2025.5 - 0.125, color="#888888", linestyle="--", linewidth=1)
    ax_type.text(2025.5 - 0.1, by_type["len"].min(), "test", fontsize=8, color="#555555")
    ax_type.set_xlabel("trade quarter")
    ax_type.set_ylabel("rows")
    ax_type.set_xlim(2018.8, 2026.9)
    ax_type.spines[["top", "right"]].set_visible(False)
    fig_type.tight_layout()
    fig_type
    return


@app.cell
def _(mo, pl, test, train):
    def share_contract(df):
        return (df["価格情報区分"] == "成約価格情報").mean()

    train_block = train.filter(pl.col("t") < 2025)
    holdout = train.filter(pl.col("t") >= 2025)
    mo.md(
        f"""
        成約価格情報は2021年第1四半期から現れ、2025年から件数が増えています。一方、test の最後の四半期（2026年第1四半期）は
        取引価格情報が少なく、公表が追いついていないと見られます。その結果、成約価格情報の割合は時期によって大きく違います。

        | 期間 | 成約価格情報の割合 |
        |---|---|
        | 学習に使う期間（2019〜2024年） | {share_contract(train_block):.0%} |
        | 検証に使う期間（2025年第1〜第2四半期） | {share_contract(holdout):.0%} |
        | test（2025年第3四半期〜2026年第1四半期） | {share_contract(test):.0%} |

        検証期間の割合が test に近いことも、時系列 holdout を選んだ理由の1つです。
        """
    )
    return holdout, train_block


@app.cell
def _(mo, pl, train):
    missing_cols = [
        "最寄駅：距離（分）",
        "間取り",
        "建築年",
        "都市計画",
        "用途",
        "今後の利用目的",
        "建ぺい率（％）",
        "容積率（％）",
        "改装",
        "取引の事情等",
    ]
    missing_by_type = (
        train.group_by("価格情報区分")
        .agg([pl.col(c).is_null().mean().round(3).alias(c) for c in missing_cols])
        .sort("価格情報区分")
    )
    mo.vstack(
        [
            mo.md(
                """
                埋まっている列も違います。下の表は区分ごとの欠損の割合です（1.0 は全行が欠損）。
                成約価格情報の行には `用途`、`今後の利用目的`、`建ぺい率（％）`、`容積率（％）`、`取引の事情等` がまったくありません。
                これらの列の欠損は「値が分からない」ではなく「成約価格情報の行である」という意味を持っています。
                """
            ),
            missing_by_type,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. 価格の推移

    四半期ごとに、1㎡あたりの価格（㎡単価）の中央値を区分別に並べます。総額ではなく㎡単価を見るのは、
    広い物件が多い時期と狭い物件が多い時期とで、総額の平均が見かけ上動いてしまうからです。
    """)
    return


@app.cell
def _(BLUE, ORANGE, pl, plt, train):
    trend = (
        train.drop_nulls("price_per_m2")
        .group_by("t", "価格情報区分")
        .agg(pl.col("price_per_m2").median().alias("median_ppm2"), pl.col("area").median().alias("median_area"))
        .sort("t")
    )
    fig_trend, ax_trend = plt.subplots(figsize=(10, 3.2))
    for _kind, _color, _label in (
        ("不動産取引価格情報", BLUE, "survey price"),
        ("成約価格情報", ORANGE, "contract price"),
    ):
        _rows = trend.filter(pl.col("価格情報区分") == _kind)
        ax_trend.plot(
            _rows["t"].to_numpy(),
            _rows["median_ppm2"].to_numpy() / 1e4,
            color=_color,
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax_trend.text(
            _rows["t"].to_numpy()[-1] + 0.05,
            _rows["median_ppm2"].to_numpy()[-1] / 1e4,
            _label,
            color="#333333",
            fontsize=8,
        )
    ax_trend.axvline(2025.0 - 0.125, color="#888888", linestyle="--", linewidth=1)
    ax_trend.text(2025.0 - 0.1, trend["median_ppm2"].min() / 1e4, "holdout", fontsize=8, color="#555555")
    ax_trend.set_xlabel("trade quarter")
    ax_trend.set_ylabel("median price per m2 (10k yen)")
    ax_trend.set_xlim(2018.8, 2026.4)
    ax_trend.spines[["top", "right"]].set_visible(False)
    fig_trend.tight_layout()
    fig_trend
    return (trend,)


@app.cell
def _(holdout, mo, np, train_block, trend):
    gap = holdout["log1p_price"].mean() - train_block["log1p_price"].mean()
    latest = trend.filter(trend["t"] == trend["t"].max())
    mo.md(
        f"""
        どちらの区分も㎡単価は上がり続けています。検証期間の log1p 価格の平均は、学習期間より {gap:.3f} 高く、
        価格に直すと約 {np.expm1(gap):.0%} 高い水準です。

        ここに、木のモデル（GBDT）の弱点が関わります。木は学習データで見た範囲の値しか出せません。
        そのため、学習期間より高い価格が続く未来には、低めの予測を出し続けます。これをどう補うかが S3 の論点です。

        2つの区分は㎡単価の水準も違います。直近の四半期（{latest["t"][0]:.2f}）では、
        取引価格情報の㎡単価のほうが高く、面積の中央値は小さい傾向があります（表は下）。
        区分によって、取引される物件の種類そのものが違うということです。
        """
    )
    return


@app.cell
def _(trend):
    trend.filter(trend["t"] >= 2024).sort("t", "価格情報区分")
    return


@app.cell
def _(mo, pl, train):
    non_numeric_distance = (
        train.filter(~pl.col("最寄駅：距離（分）").str.contains(r"^\d+$"))["最寄駅：距離（分）"]
        .value_counts()
        .sort("count", descending=True)
    )
    other_notation = pl.DataFrame(
        {
            "列": ["建築年", "面積（㎡）", "間取り", "建物の構造"],
            "数値でない・揺れのある値の例": [
                ", ".join(train.filter(~pl.col("建築年").str.contains(r"^\d{4}年$"))["建築年"].unique().to_list()),
                ", ".join(train.filter(pl.col("area").is_null())["面積（㎡）"].unique().to_list()),
                ", ".join(train["間取り"].drop_nulls().unique().sort().head(8).to_list()),
                ", ".join(train["建物の構造"].drop_nulls().unique().sort().to_list()),
            ],
        }
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 6. 表記ゆれ

                値は元データの表記のままです。数値に見える列にも文字が混ざっています。
                たとえば `最寄駅：距離（分）` は、30分までは数字ですが、それより遠いとレンジ（`30分～60分`、`1H～1H30`）で書かれています。
                このまま扱うと、モデルは「60分は30分より遠い」ことを知りません。S2 で数値に直します。
                """
            ),
            non_numeric_distance,
            other_notation,
        ]
    )
    return


@app.cell
def _(mo, pl, test, train):
    def unseen_rows(col: str) -> int:
        seen = train[col].drop_nulls().unique()
        return test.filter(pl.col(col).is_not_null() & ~pl.col(col).is_in(seen.implode())).height

    unseen = pl.DataFrame(
        {
            "列": ["市区町村名", "地区名", "最寄駅：名称", "間取り"],
            "train の種類数": [train[c].n_unique() for c in ["市区町村名", "地区名", "最寄駅：名称", "間取り"]],
            "test にだけある値の行数": [unseen_rows(c) for c in ["市区町村名", "地区名", "最寄駅：名称", "間取り"]],
        }
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 7. test にだけ現れる値

                場所の列（市区町村、地区、駅）は、test の値のほとんどが train にも出てきます。
                場所をカテゴリとしてそのままモデルに渡しても、test で困ることはほぼありません。
                """
            ),
            unseen,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## まとめ: 次の段階でやること

    | 見つけたこと | 対応する段階 |
    |---|---|
    | RMSLE は log1p した値の RMSE。log1p を目的変数にすれば指標をそのまま最小化できる | S1 |
    | test は train の直後の未来。CV も時系列で切る | S1（時系列 holdout） |
    | 数値の列にレンジや文字が混ざっている | S2 |
    | 価格情報区分で埋まっている列も価格水準も違い、test では成約が多い | S2（区分別の誤差を見る） |
    | 価格は上がり続けており、木のモデルは未来の高い価格を出せない | S3 |
    | 場所の値は test でもほぼ既知 | S2（カテゴリ）、S4（Target encoding）、S6（座標） |
    """)
    return


if __name__ == "__main__":
    app.run()
