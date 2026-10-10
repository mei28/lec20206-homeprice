import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from kgl import mlit
    from kgl.env import comp_dir, detect

    return comp_dir, detect, mlit, mo, np, pl, plt


@app.cell
def _(mo):
    mo.md(r"""
    # S2. 特徴量づくりのための EDA

    S0 ではデータ全体の形を見ました。この段階では、**どんな特徴量を作ればよいか** を決めるために、2つのことを調べます。

    1. 各列と価格の関係。列の値が変わると㎡単価がどう変わるか
    2. S1 のベースラインの誤差がどこに集まっているか

    価格は総額ではなく㎡単価（1㎡あたりの価格）で見ます。総額は面積に強く引っぱられるので、面積以外の列の効き方が見えにくくなるからです。
    この段階ではモデルを作らず、提出もしません。最後に、見つけたことを S3 以降の特徴量の案としてまとめます。
    """)
    return


@app.cell
def _(comp_dir, detect, mlit, pl):
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    env = detect()
    raw = pl.read_csv(comp_dir(env, "utokyo-mlapp-2026-ex1") / "train.csv", infer_schema_length=0)
    # The raw notation is parsed with kgl.mlit (tested in tests/test_mlit.py); S3 uses the same parsers.
    train = raw.with_columns(
        pl.col("id").cast(pl.Int64),
        price=pl.col("取引価格（総額）").cast(pl.Float64),
        t=mlit.trade_period("取引時期"),
        minutes=mlit.station_minutes("最寄駅：距離（分）"),
        year=mlit.build_year("建築年"),
        area=mlit.area_m2("面積（㎡）"),
        rooms=mlit.layout_rooms("間取り"),
    ).with_columns(
        ppm2=pl.col("price") / pl.col("area"),
        age=pl.col("t").floor() - pl.col("year"),
    )
    return BLUE, ORANGE, env, train


@app.cell
def _(env, pl, train):
    # S1's holdout predictions: the residual is log(1 + prediction) - log(1 + truth); negative means too low.
    oof = pl.read_parquet(env.output_dir / "s1_baseline" / "lgbm_raw" / "oof.parquet").select("id", "pred")
    holdout = train.join(oof, on="id", how="inner").with_columns(
        residual=pl.col("pred").log1p() - pl.col("price").log1p()
    )
    assert holdout.height == 21_691, holdout.height
    return (holdout,)


@app.cell
def _(pl):
    def median_ppm2(frame, key):
        """Rows and median price per m2 (10k yen) per value of `key`."""
        return (
            frame.group_by(key)
            .agg(pl.len().alias("rows"), (pl.col("ppm2").median() / 1e4).round(1).alias("median_ppm2_10k_yen"))
            .sort(key)
        )

    def residual_by(frame, key):
        """S1 holdout residuals per value of `key`: mean (bias) and RMSLE."""
        return (
            frame.group_by(key)
            .agg(
                pl.len().alias("rows"),
                pl.col("residual").mean().round(3).alias("bias"),
                (pl.col("residual") ** 2).mean().sqrt().round(3).alias("rmsle"),
            )
            .sort(key)
        )

    return median_ppm2, residual_by


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 駅からの距離

    S1 では `30分～60分` のようなレンジを欠損にしていました。ここでは `kgl.mlit` で数値に直しています（レンジは中央の値、`2H～` は 120 分）。
    5分刻みで㎡単価の中央値を並べると、駅から遠いほどはっきり安くなります。
    """)
    return


@app.cell
def _(BLUE, pl, plt, train):
    by_minutes = (
        train.drop_nulls("minutes")
        .filter(pl.col("minutes") < 30)
        .group_by("minutes")
        .agg(pl.len().alias("rows"), pl.col("ppm2").median().alias("ppm2"))
        .sort("minutes")
    )
    fig_minutes, ax_minutes = plt.subplots(figsize=(8, 3))
    ax_minutes.plot(by_minutes["minutes"], by_minutes["ppm2"] / 1e4, color=BLUE, linewidth=2, marker="o", markersize=4)
    ax_minutes.set_xlabel("walking minutes to the nearest station (0-29)")
    ax_minutes.set_ylabel("median price per m2 (10k yen)")
    ax_minutes.spines[["top", "right"]].set_visible(False)
    fig_minutes.tight_layout()
    fig_minutes
    return


@app.cell
def _(median_ppm2, mo, pl, train):
    far = median_ppm2(train.filter(pl.col("minutes") >= 30), "minutes")
    no_station = train.filter(pl.col("minutes").is_null())
    mo.vstack(
        [
            mo.md(f"""
            30分を超えるレンジの行は少なく（下の表）、値もばらつきます。駅の情報がない行（{no_station.height:,} 行）の㎡単価の中央値は
            {no_station["ppm2"].median() / 1e4:.1f} 万円で、駅に近い行よりかなり安い水準です。
            """),
            far,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 築年数と建築年

    建築年から、取引した時点での築年数（取引年 − 建築年）を作れます。2019年と2024年の取引について、築年数ごとの㎡単価を並べます。
    """)
    return


@app.cell
def _(BLUE, ORANGE, pl, plt, train):
    by_age = (
        train.drop_nulls("age")
        .filter(pl.col("t").floor().is_in([2019, 2024]) & pl.col("age").is_between(0, 55))
        .with_columns(trade_year=pl.col("t").floor().cast(pl.Int32), age_bin=(pl.col("age") // 5 * 5))
        .group_by("trade_year", "age_bin")
        .agg(pl.col("ppm2").median())
        .sort("age_bin")
    )
    fig_age, ax_age = plt.subplots(figsize=(8, 3))
    for _year, _color in ((2019, BLUE), (2024, ORANGE)):
        _rows = by_age.filter(pl.col("trade_year") == _year)
        ax_age.plot(_rows["age_bin"], _rows["ppm2"] / 1e4, color=_color, linewidth=2, marker="o", markersize=4)
        ax_age.text(
            _rows["age_bin"][-1] + 1, _rows["ppm2"][-1] / 1e4, f"traded in {_year}", fontsize=8, color="#333333"
        )
    ax_age.set_xlabel("age at trade (years, 5-year bins)")
    ax_age.set_ylabel("median price per m2 (10k yen)")
    ax_age.set_xlim(-2, 66)
    ax_age.spines[["top", "right"]].set_visible(False)
    fig_age.tight_layout()
    fig_age
    return


@app.cell
def _(comp_dir, env, mlit, mo, pl, train):
    test_years = pl.read_csv(comp_dir(env, "utokyo-mlapp-2026-ex1") / "test.csv", infer_schema_length=0).select(
        mlit.build_year("建築年").alias("year")
    )
    new_in_test = test_years.filter(pl.col("year") >= 2025).height
    new_in_train = train.filter(pl.col("year") >= 2025).height
    negative_age = train.filter(pl.col("age") < 0).height
    mo.md(f"""
    どちらの年も、築年数が進むほど㎡単価は下がり、築30年ほどで下げ止まります。2024年の線は全体に上にあり、これは S0 で見た価格の上昇です。

    築年数と建築年は同じ情報に見えますが、未来を予測するときには違いが出ます。test には建築年が 2025年以降の行が {new_in_test:,} 行あります。
    train 全体では {new_in_train:,} 行しかなく、学習期間（2024年まで）にはほとんどありません。木のモデルは見たことのない建築年をうまく扱えません。
    築年数なら、これらの行も「築0〜1年」という見慣れた値になります。

    なお、train には建築年が取引年より後の行（築年数が負）が {negative_age} 行あります。完成前に取引された物件と考えられます。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 面積と間取り

    面積ごとの㎡単価は、狭い部屋ほど高く、100㎡を超えると再び高くなります。単身向けの狭い部屋と、広い高級物件で単価の決まり方が違うようです。
    """)
    return


@app.cell
def _(median_ppm2, mo, pl, train):
    area_bins = median_ppm2(
        train.with_columns(area_bin=pl.col("area").cut([20, 30, 40, 50, 60, 70, 80, 100, 150])), "area_bin"
    )
    rooms_table = median_ppm2(train, "rooms")
    mo.hstack([area_bins, rooms_table], justify="start", gap=2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    間取りの部屋数（右の表、`null` は「スタジオ」「メゾネット」など数字のない間取り）も、部屋数が多いほど単価が下がります。
    S1 では間取りを49種類のカテゴリとして渡していました。部屋数と L/D/K/S/R の有無に分けると、珍しい間取りにも傾向を当てはめられます。

    ## 4. 建物と都市計画
    """)
    return


@app.cell
def _(median_ppm2, mo, train):
    mo.hstack(
        [median_ppm2(train, "建物の構造"), median_ppm2(train, "都市計画"), median_ppm2(train, "改装")],
        justify="start",
        gap=2,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    - 構造は RC（鉄筋コンクリート）が多数派で、SRC（鉄骨鉄筋コンクリート）は単価がやや低めです。築年数の中央値は SRC が32年、RC が15年で、SRC の安さには古さが混ざっています。
    - 都市計画では「商業」地域の単価が高く、駅前や都心に多い用途地域を表していると読めます。
    - 改装は「改装済み」のほうが安く見えます。築年数の中央値は改装済みが28年、未改装が16年で、改装されるのは古い物件です。1つの列だけを見て因果を読まないよう注意が要ります。

    ## 5. 場所
    """)
    return


@app.cell
def _(mo, pl, train):
    wards = (
        train.group_by("市区町村名")
        .agg(pl.len().alias("rows"), (pl.col("ppm2").median() / 1e4).round(1).alias("median_ppm2_10k_yen"))
        .sort("median_ppm2_10k_yen")
    )
    districts = train.group_by("地区名").len()
    small = districts.filter(pl.col("len") < 30)
    mo.vstack(
        [
            mo.md(f"""
            市区町村ごとの㎡単価の中央値は、{wards["median_ppm2_10k_yen"].min()} 万円から {wards["median_ppm2_10k_yen"].max()} 万円まで、
            {wards["median_ppm2_10k_yen"].max() / wards["median_ppm2_10k_yen"].min():.0f} 倍の開きがあります。場所は価格を最も強く左右する列です。

            地区名は {districts.height:,} 種類あり、そのうち {small.height} 種類は30行未満です。ただし30行未満の地区に属する行は全体の
            {small["len"].sum() / train.height:.1%} だけです。地区ごとの平均価格を特徴量にする（S5 の Target encoding）場合は、行の少ない地区を
            市区町村の平均に寄せる「平滑化」が要ります。
            """),
            mo.hstack([wards.head(5), wards.tail(5)], justify="start", gap=2),
        ]
    )
    return


@app.cell
def _(mo, np, pl, train):
    cheap = train.filter(pl.col("ppm2") < 1e5)
    pricey = train.filter(pl.col("ppm2") > 5e6)
    mo.vstack(
        [
            mo.md(f"""
            ## 6. 極端な価格

            ㎡単価が 10万円未満の行が {cheap.height} 行、500万円を超える行が {pricey.height} 行あります。
            S1 で外れの大きかった行もここに含まれます。`取引の事情等` は前者の98%、後者のすべてで空欄で、手元の列からは普通の取引と見分けられません。
            学習から外すか残すかは、CV で確かめて決める候補です（log で見た誤差の大きさは {np.log(5e6 / 1e5):.1f} にもなります）。
            """),
            pl.concat([cheap.head(5), pricey.head(5)]).select(
                "価格情報区分",
                "市区町村名",
                "面積（㎡）",
                "間取り",
                "建築年",
                "取引時期",
                "取引の事情等",
                "price",
                "ppm2",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. S1 の誤差はどこに集まっているか

    S1 の holdout（2025年第1〜第2四半期）での残差（予測の log − 正解の log）を、列ごとに集計します。
    S1 の誤差の大半は `取引時期` をカテゴリで渡したことによる全体的な低め（約 -0.18）でした。ここでは、それに加えて **どこでずれ方が違うか** を見ます。
    """)
    return


@app.cell
def _(BLUE, holdout, pl, plt, residual_by):
    by_age_res = residual_by(
        holdout.drop_nulls("age")
        .filter(pl.col("age").is_between(0, 59))
        .with_columns(age_bin=pl.col("age") // 10 * 10),
        "age_bin",
    )
    by_area_res = residual_by(holdout.with_columns(area_bin=pl.col("area").cut([20, 30, 50, 70, 100])), "area_bin")
    fig_res, (ax_res_age, ax_res_area) = plt.subplots(1, 2, figsize=(10, 3))
    ax_res_age.bar(by_age_res["age_bin"].to_numpy(), by_age_res["bias"].to_numpy(), width=8, color=BLUE)
    ax_res_age.set_xlabel("age at trade (10-year bins)")
    ax_res_age.set_ylabel("mean residual (log)")
    ax_res_area.bar(range(by_area_res.height), by_area_res["bias"].to_numpy(), color=BLUE)
    ax_res_area.set_xticks(range(by_area_res.height))
    ax_res_area.set_xticklabels([str(v) for v in by_area_res["area_bin"].to_list()], fontsize=8)
    ax_res_area.set_xlabel("area (m2)")
    for _ax in (ax_res_age, ax_res_area):
        _ax.axhline(0, color="#888888", linewidth=1)
        _ax.spines[["top", "right"]].set_visible(False)
    fig_res.tight_layout()
    fig_res
    return by_age_res, by_area_res


@app.cell
def _(by_age_res, by_area_res, mo):
    mo.vstack(
        [
            mo.md("""
            新しい物件ほど、広い物件ほど、低めのずれが大きくなっています。築10年未満は築50年代よりずれが倍ほど大きく、
            100㎡を超える物件は20㎡以下の物件の3倍以上ずれています。

            これは、価格の上がり方が物件の種類によって違うことを示しています。直近では、新しい物件と広い物件の値上がりが速かったと読めます。
            S4 で価格指数を作るとき、全体で1本の指数にするか、築年数や面積の区分ごとに作るかの判断材料になります。
            """),
            mo.hstack([by_age_res, by_area_res], justify="start", gap=2),
        ]
    )
    return


@app.cell
def _(holdout, mo, pl, residual_by):
    mo.vstack(
        [
            mo.md("""
            一方、価格情報区分による違いは小さく（-0.16 と -0.19）、駅距離がレンジで書かれた行は holdout に52行しかないので、全体への影響は小さいと言えます。
            """),
            mo.hstack(
                [
                    residual_by(holdout, "価格情報区分"),
                    residual_by(
                        holdout.with_columns(range_label=pl.col("最寄駅：距離（分）").str.contains("～")), "range_label"
                    ),
                ],
                justify="start",
                gap=2,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## まとめ: S3 以降で作る特徴量

    | 見つけたこと | 特徴量の案 | 段階・Issue |
    |---|---|---|
    | 取引時期をカテゴリで渡すと、未来の四半期が欠損扱いになる（S1） | 取引時期を数値（年 + (四半期 − 1)/4）にする | S3 #12 |
    | 駅から遠いほど単価が下がる。30分超はレンジ表記 | 駅距離を分の数値にする（レンジは中央値） | S3 #9 |
    | 築年数で単価がなめらかに下がる。test には学習にない新しい建築年が多い | 築年数（取引年 − 建築年） | S3 #10 |
    | 部屋数が多いほど単価が下がる。間取りは49種類 | 部屋数と L/D/K/S/R の有無 | S3 #11 |
    | 市区町村で単価が19倍違う。地区名は1,210種類 | 場所のカテゴリ、行の少ない地区の平滑化 | S3 #13、S5 #18 |
    | 新しい物件・広い物件ほど S1 の低めのずれが大きい | 価格指数を物件の区分ごとに作る | S4 #16（新しい案の候補） |
    | ㎡単価が極端な行がある（10万円未満、500万円超） | 学習から外すかを CV で確かめる | S3（新しい案の候補） |

    最後の2つは、まだ Issue になっていない新しい案です。採用するかはユーザーが決めます。
    """)
    return


if __name__ == "__main__":
    app.run()
