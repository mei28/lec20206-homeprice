# Roadmap

Maintained by the `kaggle-journal` skill: the proposed order of work and the decisions waiting for
the user. The user decides (CLAUDE.md, Decision gates); details live in the Issues. Updated in a
wrap-up when the order or the pending decisions change.

## Where we are

| Item | Value |
|---|---|
| Our best public LB | 0.19222（S9 `stacking`、2026-10-16、CV 0.1951）。S6 は 0.19642、S5 は 0.19895、S4 は 0.19911、S3 は 0.20149 |
| Best public notebook | なし（公開ノートブック0件、2026-10-10） |
| LB top and medal line | 学生チームの1位は 0.19207、3位は 0.19328。S9 で2位（2026-10-16）。AI単独ベースライン（0.19726）は学生4チームが上回った。メダルなし |
| CV of the current baseline | S10 スタッキング 0.1950（holdout RMSLE）。S9 スタッキング 0.1951。単独の最良は S6 CatBoost 0.1990。S5 LightGBM 0.2035、S4 0.2050、S3 0.2090、S1 0.2988、下限 0.8137。S1 の leak 点検は clean |

## 方針（2026-10-10 にユーザーと決定）

- TA として、学生が目指す目安を作る。atmaCup 初心者講座のように、1段階を1本の marimo ノートブックにし、解説もその中に書く。最終的には上位を狙える水準まで進めるが、順位表には段階ごとに小出しにする。
- 提出は段階ごとに最大1回。出すかどうかと時期は毎回ユーザーが決める（1日1回まで）。
- CV は時系列 holdout（`folds/holdout_2025q1.parquet`、#4）。提出用モデルは全期間で学習し直す。
- ディスカッションには、載せられるときに文章・コード断片・図・CV と LB を載せる。ノートブック全体は締切後に公開する。
- AI の使い方は教材に含めない。

## What drives the score

仮説の段階。根拠は `docs/competition.md` の Prior art と `docs/validation.md`。

- 場所（市区町村、地区、駅）。㎡単価の大半を決める。S2 のカテゴリ特徴量、S4 の TE、S6 の座標。
- 取引時期の扱いと時間トレンド。S3 で取引時期を数値にして 0.2988 → 0.2090（#12）、S4 でヘドニック指数を外挿して 0.2050（#16）。S4 の後は四半期ごとのずれがほぼ消え（-0.007、-0.011）、築年数による差（新しい物件ほど低め）が残る。test 用の指数の選択（ヘドニックか中央値か）は holdout で確かめられず、最初の LB で確かめる。残るずれは四半期が進むほど大きく（2025Q1 -0.031、Q2 -0.050）、新しい物件・広い物件ほど大きい。S4（#16、#32）。以下は S1 時点の分析:S1 の予測は holdout で平均 -0.182（約 17% 低め）ずれていた。その大半は、`取引時期` をカテゴリで渡したため未知の四半期が欠損扱いになったことが原因で、2024年第4四半期として渡すだけで 0.2988 → 0.2108（ずれ -0.040）になる（#6、#12）。残るずれが木の外挿できない分で、S3（#16）の課題。
- 価格情報区分の構成のずれ。成約の割合は学習期間 43%、検証期間 62%、test 67%。S2 の区分別の誤差分析。

## Proposed order

2026-10-10 に改訂。EDA の段階（S2）を足し、GBDT の比較（S6）と MLP・古典モデル（S7）を分けた。日付は目安。序盤は 10/12 21:35 JST まで、中盤は 10/18 02:23 JST まで。外部データ（S8）は中盤のうちに入れる。

| 段階 | ノートブック | 内容 | Issues | 目安 |
|---|---|---|---|---|
| S0 | `s0_data.py` | データを知る | #5 | 済 |
| S1 | `s1_baseline.py` | ベースライン（log1p、LightGBM、時系列 holdout） | #6 #7 #8 | 済（CV 0.2988、未提出） |
| S2 | `s2_eda.py` | 特徴量づくりのための EDA（提出なし） | #26 | 済 |
| S3 | `s3_features.py` | 前処理と特徴量 | #9 #10 #11 #12 #13 #14 #31 | 済（CV 0.2090、#12 #9 採用、提出 LB 0.20149） |
| S4 | `s4_trend.py` | 価格指数と時間トレンド（学習データ内の指数で実質価格化、区分ごとの指数、㎡単価、重み） | #15 #16 #17 #32 | 済（CV 0.2050、ヘドニック指数を採用、提出 LB 0.19911） |
| S5 | `s5_te.py` | Target encoding、集約特徴、件数 | #18 #19 #33 | 済（CV 0.2035、3つ全部を採用、提出 LB 0.19895）。特徴量一式は `kgl.feature_set` |
| S6 | `s6_gbdt.py` | GBDT の比較（LightGBM、XGBoost、CatBoost） | #27 #24 | 済（CatBoost 0.1990、XGBoost 0.2103、CatBoost を提出 LB 0.19642） |
| S7 | `s7_mlp_classic.py` | MLP と古典モデル（RandomForest、Ridge） | #20 #28 #29 | 済（MLP 0.2168、RF 0.2346、Ridge 0.2652） |
| S8 | `s8_external.py` | 外部データ（駅、町丁目の座標） | #21 #22 | 済（差は偶然の範囲、不採用） |
| S9 | `s9_ensemble.py` | アンサンブル（平均、重み付き平均、stacking） | #23 #30 | 済（stacking 0.1951 を採用、提出 LB 0.19222。単純平均は 0.2054 で単独より悪い） |
| S10 | `s10_final.py` | 仕上げ（LightGBM の seed 平均、CatBoost の本数、最終のアンサンブル、最終2提出の候補） | #25 | 済（stacking 0.1950、重み付き 0.1959） |
| S11 | `s11_pseudo.py` | pseudo label（S10 の予測を test の仮の正解にして LightGBM を学習し直す） | #34 | 済（生徒単独は S5 より -0.0063〜-0.0069 だが、S5 と教師の単純な平均と同じ。スタッキングに戻すと差なし、不採用） |
| S12 | `s12_black_magic.py` | 黒魔術（MLP・LightGBM のスタッキングと、スタッキングの平均。俵さんの記事にならう） | #35 | 済（スタッキングの平均 0.1938、S10 より -0.0012。採用し、最終提出の候補に追加） |

提出は段階ごとに最大1回で、出すかどうかはユーザーが決める。S1 は「もう少し向上させてから」で見送り（2026-10-10）。提出枠は 10/10〜10/21 の12回。最初から最良を出さず、段階の順に1日1回ずつ少しずつ出す（2026-10-11 ユーザー）。

## Deprioritized Issues

## Decisions for the user

- 次の提出をいつ、どれにするか（S3 は 2026-10-10、S4 は 2026-10-11、S5 は 2026-10-14、S6 は 2026-10-15、S9 は 2026-10-16 に提出済み。段階の順なら次は S12 の黒魔術）。作成・検査済みの提出ファイル: S1 `lgbm_raw`（0.2988）、S3 `plus_notation`（0.2090）、S4 `hedonic_linear`（0.2050）、S5 `all_three`（0.2035）、S6 `catboost`（0.1990）、S6 `xgboost`（0.2103）、S7 `mlp`（0.2168）ほか、S9 `stacking`（0.1951）・`weighted_average`（0.1960）・`simple_average`（0.2054）、S10 `stacking`（0.1950）・`weighted_average`（0.1959）・`lightgbm_seed_avg`（0.2022）・`catboost_long`（0.1990）
- 最終2提出の候補: S12 `stacker_average`（CV 0.1938、最良）、S10 `stacking`（0.1950）、S10 `weighted_average`（0.1959、重みが0以上で保守的）。どちらも S4 のヘドニック指数の外挿が前提。S4 の提出で CV と LB は同じ向きに動いた（CV -0.0040、LB -0.0024、#3）
- S9: stacking は「2025Q1 で fit して Q2 で採点、逆向きも」の方式で進める（2026-10-10、ユーザーの「ガンガン進めていい」を受けて決定）
- 各段階のディスカッション投稿の有無
