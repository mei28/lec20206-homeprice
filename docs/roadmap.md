# Roadmap

Maintained by the `kaggle-journal` skill: the proposed order of work and the decisions waiting for
the user. The user decides (CLAUDE.md, Decision gates); details live in the Issues. Updated in a
wrap-up when the order or the pending decisions change.

## Where we are

| Item | Value |
|---|---|
| Our best public LB | 未提出 |
| Best public notebook | なし（公開ノートブック0件、2026-10-10） |
| LB top and medal line | AI単独ベースライン 0.19726（Public）。メダルなし |
| CV of the current baseline | S3 `plus_notation` 0.2090（holdout RMSLE）。S1 は 0.2988、下限は 0.8137。S1 の leak 点検は clean |

## 方針（2026-10-10 にユーザーと決定）

- TA として、学生が目指す目安を作る。atmaCup 初心者講座のように、1段階を1本の marimo ノートブックにし、解説もその中に書く。最終的には上位を狙える水準まで進めるが、順位表には段階ごとに小出しにする。
- 提出は段階ごとに最大1回。出すかどうかと時期は毎回ユーザーが決める（1日1回まで）。
- CV は時系列 holdout（`folds/holdout_2025q1.parquet`、#4）。提出用モデルは全期間で学習し直す。
- ディスカッションには、載せられるときに文章・コード断片・図・CV と LB を載せる。ノートブック全体は締切後に公開する。
- AI の使い方は教材に含めない。

## What drives the score

仮説の段階。根拠は `docs/competition.md` の Prior art と `docs/validation.md`。

- 場所（市区町村、地区、駅）。㎡単価の大半を決める。S2 のカテゴリ特徴量、S4 の TE、S6 の座標。
- 取引時期の扱いと時間トレンド。S3 で取引時期を数値にして 0.2988 → 0.2090（#12）。残るずれは四半期が進むほど大きく（2025Q1 -0.031、Q2 -0.050）、新しい物件・広い物件ほど大きい。S4（#16、#32）。以下は S1 時点の分析:S1 の予測は holdout で平均 -0.182（約 17% 低め）ずれていた。その大半は、`取引時期` をカテゴリで渡したため未知の四半期が欠損扱いになったことが原因で、2024年第4四半期として渡すだけで 0.2988 → 0.2108（ずれ -0.040）になる（#6、#12）。残るずれが木の外挿できない分で、S3（#16）の課題。
- 価格情報区分の構成のずれ。成約の割合は学習期間 43%、検証期間 62%、test 67%。S2 の区分別の誤差分析。

## Proposed order

2026-10-10 に改訂。EDA の段階（S2）を足し、GBDT の比較（S6）と MLP・古典モデル（S7）を分けた。日付は目安。序盤は 10/12 21:35 JST まで、中盤は 10/18 02:23 JST まで。外部データ（S8）は中盤のうちに入れる。

| 段階 | ノートブック | 内容 | Issues | 目安 |
|---|---|---|---|---|
| S0 | `s0_data.py` | データを知る | #5 | 済 |
| S1 | `s1_baseline.py` | ベースライン（log1p、LightGBM、時系列 holdout） | #6 #7 #8 | 済（CV 0.2988、未提出） |
| S2 | `s2_eda.py` | 特徴量づくりのための EDA（提出なし） | #26 | 済 |
| S3 | `s3_features.py` | 前処理と特徴量 | #9 #10 #11 #12 #13 #14 #31 | 済（CV 0.2090、#12 #9 採用、未提出） |
| S4 | `s4_trend.py` | 価格指数と時間トレンド（学習データ内の指数で実質価格化、区分ごとの指数、㎡単価、重み） | #15 #16 #17 #32 | 10/13 |
| S5 | `s5_te.py` | Target encoding、集約特徴、件数 | #18 #19 #33 | 10/14 |
| S6 | `s6_gbdt.py` | GBDT の比較（LightGBM、XGBoost、CatBoost） | #27 #24 | 10/15 |
| S7 | `s7_mlp_classic.py` | MLP と古典モデル（RandomForest、Ridge） | #20 #28 #29 | 10/16 |
| S8 | `s8_external.py` | 外部データ（駅、町丁目の座標） | #21 #22 | 10/17 |
| S9 | `s9_ensemble.py` | アンサンブル（平均、重み付き平均、stacking） | #23 #30 | 10/18〜19 |
| S10 | `s10_final.py` | 仕上げ（再学習、seed 平均、最終2提出） | #25 | 10/20〜21 |

提出は段階ごとに最大1回で、出すかどうかはユーザーが決める。S1 は「もう少し向上させてから」で見送り（2026-10-10）。提出枠は 10/10〜10/21 の12回。

## Deprioritized Issues

## Decisions for the user

- いつ最初の提出をするか（S3 の `output/s3_features/plus_notation/submission.csv` は作成・検査済み、holdout 0.2090）
- #10・#11 を不採用として close するか、#31 を不採用として close するか
- S7: MLP のライブラリ（PyTorch か scikit-learn の MLPRegressor）。#20 で決める
- S8: 使う外部データの出典。#21 #22 で決める
- 各段階のディスカッション投稿の有無
