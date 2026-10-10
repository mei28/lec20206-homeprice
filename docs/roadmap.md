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
| CV of the current baseline | S1 `lgbm_raw` 0.2988（holdout RMSLE、下限は 0.8137） |

## 方針（2026-10-10 にユーザーと決定）

- TA として、学生が目指す目安を作る。atmaCup 初心者講座のように、1段階を1本の marimo ノートブックにし、解説もその中に書く。最終的には上位を狙える水準まで進めるが、順位表には段階ごとに小出しにする。
- 提出は1段階1回。毎回ユーザーの OK を得てから出す（1日1回まで）。
- CV は時系列 holdout（`folds/holdout_2025q1.parquet`、#4）。提出用モデルは全期間で学習し直す。
- ディスカッションには、載せられるときに文章・コード断片・図・CV と LB を載せる。ノートブック全体は締切後に公開する。
- AI の使い方は教材に含めない。

## What drives the score

仮説の段階。根拠は `docs/competition.md` の Prior art と `docs/validation.md`。

- 場所（市区町村、地区、駅）。㎡単価の大半を決める。S2 のカテゴリ特徴量、S4 の TE、S6 の座標。
- 時間トレンドの外挿。検証期間の log1p 価格は学習期間より平均 0.197 高く、木はこれを外挿できない。S1 の予測は holdout で平均 -0.182（約 17% 低め）ずれており、このずれだけを除けば 0.2988 → 0.237 の計算（#6、#16）。S3。
- 価格情報区分の構成のずれ。成約の割合は学習期間 43%、検証期間 62%、test 67%。S2 の区分別の誤差分析。

## Proposed order

日付は1日1段階で割り振った目安。序盤は 10/12 21:35 JST まで、中盤は 10/18 02:23 JST まで。

| 段階 | ノートブック | 内容 | Issues | 提出の目安 |
|---|---|---|---|---|
| S0 | `s0_data.py` | データを知る | #5 | 提出なし |
| S1 | `s1_baseline.py` | ベースライン（log1p、LightGBM、時系列 holdout） | #6 #7 #8 | 10/10〜11 |
| S2 | `s2_features.py` | 前処理と特徴量 | #9 #10 #11 #12 #13 #14 | 10/12 |
| S3 | `s3_trend.py` | 時間トレンド（㎡単価、外挿の補正、重み） | #15 #16 #17 | 10/13 |
| S4 | `s4_te.py` | Target encoding と集約特徴 | #18 #19 | 10/14 |
| S5 | `s5_mlp.py` | MLP | #20 | 10/15〜16 |
| S6 | `s6_external.py` | 外部データ（駅、町丁目の座標） | #21 #22 | 10/17 |
| S7 | `s7_ensemble.py` | アンサンブル | #23 #24 | 10/18〜19 |
| S8 | `s8_final.py` | 仕上げ（再学習、seed 平均、最終2提出） | #25 | 10/20〜21 |

提出枠は 10/10〜10/21 の12回。段階の提出は8回なので、4回の余裕がある。

## Deprioritized Issues

## Decisions for the user

- S1 の提出（`output/s1_baseline/lgbm_raw/submission.csv`、holdout 0.2988）を出してよいか
- S5: MLP のライブラリ（PyTorch か scikit-learn の MLPRegressor）。#20 で決める
- S6: 使う外部データの出典。#21 #22 で決める
- 各段階のディスカッション投稿の有無
