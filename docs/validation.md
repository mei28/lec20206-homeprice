# Validation

Maintained by the `kaggle-validation` skill. A score is trusted only after this file says why.

## How the test set was built

- 時間で分割されている。train は 2019年第1四半期〜2025年第2四半期、test は直後の 2025年第3四半期〜2026年第1四半期（Data タブの説明と、`取引時期` の値域の両方で確認）。
- 行は並べ替えたうえで `id` を振り直している。`id` の順に意味はない。
- `価格情報区分` の構成が時期で変わる。成約価格情報は 2021Q1 から現れ、2025Q1 から件数が増える。成約の割合は train 全体で 45%（2024Q4 までに限ると 43%）、2025Q1〜Q2 で 62%、test で 67%。test の 2026Q1 は取引価格情報が 1,487 件と少なく、公表の遅れと見られる。
- Public と Private の分け方（ランダムか、時期か）は公表されていない。

## Decision flow result

- Strategy: `TimeHoldout`（時系列 holdout。`kgl.cv` に追加した戦略）
- K: 1（検証期間は1つ）
- Stratify / group / time column: time column = `取引時期`、2025年第1四半期以降を検証（fold 0）、それ以前を学習（fold -1）
- Reason: test は train の直後の未来の期間なので、学習期間より後ろを検証に使う。直近2四半期は `価格情報区分` の構成も test に近い。複数 fold の時系列分割より説明が短く、講座の「CV は test の作り方に合わせる」をそのまま示せる。提出用モデルは全期間で学習し直す（2026-10-10 にユーザーが決定）。

## Fold file

- `folds/holdout_2025q1.parquet` (never overwritten; new design = new name)
- Command: `just folds holdout_2025q1 --strategy TimeHoldout --id-col id --time-col 取引時期 --holdout-from 2025年第1四半期 --n-splits 1`

## Fold diagnostics

`kgl.cv.check_folds(df, "fold", time_col="取引時期")` は通過（fold 0 の最初の期間が fold -1 の最後の期間より後）。

| fold | rows | period | mean log1p(price) | std log1p(price) | 成約価格情報の割合 |
|---|---|---|---|---|---|
| -1 (train) | 180,632 | 2019Q1〜2024Q4 | 17.423 | 0.728 | 0.428 |
| 0 (valid) | 21,691 | 2025Q1〜2025Q2 | 17.619 | 0.797 | 0.620 |

検証期間は学習期間より log1p 価格の平均が 0.197 高い（価格水準で約 22% 高い）。時間トレンドを外挿できないモデルは、この差の分だけ低めに予測する。S4（時間トレンド）の論点。

## Leak checklist

S1（`s1_baseline`、2026-10-10）の時点。詳細は `docs/reviews/20261010-s1_baseline-leak.md`（判定 clean）。

| Item | Status | Evidence |
|---|---|---|
| Test information in train (joins, global statistics) | ok | カテゴリの語彙だけ train と test の値から作る（目的変数は使わない）。学習行にないカテゴリは欠損と同じ枝に行くので、語彙を train だけで作っても予測は同じ |
| Statistics fit outside the fold loop (scalers, encoders) | ok | S1 は行単位の変換だけ。S5 の TE と S7 の標準化で再確認する |
| Target encoding outside folds | ok（S1 時点で TE なし） | S5（#18）で、holdout 用は fold -1 だけから作る |
| Feature selection outside folds | ok | 列は手で選ぶ。重要度は表示だけ |
| Time leakage (future rows in features, folds not time ordered) | ok | fold は時間順、シャッフルなし。early stopping を holdout で行うことによる楽観は 0.0005 未満（入れ子で確認） |
| Metric re-implementation matches the competition text | ok | `rmsle` は sklearn の `root_mean_squared_log_error`、手計算のテストあり |
| Submission id order equals the sample | ok | `validate_submission` を書き出し前に呼ぶ |

CV と test の条件の差（リークではないが LB とのずれの要因）:

- 予測先の長さ: holdout は学習終了の 1〜2 四半期先、test は 1〜3 四半期先。
- 成約価格情報の割合: holdout 0.620、test 0.673。
- S1 では `取引時期` をカテゴリで渡したため、未知の四半期が欠損扱いになり、2021年ごろの水準で予測された。最後の既知四半期に置き換えたときの予測の上がり幅は、holdout モデルで +0.142、提出モデルで +0.174（log）。test のほうがずれが大きく、S1 の LB は CV より悪く出る見込み。S3（#12）で数値に直す。

## LB noise

Public LB の行数は非公開。RMSLE の値は 0.2 前後。最初の数回の提出で CV と LB の差を見てから書く。

最初の提出（S3、2026-10-10）では LB が CV より 0.0075 良かった。考えられる理由は2つあるが、1回では切り分けられない。
- test は成約価格情報の割合が高く（67%、holdout は 62%）、holdout では成約の行のほうが誤差が小さかった（S4 で 0.188 と 0.229）
- Public の行が、学習期間に近い 2025年第3四半期に偏っている可能性
価格指数の外挿を入れた S4 以降を出したときに、CV と LB が同じ向きに動くかで確かめる。

## CV vs LB

| exp | run | cv | public lb | gap | note |
|---|---|---|---|---|---|
| s3_features | plus_notation | 0.2090 | 0.20149 | -0.0075 | 最初の提出（2026-10-10）。LB のほうが良い。価格指数の外挿はまだ入っていない段階 |

## Adversarial validation

## Fold version history

| name | strategy | reason for the new version | date |
|---|---|---|---|
| holdout_2025q1 | TimeHoldout (2025Q1〜Q2 を検証) | 最初の設計 | 2026-10-10 |
