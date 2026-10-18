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

S14 以降のハイパーパラメータ探索（2026-10-17 にユーザーと決定、#38）:

- 探索では何百回も設定を比べる。同じ holdout で選ぶと、holdout にたまたま合う設定が選ばれ、CV が楽観的になる。採用のルール（対応のある bootstrap の範囲が 0 を含まない）も当てにならなくなる。
- そこで、もう1段内側の時系列分割で探す。2024Q2 までで学習し、2024Q3〜Q4 で各試行を採点する。2025Q1〜Q2 の holdout は、選んだ設定ごとに1回だけ使う。
- 内側の分割の特徴量は 2024Q4 までの行だけから作る。価格指数と TE は 2024Q2 までの行から作る。目的変数を使わない集約特徴は 2024Q3〜Q4 を含み、それより後を含まない。holdout と同じ扱い。
- 探索した設定の改善が、内側（2024Q3〜Q4）と holdout（2025Q1〜Q2）でどれだけ違うかをモデルごとに記録する。holdout での改善がずっと小さければ、探索が内側に合わせ込んだということ。
- 内側の予測は、S15（#43）で2段目の学習行を増やすのにも使う。

## Fold file

- `folds/holdout_2025q1.parquet` (never overwritten; new design = new name)
- Command: `just folds holdout_2025q1 --strategy TimeHoldout --id-col id --time-col 取引時期 --holdout-from 2025年第1四半期 --n-splits 1`
- `folds/time_2024q3_2025q1.parquet`（S14 以降、#38）: fold -1 は 2019Q1〜2024Q2、fold 0 は 2024Q3〜Q4（探索用）、fold 1 は 2025Q1〜Q2（holdout、`holdout_2025q1` の fold 0 と同じ行）
- Command: `just folds time_2024q3_2025q1 --strategy TimeHoldout --id-col id --time-col 取引時期 --holdout-from 2024年第3四半期 2025年第1四半期 --n-splits 1`

## Fold diagnostics

`kgl.cv.check_folds(df, "fold", time_col="取引時期")` は通過（fold 0 の最初の期間が fold -1 の最後の期間より後）。

| fold | rows | period | mean log1p(price) | std log1p(price) | 成約価格情報の割合 |
|---|---|---|---|---|---|
| -1 (train) | 180,632 | 2019Q1〜2024Q4 | 17.423 | 0.728 | 0.428 |
| 0 (valid) | 21,691 | 2025Q1〜2025Q2 | 17.619 | 0.797 | 0.620 |

検証期間は学習期間より log1p 価格の平均が 0.197 高い（価格水準で約 22% 高い）。時間トレンドを外挿できないモデルは、この差の分だけ低めに予測する。S4（時間トレンド）の論点。

`time_2024q3_2025q1`（S14 以降、#38）。fold 1 の行は `holdout_2025q1` の fold 0 と一致し、fold -1 と 0 を合わせると `holdout_2025q1` の fold -1 と一致する（確認済み）。`check_folds` も通過。

| fold | rows | period | mean log1p(price) | std log1p(price) | 成約価格情報の割合 |
|---|---|---|---|---|---|
| -1 (探索用の学習) | 162,701 | 2019Q1〜2024Q2 | 17.406 | 0.721 | 0.416 |
| 0 (探索の評価) | 17,931 | 2024Q3〜2024Q4 | 17.572 | 0.766 | 0.537 |
| 1 (holdout) | 21,691 | 2025Q1〜2025Q2 | 17.619 | 0.797 | 0.620 |

成約の割合は fold 0 で 0.537 と、学習期間（0.416）と holdout（0.620）の間にある。探索で選ぶ設定は、成約の行が holdout より少ない状態で選ばれる。

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

S13（`s13_tabular_dl`、2026-10-16）で再点検した。詳細は `docs/reviews/20261016-s13_tabular_dl-leak.md`（判定 clean）。

- TabPFN の holdout 用の文脈は 2024Q4 までの直近5万行で、holdout の行も 2025 年の行も含まない。前処理は文脈から作る。
- S12 のスタッキングを作り直すと S12 を再現し、TabPFN を足した改善は -0.0028（-0.0034〜-0.0022）。時間の順向き（Q1 で学習して Q2 を予測）だけでも -0.0029。
- S5 からの非対称（low）: 目的変数を使わない地区・駅の集約特徴と件数は、holdout の行も含む全学習行から作る。test の行は自分の集約に入らない。LightGBM では holdout が 0.0004 楽観的になる。全メンバーに同じようにかかり、メンバーの比較は変わらないので、直さずに記録する。

S14（`s14_tuning`、2026-10-17）で再点検した。詳細は `docs/reviews/20261017-s14_tuning-leak.md`（判定 clean）。

- 探索は 2024Q4 までの行だけで特徴量を作り、2024Q3〜Q4 で採点する。holdout（2025Q1〜Q2）の行は、探索のどの入力にも入らない。holdout を使ったのは、選んだ設定の確認の1回だけ。
- 内側の分割にも、S5 からの集約特徴の非対称が同じ形で入る。LightGBM では内側が 0.0006 楽観的になる（holdout は 0.0004）。内側の基準と全試行に同じようにかかる。
- S14 から XGBoost は GPU で学習する。S6 の設定を GPU で学習すると 0.2110 で、CPU（0.2103）より少し悪い。比べ方は探索した側に不利なので、探索の改善 -0.0106 は控えめな値（GPU どうしで比べると -0.0112）。
- LightGBM の探索は、途中で止めて再開した。止めたときの試行が RUNNING のまま残り、Optuna 5.0 の TPE の既定（`constant_liar=True`）でその後の提案に影響した。holdout の情報は入らないが、空の DB から同じ順番にはならない。S16 では `constant_liar=False` にした。

S17（`s17_tabpfn_context`、2026-10-18）で再点検した。詳細は `docs/reviews/20261018-s17_tabpfn_context-leak.md`（判定 clean）。

- TabPFN の文脈（直近12万行、市区町村ごとの全期間）は、どれも学習側の行だけから選ぶ。評価する行との重なりは 0。文脈の最新の期間は、内側 2024Q2、holdout 2024Q4、test 2025Q2。
- 改善の幅は内側と holdout でほぼ同じ（12万行の文脈の単独で -0.0060 と -0.0062）。
- 内側のスタッキングは S15 の 2024Q3〜Q4 の予測で組む。そのメンバーは S14 で探索した設定で、設定の選択と early stopping に 2024Q3〜Q4 を使っている。holdout と test で組むのは S13 のメンバー。内側では TabPFN 以外の列が楽観的で、組も違う。2本目の TabPFN を足す差（-0.00027）は、この楽観で入れ替わりうる大きさ。後の段階でも block 予測で選ぶなら、この食い違いを本文に書く。
- 追加の採点のコードを、規則の文どおりに直した（ccfae7f）。直したのは、直す前の版の holdout（0.18751、追加なし）を記録した後だった。採否は直した後の結果（0.18697）で決め、直す前と後を holdout で比べて選んではいない。どちらも S13 より良い（-0.0035、-0.0041）。
- `validate_submission` は NaN と無限大を通していた。S17 から弾くようにした。

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

S4（2026-10-11）では CV と LB が同じ向きに動いた（CV -0.0040、LB -0.0024）。LB の改善が小さい理由はまだ切り分けられない。S4 の改善には実質化して学習する効果と test 期間への外挿の効果が混ざっていて、提出1回ではどちらが LB に効いたか分からない。

S5（2026-10-14）は CV -0.0015 に対して LB -0.00016 で、LB はほとんど動かなかった。LB と CV の差は 0.0075、0.0059、0.0046 と縮んでいる。CV の改善のうち LB に出る割合は、0.001 台の改善では小さいと見ておく。

S6（2026-10-15）は CV -0.0045 に対して LB -0.0025。S3 から S6 までで CV は 0.0100、LB は 0.0051 良くなり、LB は CV のおよそ半分しか動いていない。向きは4回とも一致しているので、CV で選ぶ方針は変えない。半分になる理由（test が holdout より先の期間であること、Public の行の偏り）は、Public の結果だけでは切り分けられない。

S9（2026-10-16）は CV -0.0039 に対して LB -0.0042 で、初めて LB の改善が CV を上回った。単独モデルの改善（S4〜S6）は LB で半分ほどだったのに対し、アンサンブルの改善はそのまま LB に出た。5回分で直線を引き直すと LB ≈ 0.068 + 0.64 × CV。

S12（2026-10-17）は CV -0.0013 に対して LB -0.0008 で、LB は CV の 0.63 倍動いた。スタッキングの改善でも、単独モデルの段階と同じくらいの割合に戻った。6回分で直線を引き直すと LB ≈ 0.061 + 0.67 × CV で、直線からの外れの標準偏差は 0.0008。S13 の `stacker_average_with_tabpfn`（CV 0.1910）は、直線では 0.1899、S12 の実績からの差では 0.1895 の見込み。

S13（2026-10-18）は CV -0.0028 に対して LB -0.0029 で、LB は CV の 1.04 倍動いた。LB 0.18848 は、6点の直線の見込み 0.1899 より 0.0015 良い。外れの標準偏差 0.0008 のおよそ2倍。改善がそのまま LB に出たのは、S9（6モデルのアンサンブル）と S13（TabPFN を足した）の2回で、どちらも性質の違うメンバーを足した段階だった。特徴量・単独モデル・スタッカーの改善（S4〜S6、S12）は半分ほどしか出ていない。TabPFN は直近 5 万行を文脈にして予測するので、test の時期へのずれに強い可能性があるが、1回では確かめられない。7点で直線を引き直すと LB ≈ 0.052 + 0.72 × CV で、外れの標準偏差は 0.0008。

CV と LB の関係は、test の予測の後処理には使わない（2026-10-17 にユーザーと決定）。

- 直線は提出ごとのスコアどうしの関係で、どの行をどちら向きに直せばよいかの情報を持たない。
- LB の値から全体のずらし幅を決める方法（LB probing）は、Public には効く。予測を log で c だけずらすと RMSLE² が「元の値 + 2c × 平均誤差 + c²」になるので、2回の提出から Public の行の平均誤差を逆算できる。しかし、test の正解の情報を LB から取り出す方法で、教材にそぐわない。Public と Private の分け方も分からないので、Private に合う保証もない。
- 訓練データだけで決める後処理にも、ほぼ余地がない。holdout では、予測が学習期間の終わりから離れるほど低めにずれる（CatBoost は1四半期先 -0.0036、2四半期先 -0.0086、TabPFN は -0.0020、-0.0064）。しかし、スタッキングが 1〜2 四半期先の平均のずれを学ぶので、その分は直っている（S13 の holdout 全体で +0.0002）。test で残るのは3四半期先の 2026Q1 の -0.0075 ほどで、直しても RMSLE は 0.0001 も変わらない。

## CV vs LB

| exp | run | cv | public lb | gap | note |
|---|---|---|---|---|---|
| s3_features | plus_notation | 0.2090 | 0.20149 | -0.0075 | 最初の提出（2026-10-10）。LB のほうが良い。価格指数の外挿はまだ入っていない段階 |
| s4_trend | hedonic_linear | 0.2050 | 0.19911 | -0.0059 | 2回目（2026-10-11）。ヘドニック価格指数で実質化し外挿。CV は 0.0040、LB は 0.0024 良くなり、向きは一致 |
| s5_te | all_three | 0.2035 | 0.19895 | -0.0046 | 3回目（2026-10-14）。Target encoding・集約特徴・件数。CV は 0.0015 良くなったが、LB は 0.00016 しか動かなかった |
| s6_gbdt | catboost | 0.1990 | 0.19642 | -0.0026 | 4回目（2026-10-15）。CatBoost に切り替え。CV は 0.0045、LB は 0.0025 良くなった |
| s9_ensemble | stacking | 0.1951 | 0.19222 | -0.0029 | 5回目（2026-10-16）。6モデルの Ridge スタッキング。CV は 0.0039、LB は 0.0042 良くなり、初めて LB のほうが大きく動いた |
| s12_black_magic | stacker_average | 0.1938 | 0.19139 | -0.0024 | 6回目（2026-10-17）。4種類のスタッカー（Ridge、種類別 Ridge、MLP、LightGBM）の平均。CV は 0.0013、LB は 0.0008 良くなった |
| s13_tabular_dl | stacker_average_with_tabpfn | 0.1910 | 0.18848 | -0.0025 | 7回目（2026-10-18）。S12 のスタッキングの平均に TabPFN-3.5 を足した。CV は 0.0028、LB は 0.0029 良くなり、直線の見込みより 0.0015 良い |

## Adversarial validation

## Fold version history

| name | strategy | reason for the new version | date |
|---|---|---|---|
| holdout_2025q1 | TimeHoldout (2025Q1〜Q2 を検証) | 最初の設計 | 2026-10-10 |
| time_2024q3_2025q1 | TimeHoldout（境目2つ。2024Q3〜Q4 を探索用、2025Q1〜Q2 を holdout） | Optuna の探索で holdout を使い回さないため（#38） | 2026-10-17 |
