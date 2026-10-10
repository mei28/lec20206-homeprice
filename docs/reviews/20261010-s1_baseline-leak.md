# Leak review: s1_baseline (lgbm_raw)

Reviewer: leak-reviewer instructions (`.claude/agents/leak-reviewer.md`), run as a general-purpose agent on 2026-10-10 against commit 60bf453.
The finding 5a was reproduced by hand before acting on it (holdout RMSLE 0.2988 → 0.2108 with `取引時期` set to 2024年第4四半期; 0.2988 with it set to missing).

Verdict: clean

学習と検証の間にも、train と test の間にもリークは見つからなかった。CV 0.2988 はリークで良く見えている数字ではない。一方で、`取引時期` をカテゴリとして渡したため、未来の四半期が欠損扱いになっている（5a、medium）。これが CV と test の差を生み、LB は CV より悪く出る見込みが高い。

数値は、notebook と同じ設定でメモリ上で再学習して測った。再学習の結果は記録と一致した。best_iteration は 1,276、holdout は 0.29880、submission.csv との差は log で 0.0。

| # | Item | Severity | Location | Evidence |
|---|---|---|---|---|
| 1 | Test information in train | none | `vocab`, `X_test`, submission ids | 語彙は train と test の値から作るが、目的変数は使わない。LightGBM のビンは学習行だけから作られ、学習行にないカテゴリは予測時に欠損と同じ枝へ行く。holdout の `取引時期` を null にした予測は元と完全に一致した。train と test の結合や全体統計はない。 |
| 2a | Statistics fit outside the fold loop | none | `floor_pred`, `to_features` | 中央値は学習 fold だけから計算。数値列は行単位の変換で、scaler、imputer、encoder の fit はない。 |
| 2b | Early stopping on the holdout | low | `lgb.train(..., valid_sets=[valid_set])` | 2,500 本まで測ると最小は 1,276 本（0.29880）で early stopping と同じ。1,000〜1,600 本で 0.29880〜0.29920。入れ子（Q1 で本数を選び Q2 で測る）でも最良本数は 1,276 本。楽観の幅は 0.0005 未満。 |
| 3 | Target encoding outside folds | none | `to_features` | 目的変数から作った特徴量はない。 |
| 4 | Feature selection outside folds | none | `NUMERIC`, `CATEGORICAL` | 列は手で選んでいる。重要度は表示にだけ使う。 |
| 5a | `取引時期` as a categorical (CV vs test gap) | medium | `CATEGORICAL` | 学習にない四半期を持つ行は holdout でも test でも 100%。LightGBM はこれを欠損と同じ枝に送り、予測は 2021Q4〜2022Q1 相当の水準になる。holdout の `取引時期` を 2024Q4 に置き換えると RMSLE は 0.2988 → 0.2108、平均誤差は −0.182 → −0.041。最後の既知四半期に置き換えたときの予測の上がり幅は、holdout モデルで +0.142、提出モデル（test、2025Q2 に置換）で +0.174。リークではない。 |
| 5b | Time ordering and horizon | low | `src/kgl/cv.py`, `cv.split(folds, 0, time=True)` | シャッフルなし。fold −1 は 2019Q1〜2024Q4、fold 0 は 2025Q1〜Q2。特徴量は行単位の変換だけ。残る差: 予測先は holdout が 1〜2 四半期先、test が 1〜3 四半期先。成約価格情報の割合は holdout 0.620、test 0.673。 |
| 5c | Refit on all periods with scaled rounds | low | `refit_rounds` | 1,276 × 202,323 / 180,632 = 1,429。holdout の曲線では 1,429 本で 0.29901（最良との差 0.0002）。提出モデルは CV では評価できない。 |
| 6 | Metric | none | `src/kgl/metrics.py` | `root_mean_squared_log_error` は competition.md の定義と一致。負の予測で ValueError。oof.parquet から再計算した RMSLE は metrics.json と一致。 |
| 7a | Submission integrity | low | `validate_submission` | 書き出し前に行数、列、id 順、欠損を確認。29,341 行、`[id, price]`、id 順は sample と一致、欠損と非有限値なし。price は 1.91e6〜1.63e9 円。price は Float64、sample は Int64 だが、回帰の提出なので問題ない。 |
| 7b | OOF artifact | low | `holdout_ids` | oof.parquet の `取引価格（総額）` が String 型だった。 |
| 8 | Fold file integrity | none | `folds/holdout_2025q1.json` | sidecar は validation.md と一致。id は一意で fold ファイルと一致。件数は −1 が 180,632、0 が 21,691。 |

## Recommended fixes

1. 5a: `取引時期` を順序のある数値で渡す（S2、#12）。数値にしても木の予測は最後の既知期間の水準で頭打ちになり、残る −0.04 前後のずれは S3 の課題として残る。
2. 5a: S1 notebook の「ずれは価格の上昇が原因」という説明を直す。
3. 7b: 目的変数を Float64 にキャストしてから oof.parquet に書く。
4. 5b、5c: `docs/validation.md` の Leak checklist を埋め、予測先の長さの差、成約割合の差、5a の上がり幅を記録する。最初の LB で CV との差を確かめる。
5. 2b: 早期終了の説明に、測った楽観の幅を添える。

## Actions taken (2026-10-10)

- 2, 3, 5: S1 notebook を修正した。原因を確かめるセル（取引時期を欠損・2024年第4四半期に置き換えた予測）を追加し、説明を直した。S1 のモデル自体は「最小限の変換」の基準として変えない。
- 4: `docs/validation.md` の Leak checklist に反映した。
- 1: S2 の #12 で扱う。
