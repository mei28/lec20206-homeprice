# Leak review: s17_tabpfn_context (tabpfn_context_large, tabpfn_by_municipality, stacker_average_s17)

Reviewer: leak-reviewer instructions, run as a subagent on 2026-10-18 against HEAD ccfae7f. The working tree had no changes under `notebooks/`.
The notebook was not run, and nothing was written under `output/`. The rerun started at 13:02:17 was still going during the review. Its `test_by_municipality.parquet` and the new `stacker_average_s17` did not exist yet.
Numbers come from the saved files in `output/s17_tabpfn_context/` and `output/s15_more_stack_rows/`. These were recomputed in memory on the CPU: the contexts, the offsets, the inner-split selection, and the holdout side of the restack.

Verdict: clean

S17 で新しく入った部分に、holdout（fold 1、2025Q1〜Q2）の正解も test 期間の情報も入っていない。

- 文脈はどれも、渡された学習側の行（`tr_in`、`tr_idx`、全 train）から選んでいる。重なりを再計算すると、評価する行との重なりは 0。最新の期間は、内側 2024Q2、holdout 2024Q4、test 2025Q2。
- `by_municipality` の町ごとの文脈も、同じ学習側の行の部分集合になっている。50,000 行の上限に届く町はなかった。
- 置き換えと追加は、2024Q3〜Q4 の数字だけで決めている。
- S15 の block 予測は、2024Q2 までの行だけで学習している。Ridge は学習し直すと完全に一致した。

改善の幅は、内側と holdout でほぼ同じ。これもリークがない裏付けになる。

- `context_large` 単独では、内側 −0.0060、holdout −0.0062。
- `by_municipality` 単独では、どちらも +0.0006。

重大度が low の点が5つある。いちばん大きいのは 9a。

- 9a: 追加の規則のコードを直したのは、直す前の版の holdout（0.18751）が記録された後だった。
- 規則の文は最初の commit からあり、直した結果も内側の数字で決まる。それでも、直した結果、holdout は 0.18697（−0.00054）に良くなった。
- holdout は、直したかどうかの分かれ道にだけ関わった可能性がある。直す前と後のどちらを採るかを holdout で選ばなければ、楽観はこの1回の分岐に限られる。

| # | Item | Severity | Location | Evidence |
|---|---|---|---|---|
| 1a | 文脈に holdout・test の行が入るか（`context_large`、`s13`、`estimators_16`） | none | `tabpfn_predict`（L241-245: `everything = np.arange(len(X_context))`、`context = incontext.most_recent(everything, periods, variant["context_rows"])`）。呼び出しは L283 `fs_in.X.iloc[tr_in]`、L420 `fs.X.iloc[tr_idx], y_real[tr_idx]`、L426 `fs.X_refit, y_real_refit` | 文脈の位置は `X_context` の中の位置で、`X_apply` の行は入りえない。再計算した文脈（いずれも 120,000 行）は次のとおり。境目の四半期は一部だけ入る。行の順に意味はない（validation.md L8）。<br>・内側: 2021Q2（9,708 行中 9,580 行）〜2024Q2。fold 0 との重なり 0<br>・holdout: 2021Q3（8,790 行中 439 行）〜2024Q4。`va_idx` との重なり 0<br>・test: 2022Q2（9,129 行中 6,787 行）〜2025Q2。train の行だけ<br>文脈の目的変数は `y_real = fs.y_log - fs.offset` で、`offset` は `tr_idx` だけで作った指数 |
| 1b | `by_municipality` の町ごとの文脈 | none | L246-252 `pool = everything[town_context == town]`、`incontext.most_recent(pool, periods, variant["context_rows"])` | どの設定でも町は 50 で、上限 50,000 行に届く町はない。そのため、文脈は「その町の 2019Q1 から境目までの全行」になる。<br>・文脈の行数（最小〜最大）: 内側 5〜11,662、holdout 6〜12,894、test 9〜14,453<br>・最新の期間: 2024.25、2024.75、2025.25<br>・評価する行との重なり: 0<br>test の町はすべて train にある（`assert len(pool) > 0` は通る）。キーの `市区町村名` は元の列で、目的変数は使わない |
| 1c | 外部データ、2025Q3 以降のデータ（コンペの規則） | none | L18 の import、`~/.cache/tabpfn/` | S17 は `kgl.external` も取得系のコードも読み込まない。`feature_set.build` が読むのは train と test だけ。<br>tabpfn は 9.2.0 で、S13 のレビューと同じ。キャッシュにある重みは `tabpfn-v3.5-20260909.safetensors` の1つだけで、日付は 2026-10-16 09:23（S17 より前）。`fit_mode`、`memory_saving_mode`、`ignore_pretraining_limits`、`n_estimators` を変えても、重みは変わらない。<br>パッケージで通信するコードは `browser_auth.py` と `model_loading.py`（ライセンスと重みの取得）だけ |
| 2a | `low_memory` での TabPFN の前処理 | none | `.venv/.../tabpfn/inference.py` `InferenceEngineOnDemand.iter_outputs` | `fit_transform_ensemble_members_iterator(X_train=self.X_train, y_train=self.y_train, ...)` で文脈から作り、予測先には `em.transform_X_test(X)` を当てるだけ。<br>`fit_predict` は呼ぶたびに `TabPFNRegressor` を新しく作る。町ごとも同じ |
| 2b | 目的変数を使わない集約と件数（S13 の 2a を引き継ぐ） | low | `feature_set.py` L78 `_target_free(train_df, train_df)` | holdout の設定でも、2025Q1〜Q2 の行が地区・駅の集約と件数に入る。test の行は入らない。<br>S5 からの持ち越しで、全 variant と全メンバーに同じようにかかる。TabPFN では測り直していない（GPU は再実行で使用中だった） |
| 3 | TE と価格指数の offset（holdout と test） | none | `feature_set.py` L82-91、L95-102。notebook L284 `+ fs_in.offset[va_in]`、L421 `+ fs.offset[va_idx]`、L427 `+ fs.offset_test` | 再計算した offset は次のとおり。<br>・内側: 0.3203 と 0.3366（`tr_in` の指数を外挿）<br>・holdout: 0.3605 と 0.3777（`tr_idx` の指数を外挿）<br>・test: 0.4317、0.4543、0.4768（全行の指数を外挿）<br>holdout の offset に 2025 年の価格は入らない。全行の指数での 2025 年の値（0.3908 と 0.4159）とは違う。TE は、学習行が5分割の OOF、予測先は学習行から当てる。手順は S13 と同じ |
| 4a | variant の選択に holdout を使ったか | none | L351-372 `replacement`、`addition` | どちらも `inner_table` と `addition_table` だけ（`y_block` で採点）から求める。holdout のセル（L387 以降）は、決まった値を受け取るだけ。<br>選択の経緯は次のとおり。最初の holdout ファイルは 12:06 で、どちらもそれより前。<br>・`context_all` を足した（97b8c1b、11:29）。理由は内側の結果 0.1830 対 0.1890<br>・GPU のメモリ不足で外した（27159d2、11:33）<br>120,000 行は b7c1082（10:36）から変わっていない。L101-107 の試行は「2024Q3-Q4 の行」と書かれているが、そのスクリプトは repo にない |
| 5a | 文脈の期間の対称性 | none | 1a の再計算 | holdout の文脈は 2021Q3〜2024Q4（約13四半期）で、1〜2四半期先を予測する。test の文脈は 2022Q2〜2025Q2（約12.7四半期）で、1〜3四半期先を予測する。違いは S13 の 5a と同じ種類 |
| 5b | TabPFN どうしの水準の関係が holdout と test で違う | low（リークではない。CV と test の差） | `holdout_context_large.parquet`、`test_context_large.parquet` | log での平均の差（`context_large` − S13 の TabPFN）:<br>・holdout: Q1 −0.0042、Q2 −0.0007<br>・test: 2025Q3 +0.0061、2025Q4 +0.0066、2026Q1 +0.0031<br>スタッカーは、メンバーの水準を holdout で学ぶ。holdout の平均の誤差は、`context_large` −0.0062 と −0.0071、S13 −0.0020 と −0.0064。四半期ごとの平均のずれを除いても改善は残る（0.18853 対 0.19477） |
| 5c | 改善が内側と holdout で同じか（リークの兆候の確認） | none | 再計算 | `context_large` 単独:<br>・内側 0.18297 対 0.18901<br>・holdout 0.18864 対 0.19483。差 −0.0062（−0.0070〜−0.0053）<br>・holdout の内訳はどこでも良くなる: Q1 0.18785 対 0.1945、Q2 0.18945 対 0.19517、成約 0.1726 対 0.17979、取引 0.21222 対 0.21713<br>`by_municipality` 単独:<br>・holdout 0.19542。S13 との差 +0.0006（−0.0011〜+0.0023）。内側も +0.0006<br>スタッキングの時間の順向き（Q2 の行）: 直す前 0.18809、直した後 0.18760、S13 0.19138 |
| 5d | S15 の block 予測（内側のスタッキング） | none（リーク）／ low（内側の楽観） | `s15_more_stack_rows.py` L163-168、L276-314。notebook L274-289 | parquet の id は fold 0 の行と同じ順で、`fold` の値は 0 だけ。<br>S15 は 58422cd から実行した（`run_started_from.txt`）。S15 の notebook と `feature_set`、`incontext`、`stacking`、`ensemble`、`homeprice`、`price_index` は、それ以降変わっていない（`git diff` は空）。<br>全メンバーは `tr_in` で学習する。TabPFN は `most_recent(tr_in, ...)`。Ridge を学習し直すと、block の列との差の最大は 0.0。`fs_in` は fold ≤ 0 だけから作るので、2025 年の行はどこにも入らない。<br>ただし、GBDT は `va_in`（fold 0）で early stopping している。S14 は、GBDT と MLP の設定も fold 0 で選んだ。S17 が採点する行と同じなので、内側では TabPFN 以外の列が楽観的になる。全 variant に同じようにかかる |
| 5e | 内側と holdout でメンバーの組が違う | low（リークではない） | 本文 L266、L381。`s13_members` L461-469 | 内側のスタッキングは、S14 で探索したメンバー（S15 の block）で組む。holdout と test は S13 のメンバー（S10、S6、S7）で組む。<br>「追加」は、実際に入るのとは違う組の上で決めている。追加の差は 0.00027 しかなく、5d の楽観で入れ替わりうる大きさ |
| 6 | Metric | none | `holdout_rmsle`、`src/kgl/metrics.py` | sklearn の `root_mean_squared_log_error`（competition.md の定義と同じ）。再計算した `tabpfn_context_large` 0.18864 と、直す前の `stacker_average_s17` 0.18751 は、`metrics.json` と一致する |
| 7 | 提出ファイル | none ／ low | L446-448、L536-538 | 書き出しの前に `validate_submission` を呼ぶ。<br>直す前のスタッキングの `submission.csv`: 29,341 行、`[id, price]`、Float64、有限、id の順は sample と一致。`test_context_large`: id 一致、有限、2.11e6〜2.52e9 円。<br>low: `validate_submission` は NaN と inf を弾かない（polars の NaN は null ではない）。いまは手で確かめている |
| 8 | Fold ファイル | none | `folds/time_2024q3_2025q1.json`、`cv.attach_folds` | `TimeHoldout`、境目は [2024年第3四半期, 2025年第1四半期]、202,323 行。validation.md の L31-32 と L45-51 と一致する。fold のない行があると `attach_folds` が例外を出す。<br>再計算: `tr_idx` 180,632、`va_idx` 21,691、`tr_in` 162,701、`va_in` 17,931。`tests/test_cv.py` と `tests/test_incontext.py` は 21 件とも通過 |
| 9a | holdout を見た後の、規則のコードの修正 | low | ccfae7f（13:02:16）。`stacker_average_s17/metrics.json`（`git_commit` 27159d2、12:51:31） | 時系列:<br>1. 27159d2 で、置き換え `context_large`、追加 None、holdout 0.18751 を記録した<br>2. 11分後、ccfae7f で追加の採点を「置き換えた上に足す」に直した<br>規則の文 L345「追加: 置き換えたあとに、さらにもう1本の TabPFN を足したほうが良ければ、それも足す。」は b7c1082 からある。古いコードは「S13 の TabPFN に足した場合」と「置き換えた場合」を比べていて、文と食い違っていた。<br>内側を再計算した結果（commit メッセージと一致）:<br>・S13 の上に足す: `by_municipality` 0.18373 で、0.18241 より悪い → None<br>・`context_large` の上に足す: `by_municipality` 0.18214、`estimators_16` 0.18241 → `by_municipality` を足す<br>holdout 側を再計算した結果（直す前は保存済みの oof と差 0.0 で一致）:<br>・直した後 0.18697、直す前との差 −0.00054（−0.00085〜−0.00023）<br>・S13 との差は、直す前 −0.0035（−0.0041〜−0.0029）、直した後 −0.0041（−0.0047〜−0.0034）<br>再実行で `stacker_average_s17/` は上書きされ、直す前の記録は残らない |
| 9b | 予測のキャッシュは名前だけで引く | low | L281-287、L418-430 | `inner_/holdout_/test_{name}.parquet` は、ファイルがあれば読み直す。id の順は確かめるが、variant の設定は確かめない。<br>b7c1082 から HEAD まで、`estimators_16`、`context_large`、`by_municipality` の設定は変わっていない（変わったのは `context_all` の追加と削除だけ）。`tabpfn_predict` も同じなので、いまのキャッシュは今の設定と一致する |

## Recommended fixes

1. 9a: S17 の本文と #45 に、次を書き残す。
   - 直す前の結果（27159d2、追加なし、holdout 0.18751）。
   - コードを直したのは、その holdout を記録した後だったこと。

   採否は規則どおり直した後の結果で決め、直す前と後を holdout で比べて選ばない。どちらも S13 より良く（−0.0035、−0.0041）、範囲は 0 を含まない。そのため、S17 を採るかどうかの判断はこの分岐に左右されない。直す前の数字は、上書きされる前に Issue のコメントに残す。
2. 5d・5e: `docs/validation.md` の S17 の項に書く。
   - S17 の内側の比較は、fold 0 で設定を選び early stopping した S14 のメンバーで組んでいる。
   - 実際に入るのは S13 の組。
   - 追加の差 0.00027 は、この楽観の範囲に入る。

   後の段階でも block 予測で選ぶなら、S13 のメンバーの block 予測を作るか、この食い違いを本文に書く。
3. 9b: キャッシュの parquet に variant の設定（またはその hash）を一緒に保存し、読むときに一致を assert する。
4. 7: `kgl.io.validate_submission` に、予測の列が有限であることの確認を足す。`src/kgl` なので、先に失敗するテストを書く。
5. 5b: 結果の節に、`context_large` と S13 の TabPFN の水準の差を書く。holdout は −0.004〜−0.001、test は +0.003〜+0.007 で、向きが逆になる。
6. 再実行が終わったら、`test_by_municipality.parquet` と新しい `submission.csv` を確かめる（id の順、有限、行数）。今回のレビューの時点では、まだなかった。
7. `docs/validation.md` の Leak checklist に S17 の項を足す。

## 対応（2026-10-18）

- 1: #45 の判断のコメントと S17 の「結果を読む」に、直す前の結果と経緯を書いた。直す前の数字は `just record` のコメントにも残っている。採否は直した後の結果で決めた。
- 2、7: `docs/validation.md` の Leak checklist に S17 の項を足した。
- 3: 見送り。S17 のキャッシュは今の設定と一致していて、S17 のあとに variant の設定を変える予定はない。
- 4: `validate_submission` が NaN と無限大を弾くようにした（失敗するテストを先に書いた）。
- 5: S17 の「結果を読む」に書いた。
- 6: 再実行のあと確かめた。`stacker_average_s17/submission.csv` は 29,341 行、`[id, price]`、id の順は sample と一致、すべて有限で 2.25e6〜2.51e9 円。holdout は 0.18697 で、レビューの再計算と一致する。
