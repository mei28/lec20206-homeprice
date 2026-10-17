# Leak review: s14_tuning (xgboost_tuned, mlp_tuned, lightgbm_tuned, catboost_tuned, stacker_average_tuned)

Reviewer: leak-reviewer instructions, run as a subagent on 2026-10-17 against commit eba00f8, the commit recorded in every S14 `metrics.json`. HEAD was bc46961, and `notebooks/s14_tuning.py` had not changed since eba00f8.
The notebook was not re-run for the review. Most numbers come from `output/s14_tuning/*/oof.parquet`, `submission.csv`, and `optuna.db`. The following were recomputed in memory: the restack, the member breakdowns, the inner LightGBM baseline, an ablation of the aggregate features on the inner split, S6's XGBoost on GPU and CPU, and the tuned XGBoost on GPU.

Verdict: clean

探索にも、holdout での確認にも、スタッキングのやり直しにも、holdout（fold 1、2025Q1〜Q2）の正解や test 期間の情報は入っていない。

- 内側の設定は、fold ≤ 0 の 180,632 行だけで作っている。fold 1 の行は、探索のどの入力にも入らない。
- 内側の価格指数と TE は fold -1 だけから作る。fold 0 には、外挿した指数と fold -1 から作った TE を当てる。holdout の設定と同じ手順である。
- 試行の採点、early stopping、設定の選択は、すべて fold 0 で行う。S14 で holdout を評価したのは 17:28〜18:18 の1回だけ。
- 最後のメンバーは、S6・S7・S10 と同じ手順で作る。学習率、seed、holdout での early stopping、全行で学習し直すときの本数の規則が同じ。

保存済みの数字は、すべて再現した。

- 5つの run の oof から計算した RMSLE は、`metrics.json` と一致する。
- S13 のメンバーで組み直すと、S13 の保存済みの平均と差 0.0 で一致する。探索したメンバーに入れ替えると、`stacker_average_tuned` と差 0.0 で一致する。
- 内側の LightGBM の基準（0.199041）が一致する。探索した XGBoost の holdout（0.199754、4,049 本）も、GPU で学習し直すと完全に一致する。

XGBoost の改善 -0.0106 は、計算機器の違いから来ていない。S6 の設定を GPU で学習すると 0.2110 で、CPU（0.2103。S6 を完全に再現）より 0.0007 悪い。機器の変更は、探索した側に不利に働いている。

低い重大度の点が2つある。

- 2b: S13 の 2a（目的変数を使わない集約特徴の非対称）は、内側の分割にも同じ形で入っている。LightGBM で測ると、内側は 0.0006 楽観的になる。holdout の 0.0004 と同程度で、内側と holdout は対称のまま。
- 8b: 再開の前に止まった LightGBM の試行 64 が、RUNNING のまま DB に残っている。Optuna 5.0 の TPE は既定でこの試行を「悪い側」に数えるので、探索の道筋が変わった。holdout の情報は入っていない。ただし、空の DB から実行し直しても同じ道筋にはならない。

依頼文にある内側の改善幅のうち2つは、`metrics.json` と合わない。CatBoost は -0.0026 ではなく -0.0042、MLP は -0.0033 ではなく -0.0036。

| # | Item | Severity | Location | Evidence |
|---|---|---|---|---|
| 1a | Holdout rows in the inner setting's features | none | `s14_tuning.py` の `inner_raw = train_folded.filter(pl.col("fold") <= 0)`、`fs_in = feature_set.build(inner_raw, test_raw, tr_in, va_in, **_build)` | 再計算: 180,632 行。`tr_in` は 162,701 行（最大 2024.25）、`va_in` は 17,931 行（2024.5 と 2024.75）。`count_地区名` は `inner_raw` の件数と一致する。探索の関数（`fit_*_inner`、`inner_score`）が読むのは `fs_in`、`y_in`、`mlp_frame_in`、`tr_in`、`va_in` だけで、`fs` や `va_idx` を参照しない。`fs_in.X_refit` と `fs_in.X_test` は使わない。語彙は `inner_raw` と test の値から作る（目的変数は使わない。S1 で認めた扱い）。 |
| 1b | Trial scoring, early stopping, choice of settings | none | `inner_score`、`fit_*_inner`、`chosen`、`optuna.db` | `inner_score` は `fs_in.offset[va_in]` と `fs_in.y_log[va_in]` で採点する。early stopping も `va_in` で行う。`chosen` は `studies[name].best_trial.user_attrs["params"]` で、内側の値が最小の試行。DB に入っているのは内側の値だけ。回数は1回の時間を測って決めた。0956940 → eba00f8 の変更（XGBoost の `device` と、回数 200→400）は 10:22 に入った。この時点で走っていたのは LightGBM の試行 0〜63 だけ。最初の実行は LightGBM の探索中に止まり（exit 143）、holdout のセルまで進んでいない。 |
| 2a | Price index and TE symmetry | none | `feature_set.build`（`hedonic_index(t_train[tr_idx], ...)`、`_target_encodings(train_df[tr_idx], ...)`） | 内側の指数は `tr_in`（fold -1）だけで推定し、fold 0 には直近8四半期の直線で外挿する（offset 0.3203 と 0.3366）。TE は fold -1 の5分割 OOF で作り、fold 0 には fold -1 から当てる。`hedonic_controls` の中央値も `tr_in` から取る。holdout の設定も同じ手順で、fold -1 と 0 から作って fold 1 に当てる。 |
| 2b | Target-free aggregates on the inner split (S13 2a) | low | `_target_free(train_df, train_df)` | 内側でも `inner_raw` 全体（fold -1 と 0）から作る。そのため、評価する fold 0 の行が地区・駅の集約と件数に入る。fold 1 の行は入らない。holdout の設定と同じ形である。LightGBM（いまの設定、学習率 0.1）では 0.199041 で、集約を fold -1 だけで作ると 0.199659（+0.0006）になる。holdout の +0.0004 と同じ向きで、同じくらいの大きさ。基準も全試行も同じ条件で作る。S5 からの持ち越しで、S14 で新しく入ったものではない。 |
| 3a | Final members vs S13 members | none | `final_lightgbm`、`final_catboost`、`final_xgboost`、`final_mlp` と、S6・S7・S10 の notebook | LightGBM: 学習率 0.05、seed (42, 7, 2026)、holdout で early stopping 100。全行での学習は本数を 202,323/180,632 倍にする。ここまで S10 と同じ。上限は 10,000（S10 は 5,000）だが、どちらも届かない（S14 は 607〜764 本、S5 は 1,423 本）。CatBoost: 学習率 0.08、上限 12,000 は S10 と同じで、本数は 8,256 と 4,796。MLP: `max_iter` 60。early stopping には学習行からランダムに選んだ 10% を使い、holdout は使わない。全行での学習し直しも S7 と同じ。S14 だけ `threadpool_limits` で BLAS のスレッド数を固定しているが、情報の差ではない。XGBoost: S6 は CPU、S14 は GPU。S6 の設定を GPU で学習すると 0.21098（1,035 本）、CPU では 0.21033（1,156 本で S6 を完全に再現）。同じ GPU どうしで比べると、探索の効果は -0.0112 になる。holdout での early stopping による楽観（validation.md では 0.0005 未満）は、両方のモデルに同じくかかる。 |
| 3b | Inner gain vs holdout gain | none | `metrics.json` の `inner_current` と `inner_best` | 内側の改善: LightGBM -0.0028、CatBoost -0.0042、XGBoost -0.0138、MLP -0.0036。holdout での改善: 0.0000（-0.0011〜+0.0011）、+0.0004（-0.0004〜+0.0012）、-0.0106（-0.0127〜-0.0085）、-0.0051（-0.0066〜-0.0036）。XGBoost と MLP は、どの切り口でも良くなる。XGBoost は Q1 -0.0102、Q2 -0.0110、成約 -0.0111、取引 -0.0099。四半期ごとの平均のずれを除いても、XGBoost は 0.1996 と 0.2103、MLP は 0.2110 と 0.2151。MLP の改善のうち 0.0011 は、予測が低めにずれていたのが減った分（-0.0265 → -0.0144）。LightGBM と CatBoost は、Q1 で良く（-0.0009、-0.0006）、Q2 で悪い（+0.0009、+0.0014）。CatBoost の Q2 は範囲 +0.0003〜+0.0024 で 0 を含まない。holdout が、内側での選択の楽観を検出している。リークの兆候ではない。 |
| 4a | Restack recipe vs S13 | none | `stacker_average`、`kgl.stacking.stacker_average` | 保存済みの予測から再計算すると、S13 のメンバーでは S13 の平均と holdout・test とも差 0.0、入れ替えると `stacker_average_tuned` と差 0.0。notebook も入れ替えの前に `np.allclose(..., atol=1e-9)` で再現を確かめている。holdout の行で学習するのは、S13 と同じ4つのスタッカーと重み付き平均だけ（Q1↔Q2 の `cross_predict`、test 用は holdout の全行）。`weights()`（holdout 全行での NNLS）は表を出すためだけで、予測には使わない。S13 との差: 全行 +0.0001（-0.0002〜+0.0004）、順向きの Q2 だけ +0.0003（-0.0001〜+0.0008）、Q1 だけ -0.0001。holdout 全体での重みは TabPFN 0.558、CatBoost 0.216、LightGBM 0.205、XGBoost 0.022（S13 では XGBoost 0）。 |
| 4b | Choosing a subset of tuned members on the holdout | low | なし（再計算） | notebook は4つとも入れ替える。一部だけ入れ替えると、XGBoost と MLP で 0.19106、XGBoost だけで 0.19108、MLP だけで 0.19104、LightGBM と CatBoost で 0.19097。どれも S13 との差は ±0.0001 以内で、範囲は 0 を含む。この結果から組み合わせを選ぶと、同じ holdout で選ぶことになる。ただし楽観は 0.0001 程度。S14 は不採用なので、いまは影響しない。 |
| 5 | Metric | none | `holdout_rmsle`、`src/kgl/metrics.py` | sklearn の `root_mean_squared_log_error`。oof から再計算した値は `metrics.json` と一致する（0.199754、0.211706、0.202236、0.199411、0.191129）。 |
| 6 | Submission integrity | none | `validate_submission` を `write_submission` の前に呼ぶ | 5つの `submission.csv` はすべて 29,341 行で、列は `[id, price]`。id の順は sample と一致し、欠損も非有限値もない。価格の列は Float64 で、値は 1.64e6〜2.94e9 円。 |
| 7 | Fold file integrity | none | `folds/time_2024q3_2025q1.json`、`cv._time_holdout` | `TimeHoldout`、境目は [2024年第3四半期, 2025年第1四半期]、202,323 行、id の重複なし。再計算: fold -1 は 162,701 行（2019.0〜2024.25）、fold 0 は 17,931 行（2024.5〜2024.75）、fold 1 は 21,691 行（2025.0〜2025.25）。fold 1 は `holdout_2025q1` の fold 0 と一致し、fold ≤ 0 は同じファイルの fold -1 と一致する。境目は文字列の比較で切るが、表記が固定幅なので数値の期間と一致する。`attach_folds` は train.csv の行の順を保つので、`fs.X` の行は S13 と同じ。`docs/validation.md` の表とも一致する。 |
| 8a | Run reproducibility: commit and restart | none | `run.log`、`run_started_from.txt`、git | eba00f8 は 10:23:05 に commit され、再開は 10:23:20。notebook は eba00f8 から HEAD まで変わっていない。最初の実行（0956940、10:04）は LightGBM の試行 0〜63 の後に止まった。0956940 → eba00f8 の差は、`XgbParams.device`、`Trials.xgboost`、本文、xgboost → xgboost-cu12 だけ。LightGBM の試行は、再開の前後で同じコードとパッケージで走った。XGBoost、MLP、CatBoost の study は再開の後に作られた（11:37、12:50、15:10）。最後の学習の途中で 58422cd（17:40）が入ったが、触れたのは `kgl.tabnet` と S13 だけで、S14 は読み込まない。内側の基準と、探索した XGBoost（GPU）は完全に再現した。 |
| 8b | Stale RUNNING trial in `optuna.db` | low | `TPESampler(seed=cfg.seed)`、study `lightgbm` | lightgbm の study は COMPLETE 300、RUNNING 1（試行 64、10:22:28 開始）。ほかの study に未完了の試行はない。Optuna 5.0.0 の `TPESampler` は既定で `constant_liar=True` で、`_sample` は RUNNING の試行を「悪い側」に入れる。そのため試行 65〜300 は、試行 64 の設定（num_leaves 229、min_data_in_leaf 195 など）の周りを避けて提案された。再開で sampler も seed 42 から作り直された。`best_trial`、`_done`、`best_after` は COMPLETE だけを数えるので、回数（300）と選んだ設定は正しく求まっている。影響は2つ。空の DB から実行し直しても、LightGBM の探索の道筋は一致しない。本文の「乱数の種を固定しているので、同じ順に試します」は、中断がないときだけ成り立つ。また `metrics.json` の `trials` は `len(studies[name].trials)` なので 301 と記録された。`optuna.db` は gitignore された `output/` にあり、探索の唯一の記録である。本数や反復の上限に届いた試行はない（最大で LightGBM 1,492 本、XGBoost 5,348 本、CatBoost 6,553 本、MLP 54/60 回）。 |

## Recommended fixes

1. 8b: `notebooks/s16_tabnet_tuning.py` も、同じ `TPESampler(seed=cfg.seed)` と `load_if_exists=True` を使っている。試行は1つずつ走るので、対策は次のどちらか（ユーザーが決める）。
   - `constant_liar=False` を明示する。
   - 再開の前に RUNNING の試行を FAIL にする。

   S14 の本文には、LightGBM の探索を試行 63 の後に再開したこと、空の DB からは同じ道筋にならないことを書く。`metrics.json` の `trials` は COMPLETE の数にする。`optuna.db` は消さずに残す。
2. 3a: notebook の本文か `docs/validation.md` に、S14 から XGBoost を GPU で学習していることを書く。あわせて、S6 の設定を GPU で学習すると 0.2110（CPU では 0.2103）になることも書く。比べ方は探索した側に不利なので、-0.0106 は控えめな値である。
3. 2b: `docs/validation.md` の S14 の項に、内側の分割にも集約の非対称が同じ形で入ることを書く。大きさは LightGBM で +0.0006（holdout は +0.0004）。
4. 3b: 内側の改善幅を記録するときは、`metrics.json` の値を使う（LightGBM -0.0028、CatBoost -0.0042、XGBoost -0.0138、MLP -0.0036）。
5. 4b: 探索したメンバーの一部だけを S15 以降で使う場合は、holdout での組み合わせのスコアで選ばない。先に決めた規則（たとえば「単独での改善の範囲が 0 を含まない」なら XGBoost と MLP）で決め、その規則を記録する。
6. `docs/validation.md` の Leak checklist に S14 の行を足す。

## Actions taken (2026-10-17)

- 1: S16 は、試行を1つずつ回すので `TPESampler(seed=cfg.seed, constant_liar=False)` にした。入っている Optuna 5.0.0 で `constant_liar` の既定が True であることを確かめた。S14 の本文と `metrics.json` の `trials` は、S14 を実行し直すときに直す（探索は保存済みなので、実行し直しは基準の採点、最後の学習、スタッキングだけ）。`optuna.db` は残す。
- 2、3、6: `docs/validation.md` の Leak checklist に S14 の項を足した（GPU の XGBoost と、内側の分割の集約の非対称を含む）。
- 4: #38〜#42 の判断のコメントは `metrics.json` の値（CatBoost -0.0042、MLP -0.0036）で書いた。LightGBM は -0.0027 と書いたが、0.199041 → 0.196291 の差は -0.00275 で、`metrics.json` から丸めると -0.0028。
- 5: S15 は、探索した4つのメンバーをすべて使う S14 のメンバーの組をそのまま使う。holdout で組み合わせを選ぶことはしない。
