# Leak review: s13_tabular_dl (tabpfn, stacker_average_with_tabpfn)

Reviewer: leak-reviewer instructions, run as a subagent on 2026-10-16 against commit 4cfcd58 plus the then-untracked `notebooks/s13_tabular_dl.py`.
The notebook was not re-run for the review. TabPFN and TabNet numbers come from `output/s13_tabular_dl/*/oof.parquet` and `submission.csv`; the restacking and a LightGBM ablation were recomputed in memory.

Verdict: clean

TabPFN にも、S12 のスタッキングを作り直した部分にも、holdout の正解や test 期間の情報は入っていない。

- holdout 用の文脈は 2023Q3〜2024Q4 の5万行で、holdout の行も 2025 年の行も含まない。
- 文脈の目的変数は、`tr_idx` だけで作った価格指数で実質化している。
- holdout の行の TE と価格指数の外挿も、`tr_idx` だけから作っている。

TabPFN が CatBoost より良い幅（-0.0042）は、どの切り口でも同じように出ている。

- 2025Q1 と Q2、成約価格と取引価格のどちらでも同じくらいの幅がある。
- モデルごとに四半期ごとの平均のずれを除いても変わらない。価格の水準で得をしているわけではない。

スタッキングの作り直しは S12 の手順に列を1つ足しただけで、S12 の出力を 4e-15 の精度で再現する。S12 からの改善 -0.0028 は、時間の順向き（Q1 で学習して Q2 を予測）だけで測っても -0.0029 で同じ。

低い重大度の非対称が1つある（2a）。

- これは S5 から全メンバーに共通で、S13 で新しく入ったものではない。
- 目的変数を使わない地区・駅の集約特徴と件数は、holdout の行も含めた全学習行から作っている。一方、test の行は自分の集約に入らない。
- LightGBM で測ると、holdout は 0.0004 楽観的になる。

| # | Item | Severity | Location | Evidence |
|---|---|---|---|---|
| 1a | Test information in train: TabPFN context and inputs | none | `s13_tabular_dl.py` `most_recent`, `tabpfn_predict` | holdout 用の文脈は `tr_idx` の部分集合。再計算すると5万行で、`va_idx` との重なりは0、取引時期は 2023.5〜2024.75、2025 年以降の行は0。test 用の文脈は train の行だけから選ぶ（2024Q1〜2025Q2）。文脈の目的変数 `fs.y_log - fs.offset` の `offset` は `tr_idx` で作った指数。tabpfn 9.2.0 は既定の `fit_mode="fit_preprocessors"` で、前処理とカテゴリの `OrdinalEncoder` を `fit()` の中で文脈から作り、`predict()` は適用するだけ。呼び出しのたびに新しい `TabPFNRegressor` を作る。テレメトリのモジュールはない。語彙は train と test の値から作る（目的変数は使わない、S1 と同じ）。 |
| 1b | Model provenance (rule 2.6.a) | low | `TabPFNRegressor(device="cuda", ...)`, `model_path="auto"` | 既定の重みは `Prior-Labs/tabpfn_3_5` の `tabpfn-v3.5-20260909.safetensors` で、日付は test 期間の開始より後。パッケージは人工データでの事前学習を説明しているが、以前の版には実データで調整した `_real` の重みもあった。repo の中だけでは v3.5 の学習データを確かめられない。観察されたリークではなく、出典の記録の問題。 |
| 2a | Target-free aggregates and counts fit on rows that include the holdout | low | `src/kgl/feature_set.py` `_target_free` | holdout の設定でも `_target_free(train_df, train_df)` を使うので、地区・駅の中央値、成約の割合、件数に holdout の 21,691 行が入る。test の設定では test の行は入らない。目的変数は使わないが、非対称。holdout の行の `district_contract_share` の平均は 0.4533（`tr_idx` だけで作ると 0.4318）。2024Q4 までにない地区が 12 行、駅が 2 行あり、件数が 1 以上になる。S5 の設定の LightGBM は 0.2035、集約を `tr_idx` だけで作ると 0.20388（+0.0004）。S5 以降のすべてのメンバーに同じようにかかる。TabPFN でも CatBoost でも同じ `fs.X` を読むので、TabPFN が CatBoost より良い理由にはならない。 |
| 2b | TabNet and TabPFN preprocessing | none | `tabnet_inputs`, `tabpfn_predict` | TabNet の中央値・平均・標準偏差と目的変数の中心は `tr_idx` から。提出用は `fs.X_refit` から。TabPFN の前処理は文脈から（1a）。 |
| 3 | Target encoding and price-index offset | none | `feature_set.py`, `price_index.py` | 指数は `tr_idx` だけで推定し、holdout の時期には 2023Q1〜2024Q4 から直線で外挿する。holdout の価格は入らない。holdout の行の TE は `tr_idx` の目的変数から、学習行は5分割の OOF。提出用も同じ手順を全学習行で行う。スタッキングの行の特徴 `te_district` も holdout の設定の TE。 |
| 4 | Feature selection outside folds | none | `TabPFNParams` | 重要度や相関での特徴の選択はない。`context_rows = 50000` は #37 で事前に決めた「GPU に載る最大の直近の行」の規則で決めた。2,000 行での試し（0.2038）は候補の中から選ぶのに使っていない。 |
| 5a | Context window and offset symmetry (holdout vs test) | low | `most_recent` | holdout の文脈は 2023Q3（9,213 行中 3,529 行）〜2024Q4 の約5.4四半期で、1〜2四半期先を予測する。test の文脈は 2024Q1（9,631 行中 718 行）〜2025Q2 の約5.1四半期で、1〜3四半期先を予測する。境目の四半期は行の順で切れるが、行の順に意味はない（区の割合の差は 0.009 以内、log 価格の平均は 17.506 と 17.503）。成約の割合は、文脈で 0.54 と 0.57、予測先で 0.62 と 0.67（2026Q1 は 0.816）。指数の外挿は両方とも直近8四半期の直線。残る差は、holdout が3四半期先を測っていないことと、TabPFN が文脈より先の `trade_period` を受け取ること。TabPFN − CatBoost の予測の平均の差は、holdout で +0.0017、+0.0022、test で -0.0013、-0.0043、-0.0015。2026Q1 にずれは出ていない。 |
| 5b | TabPFN 0.1948 vs CatBoost 0.1990 | none | `output/s13_tabular_dl/tabpfn/oof.parquet` | 対応のある bootstrap で -0.0042（-0.0056〜-0.0029）。Q1 0.1945 と 0.1983、Q2 0.1952 と 0.1998、成約 0.1798 と 0.1837、取引 0.2171 と 0.2218。四半期ごとの平均のずれを除くと 0.1948 と 0.1989。誤差の相関 0.93。CatBoost は holdout で early stopping しているが TabPFN は holdout で何も選んでいないので、比べ方は TabPFN に有利ではない。 |
| 5c | Stacking direction (inherited from S12) | none for S13 | `src/kgl/ensemble.py` `cross_predict` | Q1 で学習して Q2 を予測し、Q2 で学習して Q1 を予測する（時間の逆向きも含む）。S12 との差は、全行 -0.0028（-0.0034〜-0.0022）、順向きの Q2 の行だけ -0.0029（-0.0037〜-0.0021）、逆向きの Q1 の行だけ -0.0027（-0.0037〜-0.0016）。改善は逆向きに頼っていない。 |
| 5d | TabNet early stopping on the holdout | low | TabNet の `eval_set` | TabNet の holdout の予測は holdout で選んだエポックのもの（GBDT と同じ）。`with_tabnet` と `with_both` にだけ関係し、`with_tabpfn` には関係しない。 |
| 6 | Metric | none | `holdout_rmsle`, `src/kgl/metrics.py` | sklearn の `root_mean_squared_log_error`。oof.parquet から再計算した値は、すべての run で metrics.json と一致（tabpfn 0.19483、tabnet 0.22203、with_tabpfn 0.19103、with_both 0.19112、with_tabnet 0.19405）。 |
| 7a | Restack recipe vs S12 | none | `stacker_average`, `src/kgl/stacking.py` | 保存済みのメンバーの予測から再計算すると、S12 の6メンバーで `s12_black_magic/stacker_average` を 3.6e-15 以内で再現し、`tabpfn` を足すと保存済みの `stacker_average_with_tabpfn` と完全に一致する。設定は S12 と同じで、違いはメンバーの列が1つ増えたことだけ。部品のスコアは ridge 0.19164、ridge_by_type 0.19181、mlp 0.19162、lgbm 0.19173、重み付き平均 0.19247。holdout 全体での重みは tabpfn 0.556、catboost 0.234、lightgbm 0.211、ほかは 0。 |
| 7b | Variant selection on the same holdout | low | restack のループ | 3通り（tabnet、tabpfn、both）を同じ holdout で採点している。`with_tabpfn`（0.19103）と `with_both`（0.19112）の差は 0.0001 で、どちらを選んでも楽観はほぼない。S12 との差の範囲は 0 を含まない。 |
| 7c | Submission integrity | none | `validate_submission` | 5つの `submission.csv` はすべて 29,341 行、`[id, price]`、id の順は sample と一致、欠損と非有限値なし。TabPFN の価格は 2.28e6〜2.41e9 円。 |
| 8 | Fold file integrity | none | `folds/holdout_2025q1.json` | `TimeHoldout`、2025年第1四半期から、202,323 行。`tr_idx` 180,632 行（最大 2024.75）、`va_idx` 21,691 行（最小 2025.0）。 |
| 9 | Run reproducibility | low | `notebooks/s13_tabular_dl.py` | 記録された commit 4cfcd58 に notebook が入っておらず、notebook は最後の実行の後に編集されていた。スタッキングの経路は同一であることを確認済み（7a）。TabPFN と TabNet のセルは GPU なしでは確かめられない。 |

## Recommended fixes

1. 2a: 目的変数を使わない集約特徴を、holdout と test で対称にする（ユーザーが決める）。(a) holdout 用は `tr_idx` だけで作る（LightGBM で CV +0.0004、全メンバーの再学習が要る）か、(b) 提出用を train と test の特徴から作る（目的変数は使わない）。どちらも採らないなら、測った楽観 +0.0004 を `docs/validation.md` に記録する。S13 の結論は変わらない。
2. 9: notebook を commit してから実行し直し、記録の commit に出力を作ったコードが入るようにする。
3. 1b: `docs/competition.md` に TabPFN の重みと、学習データについての記載を書く。モデルの説明だけを読み、取引データは探さない。
4. 5a: notebook の本文に、holdout は1〜2四半期先、test は3四半期先（2026Q1）まであること、TabPFN が文脈より先の `trade_period` を受け取ることを書く。
5. `docs/validation.md` の Leak checklist に S13 の行を足す。

## Actions taken (2026-10-16)

- 2: notebook を commit（a5f9507）し、その commit から通しで実行し直した。
- 3: `docs/competition.md` に、重み `tabpfn-v3.5-20260909`、非商用ライセンス、技術レポート（arXiv 2609.17895）3.3 節の「事前学習は TabPFN-3 と同じ人工データの生成手順」という記載を書いた。レポートに実データの利用は書かれていない。
- 4: S13 の「結果を読む」に、予測先の長さの違いと、test の四半期ごとの TabPFN と CatBoost の差を足した（notebook の中で計算する）。
- 1, 5: `docs/validation.md` の Leak checklist に記録した。2a は対称にする変更を行わず、楽観 +0.0004 を記録するにとどめる。全メンバーに同じようにかかり、採用の判断（メンバー間の比較）を変えないため。
- 本文で「TabPFN の強さは直近の行を選んだことから来ていない」と書くため、LightGBM を同じ直近5万行で学習する比較を notebook に足した（5b の「文脈の新しさは試していない」への対応）。scratch の結果は、全行 0.2035、直近5万行 0.2116、直近10万行 0.2072。
