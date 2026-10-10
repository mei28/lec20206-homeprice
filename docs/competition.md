# Competition

Filled by the `kaggle-onboard` skill before any training code is written.

## Platform

- URL: https://www.kaggle.com/competitions/utokyo-mlapp-2026-ex1
- 名称: 2026年度 機械学習応用 演習1：中古マンション価格予測（東京大学 馬場雪乃先生の講義コンペ）
- Type: CSV submission。Community（講義用、賞なし、メダルなし）
- Team size / merger deadline: 最大4人。講義で割り当てたグループ単位で1チーム。merger deadline は締切と同じ 2026-10-21 14:59 UTC
- 本 repo の立場: TA としての参加。目安の提示と教材化が目的で、上位は狙わない

## Task

- Input: 東京都内の中古マンション等の取引1件分の物件情報と取引情報（表形式、19列）
- Output (one row per): test の取引1件につき取引価格（総額、円）1つ
- Classes / target range: 回帰。train の `取引価格（総額）` は 50,000〜16,000,000,000 円、中央値 38,000,000 円、平均 49,370,000 円

## Data

- Files and sizes:
  - `train.csv` 32 MB、202,323 行 × 20 列（目的変数 `取引価格（総額）` を含む）
  - `test.csv` 4.2 MB、29,341 行 × 19 列
  - `sample_submission.csv` 455 KB、29,341 行（`id, price`）
  - `README.md` データ説明（Data タブと同内容）
- 列: `id`, `価格情報区分`, `市区町村コード`, `市区町村名`, `地区名`, `最寄駅：名称`, `最寄駅：距離（分）`, `間取り`, `面積（㎡）`, `建築年`, `建物の構造`, `用途`, `今後の利用目的`, `都市計画`, `建ぺい率（％）`, `容積率（％）`, `取引時期`, `改装`, `取引の事情等`。値は元データの表記のまま（全角文字、`2013年` のような文字列、欠損あり）
- Train/test split as described by the host (random, by group, by time): 時間で分割。train は 2019年第1四半期〜2025年第2四半期、test は 2025年第3四半期〜2026年第1四半期。行は並べ替えたうえで `id` を振り直している
- Public/private の分け方: 「テストデータの一部」で Public を計算するとだけ書かれている。比率と分け方（ランダムか時期か）は非公開
- 出典: 国土交通省 不動産情報ライブラリの不動産価格（取引価格・成約価格）情報、東京都・中古マンション等、2026-10-06 取得。加工は定数列（種類、都道府県名）の削除、行の並べ替えと `id` 付与、train/test 分割のみ。元データの利用条件は公共データ利用規約（PDL1.0）
- External data allowed:
  - 禁止: テスト期間（2025年第3四半期以降）の不動産取引データ。不動産情報ライブラリ（Web、API）でも他の出典でも、手作業でも AI による自動取得でも不可（Rules 2.6.a、7.d）
  - 許可: 公開済みで全参加者が無料で同じように入手できる外部データ（地理情報、駅情報など）。レポートに出典を書く（Rules 2.6.b）

## Metric

- Name (official): RMSLE（Root Mean Squared Logarithmic Error）、小さいほど良い
- Exact definition: $\sqrt{\frac{1}{n}\sum_i (\log(1+\hat{y}_i) - \log(1+y_i))^2}$。$\hat{y}_i$ は予測価格、$y_i$ は実価格（円）。全行の平均（グループ別平均ではない）。負の予測は定義外
- `kgl.metrics` name (added with a test): `rmsle`（`sklearn.metrics.root_mean_squared_log_error`。負の予測で `ValueError`）

## Submission

- Mode (`SUBMIT_MODE`): csv
- File / columns / id column / sample file: `id, price` の2列、ヘッダ行あり、test の全 29,341 行。`price` は円。予測列名は目的変数名 `取引価格（総額）` ではなく `price`。`sample_submission.csv` の `id` 順は `test.csv` と一致
- Code competition limits: 該当なし

## Timeline

- Start: 2026-10-08 15:00 UTC（2026-10-09 00:00 JST）
- 30% mark (end of early phase): 2026-10-12 12:35 UTC（2026-10-12 21:35 JST）
- 70% mark (end of mid phase): 2026-10-17 17:23 UTC（2026-10-18 02:23 JST）
- Final submission deadline: 2026-10-21 14:59 UTC（2026-10-21 23:59 JST）
- Submissions per day: 1。最終評価に選べる提出は2つ

## Rules

- Pretrained models / licenses: 特に制限なし。AI ツール（LLM、コーディングエージェント）の利用を推奨（Rules 2.6.c、7.a）
- Hand labeling: 記載なし
- Other:
  - テスト期間データ禁止のルールを、チームの AI 向け指示ファイル（CLAUDE.md、AGENTS.md など）に書くこと（Rules 2.6.a）。本 repo では `CLAUDE.md` の「Competition rules for agents」に記載
  - コンペ期間中、コードと予測をグループ外に共有しない（Rules 7.b）
  - 上位グループにはコード提出を求めることがある。現実的な水準を大きく超えるスコアは確認対象（Rules 7.c）
  - データは講義以外の目的で使用・再配布しない（Rules 2.4、7.e）
  - 発表で、使った AI ツール、AI に任せたこと、AI が誤った箇所を説明する（Rules 7.a）
  - 順位表には「AI単独ベースライン」が載っている。目標はこれを超え、差がどこから来たかを説明すること（Description）

## Prior art

コンペ内の discussion と公開ノートブックは 2026-10-10 時点で0件。外部の類似事例から集めた。詳細メモは `.tmp/prior-art.md`。

同じ国交省データのコンペ:

- ProbSpace「Re：不動産取引価格予測」（2020、東京都、時点で分割、RMSLE）https://comp.probspace.com/competitions/re_real_estate_2020 。最も近い設定。1位は LightGBM 1本に集約特徴と TE を大量に足した（Private 0.2585）https://comp.probspace.com/competitions/re_real_estate_2020/discussions/Angus_Chang-Post5ac561a421209f743fc4 。5位は地区単位の建ぺい率平均が最も効いた。上位解法まとめ https://scrapbox.io/wakame/Re%EF%BC%9A%E4%B8%8D%E5%8B%95%E7%94%A3%E5%8F%96%E5%BC%95%E4%BE%A1%E6%A0%BC%E4%BA%88%E6%B8%AC_%E3%81%8A%E3%81%95%E3%82%89%E3%81%84
- ProbSpace「不動産取引価格予測」（2020、RMSE）https://comp.probspace.com/competitions/real_estate_2020 。1位の前処理（取引時点の連続値化、駅距離レンジの数値化、和暦→西暦、間取りから部屋数）が定番になった https://comp.probspace.com/topics/chun1182-Post3748b67697bb27cb59e8
- Nishika「中古マンション価格予測」（2021〜2023、全国、log10 価格の MAE、直後2四半期がテスト）。2023秋1位は総額モデルと㎡単価モデルの平均で、一律 +1.9% の倍率補正（時間外挿）が LB で最大の改善 https://qiita.com/roniheka/items/763a14f0c6b818774359 。2023夏3位は CatBoost と総額・㎡単価の2目的変数 https://qiita.com/9en310/items/1ef8c957fd43b850162b 。2022春3位は「log 価格 − log 面積」の差分モデル https://smartbowwow.com/2022/07/nishika-2022-3rd-prize-solution.html/

指標や題材が近いもの:

- ProbSpace「民泊サービスの宿泊料金予測」（RMSLE）https://comp.probspace.com/competitions/bnb_price 。KFold の CV と LB の相関がほぼ 0 だったという議論があり、CV を test の作り方に合わせる話の実例になる
- Kaggle House Prices の入門記事2本。fold 内で前処理を fit するリーク対策と OOF を保存して blend する手順 https://qiita.com/TaichiEndoh/items/63fa9bc128fbe2a1520a 。1回の hold-out（0.0985）と LB（0.141）の差 https://qiita.com/muscle_nishimi/items/901ed94f3cdf1c8d893a
- atmaCup 初心者講座（nyk510）。#10 は同じ RMSLE で、データを知る → 特徴量 → CV → LightGBM → 重要度と提出、の構成。本 repo の講座の型

価格情報区分について:

- 成約価格情報はレインズの成約情報を加工したもの。不動産情報ライブラリは 2024-04-01 運用開始 https://www.mlit.go.jp/report/press/content/001726227.pdf 。成約価格情報は戸建とマンションのみ、2021年から https://www.homes.co.jp/cont/press/opinion/opinion_00406/

外部データの候補（無料、公開、出典明記が必要）:

- 国土数値情報 鉄道 N02（駅の位置）https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N02-2023.html 、駅別乗降客数 S12 https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-S12-2023.html 、位置参照情報（町丁目の代表点）https://nlftp.mlit.go.jp/isj/ 、駅データ.jp https://ekidata.jp/agreement.php
- 地価公示の 2026 年版や、2025Q3 以降の取引から作られた統計（不動産価格指数など）は、テスト期間の価格を反映するので使わない

## Open questions

- Public/private の分割方法（ランダムか、四半期ごとか）。CV 設計と最終提出の選び方に効く
- `価格情報区分` の比率が train と test で違う（train は成約価格情報 45%、test は 67%）。成約価格情報は 2021Q1 から現れ、2025Q1 から件数が増える。test の 2026Q1 は取引価格情報が 1,487 件しかなく（通常は約4,000件）、公表の遅れと見られる。詳細は `docs/eda.md`
