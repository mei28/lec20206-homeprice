# Submissions

| date | exp | run | cv | public_lb | message | kernel_version | file |
|---|---|---|---|---|---|---|---|
| 2026-10-10 | s3_features | plus_notation | 0.2090 | 0.20149 | S3 features: trade period as a number and parsed notation |  | submission.csv |
| 2026-10-11 | s4_trend | hedonic_linear | 0.2050 | 0.19911 | S4 hedonic price index: deflated target, index extrapolated linearly from the last 8 quarters |  | submission.csv |
| 2026-10-14 | s5_te | all_three | 0.2035 | 0.19895 | S5 target encoding, district and station aggregates, and counts on top of S4 |  | submission.csv |
| 2026-10-15 | s6_gbdt | catboost | 0.1990 | 0.19642 | S6 CatBoost on the S5 feature set |  | submission.csv |
