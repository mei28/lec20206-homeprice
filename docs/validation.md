# Validation

Maintained by the `kaggle-validation` skill. A score is trusted only after this file says why.

## How the test set was built

## Decision flow result

- Strategy:
- K:
- Stratify / group / time column:
- Reason:

## Fold file

- `folds/<name>.parquet` (never overwritten; new design = new name)
- Command:

## Fold diagnostics

Per-fold size and target distribution; group overlap check; time order check.

## Leak checklist

| Item | Status | Evidence |
|---|---|---|
| Test information in train (joins, global statistics) | | |
| Statistics fit outside the fold loop (scalers, encoders) | | |
| Target encoding outside folds | | |
| Feature selection outside folds | | |
| Time leakage (future rows in features, folds not time ordered) | | |
| Metric re-implementation matches the competition text | | |
| Submission id order equals the sample | | |

## CV vs LB

| exp | run | cv | public lb | gap | note |
|---|---|---|---|---|---|

## Adversarial validation

## Fold version history

| name | strategy | reason for the new version | date |
|---|---|---|---|
