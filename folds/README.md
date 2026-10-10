# folds/

Fold assignments shared by every experiment: `<name>.parquet` with columns `(id, fold)` and a
`<name>.json` sidecar recording the strategy and parameters. Files are git-tracked and never
overwritten; a changed design gets a new name, and `Config.folds` pins the one an experiment used.

Time-series strategies mark warm-up rows (never validated) with fold `-1`.

Build one with `just folds <name> --strategy ... --target ... --id-col ...` (see `tools/make_folds.py`).
