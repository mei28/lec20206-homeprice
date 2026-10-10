"""Submission-side inference for this experiment. The same file runs locally and inside the Kaggle kernel.

Imports are limited to the Kaggle image (numpy, polars, lightgbm) plus kgl and this experiment's run.py.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import polars as pl
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import EXP, Config, build_features, load_test, postprocess, predict  # noqa: E402

from kgl.env import comp_dir, detect, run_dir  # noqa: E402
from kgl.io import validate_submission  # noqa: E402


@dataclass
class InferConfig:
    run: str = "default"
    model_dir: Path | None = None


def main(args: InferConfig) -> None:
    env = detect()
    run_root = args.model_dir.parent if args.model_dir else run_dir(env, EXP, args.run)
    model_dir = args.model_dir or run_root / "model"
    cfg = Config.from_dict(json.loads((run_root / "config.json").read_text())["config"])

    models = [lgb.Booster(model_file=str(p)) for p in sorted(model_dir.glob("fold*.txt"))]
    if not models:
        raise FileNotFoundError(f"no fold models under {model_dir}; run `just run {EXP}` first")

    test = load_test(env, cfg)
    pred = postprocess(predict(models, build_features(test, cfg)))
    sample = pl.read_csv(comp_dir(env, cfg.comp) / cfg.sample_sub)
    # Join by id rather than inserting by position, so a test/sample row-order difference cannot pass unnoticed.
    predictions = pl.DataFrame({cfg.id_col: test[cfg.id_col], cfg.target: pred}).cast(
        {cfg.target: sample[cfg.target].dtype}
    )
    sub = sample.select(cfg.id_col).join(predictions, on=cfg.id_col, how="left")
    validate_submission(sub, sample, cfg.id_col)

    out_dir = env.output_dir if env.kind == "kaggle" else run_root
    out = out_dir / "submission.csv"
    sub.write_csv(out)
    print(f"{out} ({sub.height} rows, {len(models)} fold models)")


if __name__ == "__main__":
    main(tyro.cli(InferConfig))
