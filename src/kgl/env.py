"""Resolve input/output paths for the local checkout and the Kaggle kernel runtime."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Env:
    kind: Literal["local", "kaggle"]
    root: Path
    input_dir: Path
    output_dir: Path


def detect() -> Env:
    """Kaggle sets KAGGLE_KERNEL_RUN_TYPE inside kernels; anything else is a local checkout."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return Env(
            kind="kaggle", root=Path("/kaggle"), input_dir=Path("/kaggle/input"), output_dir=Path("/kaggle/working")
        )
    root = Path(__file__).resolve().parents[2]
    return Env(kind="local", root=root, input_dir=root / "input", output_dir=root / "output")


def comp_dir(env: Env, comp: str) -> Path:
    path = env.input_dir / comp
    if not path.is_dir():
        raise FileNotFoundError(f"competition data not found: {path} (run `just download {comp}`)")
    return path


def run_id(argv: list[str]) -> str:
    """Name a run by its CLI overrides so the same overrides reuse one directory.

    `--run NAME` names it explicitly, `--debug` always goes to `debug`, and no overrides means `default`.
    """
    if "--run" in argv:
        return argv[argv.index("--run") + 1]
    if "--debug" in argv:
        return "debug"
    if not argv:
        return "default"
    return hashlib.sha1(" ".join(argv).encode()).hexdigest()[:8]


def run_dir(env: Env, exp: str, run: str) -> Path:
    path = env.output_dir / exp / run
    path.mkdir(parents=True, exist_ok=True)
    return path
