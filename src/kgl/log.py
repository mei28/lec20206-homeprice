"""Logging, timing, and seeding shared by every experiment."""

import logging
import math
import os
import random
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import psutil


def get_logger(name: str, run_dir: Path) -> logging.Logger:
    """Console plus `run.log` in the run directory. Calling twice with the same name returns the same logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@contextmanager
def trace(title: str) -> Generator[None, None, None]:
    """Print elapsed time and memory delta of a block to stderr (after unonao/kaggle-template)."""
    t0 = time.time()
    process = psutil.Process(os.getpid())
    m0 = process.memory_info().rss / 2.0**30
    yield
    m1 = process.memory_info().rss / 2.0**30
    delta = m1 - m0
    sign = "+" if delta >= 0 else "-"
    print(f"[{m1:.1f}GB({sign}{math.fabs(delta):.1f}GB):{time.time() - t0:.1f}sec] {title}", file=sys.stderr)
