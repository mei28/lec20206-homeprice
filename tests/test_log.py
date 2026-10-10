import logging

import numpy as np

from kgl import log


def test_get_logger_writes_to_run_log(tmp_path):
    logger = log.get_logger("exp_test", tmp_path)
    logger.info("hello")
    for h in logger.handlers:
        h.flush()
    assert "hello" in (tmp_path / "run.log").read_text()


def test_get_logger_does_not_duplicate_handlers(tmp_path):
    a = log.get_logger("exp_dup", tmp_path)
    b = log.get_logger("exp_dup", tmp_path)
    assert a is b
    assert sum(isinstance(h, logging.FileHandler) for h in a.handlers) == 1


def test_seed_everything_makes_numpy_reproducible():
    log.seed_everything(7)
    first = np.random.rand(3)
    log.seed_everything(7)
    second = np.random.rand(3)
    assert first.tolist() == second.tolist()


def test_trace_reports_elapsed_time(capsys):
    with log.trace("sleep"):
        pass
    err = capsys.readouterr().err
    assert "sleep" in err and "sec" in err
