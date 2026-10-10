from pathlib import Path

import pytest

from kgl import env as kenv

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_detect_local_when_not_on_kaggle(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    e = kenv.detect()
    assert e.kind == "local"
    assert e.root == REPO_ROOT
    assert e.input_dir == REPO_ROOT / "input"
    assert e.output_dir == REPO_ROOT / "output"


def test_detect_kaggle_when_kernel_env_set(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    e = kenv.detect()
    assert e.kind == "kaggle"
    assert e.input_dir == Path("/kaggle/input")
    assert e.output_dir == Path("/kaggle/working")


def test_comp_dir_returns_existing_dir(tmp_path):
    (tmp_path / "input" / "titanic").mkdir(parents=True)
    e = kenv.Env(kind="local", root=tmp_path, input_dir=tmp_path / "input", output_dir=tmp_path / "output")
    assert kenv.comp_dir(e, "titanic") == tmp_path / "input" / "titanic"


def test_comp_dir_missing_raises(tmp_path):
    e = kenv.Env(kind="local", root=tmp_path, input_dir=tmp_path / "input", output_dir=tmp_path / "output")
    with pytest.raises(FileNotFoundError):
        kenv.comp_dir(e, "titanic")


def test_run_id_default_when_no_overrides():
    assert kenv.run_id([]) == "default"


def test_run_id_debug_flag_wins():
    assert kenv.run_id(["--seed", "1", "--debug"]) == "debug"


def test_run_id_hash_is_stable_and_distinct():
    a = kenv.run_id(["--model-params.n-estimators", "500"])
    b = kenv.run_id(["--model-params.n-estimators", "500"])
    c = kenv.run_id(["--model-params.n-estimators", "600"])
    assert a == b
    assert a != c
    assert len(a) == 8 and all(ch in "0123456789abcdef" for ch in a)


def test_run_id_explicit_run_name_overrides():
    assert kenv.run_id(["--seed", "1", "--run", "foo"]) == "foo"


def test_run_dir_is_created(tmp_path):
    e = kenv.Env(kind="local", root=tmp_path, input_dir=tmp_path / "input", output_dir=tmp_path / "output")
    d = kenv.run_dir(e, "exp000_baseline", "default")
    assert d == tmp_path / "output" / "exp000_baseline" / "default"
    assert d.is_dir()
