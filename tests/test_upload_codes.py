import json

from tools import upload_codes, upload_model


def _repo(tmp_path):
    (tmp_path / "src" / "kgl").mkdir(parents=True)
    (tmp_path / "src" / "kgl" / "env.py").write_text("x=1\n")
    (tmp_path / "src" / "kgl" / "__pycache__").mkdir()
    (tmp_path / "src" / "kgl" / "__pycache__" / "a.pyc").write_bytes(b"x")
    (tmp_path / "experiments" / "exp000_baseline").mkdir(parents=True)
    (tmp_path / "experiments" / "exp000_baseline" / "run.py").write_text("print(1)\n")
    (tmp_path / "experiments" / "exp001_other").mkdir(parents=True)
    (tmp_path / "experiments" / "exp001_other" / "run.py").write_text("print(2)\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='kgl'\n")
    run = tmp_path / "output" / "exp000_baseline" / "default"
    (run / "model").mkdir(parents=True)
    (run / "model" / "fold0.txt").write_text("tree\n")
    (run / "config.json").write_text("{}\n")
    (run / "metrics.json").write_text("{}\n")
    (run / "oof.parquet").write_bytes(b"not needed")
    return tmp_path


def test_stage_codes_copies_kgl_the_experiment_and_pyproject_only(tmp_path):
    root = _repo(tmp_path)
    stage = upload_codes.stage(root, "exp000_baseline", user="mei28", comp="titanic")
    files = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
    assert files == ["dataset-metadata.json", "experiments/exp000_baseline/run.py", "pyproject.toml", "src/kgl/env.py"]
    assert json.loads((stage / "dataset-metadata.json").read_text())["id"] == "mei28/titanic-codes"


def test_stage_codes_starts_from_a_clean_directory(tmp_path):
    root = _repo(tmp_path)
    stage = upload_codes.stage(root, "exp000_baseline", user="mei28", comp="titanic")
    (stage / "stale.txt").write_text("old\n")
    upload_codes.stage(root, "exp000_baseline", user="mei28", comp="titanic")
    assert not (stage / "stale.txt").exists()


def test_stage_model_copies_model_config_and_metrics(tmp_path):
    root = _repo(tmp_path)
    stage = upload_model.stage(root, "exp000_baseline", "default", user="mei28", comp="titanic")
    files = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
    assert files == ["config.json", "dataset-metadata.json", "metrics.json", "model/fold0.txt"]
    meta = json.loads((stage / "dataset-metadata.json").read_text())
    assert meta["id"] == "mei28/titanic-exp000-baseline-model"
    assert meta["title"] == "titanic-exp000_baseline-model"


def test_model_title_uses_override_when_too_long(tmp_path):
    root = _repo(tmp_path)
    stage = upload_model.stage(root, "exp000_baseline", "default", user="mei28", comp="titanic", title="short-title")
    assert json.loads((stage / "dataset-metadata.json").read_text())["title"] == "short-title"
