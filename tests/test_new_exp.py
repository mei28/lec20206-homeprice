import pytest

from tools import new_exp


def _seed_base(experiments: "pytest.TempPathFactory"):
    base = experiments / "exp000_baseline"
    base.mkdir(parents=True)
    (base / "run.py").write_text(
        '"""exp000_baseline: LightGBM on raw columns.\n\nbase: none\nhypothesis: a baseline\n"""\nprint(1)\n'
    )
    (base / "infer.py").write_text("print(2)\n")
    (base / "__pycache__").mkdir()
    (base / "__pycache__" / "run.cpython-312.pyc").write_bytes(b"x")
    return base


@pytest.mark.parametrize("bad", ["exp1_x", "exp0001_x", "Exp001_x", "exp001", "exp001-x", "001_x"])
def test_invalid_names_are_rejected(bad):
    with pytest.raises(ValueError):
        new_exp.validate_name(bad)


def test_valid_name_passes():
    new_exp.validate_name("exp012_lgbm_te")


def test_create_copies_only_the_scripts_and_rewrites_base(tmp_path):
    experiments = tmp_path / "experiments"
    _seed_base(experiments)
    created = new_exp.create(experiments, "exp001_more_trees", base="exp000_baseline")
    assert created == experiments / "exp001_more_trees"
    assert sorted(p.name for p in created.iterdir()) == ["infer.py", "run.py"]
    text = (created / "run.py").read_text()
    assert text.startswith('"""exp001_more_trees:')
    assert "\nbase: exp000_baseline\n" in text
    assert "hypothesis:" in text


def test_create_refuses_existing_directory(tmp_path):
    experiments = tmp_path / "experiments"
    _seed_base(experiments)
    with pytest.raises(FileExistsError):
        new_exp.create(experiments, "exp000_baseline", base="exp000_baseline")


def test_create_requires_an_existing_base(tmp_path):
    experiments = tmp_path / "experiments"
    _seed_base(experiments)
    with pytest.raises(FileNotFoundError):
        new_exp.create(experiments, "exp001_x", base="exp999_missing")
