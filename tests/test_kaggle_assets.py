import json

import pytest

from tools import kaggle_assets as ka


def test_slugify_lowercases_and_replaces_underscores():
    assert ka.slugify("titanic-exp000_baseline-model") == "titanic-exp000-baseline-model"
    assert ka.slugify("CMI_Detect Behavior") == "cmi-detect-behavior"


def test_metadata_has_id_title_and_license():
    meta = ka.dataset_metadata("mei28", "titanic-codes")
    assert meta == {"id": "mei28/titanic-codes", "title": "titanic-codes", "licenses": [{"name": "CC0-1.0"}]}


def test_metadata_rejects_titles_over_50_chars():
    with pytest.raises(ValueError, match="50"):
        ka.dataset_metadata("mei28", "x" * 51)


def test_write_metadata_creates_dataset_metadata_json(tmp_path):
    ka.write_dataset_metadata(tmp_path, "mei28", "titanic-codes")
    assert json.loads((tmp_path / "dataset-metadata.json").read_text())["id"] == "mei28/titanic-codes"


class _Api:
    def __init__(self, statuses):
        self.statuses = list(statuses)

    def dataset_status(self, dataset):
        return self.statuses.pop(0)


def test_wait_ready_returns_when_status_is_ready():
    ka.wait_ready(_Api(["blobs_received", "blobs_decompressed", "ready"]), "mei28/titanic-codes", interval=0)


def test_wait_ready_raises_on_failed_status():
    with pytest.raises(RuntimeError, match="failed"):
        ka.wait_ready(_Api(["failed"]), "mei28/titanic-codes", interval=0)


def test_copy_tree_skips_pycache(tmp_path):
    src = tmp_path / "src" / "kgl"
    (src / "__pycache__").mkdir(parents=True)
    (src / "env.py").write_text("x = 1\n")
    (src / "__pycache__" / "env.cpython-312.pyc").write_bytes(b"x")
    ka.copy_tree(src, tmp_path / "stage" / "src" / "kgl")
    assert sorted(p.name for p in (tmp_path / "stage" / "src" / "kgl").rglob("*")) == ["env.py"]


def test_dataset_exists_is_false_on_404_and_reraises_other_http_errors():
    import requests

    class _Resp:
        def __init__(self, code):
            self.status_code = code

    class _Api404:
        def dataset_status(self, dataset):
            raise requests.exceptions.HTTPError("not found", response=_Resp(404))

    class _Api500:
        def dataset_status(self, dataset):
            raise requests.exceptions.HTTPError("boom", response=_Resp(500))

    assert ka.dataset_exists(_Api404(), "mei28/x") is False
    with pytest.raises(requests.exceptions.HTTPError):
        ka.dataset_exists(_Api500(), "mei28/x")
    assert ka.dataset_exists(_Api(["ready"]), "mei28/x") is True
