"""Shared helpers for the Kaggle Datasets that carry code and models into a competition kernel."""

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import requests

TITLE_MAX = 50
LICENSE = [{"name": "CC0-1.0"}]


def slugify(text: str) -> str:
    """Kaggle slugs: lowercase letters, digits, hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def dataset_metadata(user: str, title: str) -> dict[str, Any]:
    if len(title) > TITLE_MAX:
        raise ValueError(f"dataset title must be at most {TITLE_MAX} characters, got {len(title)}: {title!r}")
    return {"id": f"{user}/{slugify(title)}", "title": title, "licenses": LICENSE}


def write_dataset_metadata(folder: Path, user: str, title: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "dataset-metadata.json"
    path.write_text(json.dumps(dataset_metadata(user, title), indent=2) + "\n")
    return path


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)


def wait_ready(api: Any, dataset: str, interval: float) -> None:
    """Poll `dataset_status` until the latest version is ready; Kaggle attaches only ready versions to kernels."""
    while True:
        status = api.dataset_status(dataset)
        if status == "ready":
            return
        if status in {"failed", "deleted"}:
            raise RuntimeError(f"dataset {dataset} is {status}")
        time.sleep(interval)


def dataset_exists(api: Any, dataset: str) -> bool:
    try:
        api.dataset_status(dataset)
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return False
        raise
    return True


def create_or_version(api: Any, folder: Path, dataset: str, notes: str) -> None:
    if dataset_exists(api, dataset):
        api.dataset_create_version(str(folder), version_notes=notes, dir_mode="zip", convert_to_csv=False, quiet=True)
    else:
        api.dataset_create_new(str(folder), public=False, dir_mode="zip", convert_to_csv=False, quiet=True)
