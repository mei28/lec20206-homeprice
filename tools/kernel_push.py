"""Render the thin submission notebook, export it to ipynb, write kernel-metadata.json, and push it.

Usage: uv run python tools/kernel_push.py exp000_baseline [--deps titanic-deps] [--gpu]
Requires the codes and model Datasets to exist and be ready (see upload_codes.py, upload_model.py).
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import tyro
from tyro.conf import Positional

from tools import kaggle_assets as ka
from tools.upload_model import default_title

VERSION_RE = re.compile(r"Kernel version (\d+)")


def render(template: str, *, codes: str, model: str, exp: str, deps: str) -> str:
    return Template(template).substitute(CODES=codes, MODEL=model, EXP=exp, DEPS=deps)


def kernel_metadata(
    user: str, comp: str, *, codes: str, model: str, deps: str | None = None, gpu: bool = False
) -> dict[str, Any]:
    title = f"{comp}-sub"
    if len(title) > ka.TITLE_MAX:
        raise ValueError(f"kernel title must be at most {ka.TITLE_MAX} characters, got {len(title)}: {title!r}")
    sources = [f"{user}/{codes}", f"{user}/{model}"] + ([f"{user}/{deps}"] if deps else [])
    return {
        "id": f"{user}/{ka.slugify(title)}",
        "title": title,
        "code_file": "sub.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": sources,
        "competition_sources": [comp],
        "kernel_sources": [],
        "model_sources": [],
    }


def parse_version(push_output: str) -> int:
    match = VERSION_RE.search(push_output)
    if match is None:
        raise RuntimeError(f"could not find the kernel version in the push output:\n{push_output}")
    return int(match.group(1))


@dataclass
class Args:
    exp: Positional[str]
    run: str = "default"
    deps: str | None = None
    gpu: bool = False
    model_title: str | None = None
    comp: str = os.environ.get("COMP", "")
    user: str = os.environ.get("KAGGLE_USERNAME", "")


def main(args: Args) -> None:
    if not args.comp or not args.user:
        raise SystemExit("COMP and KAGGLE_USERNAME must be set (see .env)")
    root = Path(__file__).resolve().parents[1]
    sub = root / "sub"
    codes = ka.slugify(f"{args.comp}-codes")
    model = ka.slugify(args.model_title or default_title(args.comp, args.exp))

    (sub / "sub.py").write_text(
        render((sub / "sub.template.py").read_text(), codes=codes, model=model, exp=args.exp, deps=args.deps or "")
    )
    subprocess.run(
        [
            "uv",
            "run",
            "marimo",
            "export",
            "ipynb",
            str(sub / "sub.py"),
            "-o",
            str(sub / "sub.ipynb"),
            "--sort",
            "top-down",
            "-f",
        ],
        check=True,
        cwd=root,
    )
    meta = kernel_metadata(args.user, args.comp, codes=codes, model=model, deps=args.deps, gpu=args.gpu)
    (sub / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    result = subprocess.run(
        ["uv", "run", "kaggle", "kernels", "push", "-p", str(sub)], capture_output=True, text=True, cwd=root
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        raise RuntimeError(f"kernels push failed:\n{result.stderr}")
    version = parse_version(result.stdout)
    (sub / ".last_version").write_text(f"{version}\n")
    print(f"pushed {meta['id']} version {version}; check with: just kernel-status")


if __name__ == "__main__":
    main(tyro.cli(Args))
