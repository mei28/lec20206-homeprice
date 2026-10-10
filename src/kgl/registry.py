"""Resolve classes and functions by name so a Config can name its parts as strings.

Two namespaces: dotted import paths for library objects (`"lightgbm.LGBMClassifier"`) and a
registry of experiment-local functions filled by the `@register(name)` decorator.
"""

import importlib
from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, Callable[..., Any]] = {}


def resolve(path: str) -> Any:
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"expected a dotted path like 'lightgbm.LGBMClassifier', got {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get(name: str) -> Callable[..., Any]:
    if name not in _REGISTRY:
        raise KeyError(f"{name!r} is not registered; known names: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def build(spec: str, **params: Any) -> Any:
    return resolve(spec)(**params)
