"""Experiment trackers behind one small interface. metrics.json is the record; a tracker is optional."""

from typing import Any, Protocol


class Tracker(Protocol):
    def start(self, run_name: str, config: dict[str, Any]) -> None: ...

    def log(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    def finish(self) -> None: ...


class NoneTracker:
    def __init__(self, project: str) -> None:
        self.project = project

    def start(self, run_name: str, config: dict[str, Any]) -> None:
        return None

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        return None

    def finish(self) -> None:
        return None


_BACKENDS: dict[str, type] = {"none": NoneTracker}


def make_tracker(name: str, project: str, **kwargs: Any) -> Tracker:
    if name not in _BACKENDS:
        raise KeyError(f"unknown tracker {name!r}; known: {sorted(_BACKENDS)}")
    return _BACKENDS[name](project=project, **kwargs)
