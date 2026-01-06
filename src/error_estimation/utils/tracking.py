from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from contextlib import contextmanager

import mlflow


def _coerce_param(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return "null"
    return json.dumps(value, sort_keys=True)


def flatten_config(cfg: dict, prefix: str = "") -> dict[str, str | int | float | bool]:
    items: dict[str, str | int | float | bool] = {}
    for key, value in cfg.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            items.update(flatten_config(value, prefix=full_key))
        elif isinstance(value, (list, tuple)):
            if len(value) <= 10:
                items[full_key] = _coerce_param(value)
            else:
                items[full_key] = f"list(len={len(value)})"
        else:
            items[full_key] = _coerce_param(value)
    return items


@dataclass
class MLflowTracker:
    enabled: bool
    experiment_name: str
    tracking_uri: str | None = None
    run_name: str | None = None

    def __enter__(self) -> "MLflowTracker":
        if not self.enabled:
            self.run = None
            return self
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self.run is not None:
            mlflow.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_tags(self, tags: dict[str, Any]) -> None:
        if not self.enabled:
            return
        mlflow.set_tags({k: _coerce_param(v) for k, v in tags.items()})

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if not self.enabled:
            return
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, path: str | Path, artifact_path: str | None = None) -> None:
        if not self.enabled:
            return
        mlflow.log_artifacts(str(path), artifact_path=artifact_path)

    @contextmanager
    def child_run(self, run_name: str | None = None, tags: dict[str, Any] | None = None):
        if not self.enabled:
            yield None
            return
        with mlflow.start_run(run_name=run_name, nested=True):
            if tags:
                self.log_tags(tags)
            yield
