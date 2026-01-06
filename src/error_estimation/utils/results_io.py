from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRIC_BASE_KEYS = {
    "fpr",
    "tpr",
    "thr",
    "roc_auc",
    "accuracy",
    "aurc",
    "aupr_in",
    "aupr_out",
    "model_acc",
    "aupr_err",
    "aupr_success",
}


def _coerce_value(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _coerce_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_value(v) for v in value]
    return value


def build_run_meta(
    data_cfg: dict,
    model_cfg: dict,
    detection_cfg: dict,
    seed_split: int,
    n_cal: int | None = None,
    mode: str | None = None,
    run_tag: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        "dataset": data_cfg.get("name"),
        "model": model_cfg.get("model_name"),
        "preprocessor": model_cfg.get("preprocessor"),
        "postprocessor": detection_cfg.get("name"),
        "seed_split": seed_split,
        "n_res": data_cfg.get("n_samples", {}).get("res"),
        "n_cal": n_cal if n_cal is not None else data_cfg.get("n_samples", {}).get("cal"),
        "n_test": data_cfg.get("n_samples", {}).get("test"),
        "mode": mode,
        "run_tag": run_tag,
    }
    if extra:
        meta.update(extra)
    return meta


def select_best_row(results: pd.DataFrame, metric: str, direction: str) -> pd.Series:
    if results is None or results.empty:
        raise ValueError("No results available to select the best row.")

    candidates = [
        f"{metric}_cal",
        f"{metric}_val_cross",
        f"{metric}_test",
        metric,
    ]
    chosen_col = next((col for col in candidates if col in results.columns), None)
    if chosen_col is None:
        return results.iloc[0]

    values = pd.to_numeric(results[chosen_col], errors="coerce")
    if not values.notna().any():
        return results.iloc[0]
    idx = values.idxmin() if direction == "min" else values.idxmax()
    return results.loc[idx]


def _extract_metric_columns(row: dict[str, Any]) -> set[str]:
    metric_cols = set()
    for key in row:
        if "_" not in key:
            continue
        base, _suffix = key.rsplit("_", 1)
        if base in METRIC_BASE_KEYS:
            metric_cols.add(key)
    return metric_cols


def extract_metrics_by_split(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for key, value in row.items():
        if "_" not in key:
            continue
        base, suffix = key.rsplit("_", 1)
        if base not in METRIC_BASE_KEYS:
            continue
        metrics.setdefault(suffix, {})[base] = _coerce_value(value)
    return metrics


def build_metrics_payload(meta: dict[str, Any], row: pd.Series) -> dict[str, Any]:
    row_dict = {k: _coerce_value(v) for k, v in row.to_dict().items()}
    metric_cols = _extract_metric_columns(row_dict)
    metrics = extract_metrics_by_split(row_dict)
    config = {
        k: v for k, v in row_dict.items() if k not in metric_cols and k not in meta
    }
    return {
        "meta": meta,
        "config": config,
        "metrics": metrics,
    }


def write_metrics_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_summary_row(meta: dict[str, Any], row: pd.Series) -> dict[str, Any]:
    summary = {k: _coerce_value(v) for k, v in meta.items()}
    for key, value in row.to_dict().items():
        summary[key] = _coerce_value(value)
    return summary


def append_summary_csv(path: str | Path, row: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            header_line = f.readline().rstrip("\n")
        existing_cols = header_line.split(",") if header_line else []
        if existing_cols:
            df = df.reindex(columns=existing_cols + [c for c in df.columns if c not in existing_cols])
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)
    return path


def flatten_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for split, values in metrics.items():
        for key, value in values.items():
            if value is None:
                continue
            try:
                flat[f"{key}_{split}"] = float(value)
            except (TypeError, ValueError):
                continue
    return flat
