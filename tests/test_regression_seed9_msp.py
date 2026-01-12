import json
import os
from pathlib import Path

import pytest


BASELINE = {
    "fpr_test": 0.27032690695725065,
    "roc_auc_test": 0.9256646961074102,
    "temperature": 0.7,
    "magnitude": 0.0,
    "n_cal": 1000,
}


def _find_latest_metrics(root: Path) -> Path | None:
    candidates = list(root.glob("**/seed-split-9/metrics.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def test_msp_seed9_regression():
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "results_main" / "msp" / "cifar10_resnet34"
    results_root = Path(os.environ.get("EE_RESULTS_ROOT", default_root))
    metrics_path = _find_latest_metrics(results_root)
    if metrics_path is None:
        pytest.skip(f"No seed-9 metrics.json under {results_root}")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    meta = payload.get("meta", {})
    test_metrics = payload.get("metrics", {}).get("test", {})

    assert config.get("temperature") == BASELINE["temperature"]
    assert config.get("magnitude") == BASELINE["magnitude"]
    assert meta.get("n_cal") == BASELINE["n_cal"]

    assert abs(test_metrics.get("fpr", 0.0) - BASELINE["fpr_test"]) <= 0.02
    assert abs(test_metrics.get("roc_auc", 0.0) - BASELINE["roc_auc_test"]) <= 0.005
