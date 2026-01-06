import json

import pandas as pd

from error_estimation.utils.results_io import (
    append_summary_csv,
    build_metrics_payload,
    build_run_meta,
    build_summary_row,
    flatten_metrics,
    select_best_row,
    write_metrics_json,
)


def test_results_io_metrics_and_summary(tmp_path):
    results = pd.DataFrame(
        [
            {"temperature": 1.0, "fpr_cal": 0.4, "fpr_test": 0.35, "accuracy_cal": 0.7, "accuracy_test": 0.72},
            {"temperature": 2.0, "fpr_cal": 0.5, "fpr_test": 0.45, "accuracy_cal": 0.69, "accuracy_test": 0.71},
        ]
    )
    best_row = select_best_row(results, "fpr", "min")
    meta = build_run_meta(
        data_cfg={"name": "cifar10", "n_samples": {"res": 0, "cal": 5000, "test": 5000}},
        model_cfg={"model_name": "resnet34", "preprocessor": "ce"},
        detection_cfg={"name": "doctor"},
        seed_split=9,
        n_cal=5000,
        mode="search",
        run_tag="unit-test",
    )
    payload = build_metrics_payload(meta, best_row)
    metrics_path = write_metrics_json(tmp_path / "metrics.json", payload)
    assert metrics_path.exists()

    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert loaded["metrics"]["cal"]["fpr"] == 0.4
    assert loaded["metrics"]["test"]["fpr"] == 0.35

    summary_path = append_summary_csv(tmp_path / "summary.csv", build_summary_row(meta, best_row))
    assert summary_path.exists()
    summary_df = pd.read_csv(summary_path)
    assert "fpr_cal" in summary_df.columns
    assert summary_df.iloc[0]["seed_split"] == 9

    flat = flatten_metrics(payload["metrics"])
    assert flat["fpr_cal"] == 0.4
