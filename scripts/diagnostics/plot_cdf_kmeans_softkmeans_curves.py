#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _to_list(val) -> list[float]:
    if isinstance(val, list):
        return [float(x) for x in val]
    if isinstance(val, str) and val.startswith("["):
        return [float(x) for x in ast.literal_eval(val)]
    return [float(val)]


def _load_grid(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("fpr_test", "roc_auc_test"):
        df[col] = df[col].apply(_to_list)
        df[f"{col}_mean"] = df[col].apply(lambda x: float(np.mean(x)))
    return df


def _aggregate_curves(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}
    mean_df = df[df["score"] == "mean"].groupby("n_clusters", as_index=False)[
        ["fpr_test_mean", "roc_auc_test_mean"]
    ].mean()
    curves["mean"] = mean_df

    for alpha in (0.05, 0.1, 0.5):
        sub = df[(df["score"] == "upper") & (df["alpha"] == alpha)]
        if sub.empty:
            continue
        curves[f"upper_{alpha:g}"] = sub.groupby("n_clusters", as_index=False)[
            ["fpr_test_mean", "roc_auc_test_mean"]
        ].mean()
    return curves


def _plot_curves(
    *,
    curves: dict[str, pd.DataFrame],
    metric: str,
    out_path: Path,
    title: str,
    y_label: str,
    raw_score: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    styles = {
        "mean": {"marker": "o", "color": "#4C78A8"},
        "upper_0.05": {"marker": "s", "color": "#F58518"},
        "upper_0.1": {"marker": "D", "color": "#54A24B"},
        "upper_0.5": {"marker": "^", "color": "#B279A2"},
    }
    labels = {
        "mean": "mean",
        "upper_0.05": "upper (alpha=0.05)",
        "upper_0.1": "upper (alpha=0.1)",
        "upper_0.5": "upper (alpha=0.5)",
    }
    order = ["mean", "upper_0.05", "upper_0.1", "upper_0.5"]
    for key in order:
        if key not in curves:
            continue
        data = curves[key].sort_values("n_clusters")
        ax.plot(
            data["n_clusters"],
            data[metric],
            label=labels.get(key, key),
            **styles.get(key, {}),
        )
    if raw_score is not None:
        ax.axhline(raw_score, color="red", lw=1.8, label="raw score")
    ax.set_xlabel("n_clusters (K)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _load_raw_metrics(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    test = metrics.get("test", {})
    fpr = test.get("fpr")
    roc = test.get("roc_auc")
    if fpr is None or roc is None:
        return None, None
    return float(fpr), float(roc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot K vs performance for CDF kmeans/softkmeans grids.")
    parser.add_argument("--kmeans-grid", required=True)
    parser.add_argument("--softkmeans-grid", required=True)
    parser.add_argument("--raw-metrics", required=True, help="metrics.json from the raw score selection")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="cdf_kmeans_softkmeans")
    args = parser.parse_args()

    _apply_style()

    out_dir = Path(args.output_dir)
    raw_fpr, raw_roc = _load_raw_metrics(Path(args.raw_metrics))

    grids = {
        "kmeans": _load_grid(Path(args.kmeans_grid)),
        "softkmeans": _load_grid(Path(args.softkmeans_grid)),
    }
    for name, grid in grids.items():
        curves = _aggregate_curves(grid)
        _plot_curves(
            curves=curves,
            metric="roc_auc_test_mean",
            out_path=out_dir / f"{args.output_prefix}_{name}_roc_auc.png",
            title=f"{name} + CDF: ROC-AUC vs K (seed 9, mean over inits)",
            y_label="ROC-AUC (test)",
            raw_score=raw_roc,
        )
        _plot_curves(
            curves=curves,
            metric="fpr_test_mean",
            out_path=out_dir / f"{args.output_prefix}_{name}_fpr.png",
            title=f"{name} + CDF: FPR@95 vs K (seed 9, mean over inits)",
            y_label="FPR@95 (test)",
            raw_score=raw_fpr,
        )


if __name__ == "__main__":
    main()
