from __future__ import annotations

import argparse
from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot grid search results for binning.")
    parser.add_argument("--search-jsonl", required=True, help="Path to search.jsonl")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    parser.add_argument("--metric", default="roc_auc_res", help="Metric column to plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_path = Path(args.search_jsonl)
    output_dir = Path(args.output_dir) if args.output_dir else search_path.parent / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    df = _load_jsonl(search_path)
    metric = args.metric
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in {search_path}")

    df = df.copy()
    for key in ("n_clusters", "n_min"):
        if key in df.columns:
            df[key] = pd.to_numeric(df[key], errors="coerce")

    if "n_min" in df.columns:
        pivot = df.pivot(index="n_min", columns="n_clusters", values=metric)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        im = ax.imshow(pivot.values, origin="lower", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.astype(int))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(int))
        ax.set_xlabel("n_clusters")
        ax.set_ylabel("n_min")
        ax.set_title(metric)
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(output_dir / f"grid_{metric}.pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        df_sorted = df.sort_values("n_clusters")
        ax.plot(df_sorted["n_clusters"], df_sorted[metric], marker="o")
        ax.set_xlabel("n_clusters")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        fig.tight_layout()
        fig.savefig(output_dir / f"curve_{metric}.pdf", bbox_inches="tight")
        plt.close(fig)

    df.to_csv(output_dir / "search_grid.csv", index=False)


if __name__ == "__main__":
    main()
