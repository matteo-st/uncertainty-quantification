from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch


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
            "legend.fontsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _parse_float_values(values: list[str]) -> list[float]:
    if len(values) == 1 and "," in values[0]:
        values = values[0].split(",")
    parsed = []
    for item in values:
        item = item.strip()
        if not item:
            continue
        parsed.append(float(item))
    if not parsed:
        raise ValueError("No temperatures provided")
    return parsed


def _compute_scores(logits: torch.Tensor, temperature: float, score: str) -> torch.Tensor:
    probs = torch.softmax(logits / temperature, dim=1)
    if score == "max_proba":
        return -probs.max(dim=1).values
    if score == "gini":
        g = torch.sum(probs**2, dim=1)
        return 1.0 - g
    if score == "margin":
        top2 = torch.topk(probs, k=2, dim=1).values
        return top2[:, 0] - top2[:, 1]
    raise ValueError(f"Unsupported score: {score}")


def _analyze_temperature(scores: torch.Tensor, k: int) -> dict[str, float]:
    unique_scores = int(torch.unique(scores).numel())
    if k <= 1:
        nonempty = int(scores.numel() > 0)
        return {
            "unique_scores": unique_scores,
            "nonempty_bins": nonempty,
            "max_bin_count": float(scores.numel()),
        }
    q = torch.linspace(0.0, 1.0, k + 1, device=scores.device)[1:-1]
    try:
        edges = torch.quantile(scores, q)
    except RuntimeError:
        edges = torch.quantile(scores.to(torch.float32), q).to(scores.dtype)
    bins = torch.bucketize(scores, edges)
    counts = torch.bincount(bins, minlength=k)
    nonempty = int((counts > 0).sum().item())
    max_count = float(counts.max().item()) if counts.numel() else 0.0
    return {
        "unique_scores": unique_scores,
        "nonempty_bins": nonempty,
        "max_bin_count": max_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze max-proba score ties vs temperature.")
    parser.add_argument("--latent-path", required=True, help="Path to latent .pt file with logits.")
    parser.add_argument("--output-dir", required=True, help="Output directory for CSV/plot.")
    parser.add_argument("--temperatures", nargs="+", required=True, help="Temperatures to evaluate.")
    parser.add_argument("--k", type=int, default=5000, help="Number of bins for quantile binning.")
    parser.add_argument("--n-samples", type=int, default=None, help="Optional subsample size.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for subsampling.")
    parser.add_argument("--scores", nargs="+", default=["max_proba"], help="Scores to analyze (max_proba/gini/margin).")
    parser.add_argument("--dtypes", nargs="+", default=["float32"], help="Dtypes to analyze.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temps = _parse_float_values(args.temperatures)
    scores = [s.strip() for s in args.scores]
    dtypes = [d.strip() for d in args.dtypes]
    latent_path = Path(args.latent_path)
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing latent file: {latent_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    pkg = torch.load(latent_path, map_location="cpu")
    logits = pkg["logits"].to(torch.float32)
    if args.n_samples is not None and args.n_samples < logits.shape[0]:
        gen = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(logits.shape[0], generator=gen)[: args.n_samples]
        logits = logits[idx]

    rows = []
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    for score_name in scores:
        for dtype_name in dtypes:
            if dtype_name not in dtype_map:
                raise ValueError(f"Unsupported dtype: {dtype_name}")
            dtype = dtype_map[dtype_name]
            eps = float(torch.finfo(dtype).eps)
            logits_cast = logits.to(dtype)
            for temperature in temps:
                score_vals = _compute_scores(logits_cast, float(temperature), score_name)
                stats = _analyze_temperature(score_vals, args.k)
                rows.append(
                    {
                        "score": score_name,
                        "dtype": dtype_name,
                        "eps": eps,
                        "temperature": float(temperature),
                        "unique_scores": stats["unique_scores"],
                        "nonempty_bins": stats["nonempty_bins"],
                        "max_bin_count": stats["max_bin_count"],
                    }
                )

    csv_path = out_dir / "max_proba_temperature_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "score",
                "dtype",
                "eps",
                "temperature",
                "unique_scores",
                "nonempty_bins",
                "max_bin_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for score_name in scores:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for dtype_name in dtypes:
            subset = [row for row in rows if row["score"] == score_name and row["dtype"] == dtype_name]
            subset.sort(key=lambda r: r["temperature"])
            temps_sorted = [row["temperature"] for row in subset]
            unique_sorted = [row["unique_scores"] for row in subset]
            nonempty_sorted = [row["nonempty_bins"] for row in subset]
            ax.plot(temps_sorted, unique_sorted, marker="o", lw=1.4, label=f"{dtype_name} unique")
            ax.plot(temps_sorted, nonempty_sorted, marker="s", lw=1.4, linestyle="--", label=f"{dtype_name} nonempty")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Count")
        ax.set_title(f"{score_name}: ties vs temperature")
        ax.legend(frameon=False, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"{score_name}_temperature_plot.png", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
