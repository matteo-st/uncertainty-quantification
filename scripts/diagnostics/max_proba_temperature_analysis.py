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


def _compute_scores(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    probs = torch.softmax(logits / temperature, dim=1)
    return -probs.max(dim=1).values


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
    edges = torch.quantile(scores, q)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temps = _parse_float_values(args.temperatures)
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
    for temperature in temps:
        scores = _compute_scores(logits, temperature)
        stats = _analyze_temperature(scores, args.k)
        rows.append(
            {
                "temperature": temperature,
                "unique_scores": stats["unique_scores"],
                "nonempty_bins": stats["nonempty_bins"],
                "max_bin_count": stats["max_bin_count"],
            }
        )

    csv_path = out_dir / "max_proba_temperature_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["temperature", "unique_scores", "nonempty_bins", "max_bin_count"])
        writer.writeheader()
        writer.writerows(rows)

    temps_sorted = [row["temperature"] for row in rows]
    unique_sorted = [row["unique_scores"] for row in rows]
    nonempty_sorted = [row["nonempty_bins"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(temps_sorted, unique_sorted, marker="o", lw=1.6, label="Unique scores")
    ax.plot(temps_sorted, nonempty_sorted, marker="s", lw=1.6, label=f"Non-empty bins (K={args.k})")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Count")
    ax.set_title("Max-proba ties vs temperature")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "max_proba_temperature_plot.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
