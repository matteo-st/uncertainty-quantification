"""
Simple Reliability Diagram showing Raw Score vs Conservative Calibrated.

Creates a figure comparing two reliability diagrams:
1. Raw score: X = mean raw score s̄(b), shows miscalibration
2. Conservative (upper): X = upper bound û(b), shows conservative calibration

Usage:
    python scripts/analysis/simple_reliability_diagram.py \
        --conservative-json results/analysis/reliability_diagram/imagenet_timm_vit_base16_doctor/reliability_stats.json \
        --raw-json results/analysis/reliability_diagram/imagenet_timm_vit_base16_doctor_raw/raw_score_reliability_stats.json \
        --output-dir results/analysis/reliability_diagram/simple_imagenet_vit_base16_doctor \
        --score-type doctor
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _apply_style() -> None:
    """Set publication-quality plot style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })


def load_json_stats(path: Path) -> dict:
    """Load JSON statistics file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_conservative_data(stats: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract data from conservative (upper) reliability stats."""
    per_bin = stats["per_bin"]
    x = np.array([b["upper_bound"] for b in per_bin])
    y = np.array([b["test_error"] for b in per_bin])
    counts = np.array([b["count_test"] for b in per_bin])
    return x, y, counts


def extract_raw_data(stats: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract data from raw score reliability stats."""
    per_bin = stats["per_bin"]
    x = np.array([b["mean_predicted_score"] for b in per_bin])
    y = np.array([b["empirical_error_rate"] for b in per_bin])
    counts = np.array([b["count"] for b in per_bin])
    return x, y, counts


def get_score_label(score_type: str) -> str:
    """Get display label for score type."""
    labels = {
        "msp": "MSP",
        "doctor": "Doctor",
        "margin": "Margin",
    }
    return labels.get(score_type, score_type.upper())


def create_simple_plot(
    conservative_data: tuple,
    raw_data: tuple,
    output_dir: Path,
    marker_size: int = 40,
    score_type: str = "msp",
) -> dict:
    """
    Create simple reliability diagram with raw and conservative only.
    """
    _apply_style()
    score_label = get_score_label(score_type)

    # Unpack data
    x_cons, y_cons, counts_cons = conservative_data
    x_raw, y_raw, counts_raw = raw_data

    # Filter to non-empty bins
    mask_cons = counts_cons > 0
    mask_raw = counts_raw > 0

    # Compute violation rates
    n_bins_cons = int(np.sum(mask_cons))
    n_bins_raw = int(np.sum(mask_raw))

    violations_cons = int(np.sum(y_cons[mask_cons] > x_cons[mask_cons]))
    violations_raw = int(np.sum(y_raw[mask_raw] > x_raw[mask_raw]))

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5.5))

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, zorder=1, label="Perfect calibration")

    # Plot raw score - orange triangles
    ax.scatter(
        x_raw[mask_raw],
        y_raw[mask_raw],
        s=marker_size,
        c="tab:orange",
        marker="^",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
        label=f"{score_label} score ({violations_raw}/{n_bins_raw} violations)",
        zorder=2,
    )

    # Plot conservatively calibrated - blue circles
    ax.scatter(
        x_cons[mask_cons],
        y_cons[mask_cons],
        s=marker_size,
        c="tab:blue",
        marker="o",
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
        label=f"Conservatively calibrated ({violations_cons}/{n_bins_cons} violations)",
        zorder=3,
    )

    # Labels
    ax.set_xlabel(r"Predicted error probability")
    ax.set_ylabel(r"Empirical test error rate")

    # Limits
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")

    # Legend
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.tight_layout()

    # Save figures
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "simple_reliability_diagram.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "simple_reliability_diagram.png", bbox_inches="tight")
    plt.close(fig)

    # Compute ECE for raw score
    weights_raw = counts_raw[mask_raw] / counts_raw[mask_raw].sum()
    ece_raw = np.sum(weights_raw * np.abs(y_raw[mask_raw] - x_raw[mask_raw]))

    summary = {
        "score_type": score_type,
        "score_label": score_label,
        "conservative": {
            "n_bins": n_bins_cons,
            "n_violations": violations_cons,
        },
        "raw_score": {
            "n_bins": n_bins_raw,
            "n_violations": violations_raw,
            "ece": float(ece_raw),
        },
    }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple reliability diagram: raw vs conservative."
    )
    parser.add_argument(
        "--conservative-json",
        type=str,
        required=True,
        help="Path to conservative (upper) reliability stats JSON.",
    )
    parser.add_argument(
        "--raw-json",
        type=str,
        required=True,
        help="Path to raw score reliability stats JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for figure.",
    )
    parser.add_argument(
        "--marker-size",
        type=int,
        default=40,
        help="Marker size for scatter points (default: 40).",
    )
    parser.add_argument(
        "--score-type",
        type=str,
        choices=["msp", "doctor", "margin"],
        default="msp",
        help="Base uncertainty score type (default: msp).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    conservative_path = Path(args.conservative_json)
    raw_path = Path(args.raw_json)
    output_dir = Path(args.output_dir)

    print(f"Loading conservative stats from: {conservative_path}")
    conservative_stats = load_json_stats(conservative_path)

    print(f"Loading raw stats from: {raw_path}")
    raw_stats = load_json_stats(raw_path)

    # Extract data
    conservative_data = extract_conservative_data(conservative_stats)
    raw_data = extract_raw_data(raw_stats)

    print(f"\nData extracted:")
    print(f"  Conservative: {len(conservative_data[0])} bins")
    print(f"  Raw: {len(raw_data[0])} bins")

    # Create plot
    summary = create_simple_plot(
        conservative_data,
        raw_data,
        output_dir,
        marker_size=args.marker_size,
        score_type=args.score_type,
    )

    print(f"\nSaved figure to: {output_dir / 'simple_reliability_diagram.pdf'}")

    # Print summary
    score_label = summary["score_label"]
    print(f"\n=== Summary ({score_label}) ===")
    print(f"Conservatively calibrated: {summary['conservative']['n_violations']}/{summary['conservative']['n_bins']} violations")
    print(f"{score_label} score: {summary['raw_score']['n_violations']}/{summary['raw_score']['n_bins']} violations (ECE={summary['raw_score']['ece']:.4f})")

    # Save summary
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
