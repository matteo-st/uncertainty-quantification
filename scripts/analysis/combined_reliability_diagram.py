"""
Combined Reliability Diagram for ICML Section 4 (Corollary 4.1).

Creates a single figure comparing three reliability diagrams:
1. Conservative (upper): X = upper bound û(b), shows conservative calibration (points below diagonal)
2. Calibrated (mean): X = mean calibration p̂_cal(b), shows calibration (points on diagonal)
3. Raw score: X = mean raw score s̄(b), shows miscalibration (points scattered)

All methods share the same Y-axis: empirical test error rate p̂_test(b).

Supports different base uncertainty scores: MSP, Doctor (Gini), and Margin.

Usage:
    python scripts/analysis/combined_reliability_diagram.py \
        --conservative-json results/analysis/reliability_diagram/imagenet_vit_base16/reliability_stats.json \
        --mean-json results/analysis/reliability_diagram/imagenet_vit_base16_mean/calibrated_mean_reliability_stats.json \
        --raw-json results/analysis/reliability_diagram/imagenet_vit_base16_raw/raw_score_reliability_stats.json \
        --output-dir results/analysis/reliability_diagram/combined \
        --score-type msp

    # For Doctor score:
    python scripts/analysis/combined_reliability_diagram.py \
        --conservative-json results/analysis/reliability_diagram/imagenet_vit_base16_doctor/reliability_stats.json \
        --mean-json results/analysis/reliability_diagram/imagenet_vit_base16_doctor_mean/calibrated_mean_reliability_stats.json \
        --raw-json results/analysis/reliability_diagram/imagenet_vit_base16_doctor_raw/raw_score_reliability_stats.json \
        --output-dir results/analysis/reliability_diagram/combined_doctor \
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
        "legend.fontsize": 9,
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
    """
    Extract data from conservative (upper) reliability stats.

    Returns:
        (x_values, y_values, counts) where:
        - x_values: upper bound û(b)
        - y_values: test error rate p̂_test(b)
        - counts: number of test samples per bin
    """
    per_bin = stats["per_bin"]
    x = np.array([b["upper_bound"] for b in per_bin])
    y = np.array([b["test_error"] for b in per_bin])
    counts = np.array([b["count_test"] for b in per_bin])
    return x, y, counts


def extract_mean_data(stats: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract data from calibrated mean reliability stats.

    Returns:
        (x_values, y_values, counts) where:
        - x_values: mean calibration error p̂_cal(b)
        - y_values: test error rate p̂_test(b)
        - counts: number of test samples per bin
    """
    per_bin = stats["per_bin"]
    x = np.array([b["mean_cal"] for b in per_bin])
    y = np.array([b["test_error"] for b in per_bin])
    counts = np.array([b["count_test"] for b in per_bin])
    return x, y, counts


def extract_raw_data(stats: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract data from raw MSP reliability stats.

    Returns:
        (x_values, y_values, counts) where:
        - x_values: mean predicted score s̄(b)
        - y_values: empirical error rate
        - counts: number of samples per bin
    """
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


def create_combined_plot(
    conservative_data: tuple,
    mean_data: tuple,
    raw_data: tuple,
    output_dir: Path,
    marker_size: int = 30,
    score_type: str = "msp",
) -> dict:
    """
    Create combined reliability diagram.

    Args:
        conservative_data: (x, y, counts) for conservative upper bound
        mean_data: (x, y, counts) for calibrated mean
        raw_data: (x, y, counts) for raw score
        output_dir: Directory to save outputs
        marker_size: Base marker size (default 30)
        score_type: Type of base uncertainty score ('msp', 'doctor', 'margin')

    Returns:
        Summary statistics dictionary
    """
    _apply_style()
    score_label = get_score_label(score_type)

    # Unpack data
    x_cons, y_cons, counts_cons = conservative_data
    x_mean, y_mean, counts_mean = mean_data
    x_raw, y_raw, counts_raw = raw_data

    # Filter to non-empty bins
    mask_cons = counts_cons > 0
    mask_mean = counts_mean > 0
    mask_raw = counts_raw > 0

    # Compute violation rates (y > x means test error exceeds predicted)
    n_bins_cons = int(np.sum(mask_cons))
    n_bins_mean = int(np.sum(mask_mean))
    n_bins_raw = int(np.sum(mask_raw))

    violations_cons = int(np.sum(y_cons[mask_cons] > x_cons[mask_cons]))
    violations_mean = int(np.sum(y_mean[mask_mean] > x_mean[mask_mean]))
    violations_raw = int(np.sum(y_raw[mask_raw] > x_raw[mask_raw]))

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5.5))

    # Diagonal reference line (perfect calibration / conservative boundary)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, zorder=1)

    # Plot raw score first (background) - orange triangles
    ax.scatter(
        x_raw[mask_raw],
        y_raw[mask_raw],
        s=marker_size,
        c="tab:orange",
        marker="^",
        alpha=0.6,
        edgecolors="white",
        linewidths=0.3,
        label=f"{score_label} ({violations_raw}/{n_bins_raw} violations)",
        zorder=2,
    )

    # Plot calibrated - green diamonds
    ax.scatter(
        x_mean[mask_mean],
        y_mean[mask_mean],
        s=marker_size,
        c="tab:green",
        marker="D",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.3,
        label=f"Calibrated ({violations_mean}/{n_bins_mean} violations)",
        zorder=3,
    )

    # Plot conservatively calibrated - blue circles
    ax.scatter(
        x_cons[mask_cons],
        y_cons[mask_cons],
        s=marker_size,
        c="tab:blue",
        marker="o",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.3,
        label=f"Conservatively calibrated ({violations_cons}/{n_bins_cons} violations)",
        zorder=4,
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
    fig.savefig(output_dir / "combined_reliability_diagram.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "combined_reliability_diagram.png", bbox_inches="tight")
    plt.close(fig)

    # Compute summary statistics
    # Mean: compute ECE
    weights_mean = counts_mean[mask_mean] / counts_mean[mask_mean].sum()
    ece_mean = np.sum(weights_mean * np.abs(y_mean[mask_mean] - x_mean[mask_mean]))

    # Raw: compute ECE
    weights_raw = counts_raw[mask_raw] / counts_raw[mask_raw].sum()
    ece_raw = np.sum(weights_raw * np.abs(y_raw[mask_raw] - x_raw[mask_raw]))

    summary = {
        "score_type": score_type,
        "score_label": score_label,
        "conservative": {
            "n_bins": n_bins_cons,
            "n_violations": violations_cons,
        },
        "calibrated_mean": {
            "n_bins": n_bins_mean,
            "n_violations": violations_mean,
            "ece": float(ece_mean),
        },
        "raw_score": {
            "n_bins": n_bins_raw,
            "n_violations": violations_raw,
            "ece": float(ece_raw),
        },
    }

    return summary


def write_caption(output_dir: Path, summary: dict) -> None:
    """Write LaTeX figure caption to file."""
    score_label = summary.get("score_label", "MSP")
    caption = r"""\textbf{Combined Reliability Diagram (%s Score).}
Comparison of three calibration approaches on ImageNet with ViT-Base16 (50 bins).
\textbf{Blue circles}: Conservatively calibrated using upper bound $\hat{u}(b)$; all points lie below the diagonal ($y=x$), demonstrating the guarantee from Corollary~4.1 with %d/%d violations.
\textbf{Green diamonds}: Calibrated using mean $\hat{p}_{\mathrm{cal}}(b)$; points cluster around the diagonal (ECE=%.4f) with %d/%d violations, showing standard calibration without conservative guarantees.
\textbf{Orange triangles}: %s score $\bar{s}(b)$; points deviate substantially from the diagonal (ECE=%.4f) with %d/%d violations, demonstrating that %s is not calibrated as an error probability estimator.
The Y-axis shows the empirical test error rate for all methods.""" % (
        score_label,
        summary["conservative"]["n_violations"],
        summary["conservative"]["n_bins"],
        summary["calibrated_mean"]["ece"],
        summary["calibrated_mean"]["n_violations"],
        summary["calibrated_mean"]["n_bins"],
        score_label,
        summary["raw_score"]["ece"],
        summary["raw_score"]["n_violations"],
        summary["raw_score"]["n_bins"],
        score_label,
    )

    caption_path = output_dir / "caption.txt"
    caption_path.write_text(caption, encoding="utf-8")
    print(f"Saved caption to: {caption_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined reliability diagram for ICML Section 4."
    )
    parser.add_argument(
        "--conservative-json",
        type=str,
        default="results/analysis/reliability_diagram/imagenet_vit_base16/reliability_stats.json",
        help="Path to conservative (upper) reliability stats JSON.",
    )
    parser.add_argument(
        "--mean-json",
        type=str,
        default="results/analysis/reliability_diagram/imagenet_vit_base16_mean/calibrated_mean_reliability_stats.json",
        help="Path to calibrated mean reliability stats JSON.",
    )
    parser.add_argument(
        "--raw-json",
        type=str,
        default="results/analysis/reliability_diagram/imagenet_vit_base16_raw/raw_score_reliability_stats.json",
        help="Path to raw MSP reliability stats JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/analysis/reliability_diagram/combined",
        help="Output directory for combined figure and caption.",
    )
    parser.add_argument(
        "--marker-size",
        type=int,
        default=30,
        help="Marker size for scatter points (default: 30).",
    )
    parser.add_argument(
        "--score-type",
        type=str,
        choices=["msp", "doctor", "margin"],
        default="msp",
        help="Base uncertainty score type: 'msp', 'doctor', or 'margin' (default: msp).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load JSON files
    conservative_path = Path(args.conservative_json)
    mean_path = Path(args.mean_json)
    raw_path = Path(args.raw_json)
    output_dir = Path(args.output_dir)

    print(f"Loading conservative stats from: {conservative_path}")
    conservative_stats = load_json_stats(conservative_path)

    print(f"Loading mean stats from: {mean_path}")
    mean_stats = load_json_stats(mean_path)

    print(f"Loading raw stats from: {raw_path}")
    raw_stats = load_json_stats(raw_path)

    # Extract data
    conservative_data = extract_conservative_data(conservative_stats)
    mean_data = extract_mean_data(mean_stats)
    raw_data = extract_raw_data(raw_stats)

    print(f"\nData extracted:")
    print(f"  Conservative: {len(conservative_data[0])} bins")
    print(f"  Mean: {len(mean_data[0])} bins")
    print(f"  Raw: {len(raw_data[0])} bins")

    # Create combined plot
    summary = create_combined_plot(
        conservative_data,
        mean_data,
        raw_data,
        output_dir,
        marker_size=args.marker_size,
        score_type=args.score_type,
    )

    print(f"\nSaved figure to: {output_dir / 'combined_reliability_diagram.pdf'}")

    # Print summary
    score_label = summary.get("score_label", args.score_type.upper())
    print(f"\n=== Summary Statistics ({score_label}) ===")
    print(f"Conservatively calibrated:")
    print(f"  Bins: {summary['conservative']['n_bins']}")
    print(f"  Violations: {summary['conservative']['n_violations']}/{summary['conservative']['n_bins']}")
    print(f"Calibrated:")
    print(f"  Bins: {summary['calibrated_mean']['n_bins']}")
    print(f"  Violations: {summary['calibrated_mean']['n_violations']}/{summary['calibrated_mean']['n_bins']}")
    print(f"  ECE: {summary['calibrated_mean']['ece']:.4f}")
    print(f"{score_label}:")
    print(f"  Bins: {summary['raw_score']['n_bins']}")
    print(f"  Violations: {summary['raw_score']['n_violations']}/{summary['raw_score']['n_bins']}")
    print(f"  ECE: {summary['raw_score']['ece']:.4f}")

    # Write caption
    write_caption(output_dir, summary)

    # Save summary JSON
    summary_path = output_dir / "combined_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
