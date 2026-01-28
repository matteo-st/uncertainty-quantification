"""
Reliability Diagrams for Uncertainty Calibration.

Three modes:
1. `calibrated` (default): Conservative calibration upper reliability diagram
   - X-axis: Upper bound û_b from conservative calibration (Hoeffding + Bonferroni)
   - Y-axis: Empirical error rate p̂_b on test set
   - Reference: Diagonal y=x (points should be below if calibration is conservative)

2. `calibrated-mean`: Calibrated mean reliability diagram (cf. Corollary 4.1)
   - X-axis: Mean calibration error p̂_cal(b) per bin from cal set
   - Y-axis: Empirical error rate p̂_test(b) on test set
   - Reference: Diagonal y=x (points should lie ON diagonal if calibrated)
   - Shows that uniform-mass binning produces a calibrated score

3. `raw-score`: Raw MSP score reliability diagram
   - X-axis: Mean transformed MSP score per bin (predicted error probability)
   - Y-axis: Empirical error rate per bin (actual error probability)
   - Reference: Diagonal y=x (perfect calibration)
   - Shows that raw MSP is miscalibrated as an error probability estimator

Usage (calibrated mode):
    python scripts/analysis/upper_reliability_diagram.py \
        --run-dir results/partition_binning/imagenet/timm_vit_base16_ce/partition/runs/msp-unif-mass-sim-grid-20260120/seed-split-1 \
        --latent-dir latent/imagenet_timm_vit_base16_ce/transform-test_n-epochs-1 \
        --n-clusters 50 \
        --alpha 0.05 \
        --output-dir results/analysis/reliability_diagram/imagenet_vit_base16

Usage (calibrated-mean mode):
    python scripts/analysis/upper_reliability_diagram.py \
        --run-dir results/partition_binning/imagenet/timm_vit_base16_ce/partition/runs/msp-unif-mass-sim-grid-20260120/seed-split-1 \
        --latent-dir latent/imagenet_timm_vit_base16_ce/transform-test_n-epochs-1 \
        --n-clusters 50 \
        --mode calibrated-mean \
        --output-dir results/analysis/reliability_diagram/imagenet_vit_base16_mean

Usage (raw-score mode):
    python scripts/analysis/upper_reliability_diagram.py \
        --run-dir results/partition_binning/imagenet/timm_vit_base16_ce/partition/runs/msp-unif-mass-sim-grid-20260120/seed-split-1 \
        --latent-dir latent/imagenet_timm_vit_base16_ce/transform-test_n-epochs-1 \
        --n-clusters 50 \
        --mode raw-score \
        --output-dir results/analysis/reliability_diagram/imagenet_vit_base16_raw
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.stats
import torch
import matplotlib.pyplot as plt
import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_stats(run_dir: Path, n_clusters: int | None) -> Tuple[Path, dict]:
    """Load partition cluster stats from run directory."""
    if n_clusters is not None:
        stats_path = run_dir / f"partition_cluster_stats_n-clusters-{n_clusters}.pt"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing {stats_path}")
    else:
        candidates = sorted(
            run_dir.glob("partition_cluster_stats_n-clusters-*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(f"No partition_cluster_stats_n-clusters-*.pt in {run_dir}")
        stats_path = candidates[-1]
    stats = torch.load(stats_path, map_location="cpu")
    return stats_path, stats


def _load_clusters_test(run_dir: Path, n_clusters: int | None) -> Tuple[Path, np.ndarray]:
    """Load test cluster assignments."""
    if n_clusters is not None:
        clusters_path = run_dir / f"clusters_test_n-clusters-{n_clusters}.pt"
        if not clusters_path.exists():
            raise FileNotFoundError(f"Missing {clusters_path}")
    else:
        candidates = sorted(
            run_dir.glob("clusters_test_n-clusters-*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(f"No clusters_test_n-clusters-*.pt in {run_dir}")
        clusters_path = candidates[-1]
    clusters = torch.load(clusters_path, map_location="cpu").view(-1).numpy().astype(int)
    return clusters_path, clusters


def _resolve_count(value, total: int, name: str) -> int:
    """Resolve count from float ratio or int."""
    if isinstance(value, float):
        if not (0 < value <= 1):
            raise ValueError(f"{name} ratio must be in (0,1], got {value}")
        return int(round(total * value))
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} count must be >= 0, got {value}")
        return value
    raise TypeError(f"{name} must be float or int, got {type(value).__name__}")


def _build_test_indices(n_total: int, n_res, n_cal, n_test, seed_split: int) -> np.ndarray:
    """Build test split indices using the same logic as experiment."""
    perm = list(range(n_total))
    if seed_split is not None:
        rng = random.Random(seed_split)
        rng.shuffle(perm)
    n_test_samples = _resolve_count(n_test, n_total, "n_test")
    test_idx = perm[n_total - n_test_samples:]
    return np.asarray(test_idx, dtype=int)


def compute_hoeffding_upper(means: np.ndarray, counts: np.ndarray, alpha: float, simultaneous: bool = True) -> np.ndarray:
    """
    Compute Hoeffding upper bounds with optional Bonferroni correction.

    Args:
        means: Per-bin calibration error means
        counts: Per-bin sample counts
        alpha: Confidence level (e.g., 0.05)
        simultaneous: If True, apply Bonferroni correction (alpha' = alpha/K)

    Returns:
        Upper bounds for each bin
    """
    K = len(means)
    alpha_eff = alpha / K if simultaneous else alpha

    # Hoeffding half-width: sqrt(log(2/alpha) / (2n))
    half_widths = np.sqrt(np.log(2.0 / alpha_eff) / (2.0 * np.maximum(counts, 1)))

    # Upper bound, clipped to [0, 1]
    uppers = np.minimum(1.0, means + half_widths)
    return uppers


def clopper_pearson_interval(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute Clopper-Pearson exact binomial confidence interval.

    Args:
        successes: Number of successes (errors)
        n: Total number of trials
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (lower, upper) bounds for the proportion
    """
    if n == 0:
        return (0.0, 1.0)

    # Lower bound: beta distribution quantile
    if successes == 0:
        lower = 0.0
    else:
        lower = scipy.stats.beta.ppf(alpha / 2, successes, n - successes + 1)

    # Upper bound: beta distribution quantile
    if successes == n:
        upper = 1.0
    else:
        upper = scipy.stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes)

    return (lower, upper)


def _serialize(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def compute_msp_score(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """
    Compute MSP (Maximum Softmax Probability) score.

    Args:
        logits: Model logits, shape (N, C)
        temperature: Temperature scaling (default: 1.0)

    Returns:
        MSP scores in range [-1, 0], where -1 is most confident
    """
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    msp = probs.max(dim=-1).values
    # MSP score convention: negate so higher uncertainty = higher score
    return -msp.numpy()


def transform_msp_to_error_prob(msp_scores: np.ndarray) -> np.ndarray:
    """
    Transform MSP scores from [-1, 0] to [0, 1] for interpretation as predicted error probability.

    Args:
        msp_scores: MSP scores in range [-1, 0]

    Returns:
        Transformed scores in range [0, 1] where 0=confident, 1=uncertain
    """
    # MSP scores are in [-1, 0], transform to [0, 1]
    # -1 (most confident) -> 0 (low predicted error)
    # 0 (least confident) -> 1 (high predicted error)
    return 1.0 + msp_scores


def uniform_mass_binning(scores: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign samples to bins with approximately equal mass (uniform-mass binning).

    Args:
        scores: Score values to bin
        n_bins: Number of bins

    Returns:
        (bin_assignments, bin_edges) where bin_assignments[i] is the bin index for sample i
    """
    n_samples = len(scores)
    # Sort indices by score
    sorted_idx = np.argsort(scores)

    # Compute target samples per bin
    samples_per_bin = n_samples // n_bins
    remainder = n_samples % n_bins

    # Assign bins
    bin_assignments = np.zeros(n_samples, dtype=int)
    current_idx = 0
    for b in range(n_bins):
        # Add one extra sample to first 'remainder' bins
        bin_size = samples_per_bin + (1 if b < remainder else 0)
        bin_indices = sorted_idx[current_idx:current_idx + bin_size]
        bin_assignments[bin_indices] = b
        current_idx += bin_size

    # Compute bin edges (for reference)
    bin_edges = np.zeros(n_bins + 1)
    bin_edges[0] = scores.min()
    current_idx = 0
    for b in range(n_bins):
        bin_size = samples_per_bin + (1 if b < remainder else 0)
        current_idx += bin_size
        if current_idx < n_samples:
            # Edge is midpoint between last sample in bin and first sample in next bin
            bin_edges[b + 1] = (scores[sorted_idx[current_idx - 1]] + scores[sorted_idx[current_idx]]) / 2
        else:
            bin_edges[b + 1] = scores.max()

    return bin_assignments, bin_edges


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
    })


def run_raw_score_mode(args: argparse.Namespace) -> None:
    """
    Generate raw MSP score reliability diagram.

    Shows that raw MSP score is not calibrated as an error probability estimator.
    """
    latent_dir = Path(args.latent_dir)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "reliability_diagram_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    _apply_style()

    n_bins = args.n_clusters if args.n_clusters else 50

    # Load latent file
    latent_path = latent_dir / "full.pt"
    if not latent_path.exists():
        candidates = list(latent_dir.glob("*.pt"))
        if candidates:
            latent_path = candidates[0]
        else:
            raise FileNotFoundError(f"No latent .pt file found in {latent_dir}")

    print(f"Loading latent from: {latent_path}")
    latent = torch.load(latent_path, map_location="cpu")
    logits = latent["logits"]
    labels = latent["labels"]
    model_preds = latent["model_preds"]

    # Compute errors (full dataset)
    errors_full = (model_preds != labels).numpy().astype(int)

    # Get dataset config for split parameters
    config_dir = run_dir / "configs"
    if not config_dir.exists():
        config_dir = run_dir.parent / "configs"
    dataset_cfg = _load_yaml(config_dir / "dataset.yml") if config_dir.exists() else {}
    n_samples = dataset_cfg.get("n_samples", {}) if dataset_cfg else {}

    # Extract split parameters
    seed_split = args.seed_split
    if seed_split is None and run_dir.name.startswith("seed-split-"):
        try:
            seed_split = int(run_dir.name.split("-")[-1])
        except ValueError:
            seed_split = None

    n_res = args.n_res if args.n_res is not None else n_samples.get("res", 0)
    n_cal = args.n_cal if args.n_cal is not None else n_samples.get("cal", 0)
    n_test = args.n_test if args.n_test is not None else n_samples.get("test", 0)

    print(f"Split params: n_res={n_res}, n_cal={n_cal}, n_test={n_test}, seed_split={seed_split}")

    # Get test subset
    n_total = errors_full.shape[0]
    test_idx = _build_test_indices(n_total, n_res, n_cal, n_test, seed_split)
    errors_test = errors_full[test_idx]
    logits_test = logits[test_idx]

    print(f"Test set size: {len(errors_test)}")
    print(f"Test error rate: {errors_test.mean():.4f}")

    # Compute MSP scores (temperature=1.0, as per experiment config)
    msp_scores = compute_msp_score(logits_test, temperature=1.0)

    # Transform to [0, 1] for interpretation as predicted error probability
    transformed_scores = transform_msp_to_error_prob(msp_scores)

    print(f"MSP score range: [{msp_scores.min():.4f}, {msp_scores.max():.4f}]")
    print(f"Transformed score range: [{transformed_scores.min():.4f}, {transformed_scores.max():.4f}]")

    # Uniform-mass binning on test set
    bin_assignments, bin_edges = uniform_mass_binning(transformed_scores, n_bins)

    print(f"Number of bins: {n_bins}")

    # Compute per-bin statistics
    counts = np.bincount(bin_assignments, minlength=n_bins).astype(float)
    score_sums = np.bincount(bin_assignments, weights=transformed_scores, minlength=n_bins)
    error_sums = np.bincount(bin_assignments, weights=errors_test, minlength=n_bins)

    # Mean score and error rate per bin
    mean_scores = np.zeros(n_bins)
    error_rates = np.zeros(n_bins)
    nonempty = counts > 0
    mean_scores[nonempty] = score_sums[nonempty] / counts[nonempty]
    error_rates[nonempty] = error_sums[nonempty] / counts[nonempty]

    # Compute Clopper-Pearson CI for each bin
    ci_lower = np.zeros(n_bins)
    ci_upper = np.zeros(n_bins)
    for b in range(n_bins):
        lo, hi = clopper_pearson_interval(
            int(error_sums[b]),
            int(counts[b]),
            alpha=args.ci_alpha,
        )
        ci_lower[b] = lo
        ci_upper[b] = hi

    # Compute calibration error (ECE-style)
    # For each bin: |mean_score - error_rate|
    calibration_errors = np.abs(mean_scores - error_rates)
    ece = np.sum((counts / counts.sum()) * calibration_errors)

    # Count "violations" (where error rate > predicted score, i.e., underconfident)
    underconfident = error_rates > mean_scores
    overconfident = error_rates < mean_scores
    n_underconfident = int(np.sum(underconfident & nonempty))
    n_overconfident = int(np.sum(overconfident & nonempty))

    print(f"\n=== Raw Score Calibration Statistics ===")
    print(f"Total bins: {n_bins}")
    print(f"Non-empty bins: {int(np.sum(nonempty))}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Underconfident bins (error > predicted): {n_underconfident}")
    print(f"Overconfident bins (error < predicted): {n_overconfident}")

    # Create reliability diagram
    fig, ax = plt.subplots(figsize=(6, 5))

    # Only plot non-empty bins
    mask = nonempty
    x_plot = mean_scores[mask]
    y_plot = error_rates[mask]
    counts_plot = counts[mask]
    ci_lo_plot = ci_lower[mask]
    ci_hi_plot = ci_upper[mask]

    # Diagonal reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="y = x (perfect calibration)", zorder=1)

    # Scatter points with size proportional to bin count
    size_scale = 300 / np.max(counts_plot) if np.max(counts_plot) > 0 else 1
    sizes = counts_plot * size_scale
    sizes = np.clip(sizes, 20, 300)

    # All points same color (blue) since this is showing miscalibration pattern
    ax.scatter(
        x_plot,
        y_plot,
        s=sizes,
        c="tab:blue",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    # Error bars (Clopper-Pearson CI)
    yerr_lo = y_plot - ci_lo_plot
    yerr_hi = ci_hi_plot - y_plot
    ax.errorbar(
        x_plot,
        y_plot,
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor="gray",
        elinewidth=0.8,
        capsize=2,
        alpha=0.5,
        zorder=2,
    )

    # Labels and title
    ax.set_xlabel(r"Mean predicted error probability $\bar{s}(b)$")
    ax.set_ylabel(r"Empirical error rate $\hat{p}(b)$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], linestyle="--", color="k", label="Perfect calibration"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=8, label="Bin (raw MSP)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, framealpha=0.9)

    # Add annotation with calibration stats
    stats_text = f"ECE: {ece:.3f}\nBins: {int(np.sum(nonempty))}"
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()

    # Save figure
    fig.savefig(output_dir / "raw_score_reliability_diagram.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "raw_score_reliability_diagram.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure to: {output_dir / 'raw_score_reliability_diagram.pdf'}")

    # Save statistics
    per_bin_stats = []
    for b in range(n_bins):
        per_bin_stats.append({
            "bin_idx": b,
            "mean_predicted_score": float(mean_scores[b]),
            "empirical_error_rate": float(error_rates[b]),
            "ci_lower": float(ci_lower[b]),
            "ci_upper": float(ci_upper[b]),
            "count": int(counts[b]),
            "calibration_error": float(calibration_errors[b]),
            "bin_edge_lower": float(bin_edges[b]),
            "bin_edge_upper": float(bin_edges[b + 1]),
        })

    summary = {
        "mode": "raw-score",
        "score_type": "msp",
        "temperature": 1.0,
        "n_bins": n_bins,
        "n_nonempty_bins": int(np.sum(nonempty)),
        "ece": float(ece),
        "n_underconfident_bins": n_underconfident,
        "n_overconfident_bins": n_overconfident,
        "total_test_samples": int(len(errors_test)),
        "test_error_rate": float(errors_test.mean()),
        "latent_path": str(latent_path),
        "ci_alpha": args.ci_alpha,
        "per_bin": per_bin_stats,
    }

    stats_output = output_dir / "raw_score_reliability_stats.json"
    stats_output.write_text(json.dumps(_serialize(summary), indent=2), encoding="utf-8")
    print(f"Saved stats to: {stats_output}")


def run_calibrated_mean_mode(args: argparse.Namespace) -> None:
    """
    Generate calibrated-mean reliability diagram (cf. Corollary 4.1).

    Shows that uniform-mass binning produces a calibrated score when using the mean.
    Points should lie ON the diagonal, demonstrating calibration.
    """
    run_dir = Path(args.run_dir)
    latent_dir = Path(args.latent_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "reliability_diagram_mean"
    output_dir.mkdir(parents=True, exist_ok=True)

    _apply_style()

    # Load partition stats from calibration
    stats_path, stats = _load_stats(run_dir, args.n_clusters)
    clusters_path, clusters_test = _load_clusters_test(run_dir, args.n_clusters)

    n_clusters = int(stats["cluster_intervals"].shape[1])
    counts_cal = stats["cluster_counts"].squeeze(0).numpy()
    means_cal = stats["cluster_error_means"].squeeze(0).numpy()

    print(f"Loaded stats from: {stats_path}")
    print(f"Number of bins: {n_clusters}")

    # Load test errors from latent file
    latent_path = latent_dir / "full.pt"
    if not latent_path.exists():
        candidates = list(latent_dir.glob("*.pt"))
        if candidates:
            latent_path = candidates[0]
        else:
            raise FileNotFoundError(f"No latent .pt file found in {latent_dir}")

    print(f"Loading latent from: {latent_path}")
    latent = torch.load(latent_path, map_location="cpu")
    errors_full = (latent["model_preds"] != latent["labels"]).numpy().astype(int)

    # Get dataset config for split parameters
    config_dir = run_dir / "configs"
    if not config_dir.exists():
        config_dir = run_dir.parent / "configs"
    dataset_cfg = _load_yaml(config_dir / "dataset.yml") if config_dir.exists() else {}
    n_samples = dataset_cfg.get("n_samples", {}) if dataset_cfg else {}

    # Extract split parameters
    seed_split = args.seed_split
    if seed_split is None and run_dir.name.startswith("seed-split-"):
        try:
            seed_split = int(run_dir.name.split("-")[-1])
        except ValueError:
            seed_split = None

    n_res = args.n_res if args.n_res is not None else n_samples.get("res", 0)
    n_cal = args.n_cal if args.n_cal is not None else n_samples.get("cal", 0)
    n_test = args.n_test if args.n_test is not None else n_samples.get("test", 0)

    print(f"Split params: n_res={n_res}, n_cal={n_cal}, n_test={n_test}, seed_split={seed_split}")

    # Get test subset errors
    n_total = errors_full.shape[0]
    test_idx = _build_test_indices(n_total, n_res, n_cal, n_test, seed_split)
    errors_test = errors_full[test_idx]

    # Verify cluster assignments match test size
    if len(clusters_test) != len(errors_test):
        print(f"Warning: clusters_test ({len(clusters_test)}) != errors_test ({len(errors_test)})")
        print("Using clusters_test length for errors_test")
        errors_test = errors_test[:len(clusters_test)]

    # Compute per-bin test error rate
    counts_test = np.bincount(clusters_test, minlength=n_clusters).astype(float)
    errors_sum_test = np.bincount(clusters_test, weights=errors_test, minlength=n_clusters).astype(float)

    # Handle empty bins
    err_rate_test = np.zeros(n_clusters)
    nonempty = counts_test > 0
    err_rate_test[nonempty] = errors_sum_test[nonempty] / counts_test[nonempty]

    # Compute Clopper-Pearson CI for each bin
    ci_lower = np.zeros(n_clusters)
    ci_upper = np.zeros(n_clusters)
    for b in range(n_clusters):
        lo, hi = clopper_pearson_interval(
            int(errors_sum_test[b]),
            int(counts_test[b]),
            alpha=args.ci_alpha,
        )
        ci_lower[b] = lo
        ci_upper[b] = hi

    # Compute calibration error (how far from diagonal)
    calibration_errors = np.abs(means_cal - err_rate_test)
    ece = np.sum((counts_test / counts_test.sum()) * calibration_errors) if counts_test.sum() > 0 else 0.0

    # Compute statistics relative to means
    overestimate = err_rate_test < means_cal  # predicted more errors than actual
    underestimate = err_rate_test > means_cal  # predicted fewer errors than actual
    n_overestimate = int(np.sum(overestimate & nonempty))
    n_underestimate = int(np.sum(underestimate & nonempty))
    n_test_total = int(np.sum(counts_test))

    print(f"\n=== Calibrated Mean Statistics ===")
    print(f"Total bins: {n_clusters}")
    print(f"Non-empty bins: {int(np.sum(nonempty))}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Overestimate bins (p̂_test < p̂_cal): {n_overestimate}")
    print(f"Underestimate bins (p̂_test > p̂_cal): {n_underestimate}")

    # Create reliability diagram
    fig, ax = plt.subplots(figsize=(6, 5))

    # Only plot non-empty bins
    mask = nonempty
    x_plot = means_cal[mask]
    y_plot = err_rate_test[mask]
    counts_plot = counts_test[mask]
    ci_lo_plot = ci_lower[mask]
    ci_hi_plot = ci_upper[mask]

    # Diagonal reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="y = x (perfect calibration)", zorder=1)

    # Scatter points with size proportional to bin count
    size_scale = 300 / np.max(counts_plot) if np.max(counts_plot) > 0 else 1
    sizes = counts_plot * size_scale
    sizes = np.clip(sizes, 20, 300)

    # Use green to indicate calibrated (on diagonal)
    ax.scatter(
        x_plot,
        y_plot,
        s=sizes,
        c="tab:green",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    # Error bars (Clopper-Pearson CI)
    yerr_lo = y_plot - ci_lo_plot
    yerr_hi = ci_hi_plot - y_plot
    ax.errorbar(
        x_plot,
        y_plot,
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor="gray",
        elinewidth=0.8,
        capsize=2,
        alpha=0.5,
        zorder=2,
    )

    # Labels and title
    ax.set_xlabel(r"Mean calibration error $\hat{p}_{\mathrm{cal}}(b)$")
    ax.set_ylabel(r"Empirical test error $\hat{p}_{\mathrm{test}}(b)$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], linestyle="--", color="k", label="Perfect calibration"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:green", markersize=8, label="Bin (calibrated mean)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, framealpha=0.9)

    # Add annotation with calibration stats
    stats_text = f"ECE: {ece:.3f}\nBins: {int(np.sum(nonempty))}"
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()

    # Save figure
    fig.savefig(output_dir / "calibrated_mean_reliability_diagram.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "calibrated_mean_reliability_diagram.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure to: {output_dir / 'calibrated_mean_reliability_diagram.pdf'}")

    # Save statistics
    per_bin_stats = []
    for b in range(n_clusters):
        per_bin_stats.append({
            "bin_idx": b,
            "mean_cal": float(means_cal[b]),
            "test_error": float(err_rate_test[b]),
            "ci_lower": float(ci_lower[b]),
            "ci_upper": float(ci_upper[b]),
            "count_cal": int(counts_cal[b]),
            "count_test": int(counts_test[b]),
            "calibration_error": float(calibration_errors[b]),
        })

    summary = {
        "mode": "calibrated-mean",
        "ci_alpha": args.ci_alpha,
        "n_clusters": n_clusters,
        "n_nonempty_bins": int(np.sum(nonempty)),
        "ece": float(ece),
        "n_overestimate_bins": n_overestimate,
        "n_underestimate_bins": n_underestimate,
        "total_test_samples": n_test_total,
        "stats_path": str(stats_path),
        "latent_path": str(latent_path),
        "per_bin": per_bin_stats,
    }

    stats_output = output_dir / "calibrated_mean_reliability_stats.json"
    stats_output.write_text(json.dumps(_serialize(summary), indent=2), encoding="utf-8")
    print(f"Saved stats to: {stats_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upper reliability diagram for conservative calibration."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory (seed-split-*) with partition outputs.",
    )
    parser.add_argument(
        "--latent-dir",
        required=True,
        help="Directory containing latent .pt file (will look for full.pt).",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters/bins to use.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Confidence level for Hoeffding bounds (default: 0.05).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for figure and stats.",
    )
    parser.add_argument(
        "--n-res",
        type=int,
        default=None,
        help="Override n_res from config.",
    )
    parser.add_argument(
        "--n-cal",
        type=int,
        default=None,
        help="Override n_cal from config.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="Override n_test from config.",
    )
    parser.add_argument(
        "--seed-split",
        type=int,
        default=None,
        help="Override seed split from directory name.",
    )
    parser.add_argument(
        "--ci-alpha",
        type=float,
        default=0.05,
        help="Significance level for Clopper-Pearson CI on test error (default: 0.05 for 95%% CI).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["calibrated", "calibrated-mean", "raw-score"],
        default="calibrated",
        help="Mode: 'calibrated' for conservative calibration diagram, 'calibrated-mean' for mean calibration diagram, 'raw-score' for raw MSP reliability diagram.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Dispatch based on mode
    if args.mode == "raw-score":
        run_raw_score_mode(args)
        return
    elif args.mode == "calibrated-mean":
        run_calibrated_mean_mode(args)
        return

    # --- Calibrated mode (default) ---
    run_dir = Path(args.run_dir)
    latent_dir = Path(args.latent_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "reliability_diagram"
    output_dir.mkdir(parents=True, exist_ok=True)

    _apply_style()

    # Load partition stats from calibration
    stats_path, stats = _load_stats(run_dir, args.n_clusters)
    clusters_path, clusters_test = _load_clusters_test(run_dir, args.n_clusters)

    n_clusters = int(stats["cluster_intervals"].shape[1])
    counts_cal = stats["cluster_counts"].squeeze(0).numpy()
    means_cal = stats["cluster_error_means"].squeeze(0).numpy()

    print(f"Loaded stats from: {stats_path}")
    print(f"Number of bins: {n_clusters}")
    print(f"Alpha for Hoeffding bounds: {args.alpha}")

    # Recompute Hoeffding upper bounds with specified alpha
    uppers = compute_hoeffding_upper(means_cal, counts_cal, args.alpha, simultaneous=True)

    # Load test errors from latent file
    latent_path = latent_dir / "full.pt"
    if not latent_path.exists():
        # Try alternative names
        candidates = list(latent_dir.glob("*.pt"))
        if candidates:
            latent_path = candidates[0]
        else:
            raise FileNotFoundError(f"No latent .pt file found in {latent_dir}")

    print(f"Loading latent from: {latent_path}")
    latent = torch.load(latent_path, map_location="cpu")
    errors_full = (latent["model_preds"] != latent["labels"]).numpy().astype(int)

    # Get dataset config for split parameters
    config_dir = run_dir / "configs"
    if not config_dir.exists():
        config_dir = run_dir.parent / "configs"
    dataset_cfg = _load_yaml(config_dir / "dataset.yml") if config_dir.exists() else {}
    n_samples = dataset_cfg.get("n_samples", {}) if dataset_cfg else {}

    # Extract split parameters
    seed_split = args.seed_split
    if seed_split is None and run_dir.name.startswith("seed-split-"):
        try:
            seed_split = int(run_dir.name.split("-")[-1])
        except ValueError:
            seed_split = None

    n_res = args.n_res if args.n_res is not None else n_samples.get("res", 0)
    n_cal = args.n_cal if args.n_cal is not None else n_samples.get("cal", 0)
    n_test = args.n_test if args.n_test is not None else n_samples.get("test", 0)

    print(f"Split params: n_res={n_res}, n_cal={n_cal}, n_test={n_test}, seed_split={seed_split}")

    # Get test subset errors
    n_total = errors_full.shape[0]
    test_idx = _build_test_indices(n_total, n_res, n_cal, n_test, seed_split)
    errors_test = errors_full[test_idx]

    # Verify cluster assignments match test size
    if len(clusters_test) != len(errors_test):
        print(f"Warning: clusters_test ({len(clusters_test)}) != errors_test ({len(errors_test)})")
        print("Using clusters_test length for errors_test")
        errors_test = errors_test[:len(clusters_test)]

    # Compute per-bin test error rate
    counts_test = np.bincount(clusters_test, minlength=n_clusters).astype(float)
    errors_sum_test = np.bincount(clusters_test, weights=errors_test, minlength=n_clusters).astype(float)

    # Handle empty bins
    err_rate_test = np.zeros(n_clusters)
    nonempty = counts_test > 0
    err_rate_test[nonempty] = errors_sum_test[nonempty] / counts_test[nonempty]

    # Compute Clopper-Pearson CI for each bin
    ci_lower = np.zeros(n_clusters)
    ci_upper = np.zeros(n_clusters)
    for b in range(n_clusters):
        lo, hi = clopper_pearson_interval(
            int(errors_sum_test[b]),
            int(counts_test[b]),
            alpha=args.ci_alpha,
        )
        ci_lower[b] = lo
        ci_upper[b] = hi

    # Compute violation statistics
    violations = err_rate_test > uppers
    violation_amounts = np.maximum(err_rate_test - uppers, 0)
    max_violation = float(np.max(violation_amounts))
    n_test_total = int(np.sum(counts_test))
    weighted_violation = float(np.sum((counts_test / n_test_total) * violation_amounts)) if n_test_total > 0 else 0.0
    n_violating = int(np.sum(violations))

    # Check if CI overlaps with upper bound (marginal violations)
    marginal_violations = (err_rate_test > uppers) & (ci_lower <= uppers)
    n_marginal = int(np.sum(marginal_violations))

    print(f"\n=== Reliability Statistics ===")
    print(f"Total bins: {n_clusters}")
    print(f"Non-empty bins: {int(np.sum(nonempty))}")
    print(f"Violating bins (p̂ > û): {n_violating}")
    print(f"Marginal violations (CI overlaps): {n_marginal}")
    print(f"Max violation: {max_violation:.4f}")
    print(f"Weighted violation: {weighted_violation:.6f}")

    # Create reliability diagram
    fig, ax = plt.subplots(figsize=(6, 5))

    # Only plot non-empty bins
    mask = nonempty
    x_plot = uppers[mask]
    y_plot = err_rate_test[mask]
    counts_plot = counts_test[mask]
    ci_lo_plot = ci_lower[mask]
    ci_hi_plot = ci_upper[mask]
    violations_plot = violations[mask]

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="y = x (ideal boundary)", zorder=1)

    # Scatter points with size proportional to bin count
    # Normalize sizes for visibility
    size_scale = 300 / np.max(counts_plot) if np.max(counts_plot) > 0 else 1
    sizes = counts_plot * size_scale
    sizes = np.clip(sizes, 20, 300)  # Min and max sizes

    # Color by violation status
    colors = np.where(violations_plot, "tab:red", "tab:blue")

    ax.scatter(
        x_plot,
        y_plot,
        s=sizes,
        c=colors,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    # Error bars (Clopper-Pearson CI)
    yerr_lo = y_plot - ci_lo_plot
    yerr_hi = ci_hi_plot - y_plot
    ax.errorbar(
        x_plot,
        y_plot,
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor="gray",
        elinewidth=0.8,
        capsize=2,
        alpha=0.5,
        zorder=2,
    )

    # Labels and title
    ax.set_xlabel(r"Conservative upper bound $\hat{u}(b)$")
    ax.set_ylabel(r"Empirical test error $\hat{p}(b)$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")

    # Add legend with custom entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], linestyle="--", color="k", label="y = x"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=8, label="Below bound"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", markersize=8, label="Above bound"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, framealpha=0.9)

    # Add annotation with violation stats
    stats_text = f"Violations: {n_violating}/{int(np.sum(nonempty))} bins\nMax: {max_violation:.3f}"
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()

    # Save figure
    fig.savefig(output_dir / "upper_reliability_diagram.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "upper_reliability_diagram.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure to: {output_dir / 'upper_reliability_diagram.pdf'}")

    # Save statistics
    per_bin_stats = []
    for b in range(n_clusters):
        per_bin_stats.append({
            "bin_idx": b,
            "upper_bound": float(uppers[b]),
            "test_error": float(err_rate_test[b]),
            "ci_lower": float(ci_lower[b]),
            "ci_upper": float(ci_upper[b]),
            "count_cal": int(counts_cal[b]),
            "count_test": int(counts_test[b]),
            "mean_cal": float(means_cal[b]),
            "violation": float(violation_amounts[b]),
            "is_violating": bool(violations[b]),
        })

    summary = {
        "alpha": args.alpha,
        "ci_alpha": args.ci_alpha,
        "n_clusters": n_clusters,
        "n_nonempty_bins": int(np.sum(nonempty)),
        "max_violation": max_violation,
        "weighted_violation": weighted_violation,
        "n_violating_bins": n_violating,
        "n_marginal_violations": n_marginal,
        "total_test_samples": n_test_total,
        "stats_path": str(stats_path),
        "latent_path": str(latent_path),
        "per_bin": per_bin_stats,
    }

    stats_output = output_dir / "reliability_stats.json"
    stats_output.write_text(json.dumps(_serialize(summary), indent=2), encoding="utf-8")
    print(f"Saved stats to: {stats_output}")


if __name__ == "__main__":
    main()
