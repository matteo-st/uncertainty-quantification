#!/usr/bin/env python3
"""
Validate Conservative Calibration with Multiple Seeds (Corollary 4.1).

Empirically validates the guarantee:
    P_{D_cal ~ P^n}(for all bins b: η(b) ≤ û(b)) ≥ 1 - α

Runs 100-200 repetitions (seeds) using pre-computed logits.
Each repetition:
1. Fixes res split (for hyperparameter selection consistency)
2. Resamples cal/test from remaining pool
3. Fits uniform-mass binning on cal
4. Computes Hoeffding upper bounds
5. Evaluates violations on test

Usage:
    python scripts/analysis/validate_conservative_calibration.py \
        --latent-dir latent/imagenet_timm_vit_base16_ce/transform-test_n-epochs-1 \
        --n-res 5000 \
        --n-cal 20000 \
        --n-eval 25000 \
        --n-bins 100 \
        --alpha 0.01 0.02 0.05 0.1 0.2 \
        --n-repetitions 200 \
        --score-type msp \
        --temperature 0.6 \
        --base-seed 1 \
        --m-min 20 \
        --output-dir results/calibration_validation/imagenet/timm_vit_base16_ce/msp/val-200seeds
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

import numpy as np
import scipy.stats
import torch
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# Score Computation Functions (from upper_reliability_diagram.py)
# ============================================================================

def compute_msp_score(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """
    Compute MSP (Maximum Softmax Probability) score.
    Higher score = more uncertain (negated MSP).
    """
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    msp = probs.max(dim=-1).values
    return -msp.numpy()


def compute_doctor_score(logits: torch.Tensor, temperature: float = 1.0, normalize: bool = True) -> np.ndarray:
    """
    Compute Doctor (Gini) score - higher = more uncertain.
    """
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    g = torch.sum(probs ** 2, dim=-1)
    if normalize:
        gini = (1.0 - g) / g
    else:
        gini = 1.0 - g
    return gini.numpy()


def compute_margin_score(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """
    Compute Margin score - higher = more uncertain (low margin).
    """
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    return (1.0 - margin).numpy()


def compute_score(logits: torch.Tensor, score_type: str, temperature: float = 1.0,
                  normalize: bool = True) -> np.ndarray:
    """Dispatch to appropriate score computation function."""
    if score_type == "msp":
        return compute_msp_score(logits, temperature)
    elif score_type == "doctor":
        return compute_doctor_score(logits, temperature, normalize)
    elif score_type == "margin":
        return compute_margin_score(logits, temperature)
    else:
        raise ValueError(f"Unknown score type: {score_type}")


# ============================================================================
# Binning Functions
# ============================================================================

def uniform_mass_binning(scores: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign samples to bins with approximately equal mass (uniform-mass binning).

    Returns:
        (bin_assignments, bin_edges)
    """
    n_samples = len(scores)
    sorted_idx = np.argsort(scores)

    samples_per_bin = n_samples // n_bins
    remainder = n_samples % n_bins

    bin_assignments = np.zeros(n_samples, dtype=int)
    current_idx = 0
    for b in range(n_bins):
        bin_size = samples_per_bin + (1 if b < remainder else 0)
        bin_indices = sorted_idx[current_idx:current_idx + bin_size]
        bin_assignments[bin_indices] = b
        current_idx += bin_size

    # Compute bin edges
    bin_edges = np.zeros(n_bins + 1)
    bin_edges[0] = scores.min()
    current_idx = 0
    for b in range(n_bins):
        bin_size = samples_per_bin + (1 if b < remainder else 0)
        current_idx += bin_size
        if current_idx < n_samples:
            bin_edges[b + 1] = (scores[sorted_idx[current_idx - 1]] + scores[sorted_idx[current_idx]]) / 2
        else:
            bin_edges[b + 1] = scores.max()

    return bin_assignments, bin_edges


def assign_to_bins(scores: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Assign new scores to existing bins defined by edges."""
    # np.digitize returns 1-indexed bins; we want 0-indexed
    # Clip to valid bin range [0, n_bins-1]
    n_bins = len(bin_edges) - 1
    assignments = np.digitize(scores, bin_edges[1:-1])  # inner edges
    return np.clip(assignments, 0, n_bins - 1)


# ============================================================================
# Statistical Functions
# ============================================================================

def compute_hoeffding_upper(means: np.ndarray, counts: np.ndarray, alpha: float,
                            simultaneous: bool = True) -> np.ndarray:
    """
    Compute Hoeffding upper bounds with optional Bonferroni correction.
    """
    K = len(means)
    alpha_eff = alpha / K if simultaneous else alpha

    # Hoeffding half-width: sqrt(log(2/alpha) / (2n))
    # Avoid division by zero for empty bins
    safe_counts = np.maximum(counts, 1)
    half_widths = np.sqrt(np.log(2.0 / alpha_eff) / (2.0 * safe_counts))

    # Upper bound, clipped to [0, 1]
    uppers = np.minimum(1.0, means + half_widths)
    return uppers


def clopper_pearson_interval(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute Clopper-Pearson exact binomial confidence interval."""
    if n == 0:
        return (0.0, 1.0)

    if successes == 0:
        lower = 0.0
    else:
        lower = scipy.stats.beta.ppf(alpha / 2, successes, n - successes + 1)

    if successes == n:
        upper = 1.0
    else:
        upper = scipy.stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes)

    return (lower, upper)


# ============================================================================
# Discriminative Metrics
# ============================================================================

def compute_roc_auc(scores: np.ndarray, errors: np.ndarray) -> float:
    """Compute ROC-AUC for error detection."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(errors)) < 2:
        return np.nan
    return roc_auc_score(errors, scores)


def compute_fpr_at_tpr(scores: np.ndarray, errors: np.ndarray, tpr_target: float = 0.95) -> float:
    """Compute FPR at fixed TPR (e.g., 95%)."""
    from sklearn.metrics import roc_curve
    if len(np.unique(errors)) < 2:
        return np.nan
    fprs, tprs, _ = roc_curve(errors, scores)
    # Find FPR at target TPR
    idxs = np.where(tprs >= tpr_target)[0]
    if len(idxs) > 0:
        return fprs[idxs[0]]
    return fprs[-1]


def compute_aurc(scores: np.ndarray, errors: np.ndarray) -> float:
    """
    Compute Area Under Risk-Coverage curve.
    Adapted from fd-shifts implementation.
    """
    n = len(errors)
    if n == 0:
        return np.nan

    # Sort by confidence (ascending score = most uncertain first, reject first)
    idx_sorted = np.argsort(scores)

    coverages = []
    risks = []

    coverage = n
    error_sum = float(np.sum(errors[idx_sorted]))

    coverages.append(coverage / n)
    risks.append(error_sum / n)

    weights = []
    tmp_weight = 0

    for i in range(len(idx_sorted) - 1):
        coverage -= 1
        error_sum -= errors[idx_sorted[i]]
        selective_risk = error_sum / (n - 1 - i) if (n - 1 - i) > 0 else 0
        tmp_weight += 1
        if i == 0 or scores[idx_sorted[i]] != scores[idx_sorted[i - 1]]:
            coverages.append(coverage / n)
            risks.append(selective_risk)
            weights.append(tmp_weight / n)
            tmp_weight = 0

    if tmp_weight > 0:
        coverages.append(0)
        risks.append(risks[-1] if risks else 0)
        weights.append(tmp_weight / n)

    # Compute AURC using trapezoidal rule with weights
    aurc = sum(
        (risks[i] + risks[i + 1]) * 0.5 * weights[i]
        for i in range(len(weights))
    )
    return aurc


# ============================================================================
# Single Repetition Runner
# ============================================================================

def run_single_repetition(
    logits: torch.Tensor,
    labels: np.ndarray,
    model_preds: np.ndarray,
    pool_idx: np.ndarray,
    seed: int,
    n_cal: int,
    n_eval: int,
    n_bins: int,
    alpha_values: List[float],
    score_type: str,
    temperature: float,
    normalize: bool,
    m_min: int = 20,
) -> Dict[str, Any]:
    """
    Run a single repetition of the calibration validation.

    Args:
        logits: Full logits tensor
        labels: Full labels array
        model_preds: Full model predictions array
        pool_idx: Indices available for cal/eval sampling (excludes res)
        seed: Random seed for this repetition
        n_cal: Number of calibration samples
        n_eval: Number of evaluation samples
        n_bins: Number of bins for uniform-mass binning
        alpha_values: List of alpha values to test
        score_type: Type of uncertainty score
        temperature: Temperature scaling
        normalize: Whether to normalize (for doctor score)
        m_min: Minimum samples in bin for reliable evaluation

    Returns:
        Dictionary with all metrics for this repetition
    """
    # 1. Resample cal/eval from pool
    rng = np.random.RandomState(seed)
    shuffled_pool = rng.permutation(pool_idx)

    if n_cal + n_eval > len(shuffled_pool):
        raise ValueError(f"n_cal ({n_cal}) + n_eval ({n_eval}) > pool size ({len(shuffled_pool)})")

    cal_idx = shuffled_pool[:n_cal]
    eval_idx = shuffled_pool[n_cal:n_cal + n_eval]

    # 2. Extract data for cal and eval
    logits_cal = logits[cal_idx]
    logits_eval = logits[eval_idx]
    errors_cal = (model_preds[cal_idx] != labels[cal_idx]).astype(int)
    errors_eval = (model_preds[eval_idx] != labels[eval_idx]).astype(int)

    # 3. Compute scores
    scores_cal = compute_score(logits_cal, score_type, temperature, normalize)
    scores_eval = compute_score(logits_eval, score_type, temperature, normalize)

    # 4. Fit uniform-mass binning on cal
    clusters_cal, bin_edges = uniform_mass_binning(scores_cal, n_bins)

    # 5. Assign eval samples to bins
    clusters_eval = assign_to_bins(scores_eval, bin_edges)

    # 6. Compute per-bin statistics on cal
    counts_cal = np.bincount(clusters_cal, minlength=n_bins).astype(float)
    error_sums_cal = np.bincount(clusters_cal, weights=errors_cal, minlength=n_bins).astype(float)
    means_cal = np.zeros(n_bins)
    nonempty_cal = counts_cal > 0
    means_cal[nonempty_cal] = error_sums_cal[nonempty_cal] / counts_cal[nonempty_cal]

    # 7. Compute per-bin statistics on eval
    counts_eval = np.bincount(clusters_eval, minlength=n_bins).astype(float)
    error_sums_eval = np.bincount(clusters_eval, weights=errors_eval, minlength=n_bins).astype(float)
    means_eval = np.zeros(n_bins)
    nonempty_eval = counts_eval > 0
    means_eval[nonempty_eval] = error_sums_eval[nonempty_eval] / counts_eval[nonempty_eval]

    # 8. Compute metrics for each alpha value
    results = {
        "seed": seed,
        "n_cal": n_cal,
        "n_eval": n_eval,
        "error_rate_cal": float(errors_cal.mean()),
        "error_rate_eval": float(errors_eval.mean()),
    }

    # Discriminative metrics on cal and eval
    results["roc_auc_cal"] = float(compute_roc_auc(scores_cal, errors_cal))
    results["roc_auc_eval"] = float(compute_roc_auc(scores_eval, errors_eval))
    results["fpr_at_95_cal"] = float(compute_fpr_at_tpr(scores_cal, errors_cal, 0.95))
    results["fpr_at_95_eval"] = float(compute_fpr_at_tpr(scores_eval, errors_eval, 0.95))
    results["aurc_cal"] = float(compute_aurc(scores_cal, errors_cal))
    results["aurc_eval"] = float(compute_aurc(scores_eval, errors_eval))

    # Per-alpha metrics
    for alpha in alpha_values:
        alpha_key = f"alpha_{alpha:.4f}".rstrip('0').rstrip('.')

        # Compute Hoeffding upper bounds (with Bonferroni)
        upper_bounds = compute_hoeffding_upper(means_cal, counts_cal, alpha, simultaneous=True)

        # Compute violations (only bins with sufficient samples)
        reliable_mask = counts_eval >= m_min
        violations = np.maximum(means_eval - upper_bounds, 0)

        # Max violation (over reliable bins)
        reliable_violations = violations.copy()
        reliable_violations[~reliable_mask] = 0
        max_violation = float(np.max(reliable_violations))

        # Weighted violation
        total_reliable = float(np.sum(counts_eval[reliable_mask]))
        if total_reliable > 0:
            weighted_violation = float(np.sum((counts_eval[reliable_mask] / total_reliable) * violations[reliable_mask]))
        else:
            weighted_violation = 0.0

        # Success indicator (no violations in any bin)
        success = int(max_violation == 0)

        # Number of violating bins
        n_violating = int(np.sum((violations > 0) & reliable_mask))

        results[f"{alpha_key}_max_violation"] = max_violation
        results[f"{alpha_key}_weighted_violation"] = weighted_violation
        results[f"{alpha_key}_success"] = success
        results[f"{alpha_key}_n_violating_bins"] = n_violating
        results[f"{alpha_key}_n_reliable_bins"] = int(np.sum(reliable_mask))

    return results


# ============================================================================
# Aggregation Functions
# ============================================================================

def aggregate_results(per_rep_results: List[Dict], alpha_values: List[float]) -> Dict[str, Any]:
    """Aggregate results across all repetitions."""
    n_reps = len(per_rep_results)

    aggregated = {
        "n_repetitions": n_reps,
        "alpha_values": alpha_values,
    }

    # Aggregate discriminative metrics
    for metric in ["roc_auc_cal", "roc_auc_eval", "fpr_at_95_cal", "fpr_at_95_eval", "aurc_cal", "aurc_eval"]:
        values = [r[metric] for r in per_rep_results if not np.isnan(r[metric])]
        if values:
            aggregated[f"{metric}_mean"] = float(np.mean(values))
            aggregated[f"{metric}_std"] = float(np.std(values))
        else:
            aggregated[f"{metric}_mean"] = np.nan
            aggregated[f"{metric}_std"] = np.nan

    # Aggregate per-alpha metrics
    for alpha in alpha_values:
        alpha_key = f"alpha_{alpha:.4f}".rstrip('0').rstrip('.')

        max_violations = [r[f"{alpha_key}_max_violation"] for r in per_rep_results]
        weighted_violations = [r[f"{alpha_key}_weighted_violation"] for r in per_rep_results]
        successes = [r[f"{alpha_key}_success"] for r in per_rep_results]
        n_violating = [r[f"{alpha_key}_n_violating_bins"] for r in per_rep_results]

        # Success rate with Clopper-Pearson CI
        n_success = sum(successes)
        success_rate = n_success / n_reps
        ci_low, ci_high = clopper_pearson_interval(n_success, n_reps, alpha=0.05)

        aggregated[f"{alpha_key}_success_rate"] = float(success_rate)
        aggregated[f"{alpha_key}_success_rate_ci_low"] = float(ci_low)
        aggregated[f"{alpha_key}_success_rate_ci_high"] = float(ci_high)
        aggregated[f"{alpha_key}_target"] = float(1 - alpha)

        # Max violation statistics
        aggregated[f"{alpha_key}_max_violation_mean"] = float(np.mean(max_violations))
        aggregated[f"{alpha_key}_max_violation_std"] = float(np.std(max_violations))
        aggregated[f"{alpha_key}_max_violation_quantiles"] = {
            "50": float(np.percentile(max_violations, 50)),
            "75": float(np.percentile(max_violations, 75)),
            "90": float(np.percentile(max_violations, 90)),
            "95": float(np.percentile(max_violations, 95)),
            "99": float(np.percentile(max_violations, 99)),
        }

        # Weighted violation statistics
        aggregated[f"{alpha_key}_weighted_violation_mean"] = float(np.mean(weighted_violations))
        aggregated[f"{alpha_key}_weighted_violation_std"] = float(np.std(weighted_violations))

        # Violating bins statistics
        aggregated[f"{alpha_key}_n_violating_bins_mean"] = float(np.mean(n_violating))
        aggregated[f"{alpha_key}_n_violating_bins_std"] = float(np.std(n_violating))

    return aggregated


# ============================================================================
# Visualization Functions
# ============================================================================

def apply_plot_style():
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


def plot_coverage_vs_alpha(aggregated: Dict, output_dir: Path):
    """Plot empirical success rate vs alpha (coverage validation)."""
    apply_plot_style()

    alpha_values = aggregated["alpha_values"]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Extract data
    success_rates = []
    ci_lows = []
    ci_highs = []
    targets = []

    for alpha in alpha_values:
        alpha_key = f"alpha_{alpha:.4f}".rstrip('0').rstrip('.')
        success_rates.append(aggregated[f"{alpha_key}_success_rate"])
        ci_lows.append(aggregated[f"{alpha_key}_success_rate_ci_low"])
        ci_highs.append(aggregated[f"{alpha_key}_success_rate_ci_high"])
        targets.append(aggregated[f"{alpha_key}_target"])

    success_rates = np.array(success_rates)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)
    targets = np.array(targets)

    # Target line (1 - alpha)
    ax.plot(alpha_values, targets, "k--", lw=2, label=r"Target: $1 - \alpha$", zorder=1)

    # Empirical success rate with CI
    ax.errorbar(
        alpha_values,
        success_rates,
        yerr=[success_rates - ci_lows, ci_highs - success_rates],
        fmt="o-",
        capsize=5,
        capthick=2,
        markersize=8,
        color="tab:blue",
        label=r"Empirical $\hat{\pi}(\alpha)$",
        zorder=2,
    )

    ax.set_xlabel(r"Significance level $\alpha$")
    ax.set_ylabel(r"Success rate $\hat{\pi}(\alpha)$")
    ax.set_xlim(-0.01, max(alpha_values) + 0.02)
    ax.set_ylim(0.75, 1.02)
    ax.legend(loc="lower left")

    # Add annotation
    n_reps = aggregated["n_repetitions"]
    ax.text(0.98, 0.02, f"n = {n_reps} repetitions", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_dir / "coverage_vs_alpha.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "coverage_vs_alpha.png", bbox_inches="tight")
    plt.close(fig)


def plot_violation_ecdf(per_rep_results: List[Dict], alpha_values: List[float], output_dir: Path):
    """Plot ECDF of max violations for each alpha."""
    apply_plot_style()

    fig, ax = plt.subplots(figsize=(6, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(alpha_values)))

    for alpha, color in zip(alpha_values, colors):
        alpha_key = f"alpha_{alpha:.4f}".rstrip('0').rstrip('.')
        max_violations = [r[f"{alpha_key}_max_violation"] for r in per_rep_results]

        # Compute ECDF
        sorted_violations = np.sort(max_violations)
        ecdf = np.arange(1, len(sorted_violations) + 1) / len(sorted_violations)

        ax.step(sorted_violations, ecdf, where="post", label=rf"$\alpha = {alpha}$", color=color, lw=2)

    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Max violation $V_r$")
    ax.set_ylabel("ECDF")
    ax.set_xlim(-0.005, None)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "max_violation_ecdf.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "max_violation_ecdf.png", bbox_inches="tight")
    plt.close(fig)


def plot_violation_histogram(per_rep_results: List[Dict], alpha: float, output_dir: Path):
    """Plot histogram of max violations for a specific alpha."""
    apply_plot_style()

    alpha_key = f"alpha_{alpha:.4f}".rstrip('0').rstrip('.')
    max_violations = [r[f"{alpha_key}_max_violation"] for r in per_rep_results]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Separate zero and non-zero violations
    n_zero = sum(1 for v in max_violations if v == 0)
    nonzero = [v for v in max_violations if v > 0]

    if nonzero:
        ax.hist(nonzero, bins=30, edgecolor="white", alpha=0.7, label=f"Violations (n={len(nonzero)})")

    ax.axvline(0, color="green", linestyle="--", lw=2, label=f"No violation (n={n_zero})")

    ax.set_xlabel("Max violation $V_r$")
    ax.set_ylabel("Count")
    ax.set_title(rf"Distribution of max violations ($\alpha = {alpha}$)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"violation_histogram_alpha_{alpha}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"violation_histogram_alpha_{alpha}.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main Functions
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate conservative calibration with multiple seeds."
    )
    parser.add_argument(
        "--latent-dir",
        type=str,
        required=True,
        help="Directory containing latent .pt file (will look for full.pt).",
    )
    parser.add_argument(
        "--n-res",
        type=int,
        default=5000,
        help="Size of fixed res split (default: 5000 for ImageNet, 1000 for CIFAR).",
    )
    parser.add_argument(
        "--n-cal",
        type=int,
        default=20000,
        help="Number of calibration samples per repetition.",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=25000,
        help="Number of evaluation samples per repetition.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=100,
        help="Number of bins for uniform-mass binning.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.1, 0.2],
        help="Alpha values to test (default: 0.01 0.02 0.05 0.1 0.2).",
    )
    parser.add_argument(
        "--n-repetitions",
        type=int,
        default=200,
        help="Number of repetitions (seeds).",
    )
    parser.add_argument(
        "--score-type",
        type=str,
        choices=["msp", "doctor", "margin"],
        default="msp",
        help="Score type: msp, doctor, or margin.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature scaling for score computation.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="For doctor score: use normalized Gini (1-g)/g.",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="For doctor score: use unnormalized Gini (1-g).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=1,
        help="Base seed for fixing res split (must match seed-split from experiments).",
    )
    parser.add_argument(
        "--m-min",
        type=int,
        default=20,
        help="Minimum samples in bin for reliable evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results.",
    )
    parser.set_defaults(normalize=True)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load latent file
    latent_dir = Path(args.latent_dir)
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
    labels = latent["labels"].numpy()
    model_preds = latent["model_preds"].numpy()

    n_total = len(labels)
    print(f"Total samples: {n_total}")
    print(f"Overall error rate: {(model_preds != labels).mean():.4f}")

    # Fix res split using base seed
    print(f"\nFixing res split with base_seed={args.base_seed}, n_res={args.n_res}")
    perm = list(range(n_total))
    rng = random.Random(args.base_seed)
    rng.shuffle(perm)

    res_idx = np.array(perm[:args.n_res])
    pool_idx = np.array(perm[args.n_res:])

    print(f"Res split size: {len(res_idx)}")
    print(f"Pool size (for cal/eval): {len(pool_idx)}")

    # Validate pool size
    if args.n_cal + args.n_eval > len(pool_idx):
        raise ValueError(f"n_cal ({args.n_cal}) + n_eval ({args.n_eval}) > pool size ({len(pool_idx)})")

    # Configuration summary
    config = {
        "latent_path": str(latent_path),
        "n_total": n_total,
        "n_res": args.n_res,
        "n_cal": args.n_cal,
        "n_eval": args.n_eval,
        "n_bins": args.n_bins,
        "alpha_values": args.alpha,
        "n_repetitions": args.n_repetitions,
        "score_type": args.score_type,
        "temperature": args.temperature,
        "normalize": args.normalize,
        "base_seed": args.base_seed,
        "m_min": args.m_min,
    }

    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Save config
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"\nSaved config to: {config_path}")

    # Run repetitions
    print(f"\nRunning {args.n_repetitions} repetitions...")
    per_rep_results = []

    for rep in range(args.n_repetitions):
        seed = args.base_seed + rep + 1  # Offset to avoid using base_seed again

        result = run_single_repetition(
            logits=logits,
            labels=labels,
            model_preds=model_preds,
            pool_idx=pool_idx,
            seed=seed,
            n_cal=args.n_cal,
            n_eval=args.n_eval,
            n_bins=args.n_bins,
            alpha_values=args.alpha,
            score_type=args.score_type,
            temperature=args.temperature,
            normalize=args.normalize,
            m_min=args.m_min,
        )
        per_rep_results.append(result)

        if (rep + 1) % 20 == 0 or rep == 0:
            print(f"  Completed {rep + 1}/{args.n_repetitions} repetitions")

    print(f"\nAll repetitions completed.")

    # Save per-repetition results
    df = pd.DataFrame(per_rep_results)
    csv_path = output_dir / "per_repetition_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved per-repetition results to: {csv_path}")

    # Aggregate results
    print("\nAggregating results...")
    aggregated = aggregate_results(per_rep_results, args.alpha)

    # Save aggregated results
    agg_path = output_dir / "aggregated_results.json"
    agg_path.write_text(json.dumps(aggregated, indent=2))
    print(f"Saved aggregated results to: {agg_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Coverage Validation Results")
    print("=" * 60)

    print(f"\nDiscriminative Metrics:")
    print(f"  ROC-AUC (eval): {aggregated['roc_auc_eval_mean']:.4f} ± {aggregated['roc_auc_eval_std']:.4f}")
    print(f"  FPR@95 (eval): {aggregated['fpr_at_95_eval_mean']:.4f} ± {aggregated['fpr_at_95_eval_std']:.4f}")
    print(f"  AURC (eval): {aggregated['aurc_eval_mean']:.4f} ± {aggregated['aurc_eval_std']:.4f}")

    print(f"\nCoverage Validation (target: π̂(α) ≥ 1 - α):")
    print("-" * 60)
    print(f"{'Alpha':>8} | {'Target':>8} | {'Success Rate':>15} | {'95% CI':>20} | {'Pass':>6}")
    print("-" * 60)

    for alpha in args.alpha:
        alpha_key = f"alpha_{alpha:.4f}".rstrip('0').rstrip('.')
        target = 1 - alpha
        success_rate = aggregated[f"{alpha_key}_success_rate"]
        ci_low = aggregated[f"{alpha_key}_success_rate_ci_low"]
        ci_high = aggregated[f"{alpha_key}_success_rate_ci_high"]
        passed = "✓" if success_rate >= target else "✗"

        print(f"{alpha:>8.3f} | {target:>8.3f} | {success_rate:>15.4f} | [{ci_low:.4f}, {ci_high:.4f}] | {passed:>6}")

    print("-" * 60)

    print(f"\nViolation Statistics (α = 0.05):")
    alpha_key = "alpha_0.05"
    print(f"  Max violation mean: {aggregated[f'{alpha_key}_max_violation_mean']:.6f}")
    print(f"  Max violation std: {aggregated[f'{alpha_key}_max_violation_std']:.6f}")
    print(f"  Max violation 95th percentile: {aggregated[f'{alpha_key}_max_violation_quantiles']['95']:.6f}")

    # Generate plots
    print("\nGenerating plots...")
    plot_coverage_vs_alpha(aggregated, plots_dir)
    plot_violation_ecdf(per_rep_results, args.alpha, plots_dir)

    # Generate histogram for alpha=0.05
    if 0.05 in args.alpha:
        plot_violation_histogram(per_rep_results, 0.05, plots_dir)

    print(f"Plots saved to: {plots_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
