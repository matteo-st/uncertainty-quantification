#!/usr/bin/env python3
"""
Plot risk-coverage curves comparing raw Doctor score vs Uniform Mass binned score.

Uses cached logits to compute scores without running full experiments.

Usage:
    python scripts/plot_risk_coverage.py --dataset cifar10 --model resnet34_ce --seed 1
    python scripts/plot_risk_coverage.py --dataset cifar100 --model densenet121_ce --seed 1

Output:
    docs/figures/risk_coverage/<dataset>_<model>_seed<seed>_risk_coverage.pdf
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.special import softmax

from error_estimation.utils.results_helper import setup_publication_style


# Stone's rule K values
STONES_RULE_K = {
    'cifar10': 30,
    'cifar100': 30,
    'imagenet': 50,
}

# Calibration set sizes
CAL_SIZES = {
    'cifar10': 4000,
    'cifar100': 4000,
    'imagenet': 20000,
}

DATASET_DISPLAY = {
    'cifar10': 'CIFAR-10',
    'cifar100': 'CIFAR-100',
    'imagenet': 'ImageNet',
}

MODEL_DISPLAY = {
    'resnet34_ce': 'ResNet-34',
    'densenet121_ce': 'DenseNet-121',
    'timm_vit_base16_ce': 'ViT-B/16',
    'timm_vit_tiny16_ce': 'ViT-Ti/16',
}


def load_cached_logits(latent_dir: Path, dataset: str, model: str):
    """Load cached logits and labels."""
    cache_path = latent_dir / f"{dataset}_{model}" / "transform-test_n-epochs-1" / "full.pt"

    if not cache_path.exists():
        raise FileNotFoundError(f"Cached logits not found: {cache_path}")

    data = torch.load(cache_path, map_location='cpu')
    logits = data['logits'].numpy()
    labels = data['labels'].numpy()

    return logits, labels


def compute_doctor_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Compute Doctor (Gini) uncertainty score.
    Higher score = more uncertain.
    """
    probs = softmax(logits / temperature, axis=1)
    # Gini impurity: 1 - sum(p^2)
    gini = 1 - np.sum(probs ** 2, axis=1)
    return gini


def compute_uniform_mass_binning(
    scores_cal: np.ndarray,
    errors_cal: np.ndarray,
    scores_test: np.ndarray,
    n_bins: int,
    alpha: float = 0.05,
    simultaneous: bool = True,
) -> np.ndarray:
    """
    Apply Uniform Mass binning with Hoeffding bound.

    Returns upper confidence bound on error probability for each test sample.
    """
    # Sort calibration scores to get bin edges
    sorted_indices = np.argsort(scores_cal)
    n_cal = len(scores_cal)
    samples_per_bin = n_cal // n_bins

    # Compute bin edges (uniform mass)
    bin_edges = [-np.inf]
    for i in range(1, n_bins):
        edge_idx = i * samples_per_bin
        bin_edges.append(scores_cal[sorted_indices[edge_idx]])
    bin_edges.append(np.inf)

    # Compute error rate and upper bound per bin
    alpha_effective = alpha / n_bins if simultaneous else alpha

    bin_upper_bounds = []
    for i in range(n_bins):
        mask = (scores_cal >= bin_edges[i]) & (scores_cal < bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin > 0:
            error_rate = errors_cal[mask].mean()
            # Hoeffding bound
            hoeffding_term = np.sqrt(np.log(1 / alpha_effective) / (2 * n_bin))
            upper_bound = min(1.0, error_rate + hoeffding_term)
        else:
            upper_bound = 1.0
        bin_upper_bounds.append(upper_bound)

    # Assign test samples to bins and get their upper bounds
    test_scores_binned = np.zeros(len(scores_test))
    for i in range(n_bins):
        mask = (scores_test >= bin_edges[i]) & (scores_test < bin_edges[i + 1])
        test_scores_binned[mask] = bin_upper_bounds[i]

    return test_scores_binned


def compute_risk_coverage_curve(errors: np.ndarray, scores: np.ndarray):
    """
    Compute risk-coverage curve.

    Returns:
        coverages: Coverage values (fraction of samples)
        risks: Risk values (error rate at each coverage)
    """
    # Sort by score (ascending - lower score = more confident)
    sorted_indices = np.argsort(scores)
    sorted_errors = errors[sorted_indices]

    n = len(errors)
    coverages = []
    risks = []

    cumsum_errors = np.cumsum(sorted_errors)

    for i in range(1, n + 1):
        coverage = i / n
        risk = cumsum_errors[i - 1] / i
        coverages.append(coverage)
        risks.append(risk)

    return np.array(coverages), np.array(risks)


def split_data(logits, labels, seed, n_cal=4000, n_test=5000):
    """Split data into calibration and test sets."""
    np.random.seed(seed)
    n_total = len(labels)
    indices = np.random.permutation(n_total)

    # Skip res split (first 1000), use next n_cal for calibration, rest for test
    cal_indices = indices[1000:1000 + n_cal]
    test_indices = indices[1000 + n_cal:1000 + n_cal + n_test]

    return (
        logits[cal_indices], labels[cal_indices],
        logits[test_indices], labels[test_indices],
    )


def main():
    parser = argparse.ArgumentParser(description='Plot risk-coverage curves.')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet34_ce', 'densenet121_ce'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--latent-dir', type=Path, default=Path('latent'))
    parser.add_argument('--output-dir', type=Path, default=Path('docs/figures/risk_coverage'))
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Alpha value for UM binning (default: 0.05)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[6, 4])
    args = parser.parse_args()

    setup_publication_style()

    # Load cached logits
    print(f"Loading cached logits for {args.dataset}/{args.model}...")
    logits, labels = load_cached_logits(args.latent_dir, args.dataset, args.model)
    print(f"  Loaded {len(labels)} samples")

    # Split data
    n_cal = CAL_SIZES[args.dataset]
    logits_cal, labels_cal, logits_test, labels_test = split_data(
        logits, labels, args.seed, n_cal=n_cal
    )
    print(f"  Calibration: {len(labels_cal)}, Test: {len(labels_test)}")

    # Compute predictions and errors
    preds_cal = logits_cal.argmax(axis=1)
    preds_test = logits_test.argmax(axis=1)
    errors_cal = (preds_cal != labels_cal).astype(float)
    errors_test = (preds_test != labels_test).astype(float)

    print(f"  Cal error rate: {errors_cal.mean():.3f}")
    print(f"  Test error rate: {errors_test.mean():.3f}")

    # Compute Doctor scores
    scores_cal = compute_doctor_score(logits_cal, args.temperature)
    scores_test = compute_doctor_score(logits_test, args.temperature)

    # Compute risk-coverage curve for raw Doctor
    cov_doctor, risk_doctor = compute_risk_coverage_curve(errors_test, scores_test)

    # Compute UM binned scores
    n_bins = STONES_RULE_K[args.dataset]
    scores_test_um = compute_uniform_mass_binning(
        scores_cal, errors_cal, scores_test,
        n_bins=n_bins, alpha=args.alpha, simultaneous=True
    )
    cov_um, risk_um = compute_risk_coverage_curve(errors_test, scores_test_um)

    # Plot
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    # Plot raw Doctor
    ax.plot(cov_doctor, risk_doctor, label='Doctor (raw)', color='#d62728', linewidth=2)

    # Plot CD (Certified Detector) curve
    ax.plot(cov_um, risk_um, label=f'CD (α={args.alpha})', color='#1f77b4', linewidth=2)

    # Add horizontal reference (random classifier)
    ax.axhline(errors_test.mean(), color='k', linestyle=':', alpha=0.5, linewidth=1, label='Random')

    ax.set_xlabel('Coverage', fontsize=12)
    ax.set_ylabel('Risk (Error Rate)', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, None])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Title with dataset/model info
    dataset_display = DATASET_DISPLAY[args.dataset]
    model_display = MODEL_DISPLAY[args.model]
    ax.set_title(f'{dataset_display} - {model_display} (seed {args.seed})', fontsize=12)

    plt.tight_layout()

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.dataset}_{args.model}_seed{args.seed}_risk_coverage.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()


if __name__ == '__main__':
    main()
