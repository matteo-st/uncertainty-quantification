#!/usr/bin/env python3
"""
Plot 2D heatmap of two uncertainty scores with empirical error rate in each cell.

Usage:
    python scripts/plot_score_heatmap.py --dataset cifar10 --model resnet34_ce \
        --score-x gini --score-y margin --n-bins 20 --seed 1

    python scripts/plot_score_heatmap.py --dataset cifar100 --model densenet121_ce \
        --score-x msp --score-y gini --n-bins 30 --split cal
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from error_estimation.utils.config import Config
from error_estimation.utils.paths import DATA_DIR, RESULTS_DIR, LATENTS_DIR


# Score computation functions
def compute_gini(logits: torch.Tensor, temperature: float = 1.0, normalize: bool = True) -> torch.Tensor:
    """Compute Gini/Doctor score (higher = more uncertain)."""
    g = torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1)
    if normalize:
        return (1 - g) / g
    return 1 - g


def compute_margin(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute Margin score (higher = more uncertain)."""
    probs = torch.softmax(logits / temperature, dim=1)
    top2 = torch.topk(probs, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return 1.0 - margin


def compute_msp(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute MSP score (higher = more uncertain)."""
    return -torch.softmax(logits / temperature, dim=1).max(dim=1)[0]


def compute_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute Entropy score (higher = more uncertain)."""
    probs = torch.softmax(logits / temperature, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1)


SCORE_FUNCTIONS = {
    "gini": compute_gini,
    "doctor": compute_gini,
    "margin": compute_margin,
    "msp": compute_msp,
    "entropy": compute_entropy,
}

SCORE_DISPLAY_NAMES = {
    "gini": "Doctor (Gini)",
    "doctor": "Doctor (Gini)",
    "margin": "Margin",
    "msp": "MSP",
    "entropy": "Entropy",
}


def load_logits(latents_dir: Path, dataset: str, model: str) -> torch.Tensor:
    """Load pre-computed logits."""
    logits_path = latents_dir / f"{dataset}_{model}" / "transform-test_n-epochs-1" / "full.pt"
    if not logits_path.exists():
        raise FileNotFoundError(f"Logits not found: {logits_path}")
    data = torch.load(logits_path, map_location="cpu")
    if isinstance(data, dict):
        return data.get("logits", data.get("output")), data.get("labels")
    return data, None


def compute_split_indices(n_total: int, n_res: int, n_cal: int, n_test: int, seed: int, split: str) -> list:
    """Compute split indices by shuffling with seed (matching dataloader.py logic)."""
    import random
    perm = list(range(n_total))
    random.seed(seed)
    random.shuffle(perm)

    if n_res == 0:
        cal_idx = perm[:n_cal]
        test_idx = perm[n_total - n_test:]
        splits = {"cal": cal_idx, "test": test_idx}
    else:
        cal_idx = perm[:n_cal]
        res_idx = perm[n_cal:n_cal + n_res]
        test_idx = perm[n_total - n_test:]
        splits = {"res": res_idx, "cal": cal_idx, "test": test_idx}

    if split not in splits:
        raise ValueError(f"Split '{split}' not available. Available: {list(splits.keys())}")
    return splits[split]


def load_split_config(dataset_config_path: Path) -> dict:
    """Load split configuration from dataset config."""
    cfg = Config(dataset_config_path)
    return {
        "n_res": cfg.get("n_samples", {}).get("res", 1000),
        "n_cal": cfg.get("n_samples", {}).get("cal", 4000),
        "n_test": cfg.get("n_samples", {}).get("test", 5000),
    }


def plot_heatmap(
    score_x: np.ndarray,
    score_y: np.ndarray,
    errors: np.ndarray,
    n_bins: int,
    score_x_name: str,
    score_y_name: str,
    title: str,
    output_path: Path,
    figsize: tuple = (10, 8),
    show_counts: bool = True,
    min_samples: int = 5,
):
    """
    Create 2D heatmap of error rates.

    Args:
        score_x: First uncertainty score (N,)
        score_y: Second uncertainty score (N,)
        errors: Binary error labels (N,)
        n_bins: Number of bins per dimension
        score_x_name: Name for x-axis
        score_y_name: Name for y-axis
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size
        show_counts: Whether to annotate cells with sample counts
        min_samples: Minimum samples in a cell to show error rate
    """
    # Create bin edges using quantiles for more uniform distribution
    x_edges = np.percentile(score_x, np.linspace(0, 100, n_bins + 1))
    y_edges = np.percentile(score_y, np.linspace(0, 100, n_bins + 1))

    # Make edges strictly increasing
    x_edges = np.unique(x_edges)
    y_edges = np.unique(y_edges)

    # Compute 2D histogram of errors and counts
    error_sum, x_edges, y_edges = np.histogram2d(
        score_x, score_y, bins=[x_edges, y_edges], weights=errors
    )
    counts, _, _ = np.histogram2d(score_x, score_y, bins=[x_edges, y_edges])

    # Compute error rate (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        error_rate = np.where(counts >= min_samples, error_sum / counts, np.nan)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Custom colormap: green (low error) -> yellow -> red (high error)
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    cmap = LinearSegmentedColormap.from_list('error_cmap', colors)
    cmap.set_bad(color='lightgray')  # For NaN values

    # Plot heatmap
    im = ax.imshow(
        error_rate.T,
        origin='lower',
        aspect='auto',
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Empirical Error Rate', shrink=0.8)

    # Annotate cells with counts if requested
    if show_counts and n_bins <= 15:
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        for i, x in enumerate(x_centers):
            for j, y in enumerate(y_centers):
                count = int(counts[i, j])
                if count >= min_samples:
                    err = error_rate[i, j]
                    text_color = 'white' if err > 0.5 else 'black'
                    ax.text(x, y, f'{err:.2f}\n({count})',
                            ha='center', va='center', fontsize=7, color=text_color)

    # Labels and title
    ax.set_xlabel(SCORE_DISPLAY_NAMES.get(score_x_name, score_x_name), fontsize=12)
    ax.set_ylabel(SCORE_DISPLAY_NAMES.get(score_y_name, score_y_name), fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add statistics text
    total_samples = len(errors)
    total_errors = errors.sum()
    overall_error_rate = total_errors / total_samples
    stats_text = f'N={total_samples:,}, Error Rate={overall_error_rate:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot 2D heatmap of uncertainty scores vs error rate')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--score-x', type=str, required=True, choices=list(SCORE_FUNCTIONS.keys()))
    parser.add_argument('--score-y', type=str, required=True, choices=list(SCORE_FUNCTIONS.keys()))
    parser.add_argument('--n-bins', type=int, default=20, help='Number of bins per dimension')
    parser.add_argument('--seed', type=int, default=1, help='Seed split to use')
    parser.add_argument('--split', type=str, default='test', choices=['res', 'cal', 'test'])
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--latents-dir', type=Path, default=LATENTS_DIR)
    parser.add_argument('--output-dir', type=Path, default=Path('docs/figures/heatmaps'))
    parser.add_argument('--min-samples', type=int, default=5, help='Min samples per cell to show')
    parser.add_argument('--no-counts', action='store_true', help='Do not show sample counts')
    args = parser.parse_args()

    print(f"Loading logits for {args.dataset}/{args.model}...")
    logits, labels = load_logits(args.latents_dir, args.dataset, args.model)

    # Load split indices
    if args.dataset == 'imagenet':
        dataset_config = Path(f'configs/datasets/imagenet/imagenet_n_res-5000_n-cal-20000.yml')
    else:
        dataset_config = Path(f'configs/datasets/{args.dataset}/{args.dataset}_n_res-1000_n-cal-4000_all-seeds.yml')

    # Load split configuration and compute indices
    split_cfg = load_split_config(dataset_config)
    n_total = len(logits)
    indices = compute_split_indices(
        n_total=n_total,
        n_res=split_cfg["n_res"],
        n_cal=split_cfg["n_cal"],
        n_test=split_cfg["n_test"],
        seed=args.seed,
        split=args.split
    )
    print(f"Using {len(indices)} samples from {args.split} split (seed {args.seed})")

    # Get subset
    logits_split = logits[indices]
    if labels is not None:
        labels_split = labels[indices]
    else:
        # Need to load labels from dataset
        from error_estimation.utils.datasets import get_dataset
        ds = get_dataset(args.dataset, DATA_DIR, split='test')
        all_labels = torch.tensor([ds[i][1] for i in range(len(ds))])
        labels_split = all_labels[indices]

    # Compute predictions and errors
    preds = logits_split.argmax(dim=1)
    errors = (preds != labels_split).float().numpy()

    # Compute scores
    print(f"Computing scores: {args.score_x}, {args.score_y}...")
    score_x = SCORE_FUNCTIONS[args.score_x](logits_split, args.temperature).numpy()
    score_y = SCORE_FUNCTIONS[args.score_y](logits_split, args.temperature).numpy()

    # Plot
    title = f'{args.dataset.upper()} - {args.model}\n{args.split} split (seed {args.seed})'
    output_filename = f'{args.dataset}_{args.model}_{args.score_x}_vs_{args.score_y}_{args.split}_seed{args.seed}.pdf'
    output_path = args.output_dir / output_filename

    plot_heatmap(
        score_x=score_x,
        score_y=score_y,
        errors=errors,
        n_bins=args.n_bins,
        score_x_name=args.score_x,
        score_y_name=args.score_y,
        title=title,
        output_path=output_path,
        show_counts=not args.no_counts,
        min_samples=args.min_samples,
    )

    # Print correlation info
    corr = np.corrcoef(score_x, score_y)[0, 1]
    print(f"Correlation between {args.score_x} and {args.score_y}: {corr:.3f}")


if __name__ == '__main__':
    main()
