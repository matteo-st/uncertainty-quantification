#!/usr/bin/env python3
"""
Plot ROC-AUC test performance vs number of clusters for Uniform Mass binning.

Shows:
- Different curves for different alpha values (score='upper')
- A curve for score='mean'
- Baseline performance of the uncertified uncertainty score

Usage:
    python scripts/plot_rocauc_vs_nclusters.py --score-name margin --dataset imagenet --model timm_vit_base16_ce
    python scripts/plot_rocauc_vs_nclusters.py --score-name msp --dataset cifar10 --model resnet34_ce

Output:
    docs/figures/<score_name>_<dataset>_<model>_rocauc_vs_nclusters.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Optional
from datetime import datetime

from error_estimation.utils.results_helper import setup_publication_style


# Default run tags
DEFAULT_RUN_TAGS = {
    'margin': {
        'baseline': 'margin-grid-20260120',
        'partition': 'margin-unif-mass-grid-20260120',
    },
    'msp': {
        'baseline': 'msp-grid-20260120',
        'partition': 'msp-unif-mass-grid-20260120',
    },
    'doctor': {
        'baseline': 'doctor-grid-20260120',
        'partition': 'doctor-unif-mass-grid-20260120',
    },
}

# Baseline postprocessor names (MSP uses 'odin' internally)
BASELINE_POSTPROCESSOR = {
    'margin': 'margin',
    'msp': 'odin',
    'doctor': 'doctor',
}

# Display names
SCORE_DISPLAY_NAMES = {
    'margin': 'Margin',
    'msp': 'MSP',
    'doctor': 'Doctor',
}

DATASET_DISPLAY_NAMES = {
    'imagenet': 'ImageNet',
    'cifar100': 'CIFAR-100',
    'cifar10': 'CIFAR-10',
}

MODEL_DISPLAY_NAMES = {
    'timm_vit_base16_ce': 'ViT-B/16',
    'timm_vit_tiny16_ce': 'ViT-Ti/16',
    'resnet34_ce': 'ResNet-34',
    'densenet121_ce': 'DenseNet-121',
}

# Colors for alpha values
ALPHA_COLORS = {
    0.01: '#1f77b4',  # blue
    0.05: '#ff7f0e',  # orange
    0.1: '#2ca02c',   # green
}

# Style for mean score
MEAN_COLOR = '#9467bd'  # purple
BASELINE_COLOR = '#d62728'  # red
STONES_RULE_COLOR = '#7f7f7f'  # gray

# Stone's rule K values: K = ceil(2 * n^(1/3)) where n = calibration set size
# CIFAR-10/100: n = 4000, K = ceil(2 * 4000^(1/3)) = ceil(2 * 15.87) = 32 ~ 30
# ImageNet: n = 50000, K = ceil(2 * 50000^(1/3)) = ceil(2 * 36.84) = 74 ~ 50
STONES_RULE_K = {
    'cifar10': 30,
    'cifar100': 30,
    'imagenet': 50,
}


def load_partition_results(
    results_dir: Path,
    dataset: str,
    model: str,
    run_tag: str,
    n_seeds: int = 9,
) -> pd.DataFrame:
    """
    Load partition results for all seeds and aggregate.

    Returns DataFrame with columns: n_clusters, alpha, score, roc_auc_mean, roc_auc_std
    """
    all_results = []

    for seed in range(1, n_seeds + 1):
        csv_path = (
            results_dir / 'partition_binning' / dataset / model / 'partition' /
            'runs' / run_tag / f'seed-split-{seed}' / 'grid_results.csv'
        )

        if not csv_path.exists():
            print(f"  [WARN] Missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df['seed'] = seed
        all_results.append(df[['n_clusters', 'alpha', 'score', 'roc_auc_test', 'seed']])

    if not all_results:
        raise ValueError(f"No results found for {dataset}/{model}")

    combined = pd.concat(all_results, ignore_index=True)

    # Aggregate by n_clusters, alpha, score
    aggregated = combined.groupby(['n_clusters', 'alpha', 'score']).agg(
        roc_auc_mean=('roc_auc_test', 'mean'),
        roc_auc_std=('roc_auc_test', 'std'),
        n_seeds=('seed', 'count'),
    ).reset_index()

    return aggregated


def load_baseline_results(
    results_dir: Path,
    dataset: str,
    model: str,
    postprocessor: str,
    run_tag: str,
    n_seeds: int = 9,
) -> dict:
    """
    Load baseline results and return best ROC-AUC (selected on res split).

    Returns dict with roc_auc_mean, roc_auc_std
    """
    roc_aucs = []

    for seed in range(1, n_seeds + 1):
        csv_path = (
            results_dir / 'baselines' / dataset / model / postprocessor /
            'runs' / run_tag / f'seed-split-{seed}' / 'grid_results.csv'
        )

        if not csv_path.exists():
            print(f"  [WARN] Missing baseline: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Select best hyperparameters on res split (minimize FPR@95)
        best_idx = df['fpr_res'].idxmin()
        roc_aucs.append(df.loc[best_idx, 'roc_auc_test'])

    if not roc_aucs:
        raise ValueError(f"No baseline results found for {dataset}/{model}/{postprocessor}")

    return {
        'roc_auc_mean': np.mean(roc_aucs),
        'roc_auc_std': np.std(roc_aucs),
        'n_seeds': len(roc_aucs),
    }


def plot_rocauc_vs_nclusters(
    partition_results: pd.DataFrame,
    baseline_results: dict,
    score_name: str,
    dataset: str,
    model: str,
    output_path: Path,
    alphas: Optional[list] = None,
    show_mean: bool = True,
    show_stones_rule: bool = True,
    figsize: tuple = (8, 6),
):
    """
    Create plot of ROC-AUC vs n_clusters.

    Args:
        partition_results: DataFrame with partition results
        baseline_results: Dict with baseline ROC-AUC
        score_name: Name of the score (e.g., 'margin', 'msp')
        dataset: Dataset name
        model: Model name
        output_path: Path to save the figure
        alphas: List of alpha values to plot (default: all available)
        show_mean: Whether to show the mean score curve
        show_stones_rule: Whether to show the Stone's rule vertical line
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get available alpha values
    available_alphas = sorted(partition_results['alpha'].unique())
    if alphas is None:
        alphas = available_alphas

    # Get n_clusters values
    n_clusters_values = sorted(partition_results['n_clusters'].unique())

    # Plot curves for each alpha (score='upper')
    for alpha in alphas:
        if alpha not in available_alphas:
            print(f"  [WARN] Alpha {alpha} not available, skipping")
            continue

        mask = (partition_results['alpha'] == alpha) & (partition_results['score'] == 'upper')
        data = partition_results[mask].sort_values('n_clusters')

        if data.empty:
            continue

        color = ALPHA_COLORS.get(alpha, 'gray')
        ax.errorbar(
            data['n_clusters'],
            data['roc_auc_mean'],
            yerr=data['roc_auc_std'],
            label=f'CD (α={alpha})',
            marker='o',
            color=color,
            capsize=3,
            linewidth=2,
            markersize=6,
        )

    # Plot mean score curve (alpha doesn't matter for mean)
    if show_mean:
        # Use first available alpha for mean (they should all be the same)
        mask = (partition_results['score'] == 'mean')
        data = partition_results[mask].groupby('n_clusters').agg(
            roc_auc_mean=('roc_auc_mean', 'first'),
            roc_auc_std=('roc_auc_std', 'first'),
        ).reset_index().sort_values('n_clusters')

        if not data.empty:
            ax.errorbar(
                data['n_clusters'],
                data['roc_auc_mean'],
                yerr=data['roc_auc_std'],
                label='CD (mean)',
                marker='s',
                color=MEAN_COLOR,
                capsize=3,
                linewidth=2,
                markersize=6,
                linestyle='--',
            )

    # Plot baseline as horizontal line
    baseline_mean = baseline_results['roc_auc_mean']
    baseline_std = baseline_results['roc_auc_std']

    ax.axhline(
        baseline_mean,
        color=BASELINE_COLOR,
        linestyle='-',
        linewidth=2,
        label=f'{SCORE_DISPLAY_NAMES.get(score_name, score_name)} (uncertified)',
    )
    ax.fill_between(
        [min(n_clusters_values), max(n_clusters_values)],
        baseline_mean - baseline_std,
        baseline_mean + baseline_std,
        color=BASELINE_COLOR,
        alpha=0.2,
    )

    # Plot Stone's rule vertical line
    if show_stones_rule and dataset in STONES_RULE_K:
        stones_k = STONES_RULE_K[dataset]
        ax.axvline(
            stones_k,
            color=STONES_RULE_COLOR,
            linestyle='--',
            linewidth=2,
            label=f"Stone's rule (K={stones_k})",
        )

    # Formatting
    ax.set_xlabel('Number of clusters (K)', fontsize=12)
    ax.set_ylabel('ROC-AUC (test)', fontsize=12)

    ax.set_xticks(n_clusters_values)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Set y-axis limits - zoom in on the actual data range
    all_means = list(partition_results['roc_auc_mean']) + [baseline_mean]
    all_stds = list(partition_results['roc_auc_std']) + [baseline_std]

    y_min_data = min(m - s for m, s in zip(all_means, all_stds))
    y_max_data = max(m + s for m, s in zip(all_means, all_stds))

    # Add small padding on each side of the data range
    data_range = y_max_data - y_min_data
    padding = max(data_range * 0.03, 0.001)  # 3% padding, at least 0.1%

    ax.set_ylim(y_min_data - padding, y_max_data + padding)

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()


def main():
    # Setup publication-quality matplotlib style
    setup_publication_style()

    parser = argparse.ArgumentParser(
        description='Plot ROC-AUC vs number of clusters for Uniform Mass binning.'
    )
    parser.add_argument(
        '--score-name',
        type=str,
        required=True,
        choices=['margin', 'msp', 'doctor'],
        help='Uncertainty score name',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['imagenet', 'cifar100', 'cifar10'],
        help='Dataset name',
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., timm_vit_base16_ce, resnet34_ce)',
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('results'),
        help='Path to results directory (default: results)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('docs/figures'),
        help='Path to output directory (default: docs/figures)',
    )
    parser.add_argument(
        '--baseline-run-tag',
        type=str,
        default=None,
        help='Run tag for baseline experiments (default: auto)',
    )
    parser.add_argument(
        '--partition-run-tag',
        type=str,
        default=None,
        help='Run tag for partition experiments (default: auto)',
    )
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=9,
        help='Number of seeds to aggregate (default: 9)',
    )
    parser.add_argument(
        '--alphas',
        type=float,
        nargs='+',
        default=None,
        help='Alpha values to plot (default: all available)',
    )
    parser.add_argument(
        '--no-mean',
        action='store_true',
        help='Do not show the mean score curve',
    )
    parser.add_argument(
        '--no-stones-rule',
        action='store_true',
        help="Do not show the Stone's rule vertical line",
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[8, 6],
        help='Figure size (width height)',
    )
    args = parser.parse_args()

    # Get run tags
    baseline_run_tag = args.baseline_run_tag or DEFAULT_RUN_TAGS[args.score_name]['baseline']
    partition_run_tag = args.partition_run_tag or DEFAULT_RUN_TAGS[args.score_name]['partition']
    baseline_postprocessor = BASELINE_POSTPROCESSOR[args.score_name]

    print(f"Loading results for {args.score_name} / {args.dataset} / {args.model}")
    print(f"  Baseline run tag: {baseline_run_tag}")
    print(f"  Partition run tag: {partition_run_tag}")

    # Load results
    partition_results = load_partition_results(
        args.results_dir,
        args.dataset,
        args.model,
        partition_run_tag,
        args.n_seeds,
    )
    print(f"  Loaded {len(partition_results)} partition configurations")

    baseline_results = load_baseline_results(
        args.results_dir,
        args.dataset,
        args.model,
        baseline_postprocessor,
        baseline_run_tag,
        args.n_seeds,
    )
    print(f"  Baseline ROC-AUC: {baseline_results['roc_auc_mean']:.4f} ± {baseline_results['roc_auc_std']:.4f}")

    # Generate output path: docs/figures/<score_name>/<partition_run_tag>/<dataset>_<model>_rocauc_vs_nclusters.pdf
    output_dir = args.output_dir / args.score_name / partition_run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{args.dataset}_{args.model}_rocauc_vs_nclusters.pdf"
    output_path = output_dir / output_filename

    # Create plot
    plot_rocauc_vs_nclusters(
        partition_results,
        baseline_results,
        args.score_name,
        args.dataset,
        args.model,
        output_path,
        alphas=args.alphas,
        show_mean=not args.no_mean,
        show_stones_rule=not args.no_stones_rule,
        figsize=tuple(args.figsize),
    )

    # Save params.yml (only once per tag folder, append if exists)
    params_path = output_dir / 'params.yml'
    if not params_path.exists():
        params_lines = [
            f"# Parameters used to generate plots in this folder",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            f"score_name: {args.score_name}",
            f"baseline_run_tag: {baseline_run_tag}",
            f"partition_run_tag: {partition_run_tag}",
            f"n_seeds: {args.n_seeds}",
            f"results_dir: {args.results_dir}",
        ]
        params_path.write_text('\n'.join(params_lines))
        print(f"Saved: {params_path}")


if __name__ == '__main__':
    main()
