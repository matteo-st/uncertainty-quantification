#!/usr/bin/env python3
"""
Multi-panel plot of ROC-AUC test performance vs number of clusters for Uniform Mass binning.

Creates a 2x3 grid with all dataset/model combinations and a shared legend.

Usage:
    python scripts/plot_rocauc_vs_nclusters_multiplot.py --score-name doctor --partition-run-tag doctor-unif-mass-sim-grid-20260120
    python scripts/plot_rocauc_vs_nclusters_multiplot.py --score-name margin --partition-run-tag margin-unif-mass-sim-grid-20260120

Output:
    docs/figures/<score_name>/<partition_run_tag>/multiplot_rocauc_vs_nclusters.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
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
STONES_RULE_K = {
    'cifar10': 30,
    'cifar100': 30,
    'imagenet': 50,
}

# Dataset/model combinations for the 2x3 grid
PLOT_CONFIGS = [
    # Row 1: ImageNet
    ('imagenet', 'timm_vit_base16_ce'),
    ('imagenet', 'timm_vit_tiny16_ce'),
    # Row 1 col 3: CIFAR-100
    ('cifar100', 'resnet34_ce'),
    # Row 2: CIFAR-100 + CIFAR-10
    ('cifar100', 'densenet121_ce'),
    ('cifar10', 'resnet34_ce'),
    ('cifar10', 'densenet121_ce'),
]


def load_partition_results(
    results_dir: Path,
    dataset: str,
    model: str,
    run_tag: str,
    n_seeds: int = 9,
) -> pd.DataFrame:
    """Load partition results for all seeds and aggregate."""
    all_results = []

    for seed in range(1, n_seeds + 1):
        csv_path = (
            results_dir / 'partition_binning' / dataset / model / 'partition' /
            'runs' / run_tag / f'seed-split-{seed}' / 'grid_results.csv'
        )

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df['seed'] = seed
        all_results.append(df[['n_clusters', 'alpha', 'score', 'roc_auc_test', 'seed']])

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

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
    """Load baseline results and return best ROC-AUC (selected on res split)."""
    roc_aucs = []

    for seed in range(1, n_seeds + 1):
        csv_path = (
            results_dir / 'baselines' / dataset / model / postprocessor /
            'runs' / run_tag / f'seed-split-{seed}' / 'grid_results.csv'
        )

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        best_idx = df['fpr_res'].idxmin()
        roc_aucs.append(df.loc[best_idx, 'roc_auc_test'])

    if not roc_aucs:
        return None

    return {
        'roc_auc_mean': np.mean(roc_aucs),
        'roc_auc_std': np.std(roc_aucs),
        'n_seeds': len(roc_aucs),
    }


def plot_single_panel(
    ax,
    partition_results: pd.DataFrame,
    baseline_results: dict,
    score_name: str,
    dataset: str,
    model: str,
    alphas: list = None,
    show_mean: bool = True,
    show_stones_rule: bool = True,
):
    """Plot a single panel of the multiplot."""
    if partition_results.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return [], []

    # Get available alpha values
    available_alphas = sorted(partition_results['alpha'].unique())
    if alphas is None:
        alphas = available_alphas

    # Get n_clusters values
    n_clusters_values = sorted(partition_results['n_clusters'].unique())

    lines = []
    labels = []

    # Plot curves for each alpha (score='upper')
    for alpha in alphas:
        if alpha not in available_alphas:
            continue

        mask = (partition_results['alpha'] == alpha) & (partition_results['score'] == 'upper')
        data = partition_results[mask].sort_values('n_clusters')

        if data.empty:
            continue

        color = ALPHA_COLORS.get(alpha, 'gray')
        line = ax.errorbar(
            data['n_clusters'],
            data['roc_auc_mean'],
            yerr=data['roc_auc_std'],
            marker='o',
            color=color,
            capsize=3,
            linewidth=2,
            markersize=5,
        )
        lines.append(line)
        labels.append(f'UM ($\\alpha$={alpha})')

    # Plot mean score curve
    if show_mean:
        mask = (partition_results['score'] == 'mean')
        data = partition_results[mask].groupby('n_clusters').agg(
            roc_auc_mean=('roc_auc_mean', 'first'),
            roc_auc_std=('roc_auc_std', 'first'),
        ).reset_index().sort_values('n_clusters')

        if not data.empty:
            line = ax.errorbar(
                data['n_clusters'],
                data['roc_auc_mean'],
                yerr=data['roc_auc_std'],
                marker='s',
                color=MEAN_COLOR,
                capsize=3,
                linewidth=2,
                markersize=5,
                linestyle='--',
            )
            lines.append(line)
            labels.append('UM (mean)')

    # Plot baseline as horizontal line
    if baseline_results:
        baseline_mean = baseline_results['roc_auc_mean']
        baseline_std = baseline_results['roc_auc_std']

        line = ax.axhline(
            baseline_mean,
            color=BASELINE_COLOR,
            linestyle='-',
            linewidth=2,
        )
        ax.fill_between(
            [min(n_clusters_values), max(n_clusters_values)],
            baseline_mean - baseline_std,
            baseline_mean + baseline_std,
            color=BASELINE_COLOR,
            alpha=0.2,
        )
        lines.append(line)
        labels.append(f'{SCORE_DISPLAY_NAMES.get(score_name, score_name)} (uncertified)')

    # Plot Stone's rule vertical line (no label for legend)
    if show_stones_rule and dataset in STONES_RULE_K:
        stones_k = STONES_RULE_K[dataset]
        ax.axvline(
            stones_k,
            color=STONES_RULE_COLOR,
            linestyle='--',
            linewidth=1.5,
        )

    # Formatting
    ax.set_xticks(n_clusters_values)
    ax.grid(True, alpha=0.3)

    # Set y-axis limits
    all_means = list(partition_results['roc_auc_mean'])
    all_stds = list(partition_results['roc_auc_std'])
    if baseline_results:
        all_means.append(baseline_results['roc_auc_mean'])
        all_stds.append(baseline_results['roc_auc_std'])

    y_min_data = min(m - s for m, s in zip(all_means, all_stds))
    y_max_data = max(m + s for m, s in zip(all_means, all_stds))

    data_range = y_max_data - y_min_data
    padding = max(data_range * 0.3, 0.005)

    ax.set_ylim(y_min_data - padding, y_max_data + padding)

    # Add subplot label
    dataset_display = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    model_display = MODEL_DISPLAY_NAMES.get(model, model)
    ax.text(0.02, 0.98, f'{dataset_display} - {model_display}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            fontweight='bold')

    return lines, labels


def main():
    parser = argparse.ArgumentParser(
        description='Create multi-panel ROC-AUC vs n_clusters plot.'
    )
    parser.add_argument(
        '--score-name',
        type=str,
        required=True,
        choices=['margin', 'msp', 'doctor'],
        help='Uncertainty score name',
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
        default=[14, 8],
        help='Figure size (width height)',
    )
    args = parser.parse_args()

    # Setup publication style
    setup_publication_style()

    # Get run tags
    baseline_run_tag = args.baseline_run_tag or DEFAULT_RUN_TAGS[args.score_name]['baseline']
    partition_run_tag = args.partition_run_tag or DEFAULT_RUN_TAGS[args.score_name]['partition']
    baseline_postprocessor = BASELINE_POSTPROCESSOR[args.score_name]

    print(f"Creating multiplot for {args.score_name}")
    print(f"  Baseline run tag: {baseline_run_tag}")
    print(f"  Partition run tag: {partition_run_tag}")

    # Create figure with 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=tuple(args.figsize))
    axes = axes.flatten()

    all_lines = []
    all_labels = []

    # Plot each panel
    for idx, (dataset, model) in enumerate(PLOT_CONFIGS):
        print(f"  Loading {dataset}/{model}...")

        partition_results = load_partition_results(
            args.results_dir,
            dataset,
            model,
            partition_run_tag,
            args.n_seeds,
        )

        baseline_results = load_baseline_results(
            args.results_dir,
            dataset,
            model,
            baseline_postprocessor,
            baseline_run_tag,
            args.n_seeds,
        )

        lines, labels = plot_single_panel(
            axes[idx],
            partition_results,
            baseline_results,
            args.score_name,
            dataset,
            model,
            alphas=args.alphas,
            show_mean=not args.no_mean,
            show_stones_rule=not args.no_stones_rule,
        )

        # Collect lines/labels for shared legend (only from first panel)
        if idx == 0:
            all_lines = lines
            all_labels = labels

    # Set common axis labels
    for idx, ax in enumerate(axes):
        if idx >= 3:  # Bottom row
            ax.set_xlabel('Number of clusters (K)', fontsize=11)
        if idx % 3 == 0:  # Left column
            ax.set_ylabel('ROC-AUC (test)', fontsize=11)

    # Add shared legend at the bottom
    fig.legend(
        all_lines, all_labels,
        loc='lower center',
        ncol=len(all_labels),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Make room for legend

    # Save figure
    output_dir = args.output_dir / args.score_name / partition_run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'multiplot_rocauc_vs_nclusters.pdf'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.close()

    # Save params.yml
    params_path = output_dir / 'params_multiplot.yml'
    params_lines = [
        f"# Parameters used to generate multiplot",
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
