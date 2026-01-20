#!/usr/bin/env python3
"""
Generate LaTeX results tables for Margin and MSP experiments.

This script compares raw scores (Margin, MSP) vs Uniform Mass binning (UM).
- Baseline: best hyperparameters selected on res split (minimize FPR@95)
- Partition: Rice Rule for K selection, score=upper, alpha=0.05

Rice Rule: K = ceil(2 * n^(1/3))
- CIFAR (n=4000): K = ceil(2 * 15.87) = 32 -> closest grid value is 30
- ImageNet (n=20000): K = ceil(2 * 27.14) = 55 -> closest grid value is 50

Usage:
    python scripts/generate_results_tables.py

Output:
    - docs/margin_results_table.tex
    - docs/msp_results_table.tex
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


# Rice Rule K values (closest grid value to ceil(2 * n^(1/3)))
RICE_K = {
    'cifar10': 30,   # n=4000, K=32, closest=30
    'cifar100': 30,  # n=4000, K=32, closest=30
    'imagenet': 50,  # n=20000, K=55, closest=50
}

# Dataset/model configurations
CONFIGS = [
    ('imagenet', 'timm_vit_base16_ce', 'ViT-B/16'),
    ('imagenet', 'timm_vit_tiny16_ce', 'ViT-Ti/16'),
    ('cifar100', 'resnet34_ce', 'ResNet-34'),
    ('cifar100', 'densenet121_ce', 'DenseNet-121'),
    ('cifar10', 'resnet34_ce', 'ResNet-34'),
    ('cifar10', 'densenet121_ce', 'DenseNet-121'),
]

DATASET_NAMES = {
    'imagenet': 'ImageNet',
    'cifar100': 'CIFAR-100',
    'cifar10': 'CIFAR-10',
}


def get_baseline_results(
    results_dir: Path,
    score_type: str,
    run_tag: str,
    n_seeds: int = 9,
) -> list[dict]:
    """
    Get baseline results, selecting best hyperparameters on res split per seed.

    Args:
        results_dir: Path to results directory
        score_type: 'margin' or 'odin' (for MSP)
        run_tag: Run tag for the experiment
        n_seeds: Number of seeds to aggregate

    Returns:
        List of dicts with dataset, model, and aggregated metrics
    """
    results = []

    for dataset, model, model_name in CONFIGS:
        roc_aucs = []
        fprs = []

        for seed in range(1, n_seeds + 1):
            csv_path = (
                results_dir / 'baselines' / dataset / model / score_type /
                'runs' / run_tag / f'seed-split-{seed}' / 'grid_results.csv'
            )

            if not csv_path.exists():
                print(f"  [WARN] Missing baseline: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            # Select best hyperparameters on res split (minimize FPR@95)
            best_idx = df['fpr_res'].idxmin()
            roc_aucs.append(df.loc[best_idx, 'roc_auc_test'])
            fprs.append(df.loc[best_idx, 'fpr_test'])

        if roc_aucs:
            results.append({
                'dataset': dataset,
                'model': model_name,
                'roc_auc_mean': np.mean(roc_aucs),
                'roc_auc_std': np.std(roc_aucs),
                'fpr_mean': np.mean(fprs),
                'fpr_std': np.std(fprs),
                'n_seeds': len(roc_aucs),
            })

    return results


def get_partition_results(
    results_dir: Path,
    run_tag: str,
    n_seeds: int = 9,
    alpha: float = 0.05,
    score: str = 'upper',
) -> list[dict]:
    """
    Get partition (Uniform Mass) results with Rice Rule K selection.

    Args:
        results_dir: Path to results directory
        run_tag: Run tag for the experiment
        n_seeds: Number of seeds to aggregate
        alpha: Confidence level for binning
        score: Score type ('mean' or 'upper')

    Returns:
        List of dicts with dataset, model, and aggregated metrics
    """
    results = []

    for dataset, model, model_name in CONFIGS:
        rice_k = RICE_K[dataset]
        roc_aucs = []
        fprs = []

        for seed in range(1, n_seeds + 1):
            csv_path = (
                results_dir / 'partition_binning' / dataset / model / 'partition' /
                'runs' / run_tag / f'seed-split-{seed}' / 'grid_results.csv'
            )

            if not csv_path.exists():
                print(f"  [WARN] Missing partition: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            # Filter for Rice K, specified score and alpha
            mask = (
                (df['n_clusters'] == rice_k) &
                (df['score'] == score) &
                (df['alpha'] == alpha)
            )
            filtered = df[mask]

            if filtered.empty:
                print(f"  [WARN] No match for {dataset}/{model} seed {seed} "
                      f"with K={rice_k}, score={score}, alpha={alpha}")
                continue

            row = filtered.iloc[0]
            roc_aucs.append(row['roc_auc_test'])
            fprs.append(row['fpr_test'])

        if roc_aucs:
            results.append({
                'dataset': dataset,
                'model': model_name,
                'roc_auc_mean': np.mean(roc_aucs),
                'roc_auc_std': np.std(roc_aucs),
                'fpr_mean': np.mean(fprs),
                'fpr_std': np.std(fprs),
                'n_seeds': len(roc_aucs),
            })

    return results


def format_metric(mean: float, std: float, bold: bool = False) -> str:
    """
    Format metric as .XXX±.XXX for LaTeX.

    Args:
        mean: Mean value
        std: Standard deviation
        bold: Whether to bold the value

    Returns:
        LaTeX formatted string
    """
    # Format as .XXX (3 decimal places)
    val = f".{int(mean * 1000):03d}" if mean < 1 else f"{mean:.3f}"
    std_str = f".{int(std * 1000):03d}" if std < 1 else f"{std:.3f}"

    if bold:
        return f"\\textbf{{{val}}}{{\\scriptsize $\\pm${std_str}}}"
    return f"{val}{{\\scriptsize $\\pm${std_str}}}"


def generate_latex_table(
    score_name: str,
    baseline_results: list[dict],
    partition_results: list[dict],
) -> str:
    """
    Generate LaTeX table comparing baseline vs partition results.

    Args:
        score_name: Name of the score (e.g., 'Margin', 'MSP')
        baseline_results: List of baseline result dicts
        partition_results: List of partition result dicts

    Returns:
        LaTeX table as string
    """
    # Create lookup for partition results
    partition_lookup = {
        (r['dataset'], r['model']): r for r in partition_results
    }

    lines = [
        f"% {score_name} vs Uniform Mass binning (Rice's rule). Mean ± std over 9 seeds.",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{score_name} vs.\\ Uniform Mass binning (Rice's rule). "
        "Mean $\\pm$ std over 9 seeds.}",
        f"\\label{{tab:{score_name.lower()}_vs_uniform_mass}}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{@{}llcccc@{}}",
        "\\toprule",
        "& & \\multicolumn{2}{c}{\\textbf{ROC-AUC} $\\uparrow$} "
        "& \\multicolumn{2}{c}{\\textbf{FPR@95} $\\downarrow$} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        f"\\textbf{{Dataset}} & \\textbf{{Model}} & {score_name} & UM "
        f"& {score_name} & UM \\\\",
        "\\midrule",
    ]

    for b in baseline_results:
        ds = DATASET_NAMES[b['dataset']]
        model = b['model']
        p = partition_lookup.get((b['dataset'], b['model']))

        if p is None:
            # No partition result available
            roc_b = format_metric(b['roc_auc_mean'], b['roc_auc_std'], bold=True)
            fpr_b = format_metric(b['fpr_mean'], b['fpr_std'], bold=True)
            lines.append(f"{ds} & {model} & {roc_b} & -- & {fpr_b} & -- \\\\")
        else:
            # Compare and bold the better result
            roc_b_better = b['roc_auc_mean'] > p['roc_auc_mean']
            fpr_b_better = b['fpr_mean'] < p['fpr_mean']

            roc_b = format_metric(b['roc_auc_mean'], b['roc_auc_std'], bold=roc_b_better)
            roc_p = format_metric(p['roc_auc_mean'], p['roc_auc_std'], bold=not roc_b_better)
            fpr_b = format_metric(b['fpr_mean'], b['fpr_std'], bold=fpr_b_better)
            fpr_p = format_metric(p['fpr_mean'], p['fpr_std'], bold=not fpr_b_better)

            lines.append(f"{ds} & {model} & {roc_b} & {roc_p} & {fpr_b} & {fpr_p} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX results tables for Margin and MSP experiments.'
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
        default=Path('docs'),
        help='Path to output directory (default: docs)',
    )
    parser.add_argument(
        '--margin-baseline-tag',
        default='margin-grid-20260120',
        help='Run tag for margin baseline experiments',
    )
    parser.add_argument(
        '--margin-partition-tag',
        default='margin-unif-mass-grid-20260120',
        help='Run tag for margin partition experiments',
    )
    parser.add_argument(
        '--msp-baseline-tag',
        default='msp-grid-20260120',
        help='Run tag for MSP baseline experiments',
    )
    parser.add_argument(
        '--msp-partition-tag',
        default='msp-unif-mass-grid-20260120',
        help='Run tag for MSP partition experiments',
    )
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=9,
        help='Number of seeds to aggregate (default: 9)',
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate Margin table
    print("=" * 60)
    print("Generating Margin table...")
    print("=" * 60)

    margin_baseline = get_baseline_results(
        args.results_dir, 'margin', args.margin_baseline_tag, args.n_seeds
    )
    margin_partition = get_partition_results(
        args.results_dir, args.margin_partition_tag, args.n_seeds
    )
    margin_table = generate_latex_table('Margin', margin_baseline, margin_partition)

    margin_output = args.output_dir / 'margin_results_table.tex'
    margin_output.write_text(margin_table)
    print(f"\nSaved to: {margin_output}")
    print("\n" + margin_table)

    # Generate MSP table
    print("\n" + "=" * 60)
    print("Generating MSP table...")
    print("=" * 60)

    # Note: MSP uses 'odin' as the postprocessor name in baselines
    msp_baseline = get_baseline_results(
        args.results_dir, 'odin', args.msp_baseline_tag, args.n_seeds
    )
    msp_partition = get_partition_results(
        args.results_dir, args.msp_partition_tag, args.n_seeds
    )
    msp_table = generate_latex_table('MSP', msp_baseline, msp_partition)

    msp_output = args.output_dir / 'msp_results_table.tex'
    msp_output.write_text(msp_table)
    print(f"\nSaved to: {msp_output}")
    print("\n" + msp_table)

    # Print data availability summary
    print("\n" + "=" * 60)
    print("DATA AVAILABILITY SUMMARY")
    print("=" * 60)

    print("\nMargin baseline:")
    for r in margin_baseline:
        print(f"  {r['dataset']}/{r['model']}: {r['n_seeds']} seeds")

    print("\nMargin partition (UM):")
    for r in margin_partition:
        print(f"  {r['dataset']}/{r['model']}: {r['n_seeds']} seeds")

    print("\nMSP baseline:")
    for r in msp_baseline:
        print(f"  {r['dataset']}/{r['model']}: {r['n_seeds']} seeds")

    print("\nMSP partition (UM):")
    for r in msp_partition:
        print(f"  {r['dataset']}/{r['model']}: {r['n_seeds']} seeds")


if __name__ == '__main__':
    main()
