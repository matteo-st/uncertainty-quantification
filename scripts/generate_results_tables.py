#!/usr/bin/env python3
"""
Generate LaTeX results tables for Margin, MSP, and Doctor experiments.

This script compares raw scores (Margin, MSP, Doctor) vs Uniform Mass binning (UM).
- Baseline: best hyperparameters selected on res split (minimize FPR@95)
- Partition: Rice Rule for K selection, score=upper, alpha=0.05

Rice Rule: K = ceil(2 * n^(1/3))
- CIFAR (n=4000): K = ceil(2 * 15.87) = 32 -> closest grid value is 30
- ImageNet (n=20000): K = ceil(2 * 27.14) = 55 -> closest grid value is 50

Usage:
    python scripts/generate_results_tables.py

Output:
    docs/tables/<tag>/
    ├── table.tex          # LaTeX table
    └── params.yml         # Parameters used to generate the table
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime


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


def save_table_with_params(
    output_dir: Path,
    tag: str,
    table_content: str,
    params: dict,
) -> Path:
    """
    Save table and parameters to a tagged subfolder.

    Args:
        output_dir: Base output directory (e.g., docs/tables)
        tag: Tag for the subfolder (e.g., 'margin_vs_um_20260120')
        table_content: LaTeX table content
        params: Dictionary of parameters used to generate the table

    Returns:
        Path to the created subfolder
    """
    # Create subfolder
    table_dir = output_dir / tag
    table_dir.mkdir(parents=True, exist_ok=True)

    # Save table
    table_path = table_dir / 'table.tex'
    table_path.write_text(table_content)

    # Save parameters as YAML-like format
    params_path = table_dir / 'params.yml'
    params_lines = [
        f"# Parameters used to generate this table",
        f"# Generated: {datetime.now().isoformat()}",
        "",
    ]
    for key, value in params.items():
        if isinstance(value, dict):
            params_lines.append(f"{key}:")
            for k, v in value.items():
                params_lines.append(f"  {k}: {v}")
        else:
            params_lines.append(f"{key}: {value}")

    params_path.write_text('\n'.join(params_lines))

    return table_dir


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
        default=Path('docs/tables'),
        help='Path to output directory (default: docs/tables)',
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
        '--doctor-baseline-tag',
        default='doctor-grid-20260120',
        help='Run tag for Doctor baseline experiments',
    )
    parser.add_argument(
        '--doctor-partition-tag',
        default='doctor-unif-mass-grid-20260120',
        help='Run tag for Doctor partition experiments',
    )
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=9,
        help='Number of seeds to aggregate (default: 9)',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Alpha for partition binning (default: 0.05)',
    )
    parser.add_argument(
        '--score',
        type=str,
        default='upper',
        choices=['mean', 'upper'],
        help='Score type for partition binning (default: upper)',
    )
    parser.add_argument(
        '--tag-suffix',
        type=str,
        default='',
        help='Optional suffix to add to table tags',
    )
    parser.add_argument(
        '--scores',
        type=str,
        nargs='+',
        default=['margin', 'msp', 'doctor'],
        choices=['margin', 'msp', 'doctor'],
        help='Which score tables to generate (default: all)',
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Common partition parameters
    partition_params = {
        'alpha': args.alpha,
        'score': args.score,
        'k_selection': 'rice_rule',
        'rice_k': RICE_K,
    }

    # Generate Margin table
    margin_baseline, margin_partition = None, None
    if 'margin' in args.scores:
        print("=" * 60)
        print("Generating Margin table...")
        print("=" * 60)

        margin_baseline = get_baseline_results(
            args.results_dir, 'margin', args.margin_baseline_tag, args.n_seeds
        )
        margin_partition = get_partition_results(
            args.results_dir, args.margin_partition_tag, args.n_seeds,
            alpha=args.alpha, score=args.score,
        )
        margin_table = generate_latex_table('Margin', margin_baseline, margin_partition)

        # Save with parameters
        margin_tag = f"margin_vs_um{args.tag_suffix}" if args.tag_suffix else "margin_vs_um"
        margin_params = {
            'score_name': 'Margin',
            'baseline': {
                'postprocessor': 'margin',
                'run_tag': args.margin_baseline_tag,
                'selection': 'best on res (minimize FPR@95)',
            },
            'partition': {
                'run_tag': args.margin_partition_tag,
                **partition_params,
            },
            'n_seeds': args.n_seeds,
            'results_dir': str(args.results_dir),
        }
        margin_dir = save_table_with_params(args.output_dir, margin_tag, margin_table, margin_params)
        print(f"\nSaved to: {margin_dir}/")
        print(f"  - table.tex")
        print(f"  - params.yml")
        print("\n" + margin_table)

    # Generate MSP table
    msp_baseline, msp_partition = None, None
    if 'msp' in args.scores:
        print("\n" + "=" * 60)
        print("Generating MSP table...")
        print("=" * 60)

        # Note: MSP uses 'odin' as the postprocessor name in baselines
        msp_baseline = get_baseline_results(
            args.results_dir, 'odin', args.msp_baseline_tag, args.n_seeds
        )
        msp_partition = get_partition_results(
            args.results_dir, args.msp_partition_tag, args.n_seeds,
            alpha=args.alpha, score=args.score,
        )
        msp_table = generate_latex_table('MSP', msp_baseline, msp_partition)

        # Save with parameters
        msp_tag = f"msp_vs_um{args.tag_suffix}" if args.tag_suffix else "msp_vs_um"
        msp_params = {
            'score_name': 'MSP',
            'baseline': {
                'postprocessor': 'odin',
                'run_tag': args.msp_baseline_tag,
                'selection': 'best on res (minimize FPR@95)',
            },
            'partition': {
                'run_tag': args.msp_partition_tag,
                **partition_params,
            },
            'n_seeds': args.n_seeds,
            'results_dir': str(args.results_dir),
        }
        msp_dir = save_table_with_params(args.output_dir, msp_tag, msp_table, msp_params)
        print(f"\nSaved to: {msp_dir}/")
        print(f"  - table.tex")
        print(f"  - params.yml")
        print("\n" + msp_table)

    # Generate Doctor table
    doctor_baseline, doctor_partition = None, None
    if 'doctor' in args.scores:
        print("\n" + "=" * 60)
        print("Generating Doctor table...")
        print("=" * 60)

        doctor_baseline = get_baseline_results(
            args.results_dir, 'doctor', args.doctor_baseline_tag, args.n_seeds
        )
        doctor_partition = get_partition_results(
            args.results_dir, args.doctor_partition_tag, args.n_seeds,
            alpha=args.alpha, score=args.score,
        )
        doctor_table = generate_latex_table('Doctor', doctor_baseline, doctor_partition)

        # Save with parameters
        doctor_tag = f"doctor_vs_um{args.tag_suffix}" if args.tag_suffix else "doctor_vs_um"
        doctor_params = {
            'score_name': 'Doctor',
            'baseline': {
                'postprocessor': 'doctor',
                'run_tag': args.doctor_baseline_tag,
                'selection': 'best on res (minimize FPR@95)',
            },
            'partition': {
                'run_tag': args.doctor_partition_tag,
                **partition_params,
            },
            'n_seeds': args.n_seeds,
            'results_dir': str(args.results_dir),
        }
        doctor_dir = save_table_with_params(args.output_dir, doctor_tag, doctor_table, doctor_params)
        print(f"\nSaved to: {doctor_dir}/")
        print(f"  - table.tex")
        print(f"  - params.yml")
        print("\n" + doctor_table)

    # Print data availability summary
    print("\n" + "=" * 60)
    print("DATA AVAILABILITY SUMMARY")
    print("=" * 60)

    if margin_baseline:
        print("\nMargin baseline:")
        for r in margin_baseline:
            print(f"  {r['dataset']}/{r['model']}: {r['n_seeds']} seeds")

    if margin_partition:
        print("\nMargin partition (UM):")
        for r in margin_partition:
            print(f"  {r['dataset']}/{r['model']}: {r['n_seeds']} seeds")

    if msp_baseline:
        print("\nMSP baseline:")
        for r in msp_baseline:
            print(f"  {r['dataset']}/{r['model']}: {r['n_seeds']} seeds")

    if msp_partition:
        print("\nMSP partition (UM):")
        for r in msp_partition:
            print(f"  {r['dataset']}/{r['model']}: {r['n_seeds']} seeds")

    if doctor_baseline:
        print("\nDoctor baseline:")
        for r in doctor_baseline:
            print(f"  {r['dataset']}/{r['model']}: {r['n_seeds']} seeds")

    if doctor_partition:
        print("\nDoctor partition (UM):")
        for r in doctor_partition:
            print(f"  {r['dataset']}/{r['model']}: {r['n_seeds']} seeds")


if __name__ == '__main__':
    main()
