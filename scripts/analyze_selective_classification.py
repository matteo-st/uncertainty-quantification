#!/usr/bin/env python3
"""
Analyze selective classification guarantees for uniform mass binning.

This script verifies that Hoeffding confidence intervals on per-bin error rates
provide valid coverage on held-out test data.

Key Metrics:
1. Per-bin guarantee: For each bin z, check if test error rate exceeds upper bound
   - Expected violation rate should be approximately alpha (e.g., 5%)
2. Selective classification: For threshold tau, check if error among accepted samples
   is bounded by the predicted upper confidence bound
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import numpy as np
import pandas as pd
import torch


def load_cluster_stats(stats_path: str) -> Dict[str, torch.Tensor]:
    """Load cluster statistics from partition_cluster_stats_*.pt file."""
    data = torch.load(stats_path, map_location="cpu")
    return {
        "counts": data["cluster_counts"],
        "means": data["cluster_error_means"],
        "vars": data["cluster_error_vars"],
        "intervals": data["cluster_intervals"],  # (bs, K, 2) [lower, upper]
    }


def load_test_clusters(clusters_path: str) -> torch.Tensor:
    """Load test sample bin assignments from clusters_test_*.pt file."""
    return torch.load(clusters_path, map_location="cpu")


def load_test_errors_from_latent(
    latent_path: str,
    n_test: int,
    seed_split: int,
    n_res: int = 0,
    n_cal: int = 0,
) -> torch.Tensor:
    """
    Load test error labels from cached latent file.

    The latent file contains the full dataset. We need to:
    1. Shuffle with the same seed_split used during training
    2. Skip res+cal samples to get the test split
    3. Return the error labels for test samples
    """
    pkg = torch.load(latent_path, map_location="cpu")
    all_labels = pkg["labels"]
    all_model_preds = pkg["model_preds"]
    all_detector_labels = (all_model_preds != all_labels).int()

    # Shuffle indices with the same seed
    N = len(all_detector_labels)
    indices = list(range(N))
    import random
    random.seed(seed_split)
    random.shuffle(indices)

    # Get test indices
    test_start = n_res + n_cal
    test_end = test_start + n_test
    test_indices = indices[test_start:test_end]

    return all_detector_labels[test_indices]


def verify_per_bin_guarantee(
    test_bin_assignments: torch.Tensor,
    test_errors: torch.Tensor,
    bin_uppers: torch.Tensor,
) -> Tuple[int, int, float, List[Dict]]:
    """
    For each bin, check if test error rate exceeds upper bound.
    Expected violation rate: alpha (e.g., 5%)

    Returns:
        violations: number of bins where test error > upper bound
        total_bins: total number of non-empty bins
        violation_rate: violations / total_bins
        bin_details: list of dicts with per-bin info
    """
    # Handle batched data (squeeze if needed)
    if test_bin_assignments.dim() > 1:
        test_bin_assignments = test_bin_assignments.squeeze()
    if bin_uppers.dim() > 2:
        bin_uppers = bin_uppers.squeeze(0)

    K = bin_uppers.shape[0] if bin_uppers.dim() == 2 else len(bin_uppers)
    upper_bounds = bin_uppers[..., 1] if bin_uppers.dim() == 2 else bin_uppers

    violations = 0
    total_bins = 0
    bin_details = []

    for z in range(K):
        mask = test_bin_assignments == z
        n_in_bin = mask.sum().item()

        if n_in_bin > 0:
            total_bins += 1
            empirical_error = test_errors[mask].float().mean().item()
            upper_bound = upper_bounds[z].item()
            is_violation = empirical_error > upper_bound

            if is_violation:
                violations += 1

            bin_details.append({
                "bin": z,
                "n_samples": n_in_bin,
                "empirical_error": empirical_error,
                "upper_bound": upper_bound,
                "violation": is_violation,
                "gap": upper_bound - empirical_error,
            })

    violation_rate = violations / total_bins if total_bins > 0 else 0.0
    return violations, total_bins, violation_rate, bin_details


def compute_selective_classification_metrics(
    test_bin_assignments: torch.Tensor,
    test_errors: torch.Tensor,
    bin_means: torch.Tensor,
    bin_uppers: torch.Tensor,
    threshold_tau: float,
) -> Dict:
    """
    For threshold tau (reject if predicted_error > tau):
    - Coverage: fraction of samples with bin_mean <= tau
    - Risk: error rate among accepted samples
    - Upper Bound: max(bin_upper) for accepted bins
    - Violation: 1 if Risk > Upper Bound
    """
    # Handle batched data
    if test_bin_assignments.dim() > 1:
        test_bin_assignments = test_bin_assignments.squeeze()
    if bin_means.dim() > 1:
        bin_means = bin_means.squeeze()
    if bin_uppers.dim() > 2:
        bin_uppers = bin_uppers.squeeze(0)

    upper_bounds = bin_uppers[..., 1] if bin_uppers.dim() == 2 else bin_uppers

    # Get predicted error for each test sample (using bin mean as the point estimate)
    test_predictions = bin_means[test_bin_assignments]

    # Accept samples with low predicted error
    accepted = test_predictions <= threshold_tau
    coverage = accepted.float().mean().item()

    if accepted.sum().item() > 0:
        risk = test_errors[accepted].float().mean().item()
        accepted_bins = torch.unique(test_bin_assignments[accepted])
        # Upper bound is the max upper bound among accepted bins
        upper_bound = upper_bounds[accepted_bins].max().item()
        violation = int(risk > upper_bound)
    else:
        risk, upper_bound, violation = 0.0, 0.0, 0

    return {
        "threshold": threshold_tau,
        "coverage": coverage,
        "risk": risk,
        "upper_bound": upper_bound,
        "violation": violation,
        "n_accepted": accepted.sum().item(),
    }


def analyze_run(
    results_dir: Path,
    seed_split: int,
    n_clusters: int,
    latent_path: str,
    n_test: int,
    n_res: int,
    n_cal: int,
    alpha: float,
) -> Dict:
    """Analyze a single experiment run."""
    seed_dir = results_dir / f"seed-split-{seed_split}"

    # Load cluster statistics
    stats_path = seed_dir / f"partition_cluster_stats_n-clusters-{n_clusters}.pt"
    if not stats_path.exists():
        return None
    stats = load_cluster_stats(str(stats_path))

    # Load test cluster assignments
    clusters_path = seed_dir / f"clusters_test_n-clusters-{n_clusters}.pt"
    if not clusters_path.exists():
        return None
    test_clusters = load_test_clusters(str(clusters_path))

    # Load test errors
    test_errors = load_test_errors_from_latent(
        latent_path, n_test, seed_split, n_res, n_cal
    )

    # Verify per-bin guarantee
    violations, total_bins, violation_rate, bin_details = verify_per_bin_guarantee(
        test_clusters, test_errors, stats["intervals"]
    )

    # Compute selective classification metrics at various coverage levels
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]
    selective_results = []
    for tau in thresholds:
        result = compute_selective_classification_metrics(
            test_clusters, test_errors, stats["means"], stats["intervals"], tau
        )
        selective_results.append(result)

    return {
        "seed_split": seed_split,
        "n_clusters": n_clusters,
        "alpha": alpha,
        "per_bin": {
            "violations": violations,
            "total_bins": total_bins,
            "violation_rate": violation_rate,
            "expected_rate": alpha,
            "pass": violation_rate <= alpha + 0.05,  # Allow some slack for finite samples
            "bin_details": bin_details,
        },
        "selective_classification": selective_results,
        "test_error_rate": test_errors.float().mean().item(),
    }


def find_experiment_runs(
    results_base: Path,
    run_tag: str,
) -> List[Dict]:
    """Find all experiment runs matching the run tag."""
    runs = []

    # Pattern: results/<family>/<dataset>/<model>/<postprocessor>/runs/<run_tag>/
    for run_dir in results_base.glob(f"**/{run_tag}"):
        if not run_dir.is_dir():
            continue

        # Extract metadata from path
        parts = run_dir.parts
        runs_idx = parts.index("runs")
        postprocessor = parts[runs_idx - 1]
        model = parts[runs_idx - 2]
        dataset = parts[runs_idx - 3]
        family = parts[runs_idx - 4]

        # Find seed directories
        seed_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("seed-split-")]
        for seed_dir in seed_dirs:
            seed_split = int(seed_dir.name.split("-")[-1])
            runs.append({
                "path": run_dir,
                "family": family,
                "dataset": dataset,
                "model": model,
                "postprocessor": postprocessor,
                "seed_split": seed_split,
            })

    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Analyze selective classification guarantees for uniform mass binning"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base results directory",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="unif-mass-selective-verify",
        help="Run tag to analyze",
    )
    parser.add_argument(
        "--latent-dir",
        type=str,
        default="latent",
        help="Base latent/cached data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="selective_classification_analysis.json",
        help="Output file path",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        nargs="+",
        default=[10, 20, 30, 50, 100],
        help="Number of clusters to analyze",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1],
        help="Alpha values to analyze",
    )
    args = parser.parse_args()

    results_base = Path(args.results_dir)
    latent_base = Path(args.latent_dir)

    # Dataset configurations
    dataset_configs = {
        "cifar100": {
            "n_test": 5000,
            "n_res": 1000,
            "n_cal": 4000,
            "latent_pattern": "cifar100_{model}/transform-test_n-epochs-1/full.pt",
        },
        "cifar10": {
            "n_test": 5000,
            "n_res": 1000,
            "n_cal": 4000,
            "latent_pattern": "cifar10_{model}/transform-test_n-epochs-1/full.pt",
        },
        "imagenet": {
            "n_test": 25000,
            "n_res": 5000,
            "n_cal": 20000,
            "latent_pattern": "imagenet_{model}/transform-test_n-epochs-1/full.pt",
        },
    }

    # Model name mappings
    model_mappings = {
        "resnet34_ce": "resnet34_ce",
        "densenet121_ce": "densenet121_ce",
        "timm_vit_base16_ce": "timm_vit_base16_ce",
        "timm_vit_tiny16_ce": "timm_vit_tiny16_ce",
    }

    # Find all experiment runs
    experiment_runs = find_experiment_runs(results_base, args.run_tag)
    print(f"Found {len(experiment_runs)} experiment run(s)")

    all_results = []

    for run in experiment_runs:
        dataset = run["dataset"]
        model = run["model"]

        if dataset not in dataset_configs:
            print(f"Skipping unknown dataset: {dataset}")
            continue

        config = dataset_configs[dataset]
        model_name = model_mappings.get(model, model)
        latent_path = latent_base / config["latent_pattern"].format(model=model_name)

        if not latent_path.exists():
            print(f"Latent file not found: {latent_path}")
            # Try alternative path on server
            continue

        for n_clusters in args.n_clusters:
            for alpha in args.alpha:
                result = analyze_run(
                    results_dir=run["path"],
                    seed_split=run["seed_split"],
                    n_clusters=n_clusters,
                    latent_path=str(latent_path),
                    n_test=config["n_test"],
                    n_res=config["n_res"],
                    n_cal=config["n_cal"],
                    alpha=alpha,
                )

                if result is not None:
                    result.update({
                        "dataset": dataset,
                        "model": model,
                    })
                    all_results.append(result)
                    print(
                        f"{dataset}/{model} seed={run['seed_split']} K={n_clusters} alpha={alpha}: "
                        f"violations={result['per_bin']['violations']}/{result['per_bin']['total_bins']} "
                        f"({result['per_bin']['violation_rate']:.2%})"
                    )

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Print summary tables
    if all_results:
        print_summary_tables(all_results)


def print_summary_tables(results: List[Dict]):
    """Print summary tables to stdout."""
    print("\n" + "=" * 80)
    print("Table 1: Per-Bin Guarantee Verification")
    print("=" * 80)

    # Group by dataset, model, K, alpha
    df = pd.DataFrame([
        {
            "dataset": r["dataset"],
            "model": r["model"],
            "K": r["n_clusters"],
            "alpha": r["alpha"],
            "seed": r["seed_split"],
            "violations": r["per_bin"]["violations"],
            "total_bins": r["per_bin"]["total_bins"],
            "violation_rate": r["per_bin"]["violation_rate"],
        }
        for r in results
    ])

    # Aggregate across seeds
    summary = df.groupby(["dataset", "model", "K", "alpha"]).agg({
        "violations": ["mean", "std"],
        "total_bins": "first",
        "violation_rate": ["mean", "std"],
    }).reset_index()
    summary.columns = ["dataset", "model", "K", "alpha",
                       "violations_mean", "violations_std",
                       "total_bins", "violation_rate_mean", "violation_rate_std"]

    print("\n{:<12} {:<20} {:>5} {:>6} {:>12} {:>10} {:>12}".format(
        "Dataset", "Model", "K", "alpha", "Violations", "Expected", "Pass?"
    ))
    print("-" * 80)
    for _, row in summary.iterrows():
        expected = row["alpha"] * row["total_bins"]
        passed = "Yes" if row["violation_rate_mean"] <= row["alpha"] + 0.05 else "NO"
        print("{:<12} {:<20} {:>5} {:>6.2f} {:>5.1f}/{:<5} {:>10.1f} {:>12}".format(
            row["dataset"],
            row["model"].replace("_ce", ""),
            int(row["K"]),
            row["alpha"],
            row["violations_mean"],
            int(row["total_bins"]),
            expected,
            passed,
        ))

    print("\n" + "=" * 80)
    print("Table 2: Selective Classification at Different Coverage Levels")
    print("=" * 80)

    # Build selective classification table
    selective_rows = []
    for r in results:
        for sc in r["selective_classification"]:
            selective_rows.append({
                "dataset": r["dataset"],
                "model": r["model"],
                "K": r["n_clusters"],
                "alpha": r["alpha"],
                "seed": r["seed_split"],
                "threshold": sc["threshold"],
                "coverage": sc["coverage"],
                "risk": sc["risk"],
                "upper_bound": sc["upper_bound"],
                "violation": sc["violation"],
            })

    selective_df = pd.DataFrame(selective_rows)

    # Aggregate across seeds for a specific K and alpha
    for (dataset, model, K, alpha), group in selective_df.groupby(["dataset", "model", "K", "alpha"]):
        print(f"\n{dataset} / {model.replace('_ce', '')} (K={K}, alpha={alpha}):")
        print("{:>10} {:>10} {:>10} {:>12} {:>10}".format(
            "Coverage", "Risk", "Upper", "Violation", "N Seeds"
        ))
        print("-" * 55)

        agg = group.groupby("threshold").agg({
            "coverage": "mean",
            "risk": "mean",
            "upper_bound": "mean",
            "violation": ["sum", "count"],
        })

        for thresh, row in agg.iterrows():
            violations = int(row[("violation", "sum")])
            n_seeds = int(row[("violation", "count")])
            print("{:>10.1%} {:>10.3f} {:>10.3f} {:>8}/{:<3} {:>10}".format(
                row[("coverage", "mean")],
                row[("risk", "mean")],
                row[("upper_bound", "mean")],
                violations,
                n_seeds,
                n_seeds,
            ))


if __name__ == "__main__":
    main()
