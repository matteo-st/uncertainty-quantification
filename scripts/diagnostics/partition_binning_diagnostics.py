from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import random

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from error_estimation.utils.metrics import compute_all_metrics


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_stats(run_dir: Path, n_clusters: int | None) -> Tuple[Path, dict]:
    if n_clusters is not None:
        stats_path = run_dir / f"partition_cluster_stats_n-clusters-{n_clusters}.pt"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing {stats_path}")
    else:
        candidates = sorted(run_dir.glob("partition_cluster_stats_n-clusters-*.pt"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No partition_cluster_stats_n-clusters-*.pt in {run_dir}")
        stats_path = candidates[-1]
    stats = torch.load(stats_path, map_location="cpu")
    return stats_path, stats


def _load_clusters_test(run_dir: Path, n_clusters: int | None) -> Tuple[Path, np.ndarray]:
    if n_clusters is not None:
        clusters_path = run_dir / f"clusters_test_n-clusters-{n_clusters}.pt"
        if not clusters_path.exists():
            raise FileNotFoundError(f"Missing {clusters_path}")
    else:
        candidates = sorted(run_dir.glob("clusters_test_n-clusters-*.pt"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No clusters_test_n-clusters-*.pt in {run_dir}")
        clusters_path = candidates[-1]
    clusters = torch.load(clusters_path, map_location="cpu").view(-1).numpy().astype(int)
    return clusters_path, clusters


def _gini(logits: torch.Tensor, temperature: float = 1.0, normalize: bool = False) -> torch.Tensor:
    probs = torch.softmax(logits / temperature, dim=1)
    g = torch.sum(probs ** 2, dim=1)
    if normalize:
        return (1.0 - g) / g
    return 1.0 - g


def _compute_score(
    logits: torch.Tensor,
    space: str,
    temperature: float,
    normalize_gini: bool,
) -> torch.Tensor:
    if space == "gini":
        return _gini(logits, temperature=temperature, normalize=normalize_gini)
    if space == "msp":
        probs = torch.softmax(logits / temperature, dim=1)
        return -probs.max(dim=1).values
    if space == "max_proba":
        probs = torch.softmax(logits / temperature, dim=1)
        return -probs.max(dim=1).values
    raise ValueError(f"Unsupported score space: {space}")


def _centers_and_widths(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    return centers, widths


def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostics for 1D partition binning.")
    parser.add_argument("--run-dir", required=True, help="Run directory (seed-split-*) with partition outputs.")
    parser.add_argument("--latent-test", required=True, help="Latent .pt file for the test split.")
    parser.add_argument("--output-dir", default=None, help="Output directory for diagnostics.")
    parser.add_argument("--n-clusters", type=int, default=None, help="Select a specific K when multiple stats exist.")
    parser.add_argument("--space", default=None, help="Override score space when configs are missing.")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature when configs are missing.")
    parser.add_argument("--normalize-gini", action="store_true", help="Override normalize_gini when configs are missing.")
    parser.add_argument("--n-res", type=int, default=None, help="Override n_res when configs are missing.")
    parser.add_argument("--n-cal", type=int, default=None, help="Override n_cal when configs are missing.")
    parser.add_argument("--n-test", type=int, default=None, help="Override n_test when configs are missing.")
    parser.add_argument("--seed-split", type=int, default=None, help="Override seed split when configs are missing.")
    parser.add_argument("--bin-split", choices=["res", "cal"], default=None, help="Split used to build bins.")
    return parser.parse_args()


def _resolve_count(value, total: int, name: str) -> int:
    if isinstance(value, float):
        if not (0 < value <= 1):
            raise ValueError(f"{name} ratio must be in (0,1], got {value}")
        return int(round(total * value))
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} count must be >= 0, got {value}")
        return value
    raise TypeError(f"{name} must be float or int, got {type(value).__name__}")


def _build_split_indices(n_total: int, n_res, n_cal, n_test, seed_split: int) -> np.ndarray:
    perm = list(range(n_total))
    if seed_split is not None:
        rng = random.Random(seed_split)
        rng.shuffle(perm)
    n_cal_samples = _resolve_count(n_cal, n_total, "n_cal")
    n_res_samples = _resolve_count(n_res, n_total, "n_res")
    n_test_samples = _resolve_count(n_test, n_total, "n_test")
    test_idx = perm[n_total - n_test_samples :]
    return np.asarray(test_idx, dtype=int)


def _build_indices(n_total: int, n_res, n_cal, n_test, seed_split: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = list(range(n_total))
    if seed_split is not None:
        rng = random.Random(seed_split)
        rng.shuffle(perm)
    n_cal_samples = _resolve_count(n_cal, n_total, "n_cal")
    n_res_samples = _resolve_count(n_res, n_total, "n_res")
    n_test_samples = _resolve_count(n_test, n_total, "n_test")
    cal_idx = np.asarray(perm[:n_cal_samples], dtype=int)
    res_idx = np.asarray(perm[n_cal_samples : n_cal_samples + n_res_samples], dtype=int)
    test_idx = np.asarray(perm[n_total - n_test_samples :], dtype=int)
    return res_idx, cal_idx, test_idx


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    stats_path, stats = _load_stats(run_dir, args.n_clusters)
    clusters_path, clusters_test = _load_clusters_test(run_dir, args.n_clusters)

    n_clusters = int(stats["cluster_intervals"].shape[1])
    bin_edges = stats.get("bin_edges")
    if bin_edges is not None:
        bin_edges = bin_edges.view(-1).numpy()
    else:
        bin_edges = None

    intervals = stats["cluster_intervals"].squeeze(0).numpy()
    lowers = intervals[:, 0]
    uppers = intervals[:, 1]
    means = stats["cluster_error_means"].squeeze(0).numpy()
    counts_cal = stats["cluster_counts"].squeeze(0).numpy()

    latent = torch.load(args.latent_test, map_location="cpu")
    logits = latent["logits"]
    detector_labels = (latent["model_preds"] != latent["labels"]).numpy().astype(int)

    config_dir = run_dir / "configs"
    if not config_dir.exists():
        config_dir = run_dir.parent / "configs"
    detection_cfg = _load_yaml(config_dir / "detection.yml") if config_dir.exists() else {}
    dataset_cfg = _load_yaml(config_dir / "dataset.yml") if config_dir.exists() else {}
    args_cfg = detection_cfg.get("postprocessor_args", {}) if detection_cfg else {}
    space = args.space or args_cfg.get("space", "gini")
    temperature = float(args.temperature if args.temperature is not None else args_cfg.get("temperature", 1.0))
    normalize_gini = bool(args_cfg.get("normalize", False)) or bool(args.normalize_gini)

    score_cont = _compute_score(logits, space=space, temperature=temperature, normalize_gini=normalize_gini)
    score_cont = score_cont.detach().cpu().numpy()

    scores_binned = uppers[clusters_test]

    seed_split = args.seed_split
    if seed_split is None and run_dir.name.startswith("seed-split-"):
        try:
            seed_split = int(run_dir.name.split("-")[-1])
        except ValueError:
            seed_split = None
    n_samples = dataset_cfg.get("n_samples", {}) if dataset_cfg else {}
    n_res = args.n_res if args.n_res is not None else n_samples.get("res", 0)
    n_cal = args.n_cal if args.n_cal is not None else n_samples.get("cal", 0)
    n_test = args.n_test if args.n_test is not None else n_samples.get("test", 0)

    edges = None
    edge_min = float(np.min(score_cont))
    edge_max = float(np.max(score_cont))
    if bin_edges is not None:
        if bin_edges.size == n_clusters - 1:
            res_idx, cal_idx, _ = _build_indices(
                n_total=score_cont.shape[0],
                n_res=n_res,
                n_cal=n_cal,
                n_test=n_test,
                seed_split=seed_split,
            )
            bin_split = args.bin_split
            if bin_split is None:
                bin_split = "cal" if n_cal else "res"
            if bin_split == "res":
                bin_scores = score_cont[res_idx] if res_idx.size else score_cont
            else:
                bin_scores = score_cont[cal_idx] if cal_idx.size else score_cont
            edge_min = float(np.min(bin_scores))
            edge_max = float(np.max(bin_scores))
            edges = np.concatenate([[edge_min], bin_edges, [edge_max]])
        elif bin_edges.size == n_clusters + 1:
            edges = bin_edges
    if edges is None:
        edges = np.linspace(0.0, 1.0, n_clusters + 1)
    if not np.isfinite(edges).all():
        edges = np.where(np.isneginf(edges), edge_min, edges)
        edges = np.where(np.isposinf(edges), edge_max, edges)

    if score_cont.shape[0] != scores_binned.shape[0]:
        test_idx = _build_split_indices(
            n_total=score_cont.shape[0],
            n_res=n_res,
            n_cal=n_cal,
            n_test=n_test,
            seed_split=seed_split,
        )
        score_cont = score_cont[test_idx]
        detector_labels = detector_labels[test_idx]

    metrics_cont = compute_all_metrics(score_cont, detector_labels)
    metrics_bin = compute_all_metrics(scores_binned, detector_labels)

    counts_test = np.bincount(clusters_test, minlength=uppers.shape[0]).astype(float)
    err_test = np.bincount(clusters_test, weights=detector_labels, minlength=uppers.shape[0]).astype(float)
    err_rate_test = np.zeros_like(counts_test)
    mask = counts_test > 0
    err_rate_test[mask] = err_test[mask] / counts_test[mask]

    centers, widths = _centers_and_widths(edges)
    half_widths = 0.5 * (uppers - lowers)

    bin_table = {
        "bin_idx": np.arange(uppers.shape[0]),
        "center": centers,
        "width": widths,
        "count_cal": counts_cal,
        "count_test": counts_test,
        "mean_cal": means,
        "mean_test": err_rate_test,
        "lower": lowers,
        "upper": uppers,
        "half_width": half_widths,
    }

    import pandas as pd

    df = pd.DataFrame(bin_table)
    df.to_csv(output_dir / "bin_diagnostics.csv", index=False)

    summary = {
        "stats_path": str(stats_path),
        "clusters_path": str(clusters_path),
        "space": space,
        "temperature": temperature,
        "normalize_gini": normalize_gini,
        "metrics_continuous": metrics_cont,
        "metrics_binned": metrics_bin,
    }
    (output_dir / "summary.json").write_text(json.dumps(_serialize(summary), indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(centers, uppers, color="tab:red", lw=1.2, label="Upper CI")
    ax.plot(centers, lowers, color="tab:blue", lw=1.2, label="Lower CI")
    ax.fill_between(centers, lowers, uppers, color="tab:blue", alpha=0.12)
    ax.scatter(centers, means, s=12, color="black", label=r"$\widehat{\eta}(z)$")
    ax.set_xlabel("Score bin center (quantile)")
    ax.set_ylabel("Confidence interval")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(output_dir / "ci_vs_score.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(widths, half_widths, s=14, alpha=0.7, color="tab:purple")
    ax.set_xlabel(r"Bin width $\Delta s_z$")
    ax.set_ylabel(r"Half-width $h_z$")
    fig.tight_layout()
    fig.savefig(output_dir / "width_vs_halfwidth.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    finite_widths = widths[np.isfinite(widths)]
    if finite_widths.size == 0:
        ax.text(0.5, 0.5, "No finite widths", ha="center", va="center")
    else:
        bins = 1 if np.isclose(finite_widths.min(), finite_widths.max()) else 20
        ax.hist(finite_widths, bins=bins, color="tab:gray", alpha=0.8)
    ax.set_xlabel(r"Bin width $\Delta s_z$")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "bin_width_hist.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(counts_cal + 1, counts_test + 1, s=14, alpha=0.7, color="tab:green")
    ax.set_xlabel("Count (cal)")
    ax.set_ylabel("Count (test)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_dir / "count_shift.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
