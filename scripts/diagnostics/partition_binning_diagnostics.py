from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import random

import numpy as np
import torch
import yaml

from error_estimation.utils.metrics import compute_all_metrics


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_stats(run_dir: Path) -> Tuple[Path, dict]:
    candidates = sorted(run_dir.glob("partition_cluster_stats_n-clusters-*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No partition_cluster_stats_n-clusters-*.pt in {run_dir}")
    stats_path = candidates[-1]
    stats = torch.load(stats_path, map_location="cpu")
    return stats_path, stats


def _load_clusters_test(run_dir: Path) -> Tuple[Path, np.ndarray]:
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


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path, stats = _load_stats(run_dir)
    clusters_path, clusters_test = _load_clusters_test(run_dir)

    bin_edges = stats.get("bin_edges")
    if bin_edges is not None:
        bin_edges = bin_edges.view(-1).numpy()
    else:
        n_clusters = int(stats["cluster_intervals"].shape[1])
        bin_edges = np.linspace(0.0, 1.0, n_clusters + 1)

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
    detection_cfg = _load_yaml(config_dir / "detection.yml")
    dataset_cfg = _load_yaml(config_dir / "dataset.yml")
    args_cfg = detection_cfg.get("postprocessor_args", {})
    space = args_cfg.get("space", "gini")
    temperature = float(args_cfg.get("temperature", 1.0))
    normalize_gini = bool(args_cfg.get("normalize", False))

    score_cont = _compute_score(logits, space=space, temperature=temperature, normalize_gini=normalize_gini)
    score_cont = score_cont.detach().cpu().numpy()

    scores_binned = uppers[clusters_test]

    if score_cont.shape[0] != scores_binned.shape[0]:
        seed_split = None
        if run_dir.name.startswith("seed-split-"):
            try:
                seed_split = int(run_dir.name.split("-")[-1])
            except ValueError:
                seed_split = None
        n_samples = dataset_cfg.get("n_samples", {})
        test_idx = _build_split_indices(
            n_total=score_cont.shape[0],
            n_res=n_samples.get("res", 0),
            n_cal=n_samples.get("cal", 0),
            n_test=n_samples.get("test", 0),
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

    centers, widths = _centers_and_widths(bin_edges)
    if centers.shape[0] != uppers.shape[0]:
        bin_edges = np.linspace(0.0, 1.0, uppers.shape[0] + 1)
        centers, widths = _centers_and_widths(bin_edges)
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

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.vlines(centers, lowers, uppers, colors="tab:blue", alpha=0.7, linewidth=1.0)
        ax.scatter(centers, means, s=8, color="black", label=r"$\widehat{\eta}(z)$")
        ax.set_xlabel("Score bin center")
        ax.set_ylabel("Confidence interval")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "ci_vs_score.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(widths, half_widths, s=12, alpha=0.7)
        ax.set_xlabel(r"Bin width $\Delta s_z$")
        ax.set_ylabel(r"Half-width $h_z$")
        fig.tight_layout()
        fig.savefig(output_dir / "width_vs_halfwidth.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


if __name__ == "__main__":
    main()
