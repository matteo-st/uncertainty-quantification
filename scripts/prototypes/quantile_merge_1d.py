from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from error_estimation.utils.config import Config
from error_estimation.utils.datasets import get_dataset
from error_estimation.utils.datasets.dataloader import prepare_ablation_dataloaders
from error_estimation.utils.eval import AblationDetector
from error_estimation.utils.experiment import build_latent_paths
from error_estimation.utils.helper import setup_seeds
from error_estimation.utils.metrics import compute_all_metrics
from error_estimation.utils.models import get_model
from error_estimation.utils.paths import CHECKPOINTS_DIR, DATA_DIR, LATENTS_DIR
from error_estimation.utils.postprocessors import get_postprocessor


def _parse_int_list(value: str) -> list[int]:
    if value is None:
        return []
    items = []
    for part in value.split(","):
        part = part.strip()
        if part:
            items.append(int(part))
    return items


def _assign_bins(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(scores, edges[1:-1], right=False)


def _build_edges_quantile_merge(scores: np.ndarray, k0: int, n_min: int) -> np.ndarray:
    if k0 < 1:
        raise ValueError("k0 must be >= 1")
    if n_min < 1:
        raise ValueError("n_min must be >= 1")
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        raise ValueError("scores must be non-empty")
    q = np.linspace(0.0, 1.0, k0 + 1, endpoint=True)[1:-1]
    quantiles = np.quantile(scores, q, method="linear") if scores.size > 1 else np.array([])
    edges = np.concatenate(([-np.inf], quantiles, [np.inf]))
    bin_idx = _assign_bins(scores, edges)
    counts = np.bincount(bin_idx, minlength=k0)

    merged_edges = [-np.inf]
    running = 0
    for i in range(k0):
        running += counts[i]
        if running >= n_min:
            merged_edges.append(edges[i + 1])
            running = 0
    if merged_edges[-1] != np.inf:
        merged_edges[-1] = np.inf
    if len(merged_edges) == 1:
        merged_edges.append(np.inf)

    cleaned = [merged_edges[0]]
    for edge in merged_edges[1:]:
        if edge > cleaned[-1]:
            cleaned.append(edge)
        elif edge == np.inf:
            cleaned[-1] = np.inf
    return np.array(cleaned, dtype=float)


def _upper_bounds(scores: np.ndarray, labels: np.ndarray, edges: np.ndarray, alpha: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    bin_idx = _assign_bins(scores, edges)
    k = len(edges) - 1
    counts = np.bincount(bin_idx, minlength=k).astype(float)
    sum_err = np.bincount(bin_idx, weights=labels, minlength=k)
    means = np.zeros_like(counts)
    mask = counts > 0
    means[mask] = sum_err[mask] / counts[mask]
    half = np.zeros_like(counts)
    half[mask] = np.sqrt(np.log(2.0 / alpha) / (2.0 * counts[mask]))
    upper = np.clip(means + half, 0.0, 1.0)
    upper[~mask] = 1.0
    return upper


def _scores_from_bins(scores: np.ndarray, edges: np.ndarray, upper: np.ndarray) -> np.ndarray:
    bin_idx = _assign_bins(scores, edges)
    return upper[bin_idx]


def _split_indices(n: int, val_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Need at least 2 samples to split res into train/val")
    idx = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    n_val = max(1, min(n - 1, n_val))
    return idx[n_val:], idx[:n_val]


def _collect_scores(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    latent_path: str,
    postprocessor_name: str,
    postprocessor_cfg: dict,
    cfg_dataset: dict,
) -> tuple[np.ndarray, np.ndarray]:
    detector = get_postprocessor(
        postprocessor_name=postprocessor_name,
        model=model,
        cfg=postprocessor_cfg,
        result_folder=".",
        device=device,
    )
    runner = AblationDetector(
        model=model,
        dataloader=loader,
        device=device,
        suffix="split",
        latent_path=latent_path,
        postprocessor_name=postprocessor_name,
        cfg_dataset=cfg_dataset,
        result_folder=".",
    )
    runner.get_scores([detector], n_samples=len(loader.dataset), list_configs=[postprocessor_cfg])
    scores = np.asarray(runner.scores["scores"][0], dtype=float)
    labels = np.asarray(runner.scores["detector_labels"], dtype=int)
    return scores, labels


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
    parser = argparse.ArgumentParser(description="Prototype: quantile bins + min-count merge on 1D score.")
    parser.add_argument(
        "--config-dataset",
        dest="config_dataset",
        default="configs/datasets/cifar10/cifar10_n-cal-5000_seed-split-9.yml",
    )
    parser.add_argument(
        "--config-model",
        dest="config_model",
        default="configs/models/cifar10_resnet34.yml",
    )
    parser.add_argument(
        "--config-detection",
        dest="config_detection",
        default="configs/postprocessors/doctor/cifar10_resnet34.yml",
    )
    parser.add_argument("--data-dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--checkpoints-dir", dest="checkpoints_dir", default=CHECKPOINTS_DIR)
    parser.add_argument("--latent-dir", dest="latent_dir", default=LATENTS_DIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed-split", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--k0-grid", default="10,20,50,100")
    parser.add_argument("--n-min-grid", default="50,100,200")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = Config(args.config_dataset)
    model_cfg = Config(args.config_model)
    detection_cfg = Config(args.config_detection)

    data_cfg.require(["name", "num_classes", "n_samples", "seed_split"], "dataset")
    model_cfg.require(["model_name", "preprocessor", "seed"], "model")
    detection_cfg.require(["name", "postprocessor_args", "experience_args"], "detection")

    seed_split = args.seed_split if args.seed_split is not None else data_cfg["seed_split"][0]
    setup_seeds(args.seed, seed_split)

    dataset = get_dataset(
        dataset_name=data_cfg["name"],
        model_name=model_cfg["model_name"],
        root=args.data_dir,
        preprocess=model_cfg["preprocessor"],
        shuffle=False,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(
        model_name=model_cfg["model_name"],
        dataset_name=data_cfg["name"],
        n_classes=data_cfg["num_classes"],
        model_seed=model_cfg["seed"],
        checkpoint_dir=str(Path(args.checkpoints_dir) / model_cfg["preprocessor"]),
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    res_loader, cal_loader, test_loader = prepare_ablation_dataloaders(
        dataset=dataset,
        seed_split=seed_split,
        n_res=data_cfg["n_samples"]["res"],
        n_cal=data_cfg["n_samples"]["cal"],
        n_test=data_cfg["n_samples"]["test"],
        batch_size_train=data_cfg["batch_size_train"],
        batch_size_test=data_cfg["batch_size_test"],
        cal_transform=detection_cfg["experience_args"]["transform"]["cal"],
        res_transform=detection_cfg["experience_args"]["transform"]["res"],
        data_name=data_cfg["name"],
        model_name=model_cfg["model_name"],
    )

    if res_loader is None:
        raise ValueError("n_samples.res must be > 0 to learn the resolution function")

    latent_paths = build_latent_paths(args.latent_dir, data_cfg, model_cfg, detection_cfg)
    postprocessor_name = detection_cfg["name"]
    postprocessor_cfg = detection_cfg.get("postprocessor_args", {})

    res_scores, res_labels = _collect_scores(
        model=model,
        loader=res_loader,
        device=device,
        latent_path=latent_paths["res"],
        postprocessor_name=postprocessor_name,
        postprocessor_cfg=postprocessor_cfg,
        cfg_dataset=data_cfg,
    )
    cal_scores, cal_labels = _collect_scores(
        model=model,
        loader=cal_loader,
        device=device,
        latent_path=latent_paths["cal"],
        postprocessor_name=postprocessor_name,
        postprocessor_cfg=postprocessor_cfg,
        cfg_dataset=data_cfg,
    )
    test_scores, test_labels = _collect_scores(
        model=model,
        loader=test_loader,
        device=device,
        latent_path=latent_paths["test"],
        postprocessor_name=postprocessor_name,
        postprocessor_cfg=postprocessor_cfg,
        cfg_dataset=data_cfg,
    )

    rng = np.random.default_rng(seed_split)
    res_train_idx, res_val_idx = _split_indices(len(res_scores), args.val_fraction, rng)
    res_train_scores = res_scores[res_train_idx]
    res_train_labels = res_labels[res_train_idx]
    res_val_scores = res_scores[res_val_idx]
    res_val_labels = res_labels[res_val_idx]

    k0_grid = _parse_int_list(args.k0_grid)
    n_min_grid = _parse_int_list(args.n_min_grid)
    if not k0_grid or not n_min_grid:
        raise ValueError("k0-grid and n-min-grid must be non-empty")

    best = None
    search_rows = []
    for k0 in k0_grid:
        if k0 > len(res_train_scores):
            continue
        for n_min in n_min_grid:
            if n_min > len(res_train_scores):
                continue
            edges = _build_edges_quantile_merge(res_train_scores, k0, n_min)
            upper = _upper_bounds(res_train_scores, res_train_labels, edges, args.alpha)
            val_conf = _scores_from_bins(res_val_scores, edges, upper)
            metrics = compute_all_metrics(val_conf, res_val_labels)
            roc_auc = float(metrics["roc_auc"])
            row = {"k0": k0, "n_min": n_min, "roc_auc_val": roc_auc}
            search_rows.append(row)
            if best is None or roc_auc > best["roc_auc_val"]:
                best = row

    if best is None:
        raise ValueError("No valid hyperparameter combination for the current res split")

    best_edges = _build_edges_quantile_merge(res_scores, best["k0"], best["n_min"])
    cal_upper = _upper_bounds(cal_scores, cal_labels, best_edges, args.alpha)
    test_conf = _scores_from_bins(test_scores, best_edges, cal_upper)
    test_metrics = compute_all_metrics(test_conf, test_labels)

    print("Best hyperparams:", best)
    print("Test metrics:", test_metrics)

    if args.output:
        payload = {
            "best": best,
            "search": search_rows,
            "test_metrics": test_metrics,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_serialize(payload), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
