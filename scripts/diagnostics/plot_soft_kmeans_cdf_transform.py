#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from error_estimation.utils.config import Config
from error_estimation.utils.datasets import get_dataset
from error_estimation.utils.datasets.dataloader import prepare_ablation_dataloaders
from error_estimation.utils.helper import setup_seeds
from error_estimation.utils.models import get_model
from error_estimation.utils.postprocessors.doctor_postprocessor import gini as doctor_gini


def _compute_adv_logits(
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    magnitude: float,
    temperature: float,
    normalize: bool,
) -> torch.Tensor:
    inputs = inputs.detach().requires_grad_(True)
    logits_clean = model(inputs)
    scores = doctor_gini(logits_clean, temperature=temperature, normalize=normalize)
    scores_for_loss = scores
    if torch.any(scores_for_loss <= 0):
        scores_for_loss = scores_for_loss.abs()
    loss = torch.log(scores_for_loss + 1e-12).sum()
    grad_inputs, = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)
    with torch.no_grad():
        adv = inputs + magnitude * grad_inputs.sign()
        logits_adv = model(adv)
    return logits_adv


def _compute_scores(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    magnitude: float,
    temperature: float,
    normalize: bool,
) -> np.ndarray:
    model.eval()
    scores = []
    for inputs, _ in loader:
        inputs = inputs.to(device)
        if magnitude > 0:
            logits = _compute_adv_logits(
                model=model,
                inputs=inputs,
                magnitude=magnitude,
                temperature=temperature,
                normalize=normalize,
            )
        else:
            with torch.no_grad():
                logits = model(inputs)
        with torch.no_grad():
            batch_scores = doctor_gini(logits, temperature=temperature, normalize=normalize)
        scores.append(batch_scores.detach().cpu().flatten())
    return torch.cat(scores).numpy()


def _rank_cdf(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(values.size)
    return (ranks + 0.5) / values.size


def _plot_score_distribution(
    *,
    scores: np.ndarray,
    output_path: Path,
    temperature: float,
    magnitude: float,
    normalize: bool,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(scores, bins=80, color="#4C78A8", alpha=0.9)
    ax.set_title("Res gini distribution (log y)")
    ax.set_xlabel("gini score")
    ax.set_ylabel("count")
    for q, label in [(0.5, "q50"), (0.9, "q90"), (0.99, "q99")]:
        ax.axvline(np.quantile(scores, q), color="#72B7B2", linestyle="--", linewidth=1)
        ax.text(
            np.quantile(scores, q),
            ax.get_ylim()[1] * 0.5,
            label,
            rotation=90,
            va="center",
            ha="right",
            color="#72B7B2",
        )
    ax.grid(alpha=0.2, linestyle=":", linewidth=0.7)
    ax.set_yscale("log")
    fig.suptitle(
        f"CDF transform on res (temp={temperature}, mag={magnitude}, normalize={normalize})",
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _plot_u_rug(cdf_vals: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.0))
    ax.eventplot([cdf_vals], orientation="horizontal", lineoffsets=0.0, linelengths=1.0, colors="#4C78A8")
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("u")
    ax.set_yticks([])
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_u_counts(cdf_vals: np.ndarray, output_path: Path, title: str) -> tuple[int, int]:
    unique, counts = np.unique(cdf_vals, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.scatter(unique, counts, s=10, color="#F58518", alpha=0.8)
    ax.set_xlabel("u")
    ax.set_ylabel("count per u")
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return unique.size, int(counts.max(initial=0))

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CDF transform for soft-kmeans scores.")
    parser.add_argument("--config-dataset", required=True)
    parser.add_argument("--config-model", required=True)
    parser.add_argument("--seed-split", type=int, default=9)
    parser.add_argument("--n-res", type=int, default=None)
    parser.add_argument("--n-cal", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--magnitude", type=float, required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data_cfg = Config(args.config_dataset)
    model_cfg = Config(args.config_model)

    setup_seeds(seed=1, seed_split=args.seed_split)

    dataset = get_dataset(
        dataset_name=data_cfg["name"],
        model_name=model_cfg["model_name"],
        root=os.environ.get("DATA_DIR", "./data"),
        preprocess=model_cfg["preprocessor"],
        shuffle=False,
    )

    n_res = args.n_res if args.n_res is not None else data_cfg["n_samples"]["res"]
    n_cal = args.n_cal if args.n_cal is not None else data_cfg["n_samples"]["cal"]
    n_test = args.n_test if args.n_test is not None else data_cfg["n_samples"]["test"]

    res_loader, _, _ = prepare_ablation_dataloaders(
        dataset=dataset,
        seed_split=args.seed_split,
        n_res=n_res,
        n_cal=n_cal,
        n_test=n_test,
        batch_size_train=data_cfg["batch_size_train"],
        batch_size_test=data_cfg["batch_size_test"],
        cal_transform="test",
        res_transform="test",
        data_name=data_cfg["name"],
        model_name=model_cfg["model_name"],
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(
        model_name=model_cfg["model_name"],
        dataset_name=data_cfg["name"],
        n_classes=data_cfg["num_classes"],
        model_seed=model_cfg["seed"],
        checkpoint_dir=os.path.join(
            os.environ.get("CHECKPOINTS_DIR", "checkpoints/"),
            model_cfg["preprocessor"],
        ),
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    scores = _compute_scores(
        model=model,
        loader=res_loader,
        device=device,
        magnitude=args.magnitude,
        temperature=args.temperature,
        normalize=args.normalize,
    )
    output_dir = Path(args.output_dir)
    dist_path = output_dir / "soft_kmeans_cdf_transform_res_dist.png"
    _plot_score_distribution(
        scores=scores,
        output_path=dist_path,
        temperature=args.temperature,
        magnitude=args.magnitude,
        normalize=args.normalize,
    )
    cdf_vals = _rank_cdf(scores)
    order = np.argsort(scores)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(scores[order], cdf_vals[order], color="#F58518", linewidth=2.5)
    ax.set_title("Empirical CDF (res)")
    ax.set_xlabel("gini score")
    ax.set_ylabel("u = rank(s)/n")
    ax.set_ylim(0, 1)
    for q in [0.5, 0.9, 0.99]:
        ax.axvline(np.quantile(scores, q), color="#72B7B2", linestyle="--", linewidth=1)
    ax.grid(alpha=0.2, linestyle=":", linewidth=0.7)
    fig.suptitle(
        f"CDF transform on res (temp={args.temperature}, mag={args.magnitude}, normalize={args.normalize})",
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    cdf_path = output_dir / "soft_kmeans_cdf_transform_res_cdf.png"
    fig.savefig(cdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(cdf_vals, bins=20, color="#54A24B", alpha=0.9)
    ax.set_title("CDF-transformed scores")
    ax.set_xlabel("u")
    ax.set_ylabel("count")
    ax.grid(alpha=0.2, linestyle=":", linewidth=0.7)
    fig.suptitle(
        f"CDF transform on res (temp={args.temperature}, mag={args.magnitude}, normalize={args.normalize})",
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    uhist_path = output_dir / "soft_kmeans_cdf_transform_res_u_hist.png"
    fig.savefig(uhist_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    rug_path = output_dir / "soft_kmeans_cdf_transform_res_u_rug.png"
    _plot_u_rug(
        cdf_vals,
        rug_path,
        "CDF-transformed scores (rug plot)",
    )
    counts_path = output_dir / "soft_kmeans_cdf_transform_res_u_counts.png"
    n_unique, max_count = _plot_u_counts(
        cdf_vals,
        counts_path,
        "Counts per discrete u value",
    )

    stats = {
        "n_res": int(scores.size),
        "temperature": float(args.temperature),
        "magnitude": float(args.magnitude),
        "normalize": bool(args.normalize),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "u_unique": int(n_unique),
        "u_max_count": int(max_count),
        "score_quantiles": {
            "q50": float(np.quantile(scores, 0.5)),
            "q90": float(np.quantile(scores, 0.9)),
            "q99": float(np.quantile(scores, 0.99)),
        },
    }
    with (output_dir / "soft_kmeans_cdf_transform_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
