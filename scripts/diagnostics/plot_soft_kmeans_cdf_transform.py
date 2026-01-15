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


def _plot_cdf_transform(
    *,
    scores: np.ndarray,
    cdf_vals: np.ndarray,
    output_path: Path,
    temperature: float,
    magnitude: float,
    normalize: bool,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.hist(scores, bins=50, color="#4C78A8", alpha=0.85)
    ax.set_title("Res gini distribution")
    ax.set_xlabel("gini score")
    ax.set_ylabel("count")
    for q in [0.5, 0.9, 0.99]:
        ax.axvline(np.quantile(scores, q), color="#72B7B2", linestyle="--", linewidth=1)
    ax.set_yscale("log")

    ax = axes[1]
    order = np.argsort(scores)
    ax.plot(scores[order], cdf_vals[order], color="#F58518", linewidth=2)
    ax.set_title("Empirical CDF (res)")
    ax.set_xlabel("gini score")
    ax.set_ylabel("u = rank(s)/n")
    ax.set_ylim(0, 1)

    ax = axes[2]
    ax.hist(cdf_vals, bins=20, color="#54A24B", alpha=0.85)
    ax.set_title("CDF-transformed scores")
    ax.set_xlabel("u")
    ax.set_ylabel("count")

    fig.suptitle(
        f"CDF transform on res (temp={temperature}, mag={magnitude}, normalize={normalize})",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


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
    cdf_vals = _rank_cdf(scores)

    output_dir = Path(args.output_dir)
    plot_path = output_dir / "soft_kmeans_cdf_transform_res.png"
    _plot_cdf_transform(
        scores=scores,
        cdf_vals=cdf_vals,
        output_path=plot_path,
        temperature=args.temperature,
        magnitude=args.magnitude,
        normalize=args.normalize,
    )

    stats = {
        "n_res": int(scores.size),
        "temperature": float(args.temperature),
        "magnitude": float(args.magnitude),
        "normalize": bool(args.normalize),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
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
