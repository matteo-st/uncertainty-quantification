#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from error_estimation.experiments.run_detection import _apply_raw_score_selection
from error_estimation.utils.config import Config
from error_estimation.utils.datasets import get_dataset
from error_estimation.utils.datasets.dataloader import prepare_ablation_dataloaders
from error_estimation.utils.helper import setup_seeds
from error_estimation.utils.models import get_model
from error_estimation.utils.postprocessors.doctor_postprocessor import gini as doctor_gini
from error_estimation.utils.postprocessors.partition_postprocessor import PartitionPostprocessor


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


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
    scores_for_loss = scores if torch.all(scores > 0) else scores.abs()
    loss = torch.log(scores_for_loss + 1e-12).sum()
    grad_inputs, = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)
    with torch.no_grad():
        adv = inputs + magnitude * grad_inputs.sign()
        logits_adv = model(adv)
    return logits_adv


def _collect_logits(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    magnitude: float,
    temperature: float,
    normalize: bool,
) -> torch.Tensor:
    logits_all = []
    model.eval()
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
        logits_all.append(logits.detach().cpu())
    return torch.cat(logits_all, dim=0)


def _cdf_transform(scores: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    ref_sorted = torch.sort(ref.detach().cpu().float()).values
    counts = torch.searchsorted(ref_sorted, scores.detach().cpu().float(), right=True)
    return counts.float() / float(ref_sorted.numel())


def _bin_counts(values: torch.Tensor, edges: torch.Tensor, k: int) -> np.ndarray:
    bins = torch.bucketize(values, edges.to(values.device))
    counts = torch.bincount(bins, minlength=k).cpu().numpy()
    return counts


def _summarize_counts(counts: np.ndarray) -> dict[str, float]:
    if counts.size == 0:
        return {"n_bins": 0, "n_nonempty": 0}
    return {
        "n_bins": int(counts.size),
        "n_nonempty": int(np.sum(counts > 0)),
        "min": float(counts.min()),
        "p25": float(np.quantile(counts, 0.25)),
        "median": float(np.median(counts)),
        "p75": float(np.quantile(counts, 0.75)),
        "max": float(counts.max()),
    }


def _plot_counts(counts: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(np.arange(len(counts)) + 1, counts, color="#4C78A8", alpha=0.8)
    ax.set_xlabel("bin index")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_u_distribution(u_vals: torch.Tensor, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(u_vals.numpy(), bins=40, color="#54A24B", alpha=0.85)
    ax.set_xlabel("u")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose unif-mass vs CDF+kmeans bins.")
    parser.add_argument("--config-dataset", required=True)
    parser.add_argument("--config-model", required=True)
    parser.add_argument("--config-detection", required=True)
    parser.add_argument("--seed-split", type=int, default=9)
    parser.add_argument("--k-values", nargs="+", default=["20", "30"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root-dir", default="results")
    args = parser.parse_args()

    _apply_style()

    data_cfg = Config(args.config_dataset)
    model_cfg = Config(args.config_model)
    detection_cfg = Config(args.config_detection)

    setup_seeds(seed=1, seed_split=args.seed_split)
    _apply_raw_score_selection(
        cfg_detection=detection_cfg,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        seed_split=args.seed_split,
        root_dir=args.root_dir,
    )

    dataset = get_dataset(
        dataset_name=data_cfg["name"],
        model_name=model_cfg["model_name"],
        root=os.environ.get("DATA_DIR", "./data"),
        preprocess=model_cfg["preprocessor"],
        shuffle=False,
    )

    res_loader, cal_loader, test_loader = prepare_ablation_dataloaders(
        dataset=dataset,
        seed_split=args.seed_split,
        n_res=data_cfg["n_samples"]["res"],
        n_cal=data_cfg["n_samples"]["cal"],
        n_test=data_cfg["n_samples"]["test"],
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

    temp = float(detection_cfg["postprocessor_args"]["temperature"])
    mag = float(detection_cfg["postprocessor_args"].get("magnitude", 0.0))
    normalize = bool(detection_cfg["postprocessor_args"].get("normalize", False))
    use_perturbed = bool(detection_cfg.get("experience_args", {}).get("use_perturbed_logits", False))
    if not use_perturbed:
        mag = 0.0

    res_logits = _collect_logits(
        model=model,
        loader=res_loader,
        device=device,
        magnitude=mag,
        temperature=temp,
        normalize=normalize,
    )
    cal_logits = _collect_logits(
        model=model,
        loader=cal_loader,
        device=device,
        magnitude=mag,
        temperature=temp,
        normalize=normalize,
    )
    test_logits = _collect_logits(
        model=model,
        loader=test_loader,
        device=device,
        magnitude=mag,
        temperature=temp,
        normalize=normalize,
    )

    res_scores = doctor_gini(res_logits, temperature=temp, normalize=normalize).squeeze(1)
    cal_scores = doctor_gini(cal_logits, temperature=temp, normalize=normalize).squeeze(1)
    test_scores = doctor_gini(test_logits, temperature=temp, normalize=normalize).squeeze(1)

    u_res = _cdf_transform(res_scores, res_scores)
    u_cal = _cdf_transform(cal_scores, res_scores)
    u_test = _cdf_transform(test_scores, res_scores)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_u_distribution(u_cal, out_dir / "cdf_u_cal_hist.png", "CDF u distribution (cal)")
    _plot_u_distribution(u_test, out_dir / "cdf_u_test_hist.png", "CDF u distribution (test)")

    summary: dict[str, dict[str, dict[str, float]]] = {}

    for k_str in args.k_values:
        k = int(k_str)
        summary[str(k)] = {}

        # uniform-mass edges on cal (raw score)
        q = torch.linspace(0.0, 1.0, k + 1)[1:-1]
        cal_edges = torch.quantile(cal_scores, q)
        counts_cal = _bin_counts(cal_scores, cal_edges, k)
        counts_test = _bin_counts(test_scores, cal_edges, k)
        summary[str(k)]["unif_mass_cal_bins_cal"] = _summarize_counts(counts_cal)
        summary[str(k)]["unif_mass_cal_bins_test"] = _summarize_counts(counts_test)
        _plot_counts(
            counts_cal,
            out_dir / f"unif_mass_cal_bins_k{k}_cal_counts.png",
            f"Uniform-mass (cal bins) counts on cal (K={k})",
        )
        _plot_counts(
            counts_test,
            out_dir / f"unif_mass_cal_bins_k{k}_test_counts.png",
            f"Uniform-mass (cal bins) counts on test (K={k})",
        )

        # uniform-mass edges on res (raw score, control)
        res_edges = torch.quantile(res_scores, q)
        counts_res = _bin_counts(res_scores, res_edges, k)
        counts_test_res = _bin_counts(test_scores, res_edges, k)
        summary[str(k)]["unif_mass_res_bins_res"] = _summarize_counts(counts_res)
        summary[str(k)]["unif_mass_res_bins_test"] = _summarize_counts(counts_test_res)
        _plot_counts(
            counts_test_res,
            out_dir / f"unif_mass_res_bins_k{k}_test_counts.png",
            f"Uniform-mass (res bins) counts on test (K={k})",
        )

        # uniform-width bins in u (res CDF)
        u_edges = torch.linspace(0.0, 1.0, k + 1)[1:-1]
        counts_u_cal = _bin_counts(u_cal, u_edges, k)
        counts_u_test = _bin_counts(u_test, u_edges, k)
        summary[str(k)]["cdf_uniform_bins_cal"] = _summarize_counts(counts_u_cal)
        summary[str(k)]["cdf_uniform_bins_test"] = _summarize_counts(counts_u_test)
        _plot_counts(
            counts_u_test,
            out_dir / f"cdf_uniform_bins_k{k}_test_counts.png",
            f"CDF uniform-width bins counts on test (K={k})",
        )

        # kmeans on CDF (res fit)
        cfg = dict(detection_cfg["postprocessor_args"])
        cfg["method"] = "kmeans_torch"
        cfg["n_clusters"] = k
        cfg["score_transform"] = "cdf"
        detector = PartitionPostprocessor(
            model=model,
            cfg=cfg,
            result_folder=str(out_dir),
            device=device,
        )
        detector.fit(logits=res_logits, detector_labels=torch.zeros(res_scores.numel()), fit_clustering=True)
        clusters_cal = detector.predict_clusters(logits=cal_logits).squeeze(0).cpu().numpy()
        clusters_test = detector.predict_clusters(logits=test_logits).squeeze(0).cpu().numpy()
        counts_km_cal = np.bincount(clusters_cal, minlength=k)
        counts_km_test = np.bincount(clusters_test, minlength=k)
        summary[str(k)]["kmeans_cdf_cal"] = _summarize_counts(counts_km_cal)
        summary[str(k)]["kmeans_cdf_test"] = _summarize_counts(counts_km_test)
        _plot_counts(
            counts_km_test,
            out_dir / f"kmeans_cdf_k{k}_test_counts.png",
            f"K-means + CDF counts on test (K={k})",
        )

    meta = {
        "n_res": data_cfg["n_samples"]["res"],
        "n_cal": data_cfg["n_samples"]["cal"],
        "n_test": data_cfg["n_samples"]["test"],
        "temperature": temp,
        "magnitude": mag,
        "normalize": normalize,
        "use_perturbed_logits": use_perturbed,
    }
    payload = {"meta": meta, "summary": summary}
    (out_dir / "diagnose_cdf_vs_unif_mass_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
