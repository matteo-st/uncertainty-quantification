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
from error_estimation.utils.helper import metric_direction, select_best_index, setup_seeds
from error_estimation.utils.models import get_model
from error_estimation.utils.metrics import compute_all_metrics
from error_estimation.utils.postprocessors.doctor_postprocessor import gini as doctor_gini
from error_estimation.utils.postprocessors.partition_postprocessor import PartitionPostprocessor


def _load_raw_score_selection(
    *,
    detection_cfg: Config,
    data_cfg: Config,
    model_cfg: Config,
    seed_split: int,
    root_dir: str,
) -> dict | None:
    selection = detection_cfg.get("experience_args", {}).get("raw_score_selection")
    if not selection:
        return None
    run_tag = selection.get("run_tag")
    postprocessor = selection.get("postprocessor")
    if not run_tag or not postprocessor:
        return None
    result_dir = Path(root_dir) / data_cfg["name"] / f"{model_cfg['model_name']}_{model_cfg['preprocessor']}" / postprocessor / "runs" / run_tag / f"seed-split-{seed_split}"
    metrics_path = result_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing raw-score metrics at {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return payload.get("config")


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
    return logits_adv, logits_clean


def _collect_logits(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    magnitude: float,
    temperature: float,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_all = []
    labels_all = []
    preds_all = []
    model.eval()
    for inputs, targets in loader:
        inputs = inputs.to(device)
        if magnitude > 0:
            logits_adv, logits_clean = _compute_adv_logits(
                model=model,
                inputs=inputs,
                magnitude=magnitude,
                temperature=temperature,
                normalize=normalize,
            )
            logits = logits_adv
            with torch.no_grad():
                preds = torch.argmax(logits_clean, dim=1)
        else:
            with torch.no_grad():
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)
        logits_all.append(logits.detach().cpu())
        labels_all.append(targets.detach().cpu())
        preds_all.append(preds.detach().cpu())
    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    preds_all = torch.cat(preds_all, dim=0)
    detector_labels = (preds_all != labels_all).float()
    return logits_all, detector_labels


def _prepare_detector(
    *,
    cfg: dict,
    model: torch.nn.Module,
    result_folder: str,
    device: torch.device,
) -> PartitionPostprocessor:
    return PartitionPostprocessor(model=model, cfg=cfg, result_folder=result_folder, device=device)


def _select_init(
    *,
    detector: PartitionPostprocessor,
    res_logits: torch.Tensor,
    res_labels: torch.Tensor,
    metric: str,
) -> int:
    with torch.no_grad():
        scores = detector(logits=res_logits).detach().cpu().numpy()
    metrics = compute_all_metrics(conf=scores, detector_labels=res_labels.cpu().numpy())
    values = np.asarray(metrics.get(metric), dtype=float)
    if values.ndim == 0:
        return 0
    direction = metric_direction(metric)
    return select_best_index(values, direction)


def _build_ci_dataframe(
    *,
    detector: PartitionPostprocessor,
    cal_logits: torch.Tensor,
    cal_labels: torch.Tensor,
) -> dict[str, np.ndarray]:
    embs = detector._extract_embeddings(logits=cal_logits)
    transformed = detector._apply_score_transform(embs, fit=False)
    if transformed.dim() > 1 and transformed.size(1) == 1:
        transformed = transformed.squeeze(1)
    u_vals = transformed.detach().cpu().numpy()
    clusters = detector.predict_clusters(logits=cal_logits).squeeze(0).cpu().numpy().astype(int)

    means = detector.cluster_error_means.squeeze(0).cpu().numpy()
    intervals = detector.cluster_intervals.squeeze(0).cpu().numpy()
    lower = intervals[:, 0]
    upper = intervals[:, 1]

    centers = []
    u_min = []
    u_max = []
    mean_cal = []
    lower_ci = []
    upper_ci = []

    for k in range(len(lower)):
        mask = clusters == k
        if not np.any(mask):
            continue
        u_cluster = u_vals[mask]
        centers.append(float(np.mean(u_cluster)))
        u_min.append(float(np.min(u_cluster)))
        u_max.append(float(np.max(u_cluster)))
        mean_cal.append(float(means[k]))
        lower_ci.append(float(lower[k]))
        upper_ci.append(float(upper[k]))

    return {
        "center": np.asarray(centers),
        "u_min": np.asarray(u_min),
        "u_max": np.asarray(u_max),
        "mean_cal": np.asarray(mean_cal),
        "lower": np.asarray(lower_ci),
        "upper": np.asarray(upper_ci),
    }


def _build_bin_summary(
    *,
    detector: PartitionPostprocessor,
    res_logits: torch.Tensor,
    temperature: float,
    normalize: bool,
) -> list[dict[str, float]]:
    embs = detector._extract_embeddings(logits=res_logits)
    u_vals = detector._apply_score_transform(embs, fit=False)
    if u_vals.dim() > 1 and u_vals.size(1) == 1:
        u_vals = u_vals.squeeze(1)
    u_vals = u_vals.detach().cpu().numpy()

    s_vals = doctor_gini(res_logits, temperature=temperature, normalize=normalize).detach().cpu().numpy()
    clusters = detector.predict_clusters(logits=res_logits).squeeze(0).cpu().numpy().astype(int)

    bins = []
    for k in range(detector.n_clusters):
        mask = clusters == k
        if not np.any(mask):
            continue
        u_min = float(np.min(u_vals[mask]))
        u_max = float(np.max(u_vals[mask]))
        s_min = float(np.min(s_vals[mask]))
        s_max = float(np.max(s_vals[mask]))
        bins.append(
            {
                "cluster": float(k),
                "count": float(np.sum(mask)),
                "u_min": u_min,
                "u_max": u_max,
                "u_center": float(0.5 * (u_min + u_max)),
                "s_min": s_min,
                "s_max": s_max,
                "s_width": float(s_max - s_min),
            }
        )
    bins.sort(key=lambda row: row["u_min"])
    return bins


def _plot_bins_on_s(
    *,
    s_vals: np.ndarray,
    bins: list[dict[str, float]],
    output_path: Path,
    title: str,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(s_vals, bins=60, color="#4C78A8", alpha=0.8)
    boundaries = [row["s_max"] for row in bins[:-1]]
    for edge in boundaries:
        ax.axvline(edge, color="#F58518", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("gini score (s)")
    ax.set_ylabel("count (res)")
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_bin_widths(
    *,
    bins: list[dict[str, float]],
    output_path: Path,
    title: str,
) -> None:
    idx = np.arange(1, len(bins) + 1)
    widths = [row["s_width"] for row in bins]
    counts = [row["count"] for row in bins]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(idx, widths, color="#54A24B", alpha=0.85)
    ax.set_xlabel("Bin index (sorted by u)")
    ax.set_ylabel("Width in s-space")
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return counts


def _plot_u_vs_s_ranges(
    *,
    bins: list[dict[str, float]],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for row in bins:
        ax.vlines(row["u_center"], row["s_min"], row["s_max"], color="#4C78A8", alpha=0.8)
    ax.set_xlabel("u (bin center)")
    ax.set_ylabel("s-range per bin")
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_ci_vs_s(
    *,
    s_vals: np.ndarray,
    bins: list[dict[str, float]],
    lower: np.ndarray,
    upper: np.ndarray,
    mean_cal: np.ndarray,
    output_path: Path,
    title: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    edges = []
    lower_steps = []
    upper_steps = []
    mean_steps = []
    for row in bins:
        idx = int(row["cluster"])
        edges.extend([row["s_min"], row["s_max"]])
        lower_steps.extend([lower[idx], lower[idx]])
        upper_steps.extend([upper[idx], upper[idx]])
        mean_steps.extend([mean_cal[idx], mean_cal[idx]])

    edges = np.asarray(edges)
    lower_steps = np.asarray(lower_steps)
    upper_steps = np.asarray(upper_steps)
    mean_steps = np.asarray(mean_steps)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(9.0, 6.0), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]}
    )

    ax_top.hist(s_vals, bins=60, color="#4C78A8", alpha=0.8)
    for row in bins[:-1]:
        ax_top.axvline(row["s_max"], color="#F58518", linestyle="--", linewidth=1.0)
    ax_top.set_yscale("log")
    ax_top.set_ylabel("count (res)")
    ax_top.grid(alpha=0.2, linestyle=":")
    if xlim is not None:
        ax_top.set_xlim(xlim)

    ax_bottom.plot(edges, lower_steps, color="tab:blue", lw=1.5, label="Lower CI", drawstyle="steps-post")
    ax_bottom.plot(edges, upper_steps, color="tab:red", lw=1.5, label="Upper CI", drawstyle="steps-post")
    ax_bottom.plot(edges, mean_steps, color="black", lw=1.3, label=r"$\widehat{\eta}(s)$", drawstyle="steps-post")
    ax_bottom.fill_between(edges, lower_steps, upper_steps, color="tab:blue", alpha=0.12, step="post")
    ax_bottom.set_xlabel("gini score (s)")
    ax_bottom.set_ylabel("confidence interval")
    ax_bottom.grid(alpha=0.2, linestyle=":")
    ax_bottom.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    if xlim is not None:
        ax_bottom.set_xlim(xlim)

    fig.suptitle(title, y=0.98)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_ci_vs_u(data: dict[str, np.ndarray], output_path: Path, title: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    order = np.argsort(data["center"])
    lower = data["lower"][order]
    upper = data["upper"][order]
    mean_cal = data["mean_cal"][order]
    u_min = data["u_min"][order]
    u_max = data["u_max"][order]

    edges = []
    lower_steps = []
    upper_steps = []
    mean_steps = []
    for lo, hi, lval, uval, mval in zip(u_min, u_max, lower, upper, mean_cal):
        edges.extend([lo, hi])
        lower_steps.extend([lval, lval])
        upper_steps.extend([uval, uval])
        mean_steps.extend([mval, mval])
    edges = np.asarray(edges)
    lower_steps = np.asarray(lower_steps)
    upper_steps = np.asarray(upper_steps)
    mean_steps = np.asarray(mean_steps)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(edges, lower_steps, color="tab:blue", lw=1.6, label="Lower CI", drawstyle="steps-post")
    ax.plot(edges, upper_steps, color="tab:red", lw=1.6, label="Upper CI", drawstyle="steps-post")
    ax.plot(edges, mean_steps, color="black", lw=1.4, label=r"$\widehat{\eta}(u)$", drawstyle="steps-post")
    ax.fill_between(edges, lower_steps, upper_steps, color="tab:blue", alpha=0.12, step="post")
    ax.set_xlabel("Transformed score (u)")
    ax.set_ylabel("Confidence interval")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CDF-transformed score with CI per bin.")
    parser.add_argument("--config-dataset", required=True)
    parser.add_argument("--config-model", required=True)
    parser.add_argument("--config-detection", required=True)
    parser.add_argument("--seed-split", type=int, default=9)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--score", choices=["upper", "mean"], required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--init-metric", choices=["fpr", "roc_auc"], default="fpr")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data_cfg = Config(args.config_dataset)
    model_cfg = Config(args.config_model)
    detection_cfg = Config(args.config_detection)

    setup_seeds(seed=1, seed_split=args.seed_split)

    root_dir = detection_cfg.get("experience_args", {}).get("raw_score_selection", {}).get("root_dir", "./results")
    raw_cfg = _load_raw_score_selection(
        detection_cfg=detection_cfg,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        seed_split=args.seed_split,
        root_dir=root_dir,
    )
    if raw_cfg:
        detection_cfg["postprocessor_args"]["temperature"] = raw_cfg.get("temperature")
        detection_cfg["postprocessor_args"]["magnitude"] = raw_cfg.get("magnitude")
        detection_cfg["postprocessor_args"]["normalize"] = raw_cfg.get("normalize")

    detection_cfg["postprocessor_args"]["n_clusters"] = args.n_clusters
    detection_cfg["postprocessor_args"]["alpha"] = args.alpha
    detection_cfg["postprocessor_args"]["score"] = args.score

    dataset = get_dataset(
        dataset_name=data_cfg["name"],
        model_name=model_cfg["model_name"],
        root=os.environ.get("DATA_DIR", "./data"),
        preprocess=model_cfg["preprocessor"],
        shuffle=False,
    )

    res_loader, cal_loader, _ = prepare_ablation_dataloaders(
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

    res_logits, res_labels = _collect_logits(
        model=model,
        loader=res_loader,
        device=device,
        magnitude=mag,
        temperature=temp,
        normalize=normalize,
    )
    cal_logits, cal_labels = _collect_logits(
        model=model,
        loader=cal_loader,
        device=device,
        magnitude=mag,
        temperature=temp,
        normalize=normalize,
    )

    detector = _prepare_detector(
        cfg=detection_cfg["postprocessor_args"],
        model=model,
        result_folder=str(args.output_dir),
        device=device,
    )
    detector.fit(logits=res_logits, detector_labels=res_labels, fit_clustering=True)
    best_init = _select_init(
        detector=detector,
        res_logits=res_logits,
        res_labels=res_labels,
        metric=args.init_metric,
    )
    detector.clustering_algo.best_init = int(best_init)
    detector.fit(logits=cal_logits, detector_labels=cal_labels, fit_clustering=False)

    data = _build_ci_dataframe(
        detector=detector,
        cal_logits=cal_logits,
        cal_labels=cal_labels,
    )
    output_dir = Path(args.output_dir)
    plot_path = output_dir / "soft_kmeans_cdf_ci_vs_u.png"
    title = (
        f"CI vs transformed score (K={args.n_clusters}, score={args.score}, "
        f"alpha={args.alpha}, init={args.init_metric})"
    )
    _plot_ci_vs_u(data, plot_path, title)

    bins = _build_bin_summary(
        detector=detector,
        res_logits=res_logits,
        temperature=temp,
        normalize=normalize,
    )
    s_vals = doctor_gini(res_logits, temperature=temp, normalize=normalize).detach().cpu().numpy()
    _plot_bins_on_s(
        s_vals=s_vals,
        bins=bins,
        output_path=output_dir / "soft_kmeans_cdf_bins_on_s.png",
        title="Soft-kmeans bins mapped back to s-space (res)",
    )
    _plot_bin_widths(
        bins=bins,
        output_path=output_dir / "soft_kmeans_cdf_bins_s_width.png",
        title="Bin widths in s-space (sorted by u)",
    )
    _plot_u_vs_s_ranges(
        bins=bins,
        output_path=output_dir / "soft_kmeans_cdf_bins_u_vs_s.png",
        title="u-bin centers with s-range",
    )
    lower = detector.cluster_intervals.squeeze(0).cpu().numpy()[:, 0]
    upper = detector.cluster_intervals.squeeze(0).cpu().numpy()[:, 1]
    mean_cal = detector.cluster_error_means.squeeze(0).cpu().numpy()
    _plot_ci_vs_s(
        s_vals=s_vals,
        bins=bins,
        lower=lower,
        upper=upper,
        mean_cal=mean_cal,
        output_path=output_dir / "soft_kmeans_cdf_ci_vs_s.png",
        title="CI vs s-space (bin boundaries from CDF soft-kmeans)",
    )
    for q in (0.9, 0.95, 0.99):
        zoom_max = float(np.quantile(s_vals, q))
        _plot_ci_vs_s(
            s_vals=s_vals,
            bins=bins,
            lower=lower,
            upper=upper,
            mean_cal=mean_cal,
            output_path=output_dir / f"soft_kmeans_cdf_ci_vs_s_zoom_q{int(q*100)}.png",
            title=f"CI vs s-space (zoom near 0; q{int(q*100)})",
            xlim=(float(np.min(s_vals)), zoom_max),
        )

    stats_path = output_dir / "soft_kmeans_cdf_ci_vs_u.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_clusters": args.n_clusters,
                "score": args.score,
                "alpha": args.alpha,
                "init_metric": args.init_metric,
                "temperature": temp,
                "magnitude": mag,
                "normalize": normalize,
                "bins": bins,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
