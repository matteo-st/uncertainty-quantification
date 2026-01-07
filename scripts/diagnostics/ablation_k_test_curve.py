from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from error_estimation.utils.metrics import compute_all_metrics
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
            "legend.fontsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _parse_k_values(values: list[str]) -> list[int]:
    if len(values) == 1 and "," in values[0]:
        values = values[0].split(",")
    k_vals = []
    for item in values:
        item = item.strip()
        if not item:
            continue
        k_vals.append(int(item))
    if not k_vals:
        raise ValueError("No K values provided")
    return k_vals


def _build_indices(n_total: int, n_res: int, n_cal: int, n_test: int, seed_split: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = list(range(n_total))
    rng = random.Random(seed_split)
    rng.shuffle(perm)
    cal_idx = np.asarray(perm[:n_cal], dtype=int)
    res_idx = np.asarray(perm[n_cal : n_cal + n_res], dtype=int)
    test_idx = np.asarray(perm[n_total - n_test :], dtype=int)
    return res_idx, cal_idx, test_idx


def _plot_curve(df: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    df_sorted = df.sort_values("n_clusters")
    ax.plot(df_sorted["n_clusters"], df_sorted[value_col], marker="o", lw=1.6, color="tab:blue")
    ax.set_xlabel("n_clusters (K)")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute test metrics vs K for uniform-mass binning.")
    parser.add_argument("--latent-path", required=True, help="Path to latent full.pt file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--n-res", type=int, required=True, help="Number of res samples")
    parser.add_argument("--n-cal", type=int, required=True, help="Number of cal samples")
    parser.add_argument("--n-test", type=int, required=True, help="Number of test samples")
    parser.add_argument("--seed-split", type=int, required=True, help="Seed split")
    parser.add_argument("--n-classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--k-values", nargs="+", required=True, help="List of K values (space or comma separated)")
    parser.add_argument("--space", default="gini", help="Quantizer space (gini/msp/max_proba)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for score")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha for Hoeffding bound")
    parser.add_argument("--bound", default="hoeffding", help="Bound type (hoeffding/bernstein)")
    parser.add_argument("--score", default="upper", help="Score type (upper/mean)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    k_values = _parse_k_values(args.k_values)

    latent_path = Path(args.latent_path)
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing latent file: {latent_path}")
    pkg = torch.load(latent_path, map_location="cpu")
    logits = pkg["logits"].to(torch.float32)
    labels = pkg["labels"].to(torch.int64)
    preds = pkg["model_preds"].to(torch.int64)
    detector_labels = (preds != labels).to(torch.float32)

    res_idx, cal_idx, test_idx = _build_indices(
        n_total=logits.shape[0],
        n_res=args.n_res,
        n_cal=args.n_cal,
        n_test=args.n_test,
        seed_split=args.seed_split,
    )

    res_logits = logits[res_idx]
    cal_logits = logits[cal_idx]
    test_logits = logits[test_idx]
    res_labels = detector_labels[res_idx]
    cal_labels = detector_labels[cal_idx]
    test_labels = detector_labels[test_idx]

    rows = []
    for k in k_values:
        cfg = {
            "alpha": args.alpha,
            "method": "unif-mass",
            "bound": args.bound,
            "n_classes": args.n_classes,
            "space": args.space,
            "reorder_embs": False,
            "temperature": args.temperature,
            "magnitude": 0.0,
            "pred_weights": 0,
            "score": args.score,
            "n_clusters": k,
            "init_scheme": None,
            "n_init": None,
            "max_iter": None,
            "clustering_seed": None,
        }
        dec = PartitionPostprocessor(model=None, cfg=cfg, result_folder=str(output_dir), device=torch.device("cpu"))
        dec.fit(logits=res_logits, detector_labels=res_labels, fit_clustering=True)
        dec.fit(logits=cal_logits, detector_labels=cal_labels, fit_clustering=False)
        scores = dec(logits=test_logits).detach().cpu().numpy()
        metrics = compute_all_metrics(scores, test_labels.cpu().numpy())
        row = {"n_clusters": k}
        row.update({f"{key}_test": value for key, value in metrics.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "test_curve.csv", index=False)

    _plot_curve(df, "roc_auc_test", output_dir / "test_curve_roc_auc.png", "ROC-AUC on test vs K")
    _plot_curve(df, "fpr_test", output_dir / "test_curve_fpr.png", "FPR@95 on test vs K")

    meta = {
        "latent_path": str(latent_path),
        "n_res": args.n_res,
        "n_cal": args.n_cal,
        "n_test": args.n_test,
        "seed_split": args.seed_split,
        "n_classes": args.n_classes,
        "k_values": k_values,
        "space": args.space,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "bound": args.bound,
        "score": args.score,
    }
    (output_dir / "test_curve_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
