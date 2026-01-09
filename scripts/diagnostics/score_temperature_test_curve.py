from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from error_estimation.utils.metrics import compute_all_metrics


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


def _parse_float_values(values: list[str]) -> list[float]:
    if len(values) == 1 and "," in values[0]:
        values = values[0].split(",")
    parsed = []
    for item in values:
        item = item.strip()
        if not item:
            continue
        parsed.append(float(item))
    if not parsed:
        raise ValueError("No float values provided")
    return parsed


def _build_indices(n_total: int, n_res: int, n_cal: int, n_test: int, seed_split: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = list(range(n_total))
    rng = random.Random(seed_split)
    rng.shuffle(perm)
    cal_idx = np.asarray(perm[:n_cal], dtype=int)
    res_idx = np.asarray(perm[n_cal : n_cal + n_res], dtype=int)
    test_idx = np.asarray(perm[n_total - n_test :], dtype=int)
    return res_idx, cal_idx, test_idx


def _compute_scores(logits: torch.Tensor, temperature: float, score: str) -> torch.Tensor:
    probs = torch.softmax(logits / temperature, dim=1)
    if score == "max_proba":
        return -probs.max(dim=1).values
    if score == "gini":
        g = torch.sum(probs**2, dim=1)
        return 1.0 - g
    if score == "margin":
        top2 = torch.topk(probs, k=2, dim=1).values
        return -(top2[:, 0] - top2[:, 1])
    raise ValueError(f"Unsupported score: {score}")


def _plot_metric(
    mean_rows: list[dict[str, float]],
    std_rows: list[dict[str, float]],
    score: str,
    metric: str,
    out_path: Path,
    temps: list[float],
    dtypes: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for dtype in dtypes:
        mean_vals = [row[metric] for row in mean_rows if row["score"] == score and row["dtype"] == dtype]
        std_vals = [row[metric] for row in std_rows if row["score"] == score and row["dtype"] == dtype]
        if not mean_vals:
            continue
        ax.plot(temps, mean_vals, marker="o", lw=1.5, label=f"{dtype}")
        if std_vals:
            mean_arr = np.asarray(mean_vals, dtype=float)
            std_arr = np.asarray(std_vals, dtype=float)
            ax.fill_between(temps, mean_arr - std_arr, mean_arr + std_arr, alpha=0.12)
    ax.set_xlabel("Temperature")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(f"{score}: {metric.upper()} vs temperature (test)")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate continuous score performance vs temperature.")
    parser.add_argument("--latent-path", required=True, help="Path to latent .pt file with logits.")
    parser.add_argument("--output-dir", required=True, help="Output directory for CSV/plots.")
    parser.add_argument("--n-res", type=int, required=True, help="Number of res samples.")
    parser.add_argument("--n-cal", type=int, required=True, help="Number of cal samples.")
    parser.add_argument("--n-test", type=int, required=True, help="Number of test samples.")
    parser.add_argument("--seed-splits", nargs="+", type=int, required=True, help="Seed splits to evaluate.")
    parser.add_argument("--temperatures", nargs="+", required=True, help="Temperatures to evaluate.")
    parser.add_argument("--scores", nargs="+", default=["max_proba"], help="Scores (max_proba/gini/margin).")
    parser.add_argument("--dtypes", nargs="+", default=["float32"], help="Dtypes (float16/float32/float64).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temps = _parse_float_values(args.temperatures)
    scores = [s.strip() for s in args.scores]
    dtypes = [d.strip() for d in args.dtypes]

    latent_path = Path(args.latent_path)
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing latent file: {latent_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    pkg = torch.load(latent_path, map_location="cpu")
    logits = pkg["logits"].to(torch.float32)
    labels = pkg["labels"].to(torch.int64)
    preds = pkg["model_preds"].to(torch.int64)
    detector_labels = (preds != labels).to(torch.float32)

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    rows = []
    for seed_split in args.seed_splits:
        _, _, test_idx = _build_indices(
            n_total=logits.shape[0],
            n_res=args.n_res,
            n_cal=args.n_cal,
            n_test=args.n_test,
            seed_split=seed_split,
        )
        test_logits = logits[test_idx]
        test_labels = detector_labels[test_idx]
        for score_name in scores:
            for dtype_name in dtypes:
                if dtype_name not in dtype_map:
                    raise ValueError(f"Unsupported dtype: {dtype_name}")
                dtype = dtype_map[dtype_name]
                eps = float(torch.finfo(dtype).eps)
                logits_cast = test_logits.to(dtype)
                for temperature in temps:
                    score_vals = _compute_scores(logits_cast, float(temperature), score_name)
                    metrics = compute_all_metrics(score_vals.detach().cpu().numpy(), test_labels.cpu().numpy())
                    rows.append(
                        {
                            "seed_split": seed_split,
                            "score": score_name,
                            "dtype": dtype_name,
                            "eps": eps,
                            "temperature": float(temperature),
                            "roc_auc_test": float(metrics["roc_auc"]),
                            "fpr_test": float(metrics["fpr"]),
                        }
                    )

    per_seed_path = out_dir / "score_temperature_test_per_seed.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed_split",
                "score",
                "dtype",
                "eps",
                "temperature",
                "roc_auc_test",
                "fpr_test",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    agg = {}
    for row in rows:
        key = (row["score"], row["dtype"], row["temperature"], row["eps"])
        agg.setdefault(key, {"roc_auc_test": [], "fpr_test": []})
        agg[key]["roc_auc_test"].append(row["roc_auc_test"])
        agg[key]["fpr_test"].append(row["fpr_test"])

    mean_rows = []
    std_rows = []
    for (score, dtype, temperature, eps), values in agg.items():
        mean_rows.append(
            {
                "score": score,
                "dtype": dtype,
                "eps": eps,
                "temperature": temperature,
                "roc_auc_test": float(np.mean(values["roc_auc_test"])),
                "fpr_test": float(np.mean(values["fpr_test"])),
            }
        )
        std_rows.append(
            {
                "score": score,
                "dtype": dtype,
                "eps": eps,
                "temperature": temperature,
                "roc_auc_test": float(np.std(values["roc_auc_test"], ddof=1)),
                "fpr_test": float(np.std(values["fpr_test"], ddof=1)),
            }
        )

    mean_path = out_dir / "score_temperature_test_mean.csv"
    std_path = out_dir / "score_temperature_test_std.csv"
    for path, data in ((mean_path, mean_rows), (std_path, std_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "score",
                    "dtype",
                    "eps",
                    "temperature",
                    "roc_auc_test",
                    "fpr_test",
                ],
            )
            writer.writeheader()
            writer.writerows(data)

    for score in scores:
        _plot_metric(
            mean_rows,
            std_rows,
            score=score,
            metric="roc_auc_test",
            out_path=out_dir / f"{score}_roc_auc_vs_temperature.png",
            temps=temps,
            dtypes=dtypes,
        )
        _plot_metric(
            mean_rows,
            std_rows,
            score=score,
            metric="fpr_test",
            out_path=out_dir / f"{score}_fpr_vs_temperature.png",
            temps=temps,
            dtypes=dtypes,
        )


if __name__ == "__main__":
    main()
