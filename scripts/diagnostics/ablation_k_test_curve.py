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


def _format_alpha(alpha: float) -> str:
    text = f"{alpha:.3g}"
    if "e" in text:
        return f"{alpha:g}"
    return text


def _compute_continuous_scores(logits: torch.Tensor, space: str, temperature: float) -> np.ndarray:
    if space == "gini":
        probs = torch.softmax(logits / temperature, dim=1)
        g = torch.sum(probs**2, dim=1)
        scores = 1.0 - g
        return scores.detach().cpu().numpy()
    if space in {"msp", "max_proba"}:
        probs = torch.softmax(logits / temperature, dim=1)
        scores = -probs.max(dim=1).values
        return scores.detach().cpu().numpy()
    raise ValueError(f"Unsupported score space: {space}")


def _plot_multi_curve_with_band(
    df_mean: pd.DataFrame,
    df_std: pd.DataFrame,
    value_col: str,
    out_path: Path,
    title: str,
    y_label: str,
    curve_order: list[str],
    curve_labels: dict[str, str],
    curve_styles: dict[str, dict[str, object]],
    continuous_stats: dict[str, dict[str, float]] | None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    df_mean = df_mean.copy()
    df_std = df_std.copy() if df_std is not None else None
    k_values = df_mean["n_clusters"].to_numpy() if not df_mean.empty else np.array([])
    k_min = float(np.min(k_values)) if k_values.size else None
    k_max = float(np.max(k_values)) if k_values.size else None

    for curve in curve_order:
        subset = df_mean[df_mean["curve"] == curve].sort_values("n_clusters")
        if subset.empty:
            continue
        style = curve_styles.get(curve, {})
        label = curve_labels.get(curve, curve)
        line = ax.plot(subset["n_clusters"], subset[value_col], label=label, **style)
        if df_std is not None and value_col in df_std.columns:
            std_series = (
                df_std[df_std["curve"] == curve]
                .set_index("n_clusters")
                .reindex(subset["n_clusters"])[value_col]
            )
            if std_series.notna().any():
                mean_vals = subset[value_col].to_numpy()
                std_vals = std_series.to_numpy()
                color = style.get("color", line[0].get_color())
                ax.fill_between(
                    subset["n_clusters"],
                    mean_vals - std_vals,
                    mean_vals + std_vals,
                    color=color,
                    alpha=0.12,
                )

    if continuous_stats and value_col in continuous_stats.get("mean", {}):
        mean_val = continuous_stats["mean"][value_col]
        std_val = continuous_stats.get("std", {}).get(value_col, None)
        ax.axhline(mean_val, color="red", lw=1.8, label="Continuous score")
        if std_val is not None and k_min is not None and k_max is not None:
            ax.fill_between(
                np.array([k_min, k_max]),
                mean_val - std_val,
                mean_val + std_val,
                color="red",
                alpha=0.12,
            )

    ax.set_xlabel("n_clusters (K)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute test metrics vs K for uniform-mass binning.")
    parser.add_argument("--latent-path", required=True, help="Path to latent full.pt file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--output-prefix", default="test_curve", help="Prefix for output files")
    parser.add_argument("--n-res", type=int, required=True, help="Number of res samples")
    parser.add_argument("--n-cal", type=int, required=True, help="Number of cal samples")
    parser.add_argument("--n-test", type=int, required=True, help="Number of test samples")
    parser.add_argument("--seed-split", type=int, default=None, help="Single seed split")
    parser.add_argument("--seed-splits", nargs="+", type=int, default=None, help="Multiple seed splits")
    parser.add_argument("--n-classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--k-values", nargs="+", default=None, help="List of K values (space or comma separated)")
    parser.add_argument("--k-range", nargs=3, type=int, default=None, help="Range: start end step (inclusive end)")
    parser.add_argument("--space", default="gini", help="Quantizer space (gini/msp/max_proba)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for score")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha for Hoeffding bound")
    parser.add_argument("--alpha-values", nargs="+", default=None, help="Alpha values for upper-bound curves")
    parser.add_argument("--bound", default="hoeffding", help="Bound type (hoeffding/bernstein)")
    parser.add_argument("--score", default="upper", help="Score type (upper/mean)")
    parser.add_argument("--include-mean", action="store_true", help="Add curve for empirical mean score")
    parser.add_argument("--include-continuous", action="store_true", help="Add continuous-score baseline")
    parser.add_argument("--bin-split", default="res", choices=["res", "cal"], help="Split used to build bins")
    parser.add_argument("--ci-split", default="cal", choices=["res", "cal"], help="Split used to compute CIs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    if args.k_range:
        start, end, step = args.k_range
        if step <= 0:
            raise ValueError("k-range step must be > 0")
        k_values = list(range(start, end + 1, step))
    else:
        if not args.k_values:
            raise ValueError("Provide --k-values or --k-range")
        k_values = _parse_k_values(args.k_values)

    latent_path = Path(args.latent_path)
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing latent file: {latent_path}")
    pkg = torch.load(latent_path, map_location="cpu")
    logits = pkg["logits"].to(torch.float32)
    labels = pkg["labels"].to(torch.int64)
    preds = pkg["model_preds"].to(torch.int64)
    detector_labels = (preds != labels).to(torch.float32)

    seed_splits = args.seed_splits or ([] if args.seed_split is None else [args.seed_split])
    if not seed_splits:
        raise ValueError("Provide --seed-split or --seed-splits")

    if args.bin_split == "res" and args.n_res <= 0:
        raise ValueError("bin-split=res requires n_res > 0")
    if args.bin_split == "cal" and args.n_cal <= 0:
        raise ValueError("bin-split=cal requires n_cal > 0")
    if args.ci_split == "res" and args.n_res <= 0:
        raise ValueError("ci-split=res requires n_res > 0")
    if args.ci_split == "cal" and args.n_cal <= 0:
        raise ValueError("ci-split=cal requires n_cal > 0")

    output_prefix = args.output_prefix
    rows = []
    continuous_rows = []
    alpha_values = _parse_float_values(args.alpha_values) if args.alpha_values else [args.alpha]
    multi_mode = args.alpha_values is not None or args.include_mean or args.include_continuous

    if not multi_mode:
        curve_specs = [
            {
                "curve": f"{args.score}_alpha={_format_alpha(args.alpha)}",
                "score": args.score,
                "alpha": args.alpha,
                "label": (
                    f"upper (alpha={_format_alpha(args.alpha)})"
                    if args.score == "upper"
                    else "mean (empirical)"
                ),
            }
        ]
    else:
        curve_specs = []
        score_for_alpha = "upper" if args.include_mean else args.score
        if score_for_alpha == "mean":
            curve_specs.append(
                {
                    "curve": "mean",
                    "score": "mean",
                    "alpha": alpha_values[0],
                    "label": "mean (empirical)",
                }
            )
        else:
            for alpha in alpha_values:
                curve_specs.append(
                    {
                        "curve": f"upper_alpha={_format_alpha(alpha)}",
                        "score": score_for_alpha,
                        "alpha": alpha,
                        "label": f"upper (alpha={_format_alpha(alpha)})",
                    }
                )
        if args.include_mean:
            curve_specs.append(
                {
                    "curve": "mean",
                    "score": "mean",
                    "alpha": alpha_values[0],
                    "label": "mean (empirical)",
                }
            )

    curve_order = [spec["curve"] for spec in curve_specs]
    curve_labels = {spec["curve"]: spec["label"] for spec in curve_specs}
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]
    curve_styles = {}
    for idx, spec in enumerate(curve_specs):
        color = palette[idx % len(palette)]
        if spec["score"] == "mean":
            curve_styles[spec["curve"]] = {"marker": "s", "lw": 1.6, "color": color, "linestyle": "--"}
        else:
            curve_styles[spec["curve"]] = {"marker": "o", "lw": 1.6, "color": color}

    for seed_split in seed_splits:
        res_idx, cal_idx, test_idx = _build_indices(
            n_total=logits.shape[0],
            n_res=args.n_res,
            n_cal=args.n_cal,
            n_test=args.n_test,
            seed_split=seed_split,
        )

        res_logits = logits[res_idx]
        cal_logits = logits[cal_idx]
        test_logits = logits[test_idx]
        res_labels = detector_labels[res_idx]
        cal_labels = detector_labels[cal_idx]
        test_labels = detector_labels[test_idx]

        bin_logits = res_logits if args.bin_split == "res" else cal_logits
        bin_labels = res_labels if args.bin_split == "res" else cal_labels
        ci_logits = res_logits if args.ci_split == "res" else cal_logits
        ci_labels = res_labels if args.ci_split == "res" else cal_labels

        if args.include_continuous:
            cont_scores = _compute_continuous_scores(
                logits=test_logits,
                space=args.space,
                temperature=args.temperature,
            )
            cont_metrics = compute_all_metrics(cont_scores, test_labels.cpu().numpy())
            cont_row = {"seed_split": seed_split}
            cont_row.update({f"{key}_test": value for key, value in cont_metrics.items()})
            continuous_rows.append(cont_row)

        for curve in curve_specs:
            for k in k_values:
                cfg = {
                    "alpha": curve["alpha"],
                    "method": "unif-mass",
                    "bound": args.bound,
                    "n_classes": args.n_classes,
                    "space": args.space,
                    "reorder_embs": False,
                    "temperature": args.temperature,
                    "magnitude": 0.0,
                    "pred_weights": 0,
                    "score": curve["score"],
                    "n_clusters": k,
                    "init_scheme": None,
                    "n_init": None,
                    "max_iter": None,
                    "clustering_seed": None,
                }
                dec = PartitionPostprocessor(model=None, cfg=cfg, result_folder=str(output_dir), device=torch.device("cpu"))
                dec.fit(logits=bin_logits, detector_labels=bin_labels, fit_clustering=True)
                dec.fit(logits=ci_logits, detector_labels=ci_labels, fit_clustering=False)
                scores = dec(logits=test_logits).detach().cpu().numpy().reshape(-1)
                metrics = compute_all_metrics(scores, test_labels.cpu().numpy())
                row = {
                    "n_clusters": k,
                    "seed_split": seed_split,
                    "curve": curve["curve"],
                    "alpha": curve["alpha"],
                    "score": curve["score"],
                }
                row.update({f"{key}_test": value for key, value in metrics.items()})
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"{output_prefix}_per_seed.csv", index=False)

    metric_cols = [col for col in df.columns if col.endswith("_test")]
    agg_mean = df.groupby(["curve", "n_clusters"], as_index=False)[metric_cols].mean()
    agg_std = df.groupby(["curve", "n_clusters"], as_index=False)[metric_cols].std()
    agg_mean.to_csv(output_dir / f"{output_prefix}_mean.csv", index=False)
    agg_std.to_csv(output_dir / f"{output_prefix}_std.csv", index=False)

    continuous_stats = None
    if continuous_rows:
        cont_df = pd.DataFrame(continuous_rows)
        cont_df.to_csv(output_dir / f"{output_prefix}_continuous_per_seed.csv", index=False)
        cont_mean = cont_df[metric_cols].mean().to_dict()
        cont_std = cont_df[metric_cols].std().to_dict()
        continuous_stats = {"mean": cont_mean, "std": cont_std}

    _plot_multi_curve_with_band(
        agg_mean,
        agg_std,
        "roc_auc_test",
        output_dir / f"{output_prefix}_roc_auc.png",
        "ROC-AUC on test vs K",
        "ROC-AUC (test)",
        curve_order,
        curve_labels,
        curve_styles,
        continuous_stats,
    )
    _plot_multi_curve_with_band(
        agg_mean,
        agg_std,
        "fpr_test",
        output_dir / f"{output_prefix}_fpr.png",
        "FPR@95 on test vs K",
        "FPR@95 (test)",
        curve_order,
        curve_labels,
        curve_styles,
        continuous_stats,
    )

    meta = {
        "latent_path": str(latent_path),
        "n_res": args.n_res,
        "n_cal": args.n_cal,
        "n_test": args.n_test,
        "seed_splits": seed_splits,
        "n_classes": args.n_classes,
        "k_values": k_values,
        "space": args.space,
        "temperature": args.temperature,
        "alpha_values": alpha_values,
        "bound": args.bound,
        "score": args.score,
        "include_mean": args.include_mean,
        "include_continuous": args.include_continuous,
        "bin_split": args.bin_split,
        "ci_split": args.ci_split,
        "output_prefix": output_prefix,
    }
    (output_dir / f"{output_prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
