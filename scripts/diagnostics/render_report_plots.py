from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import json


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


def _plot_ci_vs_score(df: pd.DataFrame, out_path: Path, title: str, x_label: str) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(df["center"], df["upper"], color="tab:red", lw=1.4, label="Upper CI")
    ax.plot(df["center"], df["lower"], color="tab:blue", lw=1.4, label="Lower CI")
    ax.fill_between(df["center"], df["lower"], df["upper"], color="tab:blue", alpha=0.12)
    ax.scatter(df["center"], df["mean_cal"], s=16, color="black", label=r"$\widehat{\eta}(z)$")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Confidence interval")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_width_vs_halfwidth(df: pd.DataFrame, out_path: Path, title: str, x_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.scatter(df["width"], df["half_width"], s=20, alpha=0.75, color="tab:purple")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Half-width $h_z$")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_bin_width_hist(df: pd.DataFrame, out_path: Path, title: str, x_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    finite_widths = df["width"].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(finite_widths, bins=22, color="tab:gray", alpha=0.85)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_count_shift(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    x = df["count_cal"] + 1
    y = df["count_test"] + 1
    ax.scatter(x, y, s=18, alpha=0.75, color="tab:green")
    max_val = max(float(x.max()), float(y.max()))
    ax.plot([1, max_val], [1, max_val], color="black", lw=1.0, alpha=0.6, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Count (cal)")
    ax.set_ylabel("Count (test)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _compute_score(
    logits: torch.Tensor,
    space: str,
    temperature: float,
    normalize_gini: bool,
) -> np.ndarray:
    if space == "gini":
        probs = torch.softmax(logits / temperature, dim=1)
        g = torch.sum(probs**2, dim=1)
        if normalize_gini:
            score = (1.0 - g) / g
        else:
            score = 1.0 - g
        return score.detach().cpu().numpy()
    if space in {"msp", "max_proba"}:
        probs = torch.softmax(logits / temperature, dim=1)
        score = -probs.max(dim=1).values
        return score.detach().cpu().numpy()
    raise ValueError(f"Unsupported score space: {space}")


def _build_test_indices(n_total: int, n_res: int, n_cal: int, n_test: int, seed_split: int | None) -> np.ndarray:
    perm = list(range(n_total))
    if seed_split is not None:
        rng = random.Random(seed_split)
        rng.shuffle(perm)
    test_idx = perm[n_total - n_test :]
    return np.asarray(test_idx, dtype=int)


def _build_res_indices(n_total: int, n_res: int, n_cal: int, seed_split: int | None) -> np.ndarray:
    perm = list(range(n_total))
    if seed_split is not None:
        rng = random.Random(seed_split)
        rng.shuffle(perm)
    res_idx = perm[n_cal : n_cal + n_res]
    return np.asarray(res_idx, dtype=int)


def _plot_score_distribution(
    scores: np.ndarray,
    out_path: Path,
    title: str,
    x_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.hist(scores, bins=30, density=True, color="tab:blue", alpha=0.85)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_grid_heatmap(df: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    pivot = df.pivot(index="n_min", columns="n_clusters", values=value_col)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("n_clusters (K)")
    ax.set_ylabel("n_min")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(value_col)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_curve(df: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    df_sorted = df.sort_values("n_clusters")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(df_sorted["n_clusters"], df_sorted[value_col], marker="o", lw=1.5, color="tab:blue")
    ax.set_xlabel("n_clusters (K)")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render report-ready plots from diagnostics CSVs.")
    parser.add_argument("--diagnostics-root", required=True, help="Path to diagnostics_server directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for report images.")
    parser.add_argument("--latent-path", default=None, help="Optional latent full.pt path for score histogram.")
    parser.add_argument("--seed-split", type=int, default=None, help="Seed split used to build test indices.")
    parser.add_argument("--n-res", type=int, default=0, help="Number of res samples.")
    parser.add_argument("--n-cal", type=int, default=0, help="Number of cal samples.")
    parser.add_argument("--n-test", type=int, default=0, help="Number of test samples.")
    parser.add_argument("--score-space", default="gini", help="Score space for histogram (gini/msp).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for score computation.")
    parser.add_argument("--normalize-gini", action="store_true", help="Normalize gini score.")
    args = parser.parse_args()

    root = Path(args.diagnostics_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    runs = {
        "quantile-merge-diagnostics": {
            "title": "Quantile-merge (k=50, n_min=50)",
            "x_label": "Quantile bin center (score)",
            "width_label": r"Bin width $\Delta s_z$",
        },
        "quantile-merge-ablation": {
            "title": "Quantile-merge (best grid)",
            "x_label": "Quantile bin center (score)",
            "width_label": r"Bin width $\Delta s_z$",
        },
        "unif-mass-ablation": {
            "title": "Uniform-mass (best grid)",
            "x_label": "Quantile bin center (score)",
            "width_label": r"Bin width $\Delta s_z$",
        },
    }

    res_scores = None
    if args.latent_path and args.n_res and args.n_cal:
        latent_path = Path(args.latent_path)
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing latent file: {latent_path}")
        pkg = torch.load(latent_path, map_location="cpu")
        logits = pkg["logits"]
        scores_all = _compute_score(
            logits=logits,
            space=args.score_space,
            temperature=args.temperature,
            normalize_gini=args.normalize_gini,
        )
        res_idx = _build_res_indices(
            n_total=scores_all.shape[0],
            n_res=args.n_res,
            n_cal=args.n_cal,
            seed_split=args.seed_split,
        )
        res_scores = scores_all[res_idx]

    for run_name, cfg in runs.items():
        csv_path = root / run_name / "bin_diagnostics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}")
        df = pd.read_csv(csv_path)
        centers = df["center"].to_numpy()
        widths = df["width"].to_numpy()
        if res_scores is not None and "unif-mass" in run_name:
            k = len(df)
            quantiles = np.quantile(res_scores, np.linspace(0.0, 1.0, k + 1))
            centers = 0.5 * (quantiles[:-1] + quantiles[1:])
            widths = quantiles[1:] - quantiles[:-1]
            df = df.copy()
            df["center"] = centers
            df["width"] = widths
        title = cfg["title"]
        _plot_ci_vs_score(
            df,
            out_dir / f"{run_name}_ci_vs_score.png",
            f"{title}: CI vs score",
            cfg["x_label"],
        )
        _plot_width_vs_halfwidth(
            df,
            out_dir / f"{run_name}_width_vs_halfwidth.png",
            f"{title}: width vs half-width",
            cfg["width_label"],
        )
        _plot_bin_width_hist(
            df,
            out_dir / f"{run_name}_bin_width_hist.png",
            f"{title}: bin width histogram",
            cfg["width_label"],
        )
        _plot_count_shift(
            df,
            out_dir / f"{run_name}_count_shift.png",
            f"{title}: cal vs test counts",
        )

    quantile_grid = root / "quantile-merge-ablation-search" / "search_grid.csv"
    if quantile_grid.exists():
        grid_df = pd.read_csv(quantile_grid)
        _plot_grid_heatmap(
            grid_df,
            value_col="roc_auc_res",
            out_path=out_dir / "quantile_merge_grid_roc_auc_res.png",
            title="Quantile-merge grid: ROC-AUC (res)",
        )
        _plot_grid_heatmap(
            grid_df,
            value_col="fpr_res",
            out_path=out_dir / "quantile_merge_grid_fpr_res.png",
            title="Quantile-merge grid: FPR@95 (res)",
        )

    unif_grid = root / "unif-mass-ablation-search" / "search_grid.csv"
    if unif_grid.exists():
        grid_df = pd.read_csv(unif_grid)
        _plot_curve(
            grid_df,
            value_col="roc_auc_res",
            out_path=out_dir / "unif_mass_curve_roc_auc_res.png",
            title="Uniform-mass: ROC-AUC (res) vs K",
        )
        _plot_curve(
            grid_df,
            value_col="fpr_res",
            out_path=out_dir / "unif_mass_curve_fpr_res.png",
            title="Uniform-mass: FPR@95 (res) vs K",
        )

    for dataset in ("cifar10", "cifar100"):
        search_jsonl = root / "unif-mass-grid-v2" / dataset / "search.jsonl"
        if not search_jsonl.exists():
            continue
        df = _load_jsonl(search_jsonl)
        if "n_clusters" in df.columns:
            df["n_clusters"] = pd.to_numeric(df["n_clusters"], errors="coerce")
        for metric in ("roc_auc_res", "fpr_res"):
            if metric not in df.columns:
                continue
            title = f"Uniform-mass {dataset}: {metric} vs K (res)"
            out_path = out_dir / f"unif_mass_grid_v2_{dataset}_{metric}.png"
            _plot_curve(df, value_col=metric, out_path=out_path, title=title)

    if args.latent_path:
        latent_path = Path(args.latent_path)
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing latent file: {latent_path}")
        pkg = torch.load(latent_path, map_location="cpu")
        logits = pkg["logits"]
        scores = _compute_score(
            logits=logits,
            space=args.score_space,
            temperature=args.temperature,
            normalize_gini=args.normalize_gini,
        )
        if args.n_test:
            idx = _build_test_indices(
                n_total=scores.shape[0],
                n_res=args.n_res,
                n_cal=args.n_cal,
                n_test=args.n_test,
                seed_split=args.seed_split,
            )
            scores = scores[idx]
        _plot_score_distribution(
            scores=scores,
            out_path=out_dir / "score_distribution_test.png",
            title="Continuous score distribution (test)",
            x_label=f"{args.score_space} score",
        )


if __name__ == "__main__":
    main()
