#!/usr/bin/env python3
"""
Evaluate raw LDA-combined scores WITHOUT binning.

For each base_scores combination:
1. Fit LDA on res split (supervised by error labels)
2. Apply LDA to get 1D projected score on cal and test
3. Compute FPR@95 and ROC-AUC on the raw continuous score

This isolates the effect of LDA score combination from binning.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from error_estimation.utils.config import Config
from error_estimation.utils.datasets import get_dataset
from error_estimation.utils.datasets.dataloader import prepare_ablation_dataloaders
from error_estimation.utils.models import get_model


def compute_fpr_at_tpr(scores, labels, target_tpr=0.95):
    """Compute FPR at target TPR (e.g., 95% recall of errors)."""
    thresholds = np.sort(np.unique(scores))[::-1]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    for thr in thresholds:
        pred_pos = scores >= thr
        tp = (pred_pos & (labels == 1)).sum()
        fp = (pred_pos & (labels == 0)).sum()
        tpr = tp / n_pos
        fpr = fp / n_neg
        if tpr >= target_tpr:
            return fpr
    return 1.0


def compute_gini_score(logits, temperature=1.0, magnitude=0.0, normalize=True):
    """Compute Gini impurity score (Doctor score)."""
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    if normalize:
        gini = 1.0 - (probs ** 2).sum(dim=-1)
        n_classes = probs.shape[-1]
        max_gini = 1.0 - 1.0 / n_classes
        gini = gini / max_gini
    else:
        gini = 1.0 - (probs ** 2).sum(dim=-1)
    return gini.cpu().numpy()


def compute_margin_score(logits, temperature=1.0):
    """Compute margin score (difference between top-2 probabilities)."""
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    return (1.0 - margin).cpu().numpy()


def compute_msp_score(logits, temperature=1.0):
    """Compute MSP score (max softmax probability)."""
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    msp = probs.max(dim=-1).values
    return (1.0 - msp).cpu().numpy()


def compute_entropy_score(logits, temperature=1.0):
    """Compute entropy score."""
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    probs = torch.clamp(probs, min=1e-10)
    entropy = -(probs * torch.log(probs)).sum(dim=-1)
    n_classes = probs.shape[-1]
    max_entropy = np.log(n_classes)
    entropy = entropy / max_entropy
    return entropy.cpu().numpy()


def load_score_configs(results_dir, dataset, model, seed_split):
    """Load best hyperparams for each score from LDA binning results (per-seed)."""
    configs = {}
    lda_path = results_dir / dataset / model / "lda_binning" / "runs"
    if lda_path.exists():
        # Sort by modification time (most recent first)
        runs = sorted(
            [d for d in lda_path.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True
        )
        if runs:
            seed_dir = runs[0] / f"seed-split-{seed_split}"
            search_file = seed_dir / "search.jsonl"
            if search_file.exists():
                with open(search_file) as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            if "score_configs" in record:
                                configs = record["score_configs"]
                                break
    # Defaults if not found
    if "gini" not in configs:
        configs["gini"] = {"temperature": 1.0, "magnitude": 0.0, "normalize": True}
    if "margin" not in configs:
        configs["margin"] = {"temperature": 1.0, "magnitude": 0.0}
    if "msp" not in configs:
        configs["msp"] = {"temperature": 1.0, "magnitude": 0.0}
    if "entropy" not in configs:
        configs["entropy"] = {"temperature": 1.0}
    return configs


def compute_all_scores(logits, score_configs):
    """Compute all base scores for given logits."""
    scores = {}
    cfg = score_configs.get("gini", {})
    scores["gini"] = compute_gini_score(
        logits, temperature=cfg.get("temperature", 1.0),
        magnitude=cfg.get("magnitude", 0.0), normalize=cfg.get("normalize", True)
    )
    cfg = score_configs.get("margin", {})
    scores["margin"] = compute_margin_score(logits, temperature=cfg.get("temperature", 1.0))
    cfg = score_configs.get("msp", {})
    scores["msp"] = compute_msp_score(logits, temperature=cfg.get("temperature", 1.0))
    cfg = score_configs.get("entropy", {})
    scores["entropy"] = compute_entropy_score(logits, temperature=cfg.get("temperature", 1.0))
    return scores


def evaluate_raw_score(score, labels):
    """Evaluate a raw continuous score."""
    fpr = compute_fpr_at_tpr(score, labels, target_tpr=0.95)
    try:
        roc_auc = roc_auc_score(labels, score)
    except:
        roc_auc = np.nan
    return {"fpr": fpr, "roc_auc": roc_auc}


def get_logits_and_labels(dataloader, model, device):
    """Run inference to get logits and detector labels."""
    all_logits = []
    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=-1)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = torch.cat(all_preds, dim=0)

    # Detector labels: 1 = error (wrong prediction), 0 = correct
    detector_labels = (preds != labels).long().numpy()

    return logits, detector_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoints-dir", default="./checkpoints")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output-dir", default="./results/raw_lda_evaluation")
    parser.add_argument("--seed-splits", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configurations to evaluate
    configs = [
        {
            "dataset": "cifar10", "model": "resnet34_ce",
            "dataset_cfg": "configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml",
            "model_cfg": "configs/models/cifar10_resnet34.yml",
        },
        {
            "dataset": "cifar10", "model": "densenet121_ce",
            "dataset_cfg": "configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml",
            "model_cfg": "configs/models/cifar10_densenet121.yml",
        },
        {
            "dataset": "cifar100", "model": "resnet34_ce",
            "dataset_cfg": "configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml",
            "model_cfg": "configs/models/cifar100_resnet34.yml",
        },
        {
            "dataset": "cifar100", "model": "densenet121_ce",
            "dataset_cfg": "configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml",
            "model_cfg": "configs/models/cifar100_densenet121.yml",
        },
    ]

    # Score combinations to test
    score_combinations = [
        ["gini", "margin"],
        ["gini", "msp"],
        ["gini", "entropy"],
        ["margin", "msp"],
        ["gini", "margin", "msp"],
        ["gini", "margin", "entropy"],
        ["gini", "msp", "entropy"],
        ["gini", "margin", "msp", "entropy"],
    ]

    all_results = []

    for config in configs:
        dataset_name = config["dataset"]
        model_name = config["model"]
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}, Model: {model_name}")
        print(f"{'='*60}")

        # Load configs
        cfg_dataset = Config(config["dataset_cfg"])
        cfg_model = Config(config["model_cfg"])

        # Load model once
        model = get_model(
            model_name=cfg_model["model_name"],
            dataset_name=cfg_dataset["name"],
            n_classes=cfg_dataset["num_classes"],
            model_seed=cfg_model["seed"],
            checkpoint_dir=os.path.join(args.checkpoints_dir, cfg_model["preprocessor"]),
        )
        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        for seed_split in tqdm(args.seed_splits, desc="Seeds"):
            # Load per-seed score configs from previous grids
            score_configs = load_score_configs(results_dir, dataset_name, model_name, seed_split)
            print(f"  Seed {seed_split} configs: gini T={score_configs['gini'].get('temperature')}, "
                  f"margin T={score_configs['margin'].get('temperature')}")

            # Load dataset and create dataloaders
            dataset = get_dataset(
                dataset_name=cfg_dataset["name"],
                model_name=cfg_model["model_name"],
                root=args.data_dir,
                preprocess=cfg_model["preprocessor"],
                shuffle=False,
            )

            res_loader, cal_loader, test_loader = prepare_ablation_dataloaders(
                dataset=dataset,
                seed_split=seed_split,
                n_res=cfg_dataset["n_samples"]["res"],
                n_cal=cfg_dataset["n_samples"]["cal"],
                n_test=cfg_dataset["n_samples"]["test"],
                batch_size_train=256,
                batch_size_test=256,
                cal_transform=None,
                res_transform=None,
                data_name=cfg_dataset["name"],
                model_name=cfg_model["model_name"],
            )

            # Get logits and labels for each split
            print(f"  Seed {seed_split}: Running inference...")
            logits_res, labels_res = get_logits_and_labels(res_loader, model, device)
            logits_cal, labels_cal = get_logits_and_labels(cal_loader, model, device)
            logits_test, labels_test = get_logits_and_labels(test_loader, model, device)

            # Compute all base scores
            scores_res = compute_all_scores(logits_res, score_configs)
            scores_cal = compute_all_scores(logits_cal, score_configs)
            scores_test = compute_all_scores(logits_test, score_configs)

            # Evaluate individual scores
            for score_name in ["gini", "margin", "msp", "entropy"]:
                res_metrics = evaluate_raw_score(scores_res[score_name], labels_res)
                cal_metrics = evaluate_raw_score(scores_cal[score_name], labels_cal)
                test_metrics = evaluate_raw_score(scores_test[score_name], labels_test)

                result = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed_split": seed_split,
                    "score_type": "individual",
                    "base_scores": score_name,
                    "score_config": score_configs.get(score_name, {}),
                    "fpr_res": res_metrics["fpr"],
                    "roc_auc_res": res_metrics["roc_auc"],
                    "fpr_cal": cal_metrics["fpr"],
                    "roc_auc_cal": cal_metrics["roc_auc"],
                    "fpr_test": test_metrics["fpr"],
                    "roc_auc_test": test_metrics["roc_auc"],
                }
                all_results.append(result)

            # Evaluate LDA combinations
            for base_scores in score_combinations:
                X_res = np.column_stack([scores_res[s] for s in base_scores])
                X_cal = np.column_stack([scores_cal[s] for s in base_scores])
                X_test = np.column_stack([scores_test[s] for s in base_scores])

                lda = LinearDiscriminantAnalysis(n_components=1)
                try:
                    lda.fit(X_res, labels_res)
                except Exception as e:
                    print(f"LDA fit failed for {base_scores}: {e}")
                    continue

                lda_res = lda.transform(X_res).ravel()
                lda_cal = lda.transform(X_cal).ravel()
                lda_test = lda.transform(X_test).ravel()

                # Ensure higher score = error
                if np.corrcoef(lda_res, labels_res)[0, 1] < 0:
                    lda_res = -lda_res
                    lda_cal = -lda_cal
                    lda_test = -lda_test

                res_metrics = evaluate_raw_score(lda_res, labels_res)
                cal_metrics = evaluate_raw_score(lda_cal, labels_cal)
                test_metrics = evaluate_raw_score(lda_test, labels_test)

                result = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed_split": seed_split,
                    "score_type": "lda",
                    "base_scores": "+".join(base_scores),
                    "lda_coef": lda.coef_.tolist(),
                    "fpr_res": res_metrics["fpr"],
                    "roc_auc_res": res_metrics["roc_auc"],
                    "fpr_cal": cal_metrics["fpr"],
                    "roc_auc_cal": cal_metrics["roc_auc"],
                    "fpr_test": test_metrics["fpr"],
                    "roc_auc_test": test_metrics["roc_auc"],
                }
                all_results.append(result)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "raw_lda_results.csv", index=False)

    with open(output_dir / "raw_lda_results.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    for (dataset, model), group in results_df.groupby(["dataset", "model"]):
        print(f"\n{dataset} / {model}:")
        print("-" * 50)

        ind_df = group[group["score_type"] == "individual"]
        print("\nIndividual Scores:")
        for score_name in ["gini", "margin", "msp", "entropy"]:
            score_df = ind_df[ind_df["base_scores"] == score_name]
            if len(score_df) > 0:
                print(f"  {score_name:10s}: ROC-AUC = {score_df['roc_auc_test'].mean():.4f} ± {score_df['roc_auc_test'].std():.4f}, "
                      f"FPR@95 = {score_df['fpr_test'].mean():.4f} ± {score_df['fpr_test'].std():.4f}")

        lda_df = group[group["score_type"] == "lda"]
        print("\nLDA Combinations:")
        for combo in lda_df["base_scores"].unique():
            combo_df = lda_df[lda_df["base_scores"] == combo]
            print(f"  LDA({combo:25s}): ROC-AUC = {combo_df['roc_auc_test'].mean():.4f} ± {combo_df['roc_auc_test'].std():.4f}, "
                  f"FPR@95 = {combo_df['fpr_test'].mean():.4f} ± {combo_df['fpr_test'].std():.4f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
