from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import torch

from error_estimation.evaluators import HyperparamsSearch
from error_estimation.utils.config import Config
from error_estimation.utils.datasets import get_dataset
from error_estimation.utils.datasets.dataloader import prepare_ablation_dataloaders
from error_estimation.utils.experiment import (
    build_latent_paths,
    copy_configs,
    ensure_dir,
    set_num_threads,
    write_run_metadata,
)
from error_estimation.utils.helper import setup_seeds
from error_estimation.utils.logging import setup_logging
from error_estimation.utils.models import get_model
from error_estimation.utils.paths import CHECKPOINTS_DIR, DATA_DIR, LATENTS_DIR, RESULTS_DIR
from error_estimation.utils.results_io import (
    append_summary_csv,
    build_metrics_payload,
    build_run_meta,
    build_summary_row,
    flatten_metrics,
    select_best_row,
    write_metrics_json,
)
from error_estimation.utils.tracking import MLflowTracker, flatten_config

METRIC_KEYS = [
    "fpr",
    "tpr",
    "roc_auc",
    "aurc",
    "aupr_in",
    "aupr_out",
    "accuracy",
    "model_acc",
    "aupr_err",
    "aupr_success",
    "thr",
]


def _collect_metrics(result_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics_path = result_dir / "metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return flatten_metrics(payload.get("metrics", {}))
    csv_candidates = list(result_dir.glob("results_opt-*.csv"))
    csv_candidates += list(result_dir.glob("hyperparams_results_opt-*.csv"))
    if not csv_candidates:
        return metrics

    import pandas as pd

    for csv_path in csv_candidates:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        row = df.iloc[-1]
        for key in METRIC_KEYS:
            if key in row and pd.notna(row[key]):
                try:
                    metrics[key] = float(row[key])
                except (TypeError, ValueError):
                    continue
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run error detection experiment.")
    parser.add_argument(
        "--config-dataset",
        "--config_dataset",
        dest="config_dataset",
        default="configs/datasets/cifar10/cifar10_ablation.yml",
        help="Path to the dataset config file",
    )
    parser.add_argument(
        "--config-model",
        "--config_model",
        dest="config_model",
        default="configs/models/cifar10_resnet34.yml",
        help="Path to the model config file",
    )
    parser.add_argument(
        "--config-detection",
        "--config_detection",
        dest="config_detection",
        default="configs/postprocessors/clustering/cifar10_resnet34_gmm.yml",
        help="Path to the detection config file",
    )
    parser.add_argument(
        "--root-dir",
        "--root_dir",
        dest="root_dir",
        default=RESULTS_DIR,
        help="Root directory to save results",
    )
    parser.add_argument(
        "--latent-dir",
        "--latent_dir",
        dest="latent_dir",
        default=LATENTS_DIR,
        help="Directory to save latent representations",
    )
    parser.add_argument("--data-dir", dest="data_dir", default=DATA_DIR, help="Dataset root directory")
    parser.add_argument(
        "--checkpoints-dir",
        dest="checkpoints_dir",
        default=CHECKPOINTS_DIR,
        help="Checkpoint root directory",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--gpu-id", "--gpu_id", dest="gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--metric",
        type=str,
        default="fpr",
        help="Metric to use for hyperparams selection",
    )
    parser.add_argument(
        "--quantizer-metric",
        "--quantizer_metric",
        dest="quantizer_metric",
        type=str,
        default="same",
        help="Metric to use for quantizer selection",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="search",
        help="Mode to use (search, search_res, evaluation)",
    )
    parser.add_argument(
        "--seed-splits",
        nargs="+",
        type=int,
        default=None,
        help="Optional override for seed splits",
    )
    parser.add_argument("--run-tag", default=None, help="Optional subfolder name under root_dir")
    parser.add_argument("--num-threads", type=int, default=None, help="Force CPU thread count")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs and exit")
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (overrides env)",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "error-estimation"),
        help="MLflow experiment name",
    )
    parser.add_argument("--mlflow-run-name", default=None, help="MLflow run name override")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    return parser


def run(args: argparse.Namespace) -> None:
    data_cfg = Config(args.config_dataset)
    model_cfg = Config(args.config_model)
    detection_cfg = Config(args.config_detection)

    data_cfg.require(["name", "num_classes", "n_samples", "seed_split"], "dataset")
    model_cfg.require(["model_name", "preprocessor", "seed"], "model")
    detection_cfg.require(["name", "experience_args", "postprocessor_args"], "detection")
    for key in ("transform", "n_epochs"):
        if key not in detection_cfg["experience_args"]:
            raise KeyError(f"Missing detection.experience_args.{key}")

    root_dir = Path(args.root_dir)
    if args.run_tag:
        root_dir = root_dir / args.run_tag
    ensure_dir(root_dir)

    logger = setup_logging(root_dir, level=args.log_level)
    logger.info("Starting run in %s", root_dir)

    config_paths = {
        "dataset": args.config_dataset,
        "model": args.config_model,
        "detection": args.config_detection,
    }
    copied_configs = copy_configs(root_dir, config_paths)
    metadata_path = write_run_metadata(root_dir, args, copied_configs)

    tracker = MLflowTracker(
        enabled=not args.no_mlflow,
        experiment_name=args.mlflow_experiment,
        tracking_uri=args.mlflow_uri,
        run_name=args.mlflow_run_name,
    )

    with tracker:
        tracker.log_params(
            flatten_config(
                {
                    "dataset": dict(data_cfg),
                    "model": dict(model_cfg),
                    "detection": dict(detection_cfg),
                }
            )
        )
        tracker.log_tags(
            {
                "dataset": data_cfg.get("name"),
                "model": model_cfg.get("model_name"),
                "postprocessor": detection_cfg.get("name"),
                "mode": args.mode,
                "run_tag": args.run_tag,
            }
        )
        tracker.log_artifact(metadata_path)
        for cfg_path in copied_configs.values():
            tracker.log_artifact(cfg_path)

        if args.dry_run:
            logger.info("Dry run complete (configs loaded).")
            return

        if args.num_threads is not None:
            set_num_threads(args.num_threads)

        seed_splits = args.seed_splits or data_cfg["seed_split"]
        logger.info("Seed splits: %s", seed_splits)

        for seed_split in seed_splits:
            logger.info("Running seed split %s", seed_split)
            setup_seeds(args.seed, seed_split)

            dataset = get_dataset(
                dataset_name=data_cfg["name"],
                model_name=model_cfg["model_name"],
                root=args.data_dir,
                preprocess=model_cfg["preprocessor"],
                shuffle=False,
            )

            device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
            model = get_model(
                model_name=model_cfg["model_name"],
                dataset_name=data_cfg["name"],
                n_classes=data_cfg["num_classes"],
                model_seed=model_cfg["seed"],
                checkpoint_dir=os.path.join(args.checkpoints_dir, model_cfg["preprocessor"]),
            )
            model = model.to(device)
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

            latent_paths = build_latent_paths(args.latent_dir, data_cfg, model_cfg, detection_cfg)
            run_dir = root_dir / f"seed-split-{seed_split}"
            ensure_dir(run_dir)

            cfg_detection_run = deepcopy(detection_cfg)
            if args.mode == "search_res":
                cfg_detection_run["experience_args"]["ratio_res_split"] = 1.0

            child_tags = {
                "seed_split": seed_split,
                "mode": args.mode,
                "metric": args.metric,
                "quantizer_metric": args.quantizer_metric,
            }
            with tracker.child_run(run_name=f"seed-split-{seed_split}", tags=child_tags):
                evaluator = HyperparamsSearch(
                    model=model,
                    cfg_detection=cfg_detection_run,
                    cfg_dataset=data_cfg,
                    device=device,
                    res_loader=res_loader,
                    cal_loader=cal_loader,
                    test_loader=test_loader,
                    result_folder=str(run_dir),
                    metric=args.metric,
                    quantizer_metric=args.quantizer_metric,
                    latent_paths=latent_paths,
                    seed_split=seed_split,
                    mode=args.mode,
                    verbose=False,
                )
                evaluator.run()

                best_row = select_best_row(evaluator.results, args.metric, evaluator.metric_direction)
                meta = build_run_meta(
                    data_cfg,
                    model_cfg,
                    cfg_detection_run,
                    seed_split,
                    mode=args.mode,
                    run_tag=args.run_tag,
                    extra={"result_dir": str(run_dir)},
                )
                payload = build_metrics_payload(meta, best_row)
                metrics_path = write_metrics_json(run_dir / "metrics.json", payload)
                append_summary_csv(root_dir / "summary.csv", build_summary_row(meta, best_row))

                metrics = flatten_metrics(payload.get("metrics", {}))
                if metrics:
                    tracker.log_metrics(metrics, step=seed_split)
                tracker.log_artifact(metrics_path)
                tracker.log_artifacts(run_dir, artifact_path=f"results/seed-split-{seed_split}")

        summary_path = root_dir / "summary.csv"
        if summary_path.exists():
            tracker.log_artifact(summary_path)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
