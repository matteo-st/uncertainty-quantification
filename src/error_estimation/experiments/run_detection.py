from __future__ import annotations

import argparse
import json
import os
import re
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

from error_estimation.evaluators import HyperparamsSearch
from error_estimation.utils.config import Config
from error_estimation.utils.datasets import get_dataset
from error_estimation.utils.datasets.dataloader import prepare_ablation_dataloaders
from error_estimation.utils.eval import AblationDetector
from error_estimation.utils.experiment import (
    build_latent_paths,
    copy_configs,
    ensure_dir,
    set_num_threads,
    write_run_metadata,
)
from error_estimation.utils.helper import make_grid, metric_direction, setup_seeds
from error_estimation.utils.logging import setup_logging
from error_estimation.utils.models import get_model
from error_estimation.utils.paths import CHECKPOINTS_DIR, DATA_DIR, LATENTS_DIR, RESULTS_DIR
from error_estimation.utils.postprocessors import get_postprocessor
from error_estimation.utils.results_io import (
    append_summary_csv,
    build_metrics_payload,
    build_run_meta,
    build_summary_row,
    build_results_root,
    default_run_tag,
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

_GRID_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _coerce_grid_cell(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _coerce_grid_cell(value[0])
        return value
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inner = stripped[1:-1].strip()
            if inner:
                if "np.float" in inner:
                    inner = re.sub(r"np\\.float\\d+\\(", "(", inner)
                numbers = _GRID_NUM_RE.findall(inner)
                if len(numbers) == 1:
                    try:
                        return float(numbers[0])
                    except ValueError:
                        return value
        if "np.float64" in stripped:
            cleaned = re.sub(r"np\\.float\\d+\\(", "(", stripped)
            numbers = _GRID_NUM_RE.findall(cleaned)
            if len(numbers) == 1:
                try:
                    return float(numbers[0])
                except ValueError:
                    return value
    return value


def _normalize_grid_results(df: pd.DataFrame) -> pd.DataFrame:
    if any(col.endswith("_x") for col in df.columns):
        drop_cols = set()
        rename_map = {}
        for col in df.columns:
            if col.endswith("_x"):
                base = col[:-2]
                rename_map[col] = base
                if f"{base}_y" in df.columns:
                    drop_cols.add(f"{base}_y")
        df = df.drop(columns=list(drop_cols), errors="ignore").rename(columns=rename_map)
    return df.apply(lambda col: col.map(_coerce_grid_cell))


def _dedupe_partition_grid(detection_cfg: Config, grid: list[dict]) -> list[dict]:
    if detection_cfg.get("name") != "partition":
        return grid
    grid_spec = detection_cfg.get("postprocessor_grid", {})
    if "score" not in grid_spec or "alpha" not in grid_spec:
        return grid
    alpha_values = grid_spec.get("alpha", [])
    if not alpha_values:
        return grid
    default_alpha = alpha_values[0]
    seen = set()
    pruned = []
    for cfg in grid:
        if cfg.get("score") == "mean":
            key = tuple((k, repr(cfg.get(k))) for k in sorted(cfg.keys()) if k != "alpha")
            if key in seen:
                continue
            seen.add(key)
            cfg = cfg.copy()
            cfg["alpha"] = default_alpha
        pruned.append(cfg)
    return pruned


def _resolve_indices(dataset) -> list[int] | None:
    indices = getattr(dataset, "indices", None)
    if indices is None:
        return None
    indices = list(indices)
    base_dataset = getattr(dataset, "dataset", None)
    if base_dataset is None:
        return indices
    base_indices = _resolve_indices(base_dataset)
    if base_indices is None:
        return indices
    return [base_indices[i] for i in indices]


def _unwrap_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _load_latent_values(
    *,
    dataloader,
    latent_path: str,
    model: torch.nn.Module,
    device: torch.device,
    n_epochs: int | None,
) -> dict[str, torch.Tensor] | None:
    if dataloader is None:
        return None
    split_indices = _resolve_indices(dataloader.dataset)
    full_dataset = _unwrap_dataset(dataloader.dataset)
    full_len = len(full_dataset)
    n_epochs = 1 if n_epochs is None else n_epochs

    cached = None
    if os.path.exists(latent_path):
        pkg = torch.load(latent_path, map_location="cpu")
        all_logits = pkg["logits"].to(torch.float32)
        all_labels = pkg["labels"].to(torch.int64)
        all_model_preds = pkg["model_preds"].to(torch.int64)
        expected_len = full_len * n_epochs
        if all_logits.size(0) == expected_len:
            cached = (all_logits, all_labels, all_model_preds)

    if cached is None:
        if hasattr(dataloader.dataset, "dataset"):
            full_loader = torch.utils.data.DataLoader(
                full_dataset,
                batch_size=dataloader.batch_size,
                shuffle=False,
                pin_memory=getattr(dataloader, "pin_memory", False),
                num_workers=getattr(dataloader, "num_workers", 0),
            )
        else:
            full_loader = dataloader

        all_logits = []
        all_labels = []
        all_model_preds = []
        model.eval()
        with torch.no_grad():
            for inputs, targets in full_loader:
                inputs = inputs.to(device)
                logits = model(inputs).cpu()
                model_preds = torch.argmax(logits, dim=1)
                all_logits.append(logits)
                all_labels.append(targets.cpu())
                all_model_preds.append(model_preds)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_model_preds = torch.cat(all_model_preds, dim=0)

        parent = os.path.dirname(latent_path)
        os.makedirs(parent, exist_ok=True)
        tmp = latent_path + ".tmp"
        torch.save(
            {
                "logits": all_logits.cpu(),
                "labels": all_labels.cpu(),
                "model_preds": all_model_preds.cpu(),
                "n_samples": full_len,
                "n_epochs": n_epochs,
            },
            tmp,
        )
        os.replace(tmp, latent_path)
    else:
        all_logits, all_labels, all_model_preds = cached

    if split_indices is not None:
        if len(split_indices) == 0:
            all_logits = all_logits[:0]
            all_labels = all_labels[:0]
            all_model_preds = all_model_preds[:0]
        else:
            if n_epochs > 1:
                expanded = []
                for epoch in range(n_epochs):
                    offset = epoch * full_len
                    expanded.extend([offset + idx for idx in split_indices])
                split_indices = expanded
            max_idx = max(split_indices)
            if max_idx >= all_logits.size(0):
                raise ValueError(
                    f"Latent cache {latent_path} does not cover split indices "
                    f"(max {max_idx} >= {all_logits.size(0)})."
                )
            all_logits = all_logits[split_indices]
            all_labels = all_labels[split_indices]
            all_model_preds = all_model_preds[split_indices]

    detector_labels = (all_model_preds != all_labels).float()
    return {
        "logits": all_logits.to(device),
        "detector_labels": detector_labels.to(device),
    }


def _evaluate_grid(
    *,
    run_dir: Path,
    data_cfg: Config,
    detection_cfg: Config,
    model: torch.nn.Module,
    res_loader,
    cal_loader,
    test_loader,
    device: torch.device,
    latent_paths: dict[str, str],
) -> None:
    if "postprocessor_grid" not in detection_cfg:
        raise KeyError("Missing detection.postprocessor_grid for --eval-grid")
    grid = list(make_grid(detection_cfg, key="postprocessor_grid"))
    grid = _dedupe_partition_grid(detection_cfg, grid)
    if not grid:
        raise ValueError("Empty postprocessor_grid")

    detectors = [
        get_postprocessor(
            postprocessor_name=detection_cfg["name"],
            model=model,
            cfg=cfg,
            result_folder=str(run_dir),
            device=device,
        )
        for cfg in grid
    ]

    grid_keys = list(detection_cfg["postprocessor_grid"].keys())

    if detection_cfg.get("name") == "partition":
        exp_args = detection_cfg.get("experience_args", {})
        n_epochs = exp_args.get("n_epochs", {})
        res_values = None
        if data_cfg.get("n_samples", {}).get("res", 0) > 0 and res_loader is not None:
            res_values = _load_latent_values(
                dataloader=res_loader,
                latent_path=latent_paths["res"],
                model=model,
                device=device,
                n_epochs=n_epochs.get("res", 1),
            )
        cal_values = _load_latent_values(
            dataloader=cal_loader,
            latent_path=latent_paths["cal"],
            model=model,
            device=device,
            n_epochs=n_epochs.get("cal", 1),
        )
        fit_values = res_values if res_values is not None else cal_values
        for detector in detectors:
            detector.fit(
                logits=fit_values["logits"],
                detector_labels=fit_values["detector_labels"],
                fit_clustering=True,
            )
            detector.fit(
                logits=cal_values["logits"],
                detector_labels=cal_values["detector_labels"],
                fit_clustering=False,
            )

    cal_eval = AblationDetector(
        model=model,
        dataloader=cal_loader,
        device=device,
        suffix="cal",
        latent_path=latent_paths["cal"],
        postprocessor_name=detection_cfg["name"],
        cfg_dataset=data_cfg,
        result_folder=str(run_dir),
    )
    cal_results = pd.concat(cal_eval.evaluate(grid, detectors=detectors, suffix="cal"), axis=0)

    test_eval = AblationDetector(
        model=model,
        dataloader=test_loader,
        device=device,
        suffix="test",
        latent_path=latent_paths["test"],
        postprocessor_name=detection_cfg["name"],
        cfg_dataset=data_cfg,
        result_folder=str(run_dir),
    )
    test_results = pd.concat(test_eval.evaluate(grid, detectors=detectors, suffix="test"), axis=0)

    if grid_keys:
        grid_results = pd.merge(cal_results, test_results, on=grid_keys, how="outer")
    else:
        # Raw-score / empty grids: merge by row order, keep config from cal_results.
        drop_cols = [col for col in test_results.columns if col in cal_results.columns]
        test_only = test_results.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
        grid_results = pd.concat([cal_results.reset_index(drop=True), test_only], axis=1)
    grid_results = _normalize_grid_results(grid_results)
    grid_results.to_csv(run_dir / "grid_results.csv", index=False)


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
    parser.add_argument(
        "--save-search-results",
        action="store_true",
        help="Persist full search results (search.jsonl) per seed split",
    )
    parser.add_argument(
        "--eval-grid",
        action="store_true",
        help="Evaluate the full postprocessor_grid on cal/test and save grid_results.csv",
    )
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

    results_root = build_results_root(args.root_dir, data_cfg, model_cfg, detection_cfg)
    run_tag = args.run_tag or default_run_tag()
    run_root = results_root / "runs" / run_tag
    ensure_dir(run_root)

    logger = setup_logging(run_root, level=args.log_level)
    logger.info("Starting run in %s", run_root)

    config_paths = {
        "dataset": args.config_dataset,
        "model": args.config_model,
        "detection": args.config_detection,
    }
    copied_configs = copy_configs(run_root, config_paths)
    metadata_path = write_run_metadata(run_root, args, copied_configs)

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
                "run_tag": run_tag,
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
            run_dir = run_root / f"seed-split-{seed_split}"
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
                if args.eval_grid:
                    _evaluate_grid(
                        run_dir=run_dir,
                        data_cfg=data_cfg,
                        detection_cfg=cfg_detection_run,
                        model=model,
                        res_loader=res_loader,
                        cal_loader=cal_loader,
                        test_loader=test_loader,
                        device=device,
                        latent_paths=latent_paths,
                    )
                    grid_path = run_dir / "grid_results.csv"
                    if not grid_path.exists():
                        raise FileNotFoundError(f"Missing grid results at {grid_path}")
                    results = pd.read_csv(grid_path)
                    best_row = select_best_row(results, args.metric, metric_direction(args.metric))
                    meta = build_run_meta(
                        data_cfg,
                        model_cfg,
                        cfg_detection_run,
                        seed_split,
                        mode="grid",
                        run_tag=run_tag,
                        extra={"result_dir": str(run_dir), "grid_results": str(grid_path)},
                    )
                    payload = build_metrics_payload(meta, best_row)
                    metrics_path = write_metrics_json(run_dir / "metrics.json", payload)
                    append_summary_csv(results_root / "summary.csv", build_summary_row(meta, best_row))

                    metrics = flatten_metrics(payload.get("metrics", {}))
                    if metrics:
                        tracker.log_metrics(metrics, step=seed_split)
                    tracker.log_artifact(metrics_path)
                    tracker.log_artifact(grid_path)
                    tracker.log_artifacts(run_dir, artifact_path=f"results/{run_tag}/seed-split-{seed_split}")
                    continue
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
                    save_search_results=args.save_search_results,
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
                    run_tag=run_tag,
                    extra={"result_dir": str(run_dir)},
                )
                payload = build_metrics_payload(meta, best_row)
                metrics_path = write_metrics_json(run_dir / "metrics.json", payload)
                append_summary_csv(results_root / "summary.csv", build_summary_row(meta, best_row))

                metrics = flatten_metrics(payload.get("metrics", {}))
                if metrics:
                    tracker.log_metrics(metrics, step=seed_split)
                tracker.log_artifact(metrics_path)
                tracker.log_artifacts(run_dir, artifact_path=f"results/{run_tag}/seed-split-{seed_split}")

        summary_path = results_root / "summary.csv"
        if summary_path.exists():
            tracker.log_artifact(summary_path)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
