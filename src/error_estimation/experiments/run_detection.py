from __future__ import annotations

import argparse
import json
import os
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
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
from error_estimation.utils.helper import make_grid, metric_direction, select_best_index, setup_seeds
from error_estimation.utils.logging import setup_logging
from error_estimation.utils.models import get_model
from error_estimation.utils.metrics import compute_all_metrics
from error_estimation.utils.postprocessors.doctor_postprocessor import gini as doctor_gini
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

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

_GRID_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _coerce_grid_cell(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _coerce_grid_cell(value[0])
        return [_coerce_grid_cell(item) for item in value]
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
                    inner = re.sub(r"np\\.float\\d+\\(", "", inner).replace(")", "")
                numbers = _GRID_NUM_RE.findall(inner)
                if numbers:
                    try:
                        return [float(num) for num in numbers]
                    except ValueError:
                        return value
        if "np.float64" in stripped:
            cleaned = re.sub(r"np\\.float\\d+\\(", "", stripped).replace(")", "")
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


def _resolve_raw_score_run_tag(
    *,
    runs_root: Path,
    seed_split: int,
    selection_cfg: dict,
    n_res: int | None,
    source_name: str,
) -> str:
    run_tag = selection_cfg.get("run_tag")
    if run_tag:
        return run_tag
    prefix = selection_cfg.get("run_tag_prefix")
    if prefix:
        if n_res is None:
            raise ValueError("raw_score_selection.run_tag_prefix requires n_res.")
        prefix = prefix.format(n_res=n_res)
    else:
        if n_res is None:
            raise ValueError("raw_score_selection requires n_res or run_tag.")
        prefix = f"{source_name}-res-grid-nres{n_res}-"
    candidates = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        if not run_dir.name.startswith(prefix):
            continue
        search_path = run_dir / f"seed-split-{seed_split}" / "search.jsonl"
        if search_path.exists():
            candidates.append((run_dir, search_path))
    if not candidates:
        raise FileNotFoundError(
            f"No search.jsonl found under {runs_root} for prefix '{prefix}'."
        )
    run_dir = max(candidates, key=lambda item: item[0].stat().st_mtime)[0]
    return run_dir.name


def _apply_raw_score_selection(
    *,
    cfg_detection: Config,
    data_cfg: Config,
    model_cfg: Config,
    seed_split: int,
    root_dir: str | Path,
) -> dict | None:
    selection_cfg = cfg_detection.get("experience_args", {}).get("raw_score_selection")
    if not selection_cfg:
        return None
    source_name = selection_cfg.get("postprocessor", "doctor")
    n_res = data_cfg.get("n_samples", {}).get("res")
    source_root = selection_cfg.get("root_dir", root_dir)
    source_results_root = build_results_root(source_root, data_cfg, model_cfg, {"name": source_name})
    runs_root = source_results_root / "runs"
    run_tag = _resolve_raw_score_run_tag(
        runs_root=runs_root,
        seed_split=seed_split,
        selection_cfg=selection_cfg,
        n_res=n_res,
        source_name=source_name,
    )
    search_path = runs_root / run_tag / f"seed-split-{seed_split}" / "search.jsonl"
    if not search_path.exists():
        raise FileNotFoundError(f"Missing raw score search results at {search_path}")
    results = pd.read_json(search_path, lines=True)
    metric = selection_cfg.get("metric", "fpr")
    split = selection_cfg.get("split", "res")
    metric_key = f"{metric}_{split}"
    if metric_key not in results.columns:
        raise KeyError(f"Missing metric '{metric_key}' in {search_path}")
    direction = metric_direction(metric)
    values = pd.to_numeric(results[metric_key], errors="coerce")
    if not values.notna().any():
        raise ValueError(f"No valid values for '{metric_key}' in {search_path}")
    idx = values.idxmin() if direction == "min" else values.idxmax()
    best_row = results.loc[idx].to_dict()
    for key in ("temperature", "magnitude", "normalize"):
        if key in best_row:
            cfg_detection["postprocessor_args"][key] = best_row[key]
            grid = cfg_detection.get("postprocessor_grid")
            if isinstance(grid, dict) and key in grid:
                grid[key] = [best_row[key]]
    return {
        "raw_score_source": source_name,
        "raw_score_run_tag": run_tag,
        "raw_score_metric": metric,
        "raw_score_split": split,
        "raw_score_search_path": str(search_path),
    }


def _apply_lda_score_selections(
    *,
    cfg_detection: Config,
    data_cfg: Config,
    model_cfg: Config,
    seed_split: int,
    root_dir: str | Path,
) -> dict | None:
    """
    Load best hyperparameters for each score type used in LDA binning.

    Similar to _apply_raw_score_selection, but loads configs for multiple scores.
    Each score can have its own source postprocessor and selection criteria.
    If a score's grid results are not found, it will be skipped (using defaults).

    Config format:
        experience_args:
          score_selections:
            gini:
              postprocessor: doctor
              metric: fpr
              split: res
              run_tag_prefix: doctor-res-grid-nres{n_res}-
            margin:
              postprocessor: margin
              metric: fpr
              split: res
              run_tag_prefix: margin-res-grid-nres{n_res}-
    """
    import logging
    logger = logging.getLogger(__name__)

    score_selections_cfg = cfg_detection.get("experience_args", {}).get("score_selections")
    if not score_selections_cfg:
        return None

    n_res = data_cfg.get("n_samples", {}).get("res")
    score_configs = {}
    meta = {}

    for score_name, selection_cfg in score_selections_cfg.items():
        try:
            source_name = selection_cfg.get("postprocessor", "doctor")
            source_root = selection_cfg.get("root_dir", root_dir)
            source_results_root = build_results_root(source_root, data_cfg, model_cfg, {"name": source_name})
            runs_root = source_results_root / "runs"

            run_tag = _resolve_raw_score_run_tag(
                runs_root=runs_root,
                seed_split=seed_split,
                selection_cfg=selection_cfg,
                n_res=n_res,
                source_name=source_name,
            )

            search_path = runs_root / run_tag / f"seed-split-{seed_split}" / "search.jsonl"
            grid_path = runs_root / run_tag / f"seed-split-{seed_split}" / "grid_results.csv"

            if search_path.exists():
                results = pd.read_json(search_path, lines=True)
                source_file = search_path
            elif grid_path.exists():
                results = pd.read_csv(grid_path)
                source_file = grid_path
            else:
                logger.warning(
                    f"Score selection results for '{score_name}' not found at {search_path} "
                    f"or {grid_path}. Using default hyperparameters for this score."
                )
                meta[f"score_selection_{score_name}_status"] = "not_found"
                continue
            metric = selection_cfg.get("metric", "fpr")
            split = selection_cfg.get("split", "res")
            metric_key = f"{metric}_{split}"

            if metric_key not in results.columns:
                logger.warning(
                    f"Metric '{metric_key}' not found in {source_file}. "
                    f"Using default hyperparameters for '{score_name}'."
                )
                meta[f"score_selection_{score_name}_status"] = "metric_not_found"
                continue

            direction = metric_direction(metric)
            values = pd.to_numeric(results[metric_key], errors="coerce")
            if not values.notna().any():
                logger.warning(
                    f"No valid values for '{metric_key}' in {source_file}. "
                    f"Using default hyperparameters for '{score_name}'."
                )
                meta[f"score_selection_{score_name}_status"] = "no_valid_values"
                continue

            idx = values.idxmin() if direction == "min" else values.idxmax()
            best_row = results.loc[idx].to_dict()

            # Extract hyperparams for this score
            score_config = {}
            for key in ("temperature", "magnitude", "normalize"):
                if key in best_row:
                    score_config[key] = best_row[key]

            score_configs[score_name] = score_config
            meta[f"score_selection_{score_name}_source"] = source_name
            meta[f"score_selection_{score_name}_run_tag"] = run_tag
            meta[f"score_selection_{score_name}_source_file"] = str(source_file)
            meta[f"score_selection_{score_name}_config"] = score_config
            logger.info(f"Loaded hyperparameters for '{score_name}': {score_config}")

        except Exception as e:
            logger.warning(
                f"Error loading score selection for '{score_name}': {e}. "
                f"Using default hyperparameters."
            )
            meta[f"score_selection_{score_name}_status"] = f"error: {e}"

    # Store score_configs in the detection config
    if score_configs:
        if "postprocessor_args" not in cfg_detection:
            cfg_detection["postprocessor_args"] = {}
        cfg_detection["postprocessor_args"]["score_configs"] = score_configs

    return meta if meta else None


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
    perturbation: dict | None = None,
    use_cache: bool = True,
    logits_dtype: str = "float32",
) -> dict[str, torch.Tensor] | None:
    if dataloader is None:
        return None
    torch_dtype = DTYPE_MAP.get(logits_dtype, torch.float32)
    magnitude = None
    temperature = 1.0
    normalize = False
    if perturbation:
        magnitude = float(perturbation.get("magnitude", 0.0))
        temperature = float(perturbation.get("temperature", 1.0))
        normalize = bool(perturbation.get("normalize", False))
    split_indices = _resolve_indices(dataloader.dataset)
    full_dataset = _unwrap_dataset(dataloader.dataset)
    full_len = len(full_dataset)
    n_epochs = 1 if n_epochs is None else n_epochs

    cached = None
    if use_cache and os.path.exists(latent_path) and magnitude is None:
        pkg = torch.load(latent_path, map_location="cpu")
        all_logits = pkg["logits"].to(torch_dtype)
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
        for inputs, targets in full_loader:
            inputs = inputs.to(device)
            if magnitude is not None and magnitude > 0:
                inputs = inputs.detach().requires_grad_(True)
                logits_clean = model(inputs)
                model_preds = torch.argmax(logits_clean, dim=1)
                scores = doctor_gini(logits_clean, temperature=temperature, normalize=normalize)
                scores_for_loss = scores
                if torch.any(scores_for_loss <= 0):
                    scores_for_loss = scores_for_loss.abs()
                loss = torch.log(scores_for_loss + 1e-12).sum()
                grad_inputs, = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)
                with torch.no_grad():
                    adv = inputs + magnitude * grad_inputs.sign()
                    logits = model(adv).cpu()
            else:
                with torch.no_grad():
                    logits = model(inputs).cpu()
                model_preds = torch.argmax(logits, dim=1)
            all_logits.append(logits)
            all_labels.append(targets.cpu())
            all_model_preds.append(model_preds.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_model_preds = torch.cat(all_model_preds, dim=0)

        if use_cache and magnitude is None:
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
    select_init_metric: str | None,
    logits_dtype: str = "float32",
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

    res_results = None
    if detection_cfg.get("name") == "partition":
        exp_args = detection_cfg.get("experience_args", {})
        n_epochs = exp_args.get("n_epochs", {})
        fit_partition_on_cal = exp_args.get("fit_partition_on_cal", False)
        use_perturbed_logits = exp_args.get("use_perturbed_logits", False)
        perturbation_cfg = None
        if use_perturbed_logits:
            perturbation_cfg = {
                "magnitude": detection_cfg.get("postprocessor_args", {}).get("magnitude", 0.0),
                "temperature": detection_cfg.get("postprocessor_args", {}).get("temperature", 1.0),
                "normalize": detection_cfg.get("postprocessor_args", {}).get("normalize", False),
            }
        res_values = None
        if data_cfg.get("n_samples", {}).get("res", 0) > 0 and res_loader is not None:
            res_values = _load_latent_values(
                dataloader=res_loader,
                latent_path=latent_paths["res"],
                model=model,
                device=device,
                n_epochs=n_epochs.get("res", 1),
                perturbation=perturbation_cfg,
                use_cache=not use_perturbed_logits,
                logits_dtype=logits_dtype,
            )
        cal_values = _load_latent_values(
            dataloader=cal_loader,
            latent_path=latent_paths["cal"],
            model=model,
            device=device,
            n_epochs=n_epochs.get("cal", 1),
            perturbation=perturbation_cfg,
            use_cache=not use_perturbed_logits,
            logits_dtype=logits_dtype,
        )
        if fit_partition_on_cal or res_values is None:
            fit_values = cal_values
        else:
            fit_values = res_values
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
        if res_values is not None and res_loader is not None:
            select_direction = metric_direction(select_init_metric) if select_init_metric else None
            res_labels = res_values["detector_labels"].detach().cpu().numpy()
            res_rows: list[dict[str, object]] = []
            for cfg, detector in zip(grid, detectors):
                with torch.no_grad():
                    scores = detector(logits=res_values["logits"]).detach().cpu().numpy()
                metrics = compute_all_metrics(conf=scores, detector_labels=res_labels)
                row = dict(cfg)
                for key, val in metrics.items():
                    row[f"{key}_res"] = val
                clustering_algo = getattr(detector, "clustering_algo", None)
                if clustering_algo is not None and hasattr(clustering_algo, "results"):
                    results = clustering_algo.results
                    if hasattr(results, "lower_bound"):
                        row["lower_bound_res"] = results.lower_bound.detach().cpu().numpy().tolist()
                    if hasattr(results, "inertia"):
                        row["inertia_res"] = results.inertia.detach().cpu().numpy().tolist()
                    elif hasattr(results, "means") and hasattr(results, "log_resp"):
                        try:
                            embs = detector._extract_embeddings(
                                logits=res_values["logits"]
                            ).detach()
                            if embs.dim() > 2:
                                embs = embs.squeeze(0)
                            centers = results.means.to(embs.device)
                            labels = torch.argmax(results.log_resp.to(embs.device), dim=-1)
                            inertia = clustering_algo._calculate_inertia(
                                embs,
                                centers.unsqueeze(0),
                                labels.unsqueeze(0),
                            )
                            row["inertia_res"] = inertia.squeeze(0).detach().cpu().numpy().tolist()
                        except Exception:
                            pass
                if exp_args.get("select_init_on_res") and select_init_metric and select_direction:
                    metric_values = metrics.get(select_init_metric)
                    if metric_values is not None:
                        values = np.asarray(metric_values, dtype=float)
                        if values.ndim == 0:
                            values = values.reshape(1)
                        if values.size >= 1:
                            best_idx = 0 if values.size == 1 else select_best_index(values, select_direction)
                            row["init_res"] = int(best_idx)
                            if clustering_algo is not None:
                                clustering_algo.best_init = int(best_idx)
                res_rows.append(row)
            res_results = pd.DataFrame(res_rows)

    # Evaluate res split for non-partition postprocessors (e.g., margin, odin/msp)
    if detection_cfg.get("name") != "partition":
        exp_args = detection_cfg.get("experience_args", {})
        n_epochs = exp_args.get("n_epochs", {})
        if n_epochs.get("res") is not None and data_cfg.get("n_samples", {}).get("res", 0) > 0 and res_loader is not None:
            res_eval = AblationDetector(
                model=model,
                dataloader=res_loader,
                device=device,
                suffix="res",
                latent_path=latent_paths["res"],
                postprocessor_name=detection_cfg["name"],
                cfg_dataset=data_cfg,
                result_folder=str(run_dir),
                logits_dtype=logits_dtype,
            )
            res_results = pd.concat(res_eval.evaluate(grid, detectors=detectors, suffix="res"), axis=0)

    if detection_cfg.get("name") == "partition" and use_perturbed_logits:
        test_values = _load_latent_values(
            dataloader=test_loader,
            latent_path=latent_paths["test"],
            model=model,
            device=device,
            n_epochs=n_epochs.get("test", 1),
            perturbation=perturbation_cfg,
            use_cache=False,
            logits_dtype=logits_dtype,
        )
        cal_rows = []
        test_rows = []
        for cfg, detector in zip(grid, detectors):
            cal_scores = detector(logits=cal_values["logits"]).detach().cpu().numpy()
            test_scores = detector(logits=test_values["logits"]).detach().cpu().numpy()
            cal_metrics = compute_all_metrics(
                conf=cal_scores, detector_labels=cal_values["detector_labels"].cpu().numpy()
            )
            test_metrics = compute_all_metrics(
                conf=test_scores, detector_labels=test_values["detector_labels"].cpu().numpy()
            )
            cal_metrics = {f"{key}_cal": val for key, val in cal_metrics.items()}
            test_metrics = {f"{key}_test": val for key, val in test_metrics.items()}
            cal_rows.append(pd.concat([pd.DataFrame([cfg]), pd.DataFrame([cal_metrics])], axis=1))
            test_rows.append(pd.concat([pd.DataFrame([cfg]), pd.DataFrame([test_metrics])], axis=1))
        cal_results = pd.concat(cal_rows, axis=0)
        test_results = pd.concat(test_rows, axis=0)
    else:
        cal_eval = AblationDetector(
            model=model,
            dataloader=cal_loader,
            device=device,
            suffix="cal",
            latent_path=latent_paths["cal"],
            postprocessor_name=detection_cfg["name"],
            cfg_dataset=data_cfg,
            result_folder=str(run_dir),
            logits_dtype=logits_dtype,
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
            logits_dtype=logits_dtype,
        )
        test_results = pd.concat(test_eval.evaluate(grid, detectors=detectors, suffix="test"), axis=0)

    if grid_keys:
        grid_results = pd.merge(cal_results, test_results, on=grid_keys, how="outer")
    else:
        # Raw-score / empty grids: merge by row order, keep config from cal_results.
        drop_cols = [col for col in test_results.columns if col in cal_results.columns]
        test_only = test_results.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
        grid_results = pd.concat([cal_results.reset_index(drop=True), test_only], axis=1)
    if res_results is not None:
        if grid_keys:
            grid_results = pd.merge(res_results, grid_results, on=grid_keys, how="outer")
        else:
            grid_results = pd.concat(
                [res_results.reset_index(drop=True), grid_results.reset_index(drop=True)], axis=1
            )
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
    parser.add_argument(
        "--logits-dtype",
        "--logits_dtype",
        dest="logits_dtype",
        choices=["float16", "float32", "float64"],
        default="float32",
        help="Precision for logits storage and computation (default: float32)",
    )
    parser.add_argument(
        "--results-family",
        "--results_family",
        dest="results_family",
        type=str,
        default=None,
        help="Override results family folder (e.g., 'score_combination')",
    )
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

    results_root = build_results_root(args.root_dir, data_cfg, model_cfg, detection_cfg, family=args.results_family)
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

            latent_paths = build_latent_paths(args.latent_dir, data_cfg, model_cfg, detection_cfg, logits_dtype=args.logits_dtype)
            run_dir = run_root / f"seed-split-{seed_split}"
            ensure_dir(run_dir)

            cfg_detection_run = deepcopy(detection_cfg)
            if args.mode == "search_res":
                cfg_detection_run["experience_args"]["ratio_res_split"] = 1.0
            raw_score_meta = _apply_raw_score_selection(
                cfg_detection=cfg_detection_run,
                data_cfg=data_cfg,
                model_cfg=model_cfg,
                seed_split=seed_split,
                root_dir=args.root_dir,
            )

            # For lda_binning, raw_lda, and partition with combined space,
            # load per-score hyperparameters from previous runs
            lda_score_meta = None
            postprocessor_name = cfg_detection_run.get("name")
            space = cfg_detection_run.get("postprocessor_args", {}).get("space")
            if postprocessor_name in ("lda_binning", "raw_lda") or (postprocessor_name == "partition" and space == "combined"):
                lda_score_meta = _apply_lda_score_selections(
                    cfg_detection=cfg_detection_run,
                    data_cfg=data_cfg,
                    model_cfg=model_cfg,
                    seed_split=seed_split,
                    root_dir=args.root_dir,
                )

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
                        select_init_metric=args.metric,
                        logits_dtype=args.logits_dtype,
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
                        extra={
                            "result_dir": str(run_dir),
                            "grid_results": str(grid_path),
                            **(raw_score_meta or {}),
                        },
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
                    logits_dtype=args.logits_dtype,
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
                    extra={
                        "result_dir": str(run_dir),
                        **(raw_score_meta or {}),
                        **(lda_score_meta or {}),
                    },
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
