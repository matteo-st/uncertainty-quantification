#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from error_estimation.utils.config import Config
from error_estimation.utils.datasets import get_dataset
from error_estimation.utils.datasets.dataloader import prepare_ablation_dataloaders
from error_estimation.utils.helper import setup_seeds
from error_estimation.utils.models import get_model
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
    result_dir = (
        Path(root_dir)
        / data_cfg["name"]
        / f"{model_cfg['model_name']}_{model_cfg['preprocessor']}"
        / postprocessor
        / "runs"
        / run_tag
        / f"seed-split-{seed_split}"
    )
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
    dtype: torch.dtype,
) -> torch.Tensor:
    logits_all = []
    model.eval()
    for inputs, _ in loader:
        inputs = inputs.to(device=device, dtype=dtype)
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
        logits_all.append(logits.detach())
    return torch.cat(logits_all, dim=0)


def _unique_stats(values: torch.Tensor) -> tuple[int, int]:
    vals, counts = torch.unique(values, sorted=True, return_counts=True)
    max_count = int(counts.max().item()) if counts.numel() else 0
    return int(vals.numel()), max_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze unique scores before/after score transform across float precisions."
    )
    parser.add_argument("--config-dataset", required=True)
    parser.add_argument("--config-model", required=True)
    parser.add_argument("--config-detection", required=True)
    parser.add_argument("--seed-split", type=int, default=9)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtypes", nargs="+", default=["float16", "float32", "float64"])
    parser.add_argument("--save-logits", action="store_true")
    parser.add_argument("--latent-dir", default=None)
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

    dataset = get_dataset(
        dataset_name=data_cfg["name"],
        model_name=model_cfg["model_name"],
        root=os.environ.get("DATA_DIR", "./data"),
        preprocess=model_cfg["preprocessor"],
        shuffle=False,
    )

    res_loader, _, _ = prepare_ablation_dataloaders(
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

    temp = float(detection_cfg["postprocessor_args"]["temperature"])
    mag = float(detection_cfg["postprocessor_args"].get("magnitude", 0.0))
    normalize = bool(detection_cfg["postprocessor_args"].get("normalize", False))
    use_perturbed_logits = bool(detection_cfg.get("experience_args", {}).get("use_perturbed_logits", False))

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    rows = []
    for name in args.dtypes:
        dtype = dtype_map.get(name)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {name}")
        if dtype is torch.float16 and device.type != "cuda":
            raise RuntimeError("float16 analysis requires CUDA; no GPU detected.")

        model = get_model(
            model_name=model_cfg["model_name"],
            dataset_name=data_cfg["name"],
            n_classes=data_cfg["num_classes"],
            model_seed=model_cfg["seed"],
            checkpoint_dir=os.path.join(
                os.environ.get("CHECKPOINTS_DIR", "checkpoints/"),
                model_cfg["preprocessor"],
            ),
        ).to(device=device, dtype=dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        res_logits = _collect_logits(
            model=model,
            loader=res_loader,
            device=device,
            magnitude=mag if use_perturbed_logits else 0.0,
            temperature=temp,
            normalize=normalize,
            dtype=dtype,
        )

        detector = PartitionPostprocessor(
            model=model,
            cfg=detection_cfg["postprocessor_args"],
            result_folder=args.output_dir,
            device=device,
        )
        raw_scores = detector._extract_embeddings(logits=res_logits)
        if raw_scores.dim() > 1 and raw_scores.size(1) == 1:
            raw_scores = raw_scores.squeeze(1)
        if raw_scores.dim() != 1:
            raise ValueError("Expected 1D raw scores for precision analysis.")

        u_vals = detector._apply_score_transform(raw_scores, fit=True)

        raw_scores_cpu = raw_scores.detach().cpu()
        u_vals_cpu = u_vals.detach().cpu()

        raw_unique, raw_max = _unique_stats(raw_scores_cpu)
        u_unique, u_max = _unique_stats(u_vals_cpu)
        rows.append(
            {
                "dtype": name,
                "raw_unique": raw_unique,
                "raw_max_count": raw_max,
                "u_unique": u_unique,
                "u_max_count": u_max,
            }
        )

        if args.save_logits:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            logits_path = output_dir / f"logits_res_{name}.pt"
            torch.save(
                {
                    "logits": res_logits.detach().cpu(),
                    "dtype": name,
                    "temperature": temp,
                    "magnitude": mag,
                    "normalize": normalize,
                },
                logits_path,
            )
        if args.latent_dir:
            latent_dir = Path(args.latent_dir)
            latent_dir.mkdir(parents=True, exist_ok=True)
            latent_path = latent_dir / f"latent_res_{name}.pt"
            torch.save(
                {
                    "logits": res_logits.detach().cpu(),
                    "dtype": name,
                    "temperature": temp,
                    "magnitude": mag,
                    "normalize": normalize,
                },
                latent_path,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = output_dir / "score_transform_precision_summary.csv"
    df.to_csv(out_path, index=False)

    stats_path = output_dir / "score_transform_precision_meta.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_res": int(data_cfg["n_samples"]["res"]),
                "temperature": temp,
                "magnitude": mag,
                "normalize": normalize,
                "score_transform": detection_cfg["postprocessor_args"].get("score_transform"),
                "dtypes": args.dtypes,
                "use_perturbed_logits": use_perturbed_logits,
                "save_logits": bool(args.save_logits),
                "latent_dir": args.latent_dir,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
