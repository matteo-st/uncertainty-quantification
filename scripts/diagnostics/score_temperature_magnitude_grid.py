from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from error_estimation.utils.config import Config
from error_estimation.utils.datasets import get_dataset
from error_estimation.utils.datasets.dataloader import prepare_ablation_dataloaders
from error_estimation.utils.eval import AblationDetector
from error_estimation.utils.experiment import build_latent_paths
from error_estimation.utils.helper import make_grid, setup_seeds
from error_estimation.utils.models import get_model
from error_estimation.utils.paths import CHECKPOINTS_DIR, DATA_DIR, LATENTS_DIR
from error_estimation.utils.postprocessors import get_postprocessor


def _parse_float_values(values: list[str] | None) -> list[float] | None:
    if values is None:
        return None
    if len(values) == 1 and "," in values[0]:
        values = values[0].split(",")
    parsed = []
    for item in values:
        item = item.strip()
        if not item:
            continue
        parsed.append(float(item))
    return parsed or None


def _match_value(val: float, targets: list[float] | None) -> bool:
    if targets is None:
        return True
    return any(np.isclose(val, target) for target in targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate temperature×magnitude grid on the test split."
    )
    parser.add_argument(
        "--config-dataset",
        "--config_dataset",
        dest="config_dataset",
        default="configs/datasets/cifar10/cifar10_n-cal-5000.yml",
    )
    parser.add_argument(
        "--config-model",
        "--config_model",
        dest="config_model",
        default="configs/models/cifar10_resnet34.yml",
    )
    parser.add_argument(
        "--config-detection",
        "--config_detection",
        dest="config_detection",
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-splits", nargs="+", type=int, required=True)
    parser.add_argument("--temperatures", nargs="+")
    parser.add_argument("--magnitudes", nargs="+")
    parser.add_argument("--latent-dir", default=LATENTS_DIR)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--checkpoints-dir", default=CHECKPOINTS_DIR)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = Config(args.config_dataset)
    model_cfg = Config(args.config_model)
    detection_cfg = Config(args.config_detection)

    temps = _parse_float_values(args.temperatures)
    mags = _parse_float_values(args.magnitudes)

    grid = [
        cfg
        for cfg in make_grid(detection_cfg, key="postprocessor_grid")
        if _match_value(float(cfg["temperature"]), temps)
        and _match_value(float(cfg["magnitude"]), mags)
    ]
    if not grid:
        raise ValueError("No configurations matched the requested grid.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed_split in args.seed_splits:
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
            checkpoint_dir=str(Path(args.checkpoints_dir) / model_cfg["preprocessor"]),
        )
        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        _, _, test_loader = prepare_ablation_dataloaders(
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

        seed_dir = out_dir / f"seed-split-{seed_split}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        latent_paths = build_latent_paths(args.latent_dir, data_cfg, model_cfg, detection_cfg)
        test_latent = latent_paths["test"]

        evaluator = AblationDetector(
            model=model,
            dataloader=test_loader,
            device=device,
            suffix="test",
            latent_path=test_latent,
            postprocessor_name=detection_cfg["name"],
            cfg_dataset=data_cfg,
            result_folder=str(seed_dir),
        )

        detectors = [
            get_postprocessor(
                postprocessor_name=detection_cfg["name"],
                model=model,
                cfg=cfg,
                result_folder=str(seed_dir),
                device=device,
            )
            for cfg in grid
        ]

        results = evaluator.evaluate(list_configs=grid, detectors=detectors, suffix="test")
        for frame in results:
            row = frame.iloc[0].to_dict()
            row["seed_split"] = seed_split
            all_rows.append(row)

    output_path = out_dir / "score_temperature_magnitude_test_per_seed.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
