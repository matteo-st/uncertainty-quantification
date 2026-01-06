from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import torch


def set_num_threads(n_threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)


def ensure_dir(path: str | os.PathLike) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def copy_configs(run_dir: str | os.PathLike, config_paths: dict[str, str]) -> dict[str, str]:
    run_path = ensure_dir(run_dir)
    config_dir = ensure_dir(run_path / "configs")
    copied = {}
    for name, src in config_paths.items():
        dst = config_dir / f"{name}.yml"
        shutil.copy2(src, dst)
        copied[name] = str(dst)
    return copied


def write_run_metadata(
    run_dir: str | os.PathLike,
    args: Any,
    config_paths: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> Path:
    run_path = ensure_dir(run_dir)
    payload = {
        "args": vars(args) if hasattr(args, "__dict__") else str(args),
        "configs": config_paths,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    if extra:
        payload.update(extra)
    metadata_path = run_path / "run.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata_path


def build_latent_paths(latent_dir: str, data_cfg: dict, model_cfg: dict, detection_cfg: dict) -> dict[str, str]:
    root_parts = [data_cfg["name"], model_cfg["model_name"]]
    preprocessor = model_cfg.get("preprocessor")
    if preprocessor:
        root_parts.append(preprocessor)
    latent_root = Path(latent_dir) / "_".join(root_parts)
    latent_root.mkdir(parents=True, exist_ok=True)

    def _normalize_transform(value: str | None) -> str:
        return "test" if value is None else str(value)

    def _normalize_epochs(value: int | None) -> str:
        return "1" if value is None else str(value)

    def _path(transform: str | None, n_epochs: int | None) -> str:
        transform_tag = _normalize_transform(transform)
        epochs_tag = _normalize_epochs(n_epochs)
        return str(latent_root / f"transform-{transform_tag}_n-epochs-{epochs_tag}" / "full.pt")

    exp_args = detection_cfg.get("experience_args", {})
    transforms = exp_args.get("transform", {})
    n_epochs = exp_args.get("n_epochs", {})

    return {
        "res": _path(transforms.get("res"), n_epochs.get("res")),
        "cal": _path(transforms.get("cal"), n_epochs.get("cal")),
        "test": _path(transforms.get("test", "test"), n_epochs.get("test", 1)),
    }
