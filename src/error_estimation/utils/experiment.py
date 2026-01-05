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


def build_latent_paths(latent_dir: str, data_cfg: dict, detection_cfg: dict, seed_split: int) -> dict[str, str]:
    latent_root = Path(latent_dir) / f"seed-split-{seed_split}"
    latent_root.mkdir(parents=True, exist_ok=True)
    return {
        "res": str(
            latent_root
            / (
                f"res_n-samples-{data_cfg['n_samples']['res']}"
                f"_transform-{detection_cfg['experience_args']['transform']['res']}"
                f"_n-epochs-{detection_cfg['experience_args']['n_epochs']['res']}.pt"
            )
        ),
        "cal": str(
            latent_root
            / (
                f"cal_n-samples-{data_cfg['n_samples']['cal']}"
                f"_transform-{detection_cfg['experience_args']['transform']['cal']}"
                f"_n-epochs-{detection_cfg['experience_args']['n_epochs']['cal']}.pt"
            )
        ),
        "test": str(latent_root / f"test_n-samples-{data_cfg['n_samples']['test']}.pt"),
    }
