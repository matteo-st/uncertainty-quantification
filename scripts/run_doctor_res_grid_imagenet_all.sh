#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
RUN_TAG="doctor-res-grid-nres5000-$(date +%Y%m%d)"

# ImageNet ViT-Base16
echo "Running Doctor grid search on ImageNet ViT-Base16..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/imagenet/imagenet_n_res-5000_n-cal-20000.yml \
  --config-model configs/models/imagenet_timm-vit-base16.yml \
  --config-detection configs/postprocessors/doctor/imagenet_timm-vit-base16_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

# ImageNet ViT-Tiny16
echo "Running Doctor grid search on ImageNet ViT-Tiny16..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/imagenet/imagenet_n_res-5000_n-cal-20000.yml \
  --config-model configs/models/imagenet_timm-vit-tiny16.yml \
  --config-detection configs/postprocessors/doctor/imagenet_timm-vit-tiny16_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

echo "Doctor grid search completed!"
