#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
RUN_TAG="margin-res-grid-nres1000-$(date +%Y%m%d)"

# Margin grid search on res split for all datasets/models
# Results will be used by LDA binning to select best hyperparams per score

# CIFAR-10 ResNet-34
echo "Running margin grid on CIFAR-10 ResNet-34..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_resnet34.yml \
  --config-detection configs/postprocessors/margin/cifar10_resnet34_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

# CIFAR-10 DenseNet-121
echo "Running margin grid on CIFAR-10 DenseNet-121..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_densenet121.yml \
  --config-detection configs/postprocessors/margin/cifar10_densenet121_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

# CIFAR-100 ResNet-34
echo "Running margin grid on CIFAR-100 ResNet-34..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_resnet34.yml \
  --config-detection configs/postprocessors/margin/cifar100_resnet34_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

# CIFAR-100 DenseNet-121
echo "Running margin grid on CIFAR-100 DenseNet-121..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_densenet121.yml \
  --config-detection configs/postprocessors/margin/cifar100_densenet121_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

echo "All margin grid experiments completed!"
