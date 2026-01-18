#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
RUN_TAG="kmeans-constrained-doctor-$(date +%Y%m%d)"

# K-means constrained binning on top of doctor (gini) scores
# Uses per-seed doctor hyperparameters from old_results/

# CIFAR-10 ResNet-34
echo "Running kmeans-constrained on CIFAR-10 ResNet-34..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_resnet34.yml \
  --config-detection configs/postprocessors/clustering/cifar10_resnet34/kmeans-constrained-doctor.yml \
  --eval-grid \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --results-family partition_binning \
  --logits-dtype float64 \
  --no-mlflow

# CIFAR-10 DenseNet-121
echo "Running kmeans-constrained on CIFAR-10 DenseNet-121..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_densenet121.yml \
  --config-detection configs/postprocessors/clustering/cifar10_densenet121/kmeans-constrained-doctor.yml \
  --eval-grid \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --results-family partition_binning \
  --logits-dtype float64 \
  --no-mlflow

# CIFAR-100 ResNet-34
echo "Running kmeans-constrained on CIFAR-100 ResNet-34..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_resnet34.yml \
  --config-detection configs/postprocessors/clustering/cifar100_resnet34/kmeans-constrained-doctor.yml \
  --eval-grid \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --results-family partition_binning \
  --logits-dtype float64 \
  --no-mlflow

# CIFAR-100 DenseNet-121
echo "Running kmeans-constrained on CIFAR-100 DenseNet-121..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_densenet121.yml \
  --config-detection configs/postprocessors/clustering/cifar100_densenet121/kmeans-constrained-doctor.yml \
  --eval-grid \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --results-family partition_binning \
  --logits-dtype float64 \
  --no-mlflow

echo "All kmeans-constrained experiments completed!"
