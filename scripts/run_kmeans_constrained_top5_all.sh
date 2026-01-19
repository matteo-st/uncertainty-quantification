#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
RUN_TAG="kmeans-constrained-top5-$(date +%Y%m%d)"

# K-means constrained clustering on top-5 softmax predictions
# Uses probits space with reordering to get top-5 probabilities as features

# CIFAR-10 ResNet-34
echo "Running kmeans-constrained-top5 on CIFAR-10 ResNet-34..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_resnet34.yml \
  --config-detection configs/postprocessors/clustering/cifar10_resnet34/kmeans-constrained-top5.yml \
  --eval-grid \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 10 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --results-family partition_binning \
  --logits-dtype float64 \
  --no-mlflow

# CIFAR-10 DenseNet-121
echo "Running kmeans-constrained-top5 on CIFAR-10 DenseNet-121..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_densenet121.yml \
  --config-detection configs/postprocessors/clustering/cifar10_densenet121/kmeans-constrained-top5.yml \
  --eval-grid \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 10 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --results-family partition_binning \
  --logits-dtype float64 \
  --no-mlflow

# CIFAR-100 ResNet-34
echo "Running kmeans-constrained-top5 on CIFAR-100 ResNet-34..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_resnet34.yml \
  --config-detection configs/postprocessors/clustering/cifar100_resnet34/kmeans-constrained-top5.yml \
  --eval-grid \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 10 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --results-family partition_binning \
  --logits-dtype float64 \
  --no-mlflow

# CIFAR-100 DenseNet-121
echo "Running kmeans-constrained-top5 on CIFAR-100 DenseNet-121..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_densenet121.yml \
  --config-detection configs/postprocessors/clustering/cifar100_densenet121/kmeans-constrained-top5.yml \
  --eval-grid \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 10 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --results-family partition_binning \
  --logits-dtype float64 \
  --no-mlflow

echo "All kmeans-constrained-top5 experiments completed!"
