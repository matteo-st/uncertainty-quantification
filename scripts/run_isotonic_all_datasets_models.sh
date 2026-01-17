#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
RUN_TAG="isotonic-cal-fit-doctor-allseeds-$(date +%Y%m%d)"

# Requires doctor res-grid runs for each seed (search.jsonl) under results/*/doctor/runs/.

# CIFAR-10 ResNet-34
echo "Running isotonic on CIFAR-10 ResNet-34..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_resnet34.yml \
  --config-detection configs/postprocessors/isotonic/cifar10_resnet34/isotonic-cal-fit-doctor.yml \
  --mode search \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --logits-dtype float64 \
  --no-mlflow

# CIFAR-10 DenseNet-121
echo "Running isotonic on CIFAR-10 DenseNet-121..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_densenet121.yml \
  --config-detection configs/postprocessors/isotonic/cifar10_densenet121/isotonic-cal-fit-doctor.yml \
  --mode search \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --logits-dtype float64 \
  --no-mlflow

# CIFAR-100 ResNet-34
echo "Running isotonic on CIFAR-100 ResNet-34..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_resnet34.yml \
  --config-detection configs/postprocessors/isotonic/cifar100_resnet34/isotonic-cal-fit-doctor.yml \
  --mode search \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --logits-dtype float64 \
  --no-mlflow

# CIFAR-100 DenseNet-121
echo "Running isotonic on CIFAR-100 DenseNet-121..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_densenet121.yml \
  --config-detection configs/postprocessors/isotonic/cifar100_densenet121/isotonic-cal-fit-doctor.yml \
  --mode search \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --logits-dtype float64 \
  --no-mlflow

echo "All isotonic experiments completed!"
