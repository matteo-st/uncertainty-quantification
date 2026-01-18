#!/usr/bin/env bash
set -euo pipefail

# Comprehensive script to run all LDA binning experiments with prerequisites
# This includes: margin grid, odin grid (for msp), and LDA binning with all score combinations

DATA_DIR="${DATA_DIR:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
DATE_TAG=$(date +%Y%m%d)

echo "=============================================="
echo "Starting all LDA binning experiments"
echo "Date: $(date)"
echo "=============================================="

###############################################################################
# STEP 1: Run margin grid search on res (for margin score hyperparams)
###############################################################################
echo ""
echo "=== STEP 1: Margin grid search ==="
MARGIN_TAG="margin-res-grid-nres1000-${DATE_TAG}"

echo "Running margin grid on CIFAR-10 ResNet-34..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_resnet34.yml \
  --config-detection configs/postprocessors/margin/cifar10_resnet34_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${MARGIN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

echo "Running margin grid on CIFAR-10 DenseNet-121..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_densenet121.yml \
  --config-detection configs/postprocessors/margin/cifar10_densenet121_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${MARGIN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

echo "Running margin grid on CIFAR-100 ResNet-34..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_resnet34.yml \
  --config-detection configs/postprocessors/margin/cifar100_resnet34_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${MARGIN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

echo "Running margin grid on CIFAR-100 DenseNet-121..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_densenet121.yml \
  --config-detection configs/postprocessors/margin/cifar100_densenet121_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${MARGIN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

echo "Margin grid search completed!"

###############################################################################
# STEP 2: Run odin grid search on res (for msp score hyperparams)
###############################################################################
echo ""
echo "=== STEP 2: Odin/MSP grid search ==="
ODIN_TAG="odin-res-grid-nres1000-${DATE_TAG}"

echo "Running odin grid on CIFAR-10 ResNet-34..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_resnet34.yml \
  --config-detection configs/postprocessors/msp/cifar10_resnet34_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${ODIN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

echo "Running odin grid on CIFAR-10 DenseNet-121..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_densenet121.yml \
  --config-detection configs/postprocessors/msp/cifar10_densenet121_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${ODIN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

echo "Running odin grid on CIFAR-100 ResNet-34..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_resnet34.yml \
  --config-detection configs/postprocessors/msp/cifar100_resnet34_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${ODIN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

echo "Running odin grid on CIFAR-100 DenseNet-121..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_densenet121.yml \
  --config-detection configs/postprocessors/msp/cifar100_densenet121_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${ODIN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

echo "Odin/MSP grid search completed!"

###############################################################################
# STEP 3: Run LDA binning experiments
###############################################################################
echo ""
echo "=== STEP 3: LDA Binning experiments ==="
LDA_TAG="lda-binning-grid-${DATE_TAG}"

echo "Running LDA binning on CIFAR-10 ResNet-34..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_resnet34.yml \
  --config-detection configs/postprocessors/lda_binning/cifar10_resnet34/lda-binning-grid.yml \
  --mode search \
  --run-tag "${LDA_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --logits-dtype float64 \
  --no-mlflow

echo "Running LDA binning on CIFAR-10 DenseNet-121..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_densenet121.yml \
  --config-detection configs/postprocessors/lda_binning/cifar10_densenet121/lda-binning-grid.yml \
  --mode search \
  --run-tag "${LDA_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --logits-dtype float64 \
  --no-mlflow

echo "Running LDA binning on CIFAR-100 ResNet-34..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_resnet34.yml \
  --config-detection configs/postprocessors/lda_binning/cifar100_resnet34/lda-binning-grid.yml \
  --mode search \
  --run-tag "${LDA_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --logits-dtype float64 \
  --no-mlflow

echo "Running LDA binning on CIFAR-100 DenseNet-121..."
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_densenet121.yml \
  --config-detection configs/postprocessors/lda_binning/cifar100_densenet121/lda-binning-grid.yml \
  --mode search \
  --run-tag "${LDA_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --logits-dtype float64 \
  --no-mlflow

echo ""
echo "=============================================="
echo "All LDA binning experiments completed!"
echo "Date: $(date)"
echo "=============================================="
