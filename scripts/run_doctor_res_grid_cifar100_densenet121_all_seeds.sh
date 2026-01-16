#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
RUN_TAG="doctor-res-grid-nres1000-cifar100-densenet121-allseeds-$(date +%Y%m%d)"
EVAL_TAG="doctor-eval-grid-nres1000-cifar100-densenet121-allseeds-$(date +%Y%m%d)"

python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_densenet121.yml \
  --config-detection configs/postprocessors/doctor/cifar100_densenet121_hyperparams_search.yml \
  --mode search_res \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow

# Save cal/test metrics for all grid configurations (doctor scores).
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar100_densenet121.yml \
  --config-detection configs/postprocessors/doctor/cifar100_densenet121_hyperparams_search.yml \
  --eval-grid \
  --run-tag "${EVAL_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow
