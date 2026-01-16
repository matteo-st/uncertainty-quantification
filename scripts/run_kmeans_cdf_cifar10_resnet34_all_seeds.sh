#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
RUN_TAG="kmeans-cdf-grid-nres1000-cifar10-resnet34-allseeds-$(date +%Y%m%d)"

# Requires doctor res-grid runs for each seed (search.jsonl) under results/cifar10/resnet34_ce/doctor/runs/.
python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_resnet34.yml \
  --config-detection configs/postprocessors/clustering/cifar10_resnet34/kmeans-res-fit-doctor-cdf-allseeds.yml \
  --eval-grid \
  --run-tag "${RUN_TAG}" \
  --seed-splits 1 2 3 4 5 6 7 8 9 \
  --data-dir "${DATA_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --no-mlflow
