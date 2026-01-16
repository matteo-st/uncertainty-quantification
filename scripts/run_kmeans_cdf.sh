#!/usr/bin/env bash
set -euo pipefail

python -m error_estimation.experiments.run_detection \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_resnet34.yml \
  --config-detection configs/postprocessors/clustering/cifar10_resnet34/kmeans-res-fit-doctor-cdf.yml \
  --seed-splits 9 \
  --mode search \
  --eval-grid \
  --run-tag kmeans-cdf-grid-nres1000-20260116
