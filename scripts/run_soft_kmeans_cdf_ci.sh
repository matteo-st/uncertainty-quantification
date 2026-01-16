#!/usr/bin/env bash
set -euo pipefail

python scripts/diagnostics/plot_soft_kmeans_cdf_ci.py \
  --config-dataset configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000.yml \
  --config-model configs/models/cifar10_resnet34.yml \
  --config-detection configs/postprocessors/clustering/cifar10_resnet34/soft-kmeans-res-fit-doctor-cdf.yml \
  --seed-split 9 \
  --n-clusters 20 \
  --score mean \
  --alpha 0.05 \
  --init-metric fpr \
  --output-dir results/cifar10/resnet34_ce/partition/runs/soft-kmeans-cdf-grid-nres1000-20260115/seed-split-9/analysis
