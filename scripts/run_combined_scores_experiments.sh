#!/bin/bash
# Combined Scores Binning Experiments
# Run from: ~/error_detection/uncertainty-quantification on the server
#
# Combines Doctor, Margin, and MSP scores into a 3D space and applies
# kmeans_constrained clustering for multi-dimensional binning.
#
# NOTE: This is exploratory - fitting partition on cal (same split as counting)
# is not theoretically justified. This is a first step to see if score combination
# improves performance.
#
# Usage:
#   ./scripts/run_combined_scores_experiments.sh cifar    # Run CIFAR-10/100 experiments
#   ./scripts/run_combined_scores_experiments.sh imagenet # Run ImageNet experiments
#   ./scripts/run_combined_scores_experiments.sh all      # Run all experiments
#   ./scripts/run_combined_scores_experiments.sh status   # Check running jobs
#
# Results will be saved to:
#   results/partition_binning/<dataset>/<model>/partition/runs/<run_tag>/grid_results.csv

set -e

RUN_TAG="combined-scores-kmeans-20260121"
DATASET_CIFAR10="configs/datasets/cifar10/cifar10_n_res-1000_n-cal-4000_all-seeds.yml"
DATASET_CIFAR100="configs/datasets/cifar100/cifar100_n_res-1000_n-cal-4000_all-seeds.yml"
DATASET_IMAGENET="configs/datasets/imagenet/imagenet_n_res-5000_n-cal-20000.yml"

DATA_DIR="${DATA_DIR:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"

# Ensure logs directory exists
mkdir -p logs

run_cifar() {
    echo "=== Running Combined Scores Experiments on CIFAR ==="
    echo "Grid: 5 n_clusters × 3 alphas × 2 scores = 30 combinations"
    echo "Results: results/partition_binning/<dataset>/<model>/partition/runs/${RUN_TAG}/"
    echo "Starting 4 experiments in parallel..."

    # CIFAR-10 ResNet-34
    echo "[1/4] CIFAR-10 ResNet-34..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_CIFAR10" \
        --config-model configs/models/cifar10_resnet34.yml \
        --config-detection configs/postprocessors/partition/cifar10_resnet34_combined-kmeans.yml \
        --eval-grid \
        --run-tag "$RUN_TAG" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/combined_cifar10_resnet34.log 2>&1 &
    PID1=$!
    echo "    PID: $PID1"

    # CIFAR-10 DenseNet-121
    echo "[2/4] CIFAR-10 DenseNet-121..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_CIFAR10" \
        --config-model configs/models/cifar10_densenet121.yml \
        --config-detection configs/postprocessors/partition/cifar10_densenet121_combined-kmeans.yml \
        --eval-grid \
        --run-tag "$RUN_TAG" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/combined_cifar10_densenet121.log 2>&1 &
    PID2=$!
    echo "    PID: $PID2"

    # CIFAR-100 ResNet-34
    echo "[3/4] CIFAR-100 ResNet-34..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_CIFAR100" \
        --config-model configs/models/cifar100_resnet34.yml \
        --config-detection configs/postprocessors/partition/cifar100_resnet34_combined-kmeans.yml \
        --eval-grid \
        --run-tag "$RUN_TAG" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/combined_cifar100_resnet34.log 2>&1 &
    PID3=$!
    echo "    PID: $PID3"

    # CIFAR-100 DenseNet-121
    echo "[4/4] CIFAR-100 DenseNet-121..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_CIFAR100" \
        --config-model configs/models/cifar100_densenet121.yml \
        --config-detection configs/postprocessors/partition/cifar100_densenet121_combined-kmeans.yml \
        --eval-grid \
        --run-tag "$RUN_TAG" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/combined_cifar100_densenet121.log 2>&1 &
    PID4=$!
    echo "    PID: $PID4"

    echo ""
    echo "All CIFAR experiments started!"
    echo "PIDs: $PID1, $PID2, $PID3, $PID4"
    echo ""
    echo "Monitor with:"
    echo "  tail -f logs/combined_cifar10_resnet34.log"
    echo "  tail -f logs/combined_cifar10_densenet121.log"
    echo "  tail -f logs/combined_cifar100_resnet34.log"
    echo "  tail -f logs/combined_cifar100_densenet121.log"
}

run_imagenet() {
    echo "=== Running Combined Scores Experiments on ImageNet ==="
    echo "Grid: 5 n_clusters × 3 alphas × 2 scores = 30 combinations"
    echo "Results: results/partition_binning/imagenet/<model>/partition/runs/${RUN_TAG}/"
    echo "Starting 2 experiments in parallel..."

    # ImageNet ViT-Tiny16
    echo "[1/2] ImageNet ViT-Tiny16..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-tiny16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_tiny16_combined-kmeans.yml \
        --eval-grid \
        --run-tag "$RUN_TAG" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/combined_imagenet_vit_tiny16.log 2>&1 &
    PID1=$!
    echo "    PID: $PID1"

    # ImageNet ViT-Base16
    echo "[2/2] ImageNet ViT-Base16..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-base16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_base16_combined-kmeans.yml \
        --eval-grid \
        --run-tag "$RUN_TAG" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/combined_imagenet_vit_base16.log 2>&1 &
    PID2=$!
    echo "    PID: $PID2"

    echo ""
    echo "All ImageNet experiments started!"
    echo "PIDs: $PID1, $PID2"
    echo ""
    echo "Monitor with:"
    echo "  tail -f logs/combined_imagenet_vit_tiny16.log"
    echo "  tail -f logs/combined_imagenet_vit_base16.log"
}

check_status() {
    echo "=== Checking running combined scores processes ==="
    ps aux | grep "[p]ython.*run_detection.*combined" || echo "No combined scores processes running"
    echo ""
    echo "=== Recent log activity ==="
    for log in logs/combined*.log; do
        if [ -f "$log" ]; then
            echo "--- $log (last 3 lines) ---"
            tail -3 "$log" 2>/dev/null || echo "  (empty or not found)"
        fi
    done
}

case "${1:-}" in
    cifar)
        run_cifar
        ;;
    imagenet)
        run_imagenet
        ;;
    all)
        run_cifar
        echo ""
        run_imagenet
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {cifar|imagenet|all|status}"
        echo ""
        echo "Combined Scores Binning Experiment"
        echo "Combines Doctor, Margin, and MSP scores into a 3D space"
        echo "and applies kmeans_constrained clustering."
        echo ""
        echo "  cifar    - Run CIFAR-10 and CIFAR-100 experiments"
        echo "  imagenet - Run ImageNet experiments"
        echo "  all      - Run all experiments"
        echo "  status   - Check running processes and log activity"
        echo ""
        echo "Results: results/partition_binning/<dataset>/<model>/partition/runs/${RUN_TAG}/"
        exit 1
        ;;
esac
