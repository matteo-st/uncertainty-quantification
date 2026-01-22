#!/bin/bash
# Supervised Partition Experiments (ImageNet)
# Run from: ~/error_detection/uncertainty-quantification on the server
#
# Tests supervised partition method with different score spaces:
#   - Doctor (gini) 1D
#   - Doctor + Margin (gini-margin) 2D
#
# Usage:
#   ./scripts/run_supervised_partition_experiments.sh gini         # Run 1D Doctor experiments
#   ./scripts/run_supervised_partition_experiments.sh gini-margin  # Run 2D Doctor+Margin experiments
#   ./scripts/run_supervised_partition_experiments.sh all          # Run all experiments
#   ./scripts/run_supervised_partition_experiments.sh status       # Check running jobs
#
# Results will be saved to:
#   results/partition_binning/imagenet/<model>/partition/runs/<run_tag>/

set -e

RUN_TAG_GINI="supervised-partition-gini-20260122"
RUN_TAG_GINI_MARGIN="supervised-partition-gini-margin-20260122"
DATASET_IMAGENET="configs/datasets/imagenet/imagenet_n_res-5000_n-cal-20000.yml"

DATA_DIR="${DATA_DIR:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"

mkdir -p logs

run_gini() {
    echo "=== Running Supervised Partition (1D Doctor) Experiments on ImageNet ==="
    echo "Run tag: ${RUN_TAG_GINI}"
    echo "Starting 2 experiments in parallel..."

    # ImageNet ViT-Tiny16
    echo "[1/2] ImageNet ViT-Tiny16 (supervised-partition gini)..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-tiny16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_tiny16_supervised-partition-gini.yml \
        --eval-grid \
        --run-tag "$RUN_TAG_GINI" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/supervised_partition_gini_vit_tiny16.log 2>&1 &
    PID1=$!
    echo "    PID: $PID1"

    # ImageNet ViT-Base16
    echo "[2/2] ImageNet ViT-Base16 (supervised-partition gini)..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-base16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_base16_supervised-partition-gini.yml \
        --eval-grid \
        --run-tag "$RUN_TAG_GINI" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/supervised_partition_gini_vit_base16.log 2>&1 &
    PID2=$!
    echo "    PID: $PID2"

    echo ""
    echo "Supervised Partition (gini) experiments started! PIDs: $PID1, $PID2"
    echo "Monitor: tail -f logs/supervised_partition_gini_vit_*.log"
}

run_gini_margin() {
    echo "=== Running Supervised Partition (2D Doctor+Margin) Experiments on ImageNet ==="
    echo "Run tag: ${RUN_TAG_GINI_MARGIN}"
    echo "Starting 2 experiments in parallel..."

    # ImageNet ViT-Tiny16
    echo "[1/2] ImageNet ViT-Tiny16 (supervised-partition gini-margin)..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-tiny16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_tiny16_supervised-partition-gini-margin.yml \
        --eval-grid \
        --run-tag "$RUN_TAG_GINI_MARGIN" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/supervised_partition_gini_margin_vit_tiny16.log 2>&1 &
    PID1=$!
    echo "    PID: $PID1"

    # ImageNet ViT-Base16
    echo "[2/2] ImageNet ViT-Base16 (supervised-partition gini-margin)..."
    nohup python -m error_estimation.experiments.run_detection \
        --config-dataset "$DATASET_IMAGENET" \
        --config-model configs/models/imagenet_timm-vit-base16.yml \
        --config-detection configs/postprocessors/partition/imagenet_vit_base16_supervised-partition-gini-margin.yml \
        --eval-grid \
        --run-tag "$RUN_TAG_GINI_MARGIN" \
        --results-family partition_binning \
        --seed-splits 1 2 3 4 5 6 7 8 9 \
        --data-dir "${DATA_DIR}" \
        --checkpoints-dir "${CHECKPOINTS_DIR}" \
        --logits-dtype float64 \
        --no-mlflow \
        > logs/supervised_partition_gini_margin_vit_base16.log 2>&1 &
    PID2=$!
    echo "    PID: $PID2"

    echo ""
    echo "Supervised Partition (gini-margin) experiments started! PIDs: $PID1, $PID2"
    echo "Monitor: tail -f logs/supervised_partition_gini_margin_vit_*.log"
}

check_status() {
    echo "=== Checking running supervised partition processes ==="
    ps aux | grep "[p]ython.*run_detection.*supervised-partition" || echo "No supervised partition processes running"
    echo ""
    echo "=== Recent log activity ==="
    for log in logs/supervised_partition_*.log; do
        if [ -f "$log" ]; then
            echo "--- $log (last 3 lines) ---"
            tail -3 "$log" 2>/dev/null || echo "  (empty or not found)"
        fi
    done
}

case "${1:-}" in
    gini)
        run_gini
        ;;
    gini-margin|margin)
        run_gini_margin
        ;;
    all)
        run_gini
        echo ""
        run_gini_margin
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {gini|gini-margin|all|status}"
        echo ""
        echo "Supervised Partition Experiments (ImageNet)"
        echo ""
        echo "  gini        - Run 1D Doctor experiments"
        echo "  gini-margin - Run 2D Doctor+Margin experiments"
        echo "  all         - Run all experiments"
        echo "  status      - Check running processes"
        echo ""
        echo "Results:"
        echo "  gini:        results/partition_binning/imagenet/<model>/partition/runs/${RUN_TAG_GINI}/"
        echo "  gini-margin: results/partition_binning/imagenet/<model>/partition/runs/${RUN_TAG_GINI_MARGIN}/"
        exit 1
        ;;
esac
